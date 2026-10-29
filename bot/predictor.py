"""Предсказатель для бота: ответы пользователя -> оценка риска, решение, причины, предупреждения.

Не зависит от Telegram, поэтому проверяется обычными тестами. Конфигурация лежит в models/:
decision_config_bot.json (порог, порог показа причин, пути), features_bot.json, bot_input_ranges.json.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.bot_features import NO_EMPLOYMENT_TYPES, RARE_INCOME_TYPES, answers_to_features, rare_income_message
from src.explain import GROUPS, group_shap, top_reasons
from src.models import to_lgbm

ROOT = Path(__file__).resolve().parents[1]

LABELS = {
    "AMT_INCOME_TOTAL": "доход", "AMT_CREDIT": "сумма кредита", "AMT_ANNUITY": "платёж",
    "AMT_GOODS_PRICE": "стоимость товара", "EMPLOYMENT_YEARS": "стаж, лет", "OWN_CAR_AGE": "возраст автомобиля, лет",
    "CREDIT_INCOME_RATIO": "отношение кредита к доходу", "ANNUITY_INCOME_RATIO": "отношение платежа к доходу",
    "CREDIT_TERM": "отношение платежа к сумме кредита", "CREDIT_GOODS_RATIO": "отношение кредита к стоимости товара",
    "BUREAU_COUNT": "число кредитов в других банках", "BUREAU_IS_ACTIVE_SUM": "число активных кредитов",
    "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": "задолженность в других банках", "BUREAU_LAST1Y_COUNT": "кредитов за последний год",
}
# малая нагрузка не делает оценку ненадёжной: для этих величин проверяем только верхнюю границу
# стаж 0 лет (человек только что устроился) нормален, нижняя граница для него не нужна
UPPER_ONLY = {"CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "EMPLOYMENT_YEARS"}
# подсказки пользователю при критичном отклонении: что именно проверить в ответах
HINTS = {
    "AMT_INCOME_TOTAL": "доход нужно вводить в тех же единицах и за тот же период, что и платёж (типичное значение в данных около 150 000)",
    "CREDIT_INCOME_RATIO": "проверьте сумму кредита и доход: единицы и период должны совпадать с данными",
    "ANNUITY_INCOME_RATIO": "доход и платёж должны быть указаны за один и тот же период (например, оба за месяц)",
    "CREDIT_TERM": "платёж за один период (месяц) обычно составляет 3–12% от суммы кредита, проверьте сумму и платёж",
    "CREDIT_GOODS_RATIO": "проверьте сумму кредита и стоимость товара",
}
# значения дальше чем в HARD_FACTOR раз от границы диапазона не оцениваем вовсе
HARD_FACTOR = 3.0
# нагрузка намного ниже обычной (в LOW_SOFT_FACTOR раз ниже границы) чаще всего значит, что доход и платёж
# введены за разные периоды или в разных единицах: предупреждаем, а не молча занижаем риск
LOW_SOFT = {"CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"}
LOW_SOFT_FACTOR = 0.5
DEFAULT_BAND_EDGES = (0.04, 0.10)  # низкий < 4% <= средний < 10% <= повышенный < порог отказа

DISCLAIMER = (
    "Это учебный проект. Результат не является кредитным решением или финансовой рекомендацией. "
    "Модель обучена на данных Home Credit и по вашим ответам даёт лишь приблизительную оценку. "
    "Ответы не сохраняются."
)


class OutOfRangeError(ValueError):
    """Ответы слишком далеки от данных, на которых училась модель."""


@dataclass
class Assessment:
    probability: float
    band: str
    refused: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _fmt(v: float) -> str:
    return f"{v:,.0f}".replace(",", " ") if abs(v) >= 1000 else f"{v:.3g}"


class Predictor:
    def __init__(self, root: Path = ROOT, model=None):
        root = Path(root)
        self.cfg = json.loads((root / "models" / "decision_config_bot.json").read_text())
        self.features = json.loads((root / self.cfg["features"]).read_text())
        self.limits = json.loads((root / self.cfg["input_ranges"]).read_text())
        self.model = model if model is not None else joblib.load(root / self.cfg["model"])
        if hasattr(self.model, "set_params"):  # один клиент за раз: потоки OpenMP только добавляют накладные расходы
            try:
                self.model.set_params(n_jobs=1)
            except Exception:
                pass
        self.threshold = float(self.cfg["threshold"])
        self.min_shap = float(self.cfg["min_reason_shap"])
        self.low, self.mid = self.cfg.get("risk_bands", DEFAULT_BAND_EDGES)
        if not (0 < self.low < self.mid < self.threshold):
            raise ValueError("Границы категорий риска должны удовлетворять 0 < низкий < средний < порог отказа")

    # --- категория риска ---
    def band(self, p: float) -> str:
        if p >= self.threshold:
            return "высокий"
        if p >= self.mid:
            return "повышенный"
        if p >= self.low:
            return "средний"
        return "низкий"

    # --- диапазоны входов ---
    def range_issues(self, full: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
        """(предупреждения, критичные отклонения, имена критичных величин). full: признаки из answers_to_features."""
        row = full.iloc[0].to_dict()
        row["EMPLOYMENT_YEARS"] = -row["DAYS_EMPLOYED"] / 365 if pd.notna(row["DAYS_EMPLOYED"]) else np.nan
        soft, hard, hard_names = [], [], []
        for name, lim in self.limits.items():
            v = row.get(name)
            if v is None or pd.isna(v):
                continue
            lo, hi = lim["min"], lim["max"]
            if name in LOW_SOFT and v < lo * LOW_SOFT_FACTOR:
                soft.append(f"{LABELS.get(name, name)}: {_fmt(v)} намного ниже обычного (от {_fmt(lo)}); проверьте, что доход "
                            "и платёж указаны за один и тот же период и в одних и тех же единицах")
                continue
            too_high = v > hi
            too_low = v < lo and name not in UPPER_ONLY
            if not (too_high or too_low):
                continue
            msg = f"{LABELS.get(name, name)}: {_fmt(v)} вне обычного диапазона ({_fmt(lo)} – {_fmt(hi)})"
            critical = (too_high and v > hi * HARD_FACTOR) or (too_low and lo > 0 and v < lo / HARD_FACTOR)
            (hard if critical else soft).append(msg)
            if critical:
                hard_names.append(name)
        return soft, hard, hard_names

    # --- основная оценка ---
    def assess(self, answers: dict) -> Assessment:
        if answers.get("income_type") in RARE_INCOME_TYPES:
            raise OutOfRangeError(rare_income_message(answers["income_type"]))
        full = answers_to_features(answers)  # ValueError при неверном вводе
        soft, hard, hard_names = self.range_issues(full)
        if hard:
            text = ("Некоторые ответы слишком далеки от данных, на которых училась модель, оценка была бы ненадёжной: "
                    + "; ".join(hard))
            hints = list(dict.fromkeys(HINTS[n] for n in hard_names if n in HINTS))
            if hints:
                text += ".\n\nПодсказка: " + "; ".join(hints) + "."
            raise OutOfRangeError(text)
        X = to_lgbm(full[self.features])
        p = float(self.model.predict_proba(X)[:, 1][0])
        refused = p >= self.threshold

        reasons: list[str] = []
        if refused:  # причины показываем только при отказе
            contrib = self.model.booster_.predict(X, pred_contrib=True)[:, :-1]
            found = top_reasons(group_shap(contrib, X.columns), k=3, min_shap=self.min_shap)[0]
            reasons = [text for text, _ in found]
            if answers["income_type"] in NO_EMPLOYMENT_TYPES:  # «небольшой стаж» не подходит неработающему
                reasons = ["Отсутствие трудового стажа" if t == GROUPS["employment_tenure"] else t for t in reasons]
        return Assessment(p, self.band(p), refused, reasons, soft)


def percent_text(p: float) -> str:
    return "менее 1%" if p < 0.01 else f"около {round(p * 100)}%"


def format_message(a: Assessment) -> str:
    lines = ["Ориентировочная оценка", f"Риск невозврата: {a.band} ({percent_text(a.probability)})."]
    if a.refused:
        lines.append("Модель отказала бы в кредите.")
        if a.reasons:
            lines.append("Факторы, сильнее всего повысившие оценку риска:")
            lines += [f"{i}. {r}" for i, r in enumerate(a.reasons, 1)]
    else:
        text = "Модель одобрила бы кредит."
        if a.band == "повышенный":
            text += " Риск повышенный, решение близко к границе."
        lines.append(text)
    if a.warnings:
        lines.append("\nВнимание: часть ответов вне обычного диапазона данных, оценка может быть неточной:")
        lines += [f"- {w}" for w in a.warnings]
    lines.append("\n" + DISCLAIMER)
    return "\n".join(lines)
