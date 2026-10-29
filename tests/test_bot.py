"""Тесты бота без Telegram и без LightGBM: модель подменяется заглушкой.

Запуск: python -m pytest tests -q   (или python tests/test_bot.py)
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bot.cli import run as run_cli
from bot.dialog import Dialog, parse_number
from bot.predictor import OutOfRangeError, Predictor, format_message
from src.bot_features import BOT_FEATURES, answers_to_features
from src.explain import GROUPS

ABS = ["AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE"]
FEATURES = [c for c in BOT_FEATURES if c not in ABS]
LIMITS = {
    "AMT_INCOME_TOTAL": (45_000, 630_000), "AMT_CREDIT": (62_361, 2_013_840), "AMT_ANNUITY": (5_274, 79_758),
    "AMT_GOODS_PRICE": (54_000, 1_800_000), "EMPLOYMENT_YEARS": (0.266, 34.3), "OWN_CAR_AGE": (0, 65),
    "CREDIT_INCOME_RATIO": (0.485, 14.84), "ANNUITY_INCOME_RATIO": (0.033, 0.552), "CREDIT_TERM": (0.026, 0.119),
    "CREDIT_GOODS_RATIO": (1.0, 1.528), "BUREAU_COUNT": (0, 22), "BUREAU_IS_ACTIVE_SUM": (0, 9),
    "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": (0, 8_841_679), "BUREAU_LAST1Y_COUNT": (0, 7),
}
BASE = {"income": 150_000, "credit": 400_000, "annuity": 20_000, "goods_price": 400_000,
        "income_type": "Working", "education": "Higher education", "employment_years": 5,
        "own_car": "N", "car_age": None, "own_realty": "Y", "contract_type": "Cash loans",
        "bureau_count": 2, "bureau_active": 1, "bureau_debt": 60_000, "bureau_overdue": False, "bureau_last_year": 0}


# --- заглушка модели -------------------------------------------------------------------
class FakeBooster:
    def __init__(self, contrib):
        self.contrib = contrib

    def predict(self, X, pred_contrib=True, **kwargs):
        out = np.zeros((len(X), X.shape[1] + 1))
        for j, c in enumerate(X.columns):
            out[:, j] = self.contrib.get(c, 0.0)
        return out


class FakeModel:
    def __init__(self, p, contrib=None):
        self.p = p
        self.booster_ = FakeBooster(contrib or {})

    def predict_proba(self, X):
        p = np.full(len(X), self.p)
        return np.c_[1 - p, p]


def make_predictor(p, contrib=None):
    root = Path(tempfile.mkdtemp())
    (root / "models").mkdir()
    cfg = {"threshold": 1 / 6, "min_reason_shap": 0.10, "features": "models/features_bot.json",
           "input_ranges": "models/bot_input_ranges.json", "model": "models/lgbm_bot.joblib"}
    (root / "models" / "decision_config_bot.json").write_text(json.dumps(cfg))
    (root / "models" / "features_bot.json").write_text(json.dumps(FEATURES))
    (root / "models" / "bot_input_ranges.json").write_text(
        json.dumps({k: {"min": lo, "max": hi} for k, (lo, hi) in LIMITS.items()}))
    return Predictor(root=root, model=FakeModel(p, contrib))


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return True
    return False


# --- разбор чисел ------------------------------------------------------------------------
def test_parse_number():
    assert parse_number("150000") == 150_000
    assert parse_number("150 000") == 150_000
    assert parse_number("150k") == 150_000
    assert parse_number("1,5 млн") == 1_500_000
    assert parse_number("0.5") == 0.5
    assert raises(ValueError, parse_number, "много")
    assert raises(ValueError, parse_number, "-5")


# --- диалог --------------------------------------------------------------------------------
def run(dialog, replies):
    for r in replies:
        assert dialog.current() is not None
        assert dialog.submit(r) is None, r
    return dialog


def test_full_flow_and_answers_to_features():
    d = run(Dialog(), ["1", "150000", "400000", "20000", "400000", "1", "5", "4", "нет", "да",
                       "3", "1", "60000", "нет", "0"])
    assert d.current() is None
    a = d.answers
    assert a["contract_type"] == "Cash loans" and a["income_type"] == "Working" and a["own_car"] == "N"
    assert a["car_age"] is None and a["bureau_overdue"] is False and a["bureau_count"] == 3
    X = answers_to_features(a)
    assert list(X.columns) == BOT_FEATURES and len(X) == 1


def test_pensioner_skips_tenure():
    d = Dialog()
    run(d, ["1", "100000", "200000", "10000", "-", "5"])  # 5 = пенсионер (порядок в INCOME_TYPES)
    q = d.current()
    assert q.key == "education" and d.answers["employment_years"] is None


def test_no_car_skips_age_and_no_history_skips_followups():
    d = Dialog()
    run(d, ["1", "100000", "200000", "10000", "-", "1", "3", "1", "нет", "нет", "0"])
    assert d.current() is None
    a = d.answers
    assert a["car_age"] is None and a["goods_price"] is None
    assert a["bureau_active"] == 0 and a["bureau_debt"] == 0 and a["bureau_overdue"] is False


def test_validation_errors():
    d = Dialog()
    assert d.submit("9") is not None            # нет такого варианта
    assert d.submit("1") is None
    assert d.submit("ноль") is not None         # не число
    assert d.submit("0") is not None            # доход должен быть положительным
    run(d, ["100000", "200000", "10000", "-", "1", "5", "1", "да", "5", "нет", "2"])
    assert d.current().key == "bureau_active"
    assert d.submit("3") is not None            # активных не может быть больше, чем всего (2)
    assert d.submit("1.5") is not None          # нужно целое
    assert d.submit("1") is None


# --- признаки бота --------------------------------------------------------------------------
def test_answers_to_features_formulas():
    X = answers_to_features(BASE).iloc[0]
    assert abs(X["CREDIT_INCOME_RATIO"] - 400_000 / 150_000) < 1e-9
    assert abs(X["ANNUITY_INCOME_RATIO"] - 20_000 / 150_000) < 1e-9
    assert abs(X["CREDIT_TERM"] - 0.05) < 1e-9 and abs(X["CREDIT_GOODS_RATIO"] - 1.0) < 1e-9
    assert X["DAYS_EMPLOYED"] == -5 * 365 and X["DAYS_EMPLOYED_ANOM"] == 0 and X["HAS_BUREAU"] == 1


def test_pensioner_and_no_history_features():
    X = answers_to_features({**BASE, "income_type": "Pensioner", "employment_years": None, "bureau_count": 0}).iloc[0]
    assert np.isnan(X["DAYS_EMPLOYED"]) and X["DAYS_EMPLOYED_ANOM"] == 1
    assert X["BUREAU_COUNT"] == 0 and X["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"] == 0 and X["HAS_BUREAU"] == 0


def test_invalid_answers_rejected():
    assert raises(ValueError, answers_to_features, {**BASE, "income": 0})
    assert raises(ValueError, answers_to_features, {**BASE, "income_type": "Foo"})


# --- предсказатель -------------------------------------------------------------------------
def test_bands():
    pr = make_predictor(0.05)
    assert [pr.band(p) for p in (0.01, 0.04, 0.09, 0.10, 0.15, 0.17, 0.5)] == \
        ["низкий", "средний", "средний", "повышенный", "повышенный", "высокий", "высокий"]


def test_reasons_only_on_refusal():
    contrib = {"BUREAU_HAS_OVERDUE_SUM": 0.5, "BUREAU_LAST1Y_COUNT": 0.4, "ANNUITY_INCOME_RATIO": 0.3,
               "DAYS_EMPLOYED": 0.05, "CREDIT_TERM": -0.2}
    ok = make_predictor(0.08, contrib).assess(BASE)
    assert not ok.refused and ok.reasons == []

    bad = make_predictor(0.30, contrib).assess({**BASE, "bureau_overdue": True})
    assert bad.refused and bad.band == "высокий"
    assert bad.reasons == [GROUPS["bureau_overdue"], GROUPS["bureau_recent"], GROUPS["loan_burden"]]
    assert GROUPS["employment_tenure"] not in bad.reasons  # вклад 0.05 ниже порога показа 0.10


def test_pensioner_tenure_reason_is_renamed():
    pr = make_predictor(0.30, {"DAYS_EMPLOYED": 0.8})
    a = pr.assess({**BASE, "income_type": "Pensioner", "employment_years": None})
    assert a.reasons == ["Отсутствие трудового стажа"]


def test_rare_income_types_are_not_assessed():
    pr = make_predictor(0.05)
    for t in ("Unemployed", "Student", "Businessman", "Maternity leave"):
        assert raises(OutOfRangeError, pr.assess, {**BASE, "income_type": t, "employment_years": None})
    d = Dialog()
    run(d, ["1", "150000", "400000", "20000", "-"])
    assert d.submit("8") is None  # «Не работаю»
    assert d.current() is None and "слишком редко" in d.stop_reason


def test_range_warnings_and_hard_limit():
    pr = make_predictor(0.05)
    assert pr.assess(BASE).warnings == []
    # малая нагрузка (платёж/доход ниже границы) не даёт предупреждения
    assert pr.assess({**BASE, "income": 300_000, "credit": 200_000, "annuity": 8_000, "goods_price": 200_000}).warnings == []
    # доход ниже границы: предупреждение, но оценка выдаётся
    low = pr.assess({**BASE, "income": 30_000, "credit": 100_000, "annuity": 5_500, "goods_price": 100_000})
    assert any(w.startswith("доход:") for w in low.warnings)
    # слишком далеко от данных: оценки нет
    assert raises(OutOfRangeError, pr.assess, {**BASE, "income": 5_000_000, "credit": 400_000})


def test_zero_tenure_is_allowed():
    pr = make_predictor(0.05)
    a = pr.assess({**BASE, "employment_years": 0})
    assert a.warnings == []  # только что устроился: нормальная ситуация, предупреждения нет
    assert raises(OutOfRangeError, pr.assess, {**BASE, "employment_years": 200})


def test_hard_error_contains_hint():
    pr = make_predictor(0.05)
    try:
        pr.assess({**BASE, "annuity": 100})  # платёж 100 при кредите 400000
    except OutOfRangeError as e:
        text = str(e)
        assert "Подсказка" in text and "3–12%" in text
    else:
        raise AssertionError("ожидалась ошибка диапазона")


def test_mixed_income_period_is_flagged():
    pr = make_predictor(0.05)
    # доход введён за год (x12), платёж за месяц: нагрузка выглядит в 12 раз меньше, чем есть на самом деле
    a = pr.assess({**BASE, "income": 1_800_000})
    assert any("за один и тот же период" in w for w in a.warnings)
    assert any(w.startswith("доход:") and "вне обычного" in w for w in a.warnings)


def test_message_has_disclaimer_and_no_advice():
    pr = make_predictor(0.30, {"BUREAU_HAS_OVERDUE_SUM": 0.5})
    text = format_message(pr.assess({**BASE, "bureau_overdue": True}))
    assert "Ориентировочная оценка" in text and "учебный проект" in text.lower()
    assert "Модель отказала бы" in text and "около 30%" in text
    assert "рекомендуем" not in text.lower()
    text_ok = format_message(make_predictor(0.12).assess(BASE))
    assert "Модель одобрила бы" in text_ok and "повышенный" in text_ok and "близко к границе" in text_ok


def test_cli_end_to_end_with_scripted_input():
    replies = iter(["1", "150000", "400000", "20000", "400000", "1", "5", "4", "2", "1", "3", "1", "60000", "2", "0"])
    out = []
    run_cli(make_predictor(0.30, {"BUREAU_HAS_OVERDUE_SUM": 0.5}), input_fn=lambda _: next(replies), print_fn=out.append)
    text = "\n".join(out)
    assert "Ориентировочная оценка" in text and "Модель отказала бы" in text and "учебный проект" in text.lower()


if __name__ == "__main__":
    import traceback

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print("ok  ", name)
        except Exception:
            failed.append(name)
            print("FAIL", name)
            traceback.print_exc()
    print(f"\nпройдено {len(tests) - len(failed)} из {len(tests)}")
    if failed:
        print("Не прошли:", ", ".join(failed))
        sys.exit(1)
