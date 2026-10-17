"""Причины отказа: группировка признаков и выбор топ-k причин по SHAP.

Идея: вклады SHAP аддитивны, поэтому вклад нескольких похожих признаков
(например, всех *REFUSED* из previous_application) можно просто сложить и показать
клиенту одной понятной причиной. Признаки из EXCLUDED_GROUPS в причины не попадают.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

# id группы -> формулировка причины отказа (показываем клиенту)
GROUPS: dict[str, str] = {
    "ext_rating": "Низкий внешний кредитный рейтинг",
    "bureau_overdue": "Просрочки по кредитам в других банках",
    "bureau_debt": "Высокая задолженность по кредитам в других банках",
    "bureau_recent": "Много недавно открытых кредитов в других банках",
    "bureau_active": "Большое число или сумма активных кредитов в других банках",
    "bureau_volume": "Небольшой общий объём кредитов в других банках",
    "bureau_terms": "Сроки действующих кредитов в других банках",
    "bureau_other": "Особенности кредитной истории в других банках",
    "prev_refused": "Частые отказы по предыдущим заявкам",
    "prev_count": "Много заявок на кредит за короткое время",
    "prev_terms": "Длительные сроки по прошлым кредитам",
    "prev_history": "Короткая или недавняя история в Home Credit",
    "prev_payments": "Небольшие платежи и первоначальные взносы по прошлым кредитам",
    "prev_closing": "Сроки закрытия прошлых кредитов",
    "prev_amounts": "Суммы и условия одобрения прошлых заявок",
    "prev_other": "Прочие особенности прошлых заявок",
    "inst_underpaid": "Недоплаты по платежам в прошлых кредитах",
    "inst_late": "Просрочки платежей в прошлых кредитах",
    "enquiries": "Много запросов кредитной истории",
    "no_history": "Недостаточная кредитная история",
    "loan_burden": "Высокий платёж и сумма кредита относительно дохода",
    "credit_params": "Параметры запрашиваемого кредита (срок, сумма относительно стоимости товара)",
    "employment": "Стаж и характер занятости",
    "education": "Уровень образования",
    "housing": "Характеристики жилья",
    "assets": "Отсутствие имущества (авто, недвижимость)",
    "contacts": "Недавняя смена документа, телефона или регистрации",
    "contract": "Тип запрашиваемого кредита",
    # --- группы ниже клиенту не показываем ---
    "protected": "[защищённые признаки: возраст, пол, семья]",
    "region": "[регион: возможный прокси защищённых признаков]",
    "social": "[кредитная история окружения]",
    "technical": "[технические признаки заявки]",
    "other": "[не сопоставлено]",
}

EXCLUDED_GROUPS: set[str] = {"protected", "region", "social", "technical", "other"}

# Минимальный вклад SHAP (лог-шансы), чтобы причину показывать клиенту: 0.1 примерно +10% к шансам дефолта
MIN_REASON_SHAP: float = 0.10

# Порядок важен: берётся первое совпадение.
RULES: list[tuple[str, str]] = [
    # EMPLOYED_BIRTH_RATIO = стаж / возраст, поэтому несёт возраст
    (r"^(AGE|DAYS_BIRTH|CODE_GENDER|NAME_FAMILY_STATUS|CNT_CHILDREN|CNT_FAM_MEMBERS|INCOME_PER_PERSON|EMPLOYED_BIRTH_RATIO)$", "protected"),
    (r"^(REGION_|REG_|LIVE_)", "region"),
    (r"_SOCIAL_CIRCLE$", "social"),
    (r"^(WEEKDAY_APPR_PROCESS_START|HOUR_APPR_PROCESS_START|NAME_TYPE_SUITE|FLAG_DOCUMENT_\d+)$", "technical"),
    (r"^EXT_SOURCE", "ext_rating"),
    (r"^HAS_(BUREAU|PREV|INST)$", "no_history"),
    (r"^BUREAU_.*OVERDUE", "bureau_overdue"),
    (r"^BUREAU_.*DEBT", "bureau_debt"),
    # общие суммы кредитов действуют в обратную сторону, чем суммы активных: больше история, ниже риск
    (r"^BUREAU_AMT_CREDIT_SUM_(SUM|MEAN|MAX)$", "bureau_volume"),
    (r"^BUREAU_DAYS_CREDIT_ENDDATE", "bureau_terms"),
    (r"^BUREAU_(LAST1Y_|DAYS_CREDIT_(MIN|MAX|MEAN)$)", "bureau_recent"),
    (r"^BUREAU_(ACTIVE_|IS_ACTIVE|DAYS_CREDIT_UPDATE|CNT_CREDIT_PROLONG)", "bureau_active"),
    (r"^BUREAU_", "bureau_other"),
    (r"^PREV_.*REFUSED", "prev_refused"),
    (r"^PREV_(LAST1Y_COUNT|COUNT)$", "prev_count"),
    (r"^PREV_.*CNT_PAYMENT", "prev_terms"),
    (r"^PREV_.*DAYS_DECISION", "prev_history"),
    (r"^PREV_.*(AMT_ANNUITY|AMT_DOWN_PAYMENT|RATE_DOWN_PAYMENT)", "prev_payments"),
    (r"^PREV_.*(DAYS_LAST_DUE|DAYS_TERMINATION|DAYS_FIRST)", "prev_closing"),
    (r"^PREV_.*(AMT_|APP_CREDIT_RATIO|ANNUITY_CREDIT_RATIO)", "prev_amounts"),
    (r"^PREV_", "prev_other"),
    (r"^INST_.*(UNDERPAID|PAYMENT_RATIO|PAYMENT_DIFF)", "inst_underpaid"),
    (r"^INST_", "inst_late"),
    (r"^AMT_REQ_CREDIT_BUREAU", "enquiries"),
    (r"^(CREDIT_INCOME_RATIO|ANNUITY_INCOME_RATIO|AMT_ANNUITY|AMT_CREDIT|AMT_INCOME_TOTAL)$", "loan_burden"),
    # CREDIT_TERM и AMT_GOODS_PRICE действуют немонотонно, поэтому формулировка нейтральная
    (r"^(CREDIT_GOODS_RATIO|AMT_GOODS_PRICE|CREDIT_TERM)$", "credit_params"),
    (r"^(DAYS_EMPLOYED|DAYS_EMPLOYED_ANOM|ORGANIZATION_TYPE|OCCUPATION_TYPE|NAME_INCOME_TYPE|FLAG_EMP_PHONE|FLAG_WORK_PHONE)$", "employment"),
    (r"^NAME_EDUCATION_TYPE$", "education"),
    (r"(_AVG|_MODE|_MEDI)$|^(NAME_HOUSING_TYPE|HOUSING_NAN_CNT)$", "housing"),
    (r"^(FLAG_OWN_CAR|OWN_CAR_AGE|FLAG_OWN_REALTY)$", "assets"),
    (r"^(DAYS_LAST_PHONE_CHANGE|DAYS_ID_PUBLISH|DAYS_REGISTRATION|FLAG_PHONE|FLAG_MOBIL|FLAG_EMAIL|FLAG_CONT_MOBILE)$", "contacts"),
    (r"^NAME_CONTRACT_TYPE$", "contract"),
]
_COMPILED = [(re.compile(p), g) for p, g in RULES]


def feature_group(name: str) -> str:
    for pattern, group in _COMPILED:
        if pattern.search(name):
            return group
    return "other"


def build_mapping(columns) -> pd.Series:
    """Series: признак -> id группы."""
    return pd.Series({c: feature_group(c) for c in columns}, name="group")


def group_shap(shap_values: np.ndarray, columns) -> pd.DataFrame:
    """Сумма SHAP по группам: таблица (клиенты x группы). Сумма по всем группам = сумма по признакам."""
    mapping = build_mapping(columns)
    df = pd.DataFrame(shap_values, columns=list(columns))
    return df.T.groupby(mapping.reindex(df.columns).values).sum().T


def top_reasons(
    grouped: pd.DataFrame,
    k: int = 3,
    min_shap: float = 0.0,
    excluded: set[str] | None = None,
) -> list[list[tuple[str, float]]]:
    """Для каждого клиента: до k групп с наибольшим положительным вкладом в риск.

    Возвращает список списков (текст причины, вклад SHAP в лог-шансах). Группы из `excluded`
    пропускаются, причины с вкладом <= min_shap не показываются.
    """
    excluded = EXCLUDED_GROUPS if excluded is None else excluded
    g = grouped[[c for c in grouped.columns if c not in excluded]]
    values = g.values
    names = np.array([GROUPS.get(c, c) for c in g.columns])

    k = min(k, values.shape[1])
    idx = np.argsort(-values, axis=1)[:, :k]
    out = []
    for i in range(values.shape[0]):
        reasons = [(str(names[j]), float(values[i, j])) for j in idx[i] if values[i, j] > min_shap]
        out.append(reasons)
    return out


def shap_sample(model, X: pd.DataFrame, y: pd.Series, n: int = 20_000, seed: int = 42):
    """Выборка клиентов и вклады TreeSHAP (встроенные в LightGBM, лог-шансы).

    Возвращает (Xs, ys, shap_values, base_value, proba). При тех же n и seed выборка всегда
    одна и та же, поэтому результаты разных ноутбуков сопоставимы.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = y.iloc[idx].reset_index(drop=True)
    contrib = model.booster_.predict(Xs, pred_contrib=True)
    return Xs, ys, contrib[:, :-1], contrib[:, -1], model.predict_proba(Xs)[:, 1]


def group_breakdown(shap_values: np.ndarray, Xs: pd.DataFrame, group: str, top: int = 10) -> pd.DataFrame:
    """Что внутри группы: признаки, их доля вклада и направление эффекта.

    spearman(value, SHAP) > 0: чем больше значение признака, тем выше риск; < 0: наоборот.
    Для категориальных признаков направление не определено (NaN).
    """
    mapping = build_mapping(Xs.columns)
    pos = {c: i for i, c in enumerate(Xs.columns)}
    cols = [c for c in Xs.columns if mapping[c] == group]
    mean_abs = pd.Series({c: np.abs(shap_values[:, pos[c]]).mean() for c in cols})

    rows = []
    for c in mean_abs.sort_values(ascending=False).head(top).index:
        x = Xs[c]
        corr = np.nan
        if pd.api.types.is_numeric_dtype(x) and not str(x.dtype) == "category":
            m = x.notna().values
            if m.sum() > 10:
                corr = pd.Series(x.values[m]).corr(pd.Series(shap_values[m, pos[c]]), method="spearman")
        rows.append({
            "feature": c,
            "доля в группе, %": mean_abs[c] / mean_abs.sum() * 100,
            "mean |SHAP|": mean_abs[c],
            "spearman(value, SHAP)": corr,
        })
    return pd.DataFrame(rows).round(3).set_index("feature")


def binned_effect(shap_values: np.ndarray, Xs: pd.DataFrame, feature: str, q: int = 5) -> pd.DataFrame:
    """Средний вклад SHAP по квантилям значения признака (для категорий: по категориям).

    Нужен там, где корреляция Спирмена неоднозначна (немонотонная зависимость):
    сразу видно, при каких значениях признак повышает риск, а при каких снижает.
    """
    pos = list(Xs.columns).index(feature)
    x = Xs[feature]
    d = pd.DataFrame({"value": x.values, "shap": shap_values[:, pos]})

    if not pd.api.types.is_numeric_dtype(x) or str(x.dtype) == "category":
        t = d.groupby("value", observed=True, dropna=False)["shap"].agg(["mean", "count"])
        return t.sort_values("mean", ascending=False).round(3)

    d["bin"] = pd.qcut(d["value"], q=q, duplicates="drop")
    t = d.groupby("bin", observed=True).agg(
        value_mean=("value", "mean"), shap_mean=("shap", "mean"), n=("shap", "size")
    )
    t.index = t.index.astype(str)
    missing = d[d["value"].isna()]
    if len(missing):
        t.loc["NaN"] = [np.nan, missing["shap"].mean(), len(missing)]
    return t.round(3)
