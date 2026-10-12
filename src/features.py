"""Построение признаков для Home Credit Default Risk.

Все агрегаты считаются без использования TARGET, поэтому их можно безопасно
считать на всей выборке (утечки таргета нет). Вся информация из bureau и
previous_application относится к периоду ДО текущей заявки (DAYS_* отрицательные).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "SK_ID_CURR"
TARGET_COL = "TARGET"
DAYS_PLACEHOLDER = 365243


# ---------------------------------------------------------------- helpers
def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a / b).replace([np.inf, -np.inf], np.nan)


def _flatten(agg: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """MultiIndex-колонки (признак, статистика) -> 'PREFIX_ПРИЗНАК_СТАТИСТИКА'."""
    agg.columns = [f"{prefix}{col}_{stat}".upper() for col, stat in agg.columns]
    return agg


# ------------------------------------------------------ основная таблица
def housing_duplicates(df: pd.DataFrame) -> list[str]:
    """Числовые *_MODE / *_MEDI, у которых есть двойник *_AVG (корреляция ~0.97-0.99)."""
    return [
        c for c in df.columns
        if c.endswith(("_MODE", "_MEDI"))
        and pd.api.types.is_numeric_dtype(df[c])
        and (c[:-5] + "_AVG") in df.columns
    ]


def build_application_features(
    data: pd.DataFrame, domain: bool = True, drop_housing_dups: bool = True
) -> pd.DataFrame:
    d = data.copy()

    # заглушка в стаже -> NaN + флаг (по сути "пенсионер")
    d["DAYS_EMPLOYED_ANOM"] = (d["DAYS_EMPLOYED"] == DAYS_PLACEHOLDER).astype(int)
    d["DAYS_EMPLOYED"] = d["DAYS_EMPLOYED"].replace(DAYS_PLACEHOLDER, np.nan)

    d["CODE_GENDER"] = d["CODE_GENDER"].replace("XNA", np.nan)
    d["AGE"] = -d["DAYS_BIRTH"] / 365

    if domain:
        d["CREDIT_INCOME_RATIO"] = d["AMT_CREDIT"] / d["AMT_INCOME_TOTAL"]
        d["ANNUITY_INCOME_RATIO"] = d["AMT_ANNUITY"] / d["AMT_INCOME_TOTAL"]
        d["CREDIT_TERM"] = d["AMT_ANNUITY"] / d["AMT_CREDIT"]
        d["CREDIT_GOODS_RATIO"] = d["AMT_CREDIT"] / d["AMT_GOODS_PRICE"]
        d["EMPLOYED_BIRTH_RATIO"] = d["DAYS_EMPLOYED"] / d["DAYS_BIRTH"]
        d["INCOME_PER_PERSON"] = d["AMT_INCOME_TOTAL"] / d["CNT_FAM_MEMBERS"]

        ext = d[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]]
        d["EXT_SOURCE_MEAN"] = ext.mean(axis=1)
        d["EXT_SOURCE_MIN"] = ext.min(axis=1)
        d["EXT_SOURCE_PROD"] = ext.prod(axis=1, skipna=False)

    if drop_housing_dups:
        d = d.drop(columns=housing_duplicates(d))

    return d.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------ bureau
def aggregate_bureau(bureau: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на SK_ID_CURR: история кредитов в других бюро."""
    b = bureau.copy()
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["HAS_OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)
    b["DEBT_CREDIT_RATIO"] = _safe_div(b["AMT_CREDIT_SUM_DEBT"], b["AMT_CREDIT_SUM"])

    aggs = {
        "SK_ID_BUREAU": ["count"],
        "DAYS_CREDIT": ["min", "max", "mean"],
        "CREDIT_DAY_OVERDUE": ["max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["max", "mean"],
        "CNT_CREDIT_PROLONG": ["sum", "max"],
        "AMT_CREDIT_SUM": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean", "max"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "max"],
        "DEBT_CREDIT_RATIO": ["mean", "max"],
        "DAYS_CREDIT_UPDATE": ["mean"],
        "IS_ACTIVE": ["mean", "sum"],
        "HAS_OVERDUE": ["mean", "sum"],
    }
    total = _flatten(b.groupby(ID_COL).agg(aggs), "BUREAU_")
    total = total.rename(columns={"BUREAU_SK_ID_BUREAU_COUNT": "BUREAU_COUNT"})

    # только активные кредиты
    active = _flatten(
        b[b["IS_ACTIVE"] == 1].groupby(ID_COL).agg({
            "AMT_CREDIT_SUM": ["sum", "mean"],
            "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
            "DEBT_CREDIT_RATIO": ["mean"],
            "CREDIT_DAY_OVERDUE": ["max"],
        }),
        "BUREAU_ACTIVE_",
    )

    # последний год: свежее поведение важнее старого
    recent = _flatten(
        b[b["DAYS_CREDIT"] >= -365].groupby(ID_COL).agg({
            "SK_ID_BUREAU": ["count"],
            "AMT_CREDIT_SUM": ["sum"],
            "HAS_OVERDUE": ["mean"],
        }),
        "BUREAU_LAST1Y_",
    )
    recent = recent.rename(columns={"BUREAU_LAST1Y_SK_ID_BUREAU_COUNT": "BUREAU_LAST1Y_COUNT"})

    return total.join([active, recent], how="left")


# ------------------------------------------------------ previous_application
def aggregate_previous(prev: pd.DataFrame) -> pd.DataFrame:
    """Одна строка на SK_ID_CURR: история прошлых заявок в Home Credit."""
    p = prev.copy()

    for c in ["DAYS_FIRST_DRAWING", "DAYS_FIRST_DUE", "DAYS_LAST_DUE_1ST_VERSION",
              "DAYS_LAST_DUE", "DAYS_TERMINATION"]:
        p[c] = p[c].replace(DAYS_PLACEHOLDER, np.nan)

    p["IS_APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["IS_REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    p["APP_CREDIT_RATIO"] = _safe_div(p["AMT_APPLICATION"], p["AMT_CREDIT"])
    p["ANNUITY_CREDIT_RATIO"] = _safe_div(p["AMT_ANNUITY"], p["AMT_CREDIT"])

    aggs = {
        "SK_ID_PREV": ["count"],
        "AMT_ANNUITY": ["min", "max", "mean"],
        "AMT_APPLICATION": ["min", "max", "mean"],
        "AMT_CREDIT": ["min", "max", "mean", "sum"],
        "AMT_DOWN_PAYMENT": ["max", "mean"],
        "AMT_GOODS_PRICE": ["max", "mean"],
        "RATE_DOWN_PAYMENT": ["mean", "max"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "DAYS_LAST_DUE": ["max"],
        "DAYS_TERMINATION": ["max"],
        "CNT_PAYMENT": ["mean", "sum"],
        "APP_CREDIT_RATIO": ["mean", "max"],
        "ANNUITY_CREDIT_RATIO": ["mean"],
        "IS_APPROVED": ["mean", "sum"],
        "IS_REFUSED": ["mean", "sum"],
    }
    total = _flatten(p.groupby(ID_COL).agg(aggs), "PREV_")
    total = total.rename(columns={"PREV_SK_ID_PREV_COUNT": "PREV_COUNT"})

    approved = _flatten(
        p[p["IS_APPROVED"] == 1].groupby(ID_COL).agg({
            "AMT_CREDIT": ["mean", "max", "sum"],
            "AMT_ANNUITY": ["mean"],
            "CNT_PAYMENT": ["mean"],
            "DAYS_DECISION": ["max"],
        }),
        "PREV_APPROVED_",
    )

    refused = _flatten(
        p[p["IS_REFUSED"] == 1].groupby(ID_COL).agg({
            "AMT_APPLICATION": ["mean", "max"],
            "DAYS_DECISION": ["max"],
        }),
        "PREV_REFUSED_",
    )

    # "кредитный голод": сколько заявок и отказов за последний год
    recent = _flatten(
        p[p["DAYS_DECISION"] >= -365].groupby(ID_COL).agg({
            "SK_ID_PREV": ["count"],
            "IS_REFUSED": ["mean"],
        }),
        "PREV_LAST1Y_",
    )
    recent = recent.rename(columns={"PREV_LAST1Y_SK_ID_PREV_COUNT": "PREV_LAST1Y_COUNT"})

    # статус самой свежей прошлой заявки
    last = p.sort_values("DAYS_DECISION").groupby(ID_COL).tail(1).set_index(ID_COL)
    last_feat = pd.DataFrame({
        "PREV_LAST_IS_REFUSED": last["IS_REFUSED"],
        "PREV_LAST_DAYS_DECISION": last["DAYS_DECISION"],
    })

    return total.join([approved, refused, recent, last_feat], how="left")


# --------------------------------------------------------------- сборка
def build_dataset(
    app: pd.DataFrame,
    bureau: pd.DataFrame | None = None,
    prev: pd.DataFrame | None = None,
    domain: bool = True,
) -> pd.DataFrame:
    """Основная таблица + агрегаты из связанных таблиц (left join по SK_ID_CURR)."""
    X = build_application_features(app, domain=domain)

    sources = []
    if bureau is not None:
        sources.append(("BUREAU", aggregate_bureau(bureau)))
    if prev is not None:
        sources.append(("PREV", aggregate_previous(prev)))

    for name, agg in sources:
        X = X.join(agg, on=ID_COL)
        X[f"HAS_{name}"] = X[ID_COL].isin(agg.index).astype(int)
        # у клиентов без истории счётчики и суммы = 0, а средние остаются NaN
        fill_cols = [c for c in agg.columns if c.endswith(("_COUNT", "_SUM"))]
        X[fill_cols] = X[fill_cols].fillna(0)

    if domain and bureau is not None:
        X["BUREAU_DEBT_INCOME_RATIO"] = _safe_div(
            X["BUREAU_AMT_CREDIT_SUM_DEBT_SUM"], X["AMT_INCOME_TOTAL"]
        )

    return X
