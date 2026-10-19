"""Признаки для бота: ответы пользователя -> строка признаков модели.

Формулы производных признаков совпадают с src/features.py. Это важно: расхождение между тем, как признаки
считались при обучении и как считаются в боте (train/serve skew), тихо портит качество. Совпадение проверяется
в ноутбуке 13 на реальных строках датасета.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --- признаки модели для бота (набор «анкета + история», ноутбук 12) ---
A_CORE = [
    "AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "AMT_GOODS_PRICE",
    "DAYS_EMPLOYED", "DAYS_EMPLOYED_ANOM", "NAME_INCOME_TYPE", "NAME_EDUCATION_TYPE",
    "FLAG_OWN_CAR", "OWN_CAR_AGE", "FLAG_OWN_REALTY", "NAME_CONTRACT_TYPE",
    "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO", "CREDIT_TERM", "CREDIT_GOODS_RATIO",
]
HISTORY = [
    "BUREAU_COUNT", "BUREAU_IS_ACTIVE_SUM", "BUREAU_AMT_CREDIT_SUM_DEBT_SUM",
    "BUREAU_HAS_OVERDUE_SUM", "BUREAU_LAST1Y_COUNT", "HAS_BUREAU",
]
BOT_FEATURES = A_CORE + HISTORY

# значения категорий в датасете -> подписи для пользователя
INCOME_TYPES = {
    "Working": "Работаю по найму",
    "Commercial associate": "Сотрудник коммерческой компании",
    "State servant": "Госслужащий",
    "Businessman": "Предприниматель",
    "Pensioner": "Пенсионер",
    "Student": "Студент",
    "Maternity leave": "В декрете",
    "Unemployed": "Не работаю",
}
EDUCATION_TYPES = {
    "Lower secondary": "Неполное среднее",
    "Secondary / secondary special": "Среднее / среднее специальное",
    "Incomplete higher": "Неоконченное высшее",
    "Higher education": "Высшее",
    "Academic degree": "Учёная степень",
}
CONTRACT_TYPES = {
    "Cash loans": "Кредит наличными",
    "Revolving loans": "Возобновляемый (кредитная карта)",
}
YES_NO = {"Y": "Да", "N": "Нет"}

# в датасете заглушка DAYS_EMPLOYED = 365243 встречается только у этих типов занятости
NO_EMPLOYMENT_TYPES = {"Pensioner", "Unemployed"}


def _positive(x, name: str) -> float:
    x = float(x)
    if not x > 0:
        raise ValueError(f"{name} должно быть больше нуля")
    return x


def _optional(x) -> float:
    return np.nan if x is None else float(x)


def answers_to_features(a: dict, features: list[str] | None = None) -> pd.DataFrame:
    """Ответы пользователя (словарь) -> DataFrame из одной строки с признаками модели.

    Ключи: income, credit, annuity, goods_price (может быть None), income_type, education,
    employment_years (None для пенсионера/неработающего), own_car ('Y'/'N'), car_age,
    own_realty ('Y'/'N'), contract_type, bureau_count, bureau_active, bureau_debt,
    bureau_overdue (bool), bureau_last_year.
    """
    features = features or BOT_FEATURES

    income = _positive(a["income"], "Доход")
    credit = _positive(a["credit"], "Сумма кредита")
    annuity = _positive(a["annuity"], "Платёж")
    goods = _optional(a.get("goods_price"))

    for key, allowed in [("income_type", INCOME_TYPES), ("education", EDUCATION_TYPES),
                         ("contract_type", CONTRACT_TYPES), ("own_car", YES_NO), ("own_realty", YES_NO)]:
        if a[key] not in allowed:
            raise ValueError(f"Недопустимое значение {key}: {a[key]!r}")

    if a["income_type"] in NO_EMPLOYMENT_TYPES:
        days_employed, anom = np.nan, 1
    else:
        years = a.get("employment_years")
        days_employed, anom = (np.nan if years is None else -float(years) * 365), 0

    car_age = np.nan if a["own_car"] != "Y" else _optional(a.get("car_age"))

    n_bureau = int(a.get("bureau_count") or 0)
    if n_bureau > 0:
        active = min(int(a.get("bureau_active") or 0), n_bureau)
        debt = float(a.get("bureau_debt") or 0)
        overdue = 1 if a.get("bureau_overdue") else 0
        last_year = min(int(a.get("bureau_last_year") or 0), n_bureau)
    else:
        active = debt = overdue = last_year = 0

    row = {
        "AMT_INCOME_TOTAL": income, "AMT_CREDIT": credit, "AMT_ANNUITY": annuity, "AMT_GOODS_PRICE": goods,
        "DAYS_EMPLOYED": days_employed, "DAYS_EMPLOYED_ANOM": anom,
        "NAME_INCOME_TYPE": a["income_type"], "NAME_EDUCATION_TYPE": a["education"],
        "FLAG_OWN_CAR": a["own_car"], "OWN_CAR_AGE": car_age,
        "FLAG_OWN_REALTY": a["own_realty"], "NAME_CONTRACT_TYPE": a["contract_type"],
        # производные: те же формулы, что в src/features.py
        "CREDIT_INCOME_RATIO": credit / income,
        "ANNUITY_INCOME_RATIO": annuity / income,
        "CREDIT_TERM": annuity / credit,
        "CREDIT_GOODS_RATIO": credit / goods if goods and goods > 0 else np.nan,
        # история в других банках
        "BUREAU_COUNT": n_bureau, "BUREAU_IS_ACTIVE_SUM": active, "BUREAU_AMT_CREDIT_SUM_DEBT_SUM": debt,
        "BUREAU_HAS_OVERDUE_SUM": overdue, "BUREAU_LAST1Y_COUNT": last_year, "HAS_BUREAU": int(n_bureau > 0),
    }
    return pd.DataFrame([row])[features]
