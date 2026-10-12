"""Пайплайны моделей и вспомогательные функции для кросс-валидации."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def make_logreg(X: pd.DataFrame, C: float = 0.1) -> Pipeline:
    """Логрег с импутацией, масштабированием и one-hot (нужен scikit-learn >= 1.1)."""
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=0.01)),
        ]), cat_cols),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(C=C, max_iter=2000)),
    ])


def cv_oof(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """Out-of-fold предсказания: каждая строка предсказана моделью, которая её не видела."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:, 1]
