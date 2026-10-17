"""Пайплайны моделей и вспомогательные функции для кросс-валидации."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def cv_auc(model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5):
    """ROC AUC по фолдам. Фиксированный random_state => во всех вариантах одни и те же фолды,
    поэтому результаты разных наборов признаков можно сравнивать попарно."""
    from sklearn.model_selection import cross_val_score

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X, y, cv=skf, scoring="roc_auc")


def to_lgbm(X: pd.DataFrame) -> pd.DataFrame:
    """Строковые колонки -> dtype 'category', чтобы LightGBM кодировал их сам."""
    X = X.copy()
    for c in X.select_dtypes("object").columns:
        X[c] = X[c].astype("category")
    return X


def make_lgbm(**overrides):
    """LightGBM с разумными стартовыми параметрами (позже подберём через Optuna)."""
    from lightgbm import LGBMClassifier

    params = dict(
        n_estimators=400, learning_rate=0.06, num_leaves=31,
        colsample_bytree=0.5, subsample=0.8, subsample_freq=1,
        random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
    )
    params.update(overrides)
    return LGBMClassifier(**params)


def make_random_forest(X: pd.DataFrame, **overrides) -> Pipeline:
    """Случайный лес. Пропуски -> медиана + индикатор, категории -> порядковые коды.

    min_samples_leaf=50 ограничивает глубину и делает вероятности менее "ступенчатыми"
    (полностью выросшие деревья дают очень грубые вероятности и тяжёлую модель).
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OrdinalEncoder

    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()

    pre = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
        ("cat", Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="missing")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), cat_cols),
    ])
    params = dict(
        n_estimators=300, min_samples_leaf=50, max_features="sqrt",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    params.update(overrides)
    return Pipeline([("pre", pre), ("clf", RandomForestClassifier(**params))])


def cv_predict(make_model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5, train_filter=None):
    """Кросс-валидация с OOF-предсказаниями и AUC по каждому фолду.

    make_model - функция без аргументов, возвращающая НОВУЮ необученную модель.
    Фолды те же, что в cv_auc (тот же random_state). Требует X и y с индексом 0..n-1.
    Внешний цикл последовательный: модели сами используют все ядра (n_jobs=-1).
    train_filter(tr, y) -> индексы обучающей части (например, undersampling); валидация не меняется.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(y))
    fold_auc = []
    for tr, va in skf.split(X, y):
        if train_filter is not None:
            tr = train_filter(tr, y)
        model = make_model()
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[va])[:, 1]
        oof[va] = p
        fold_auc.append(roc_auc_score(y.iloc[va], p))
    return oof, np.array(fold_auc)
