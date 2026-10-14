"""Подбор гиперпараметров LightGBM через Optuna."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.models import RANDOM_STATE, make_lgbm

# Параметры стартовой модели из make_lgbm; кладём первым испытанием, чтобы сравнивать
# результат подбора с базой в одном и том же протоколе (reg_* у базы = 0, берём минимум).
BASELINE_PARAMS = dict(
    num_leaves=31, min_child_samples=20, colsample_bytree=0.5, subsample=0.8,
    reg_alpha=1e-3, reg_lambda=1e-3,
)


def suggest_params(trial) -> dict:
    """Пространство поиска. learning_rate и n_estimators здесь не подбираем:
    темп фиксируем, а число деревьев определяет early stopping."""
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 16, 128, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 300, log=True),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 0.8),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def _fit_with_early_stopping(params, lr, X, y, tr, va, patience):
    import lightgbm as lgb

    model = make_lgbm(n_estimators=5000, learning_rate=lr, metric="auc", **params)
    model.fit(
        X.iloc[tr], y.iloc[tr],
        eval_set=[(X.iloc[va], y.iloc[va])],
        callbacks=[lgb.early_stopping(patience, first_metric_only=True, verbose=False)],
    )
    return model.best_score_["valid_0"]["auc"], model.best_iteration_


def make_objective(X: pd.DataFrame, y: pd.Series, lr: float = 0.1,
                   n_splits: int = 3, patience: int = 100):
    """Цель Optuna: средний AUC по фолдам, число деревьев подбирает early stopping.
    После каждого фолда отчитываемся pruner'у: плохие конфигурации обрываются рано."""
    import optuna

    folds = list(StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE).split(X, y))

    def objective(trial):
        params = suggest_params(trial)
        aucs, iters = [], []
        for step, (tr, va) in enumerate(folds):
            auc, it = _fit_with_early_stopping(params, lr, X, y, tr, va, patience)
            aucs.append(auc)
            iters.append(it)
            trial.report(float(np.mean(aucs)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("best_iters", iters)
        return float(np.mean(aucs))

    return objective


def find_n_estimators(X: pd.DataFrame, y: pd.Series, params: dict, lr: float,
                      n_splits: int = 5, patience: int = 100) -> list[int]:
    """Оптимальное число деревьев по фолдам для заданных параметров и темпа обучения."""
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    return [
        _fit_with_early_stopping(params, lr, X, y, tr, va, patience)[1]
        for tr, va in skf.split(X, y)
    ]
