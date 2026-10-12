"""Метрики качества для кредитного скоринга."""
from __future__ import annotations

import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import average_precision_score, roc_auc_score


def ks_stat(y_true, y_score) -> float:
    """KS-статистика: максимальное расстояние между распределениями скоров плохих и хороших."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    return ks_2samp(y_score[y_true == 1], y_score[y_true == 0]).statistic


def report(name: str, y_true, y_score) -> dict:
    auc = roc_auc_score(y_true, y_score)
    return {
        "model": name,
        "ROC AUC": round(auc, 4),
        "Gini": round(2 * auc - 1, 4),
        "PR AUC": round(average_precision_score(y_true, y_score), 4),
        "KS": round(ks_stat(y_true, y_score), 4),
    }
