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


def calibration_table(y_true, p, bins: int = 10):
    """Квантильные корзины по прогнозу: средний прогноз и фактическая доля дефолтов в каждой."""
    import pandas as pd

    d = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p)})
    d["bin"] = pd.qcut(d["p"], bins, duplicates="drop")
    return d.groupby("bin", observed=True).agg(n=("y", "size"), observed=("y", "mean"), predicted=("p", "mean"))


def ece(y_true, p, bins: int = 10) -> float:
    """Expected calibration error: средний модуль расхождения прогноза и факта по квантильным корзинам."""
    t = calibration_table(y_true, p, bins)
    return float((t["n"] / t["n"].sum() * (t["observed"] - t["predicted"]).abs()).sum())


def calibration_report(name: str, y_true, p) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss

    y_true, p = np.asarray(y_true), np.asarray(p)
    return {
        "model": name,
        "mean pred": round(float(p.mean()), 4),
        "base rate": round(float(y_true.mean()), 4),
        "Brier": round(brier_score_loss(y_true, p), 4),
        "LogLoss": round(log_loss(y_true, np.clip(p, 1e-6, 1 - 1e-6)), 4),
        "ECE": round(ece(y_true, p), 4),
    }
