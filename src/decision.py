"""Выбор порога отказа по стоимости ошибок."""
from __future__ import annotations

import numpy as np
import pandas as pd


def loss_curve(y_true, p, cost_fn: float, cost_fp: float = 1.0, n_grid: int = 200) -> pd.DataFrame:
    """Потери на одного клиента при разных порогах.

    Правило: отказываем, если p >= порога.
    FN: выдали кредит будущему неплательщику (стоит cost_fn);
    FP: отказали добросовестному клиенту (стоит cost_fp, упущенная прибыль).
    Порог -inf = отказывать всем, +inf = одобрять всех.
    """
    y, p = np.asarray(y_true), np.asarray(p)
    n, pos = len(y), int((y == 1).sum())
    grid = np.r_[-np.inf, np.unique(np.quantile(p, np.linspace(0, 1, n_grid))), np.inf]
    rows = []
    for t in grid:
        reject = p >= t
        fn = int(((~reject) & (y == 1)).sum())
        fp = int((reject & (y == 0)).sum())
        rows.append({
            "threshold": t, "approval_rate": 1 - reject.mean(), "fn": fn, "fp": fp,
            "recall": ((reject) & (y == 1)).sum() / pos, "loss": (cost_fn * fn + cost_fp * fp) / n,
        })
    return pd.DataFrame(rows)


def best_threshold(curve: pd.DataFrame) -> pd.Series:
    finite = curve[np.isfinite(curve["threshold"])]
    return finite.loc[finite["loss"].idxmin()]
