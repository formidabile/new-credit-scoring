"""Калибровка вероятностей: Platt (логистическая на логитах) и изотоническая регрессия."""
from __future__ import annotations

import numpy as np
from scipy.special import logit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

RANDOM_STATE = 42
_EPS = 1e-6


class Calibrator:
    """fit(p, y) на OOF-прогнозах, predict(p) возвращает откалиброванные вероятности."""

    def __init__(self, method: str = "platt"):
        if method not in ("platt", "isotonic"):
            raise ValueError("method: 'platt' или 'isotonic'")
        self.method = method

    def _z(self, p):
        return logit(np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)).reshape(-1, 1)

    def fit(self, p, y):
        y = np.asarray(y)
        if self.method == "platt":
            self.model_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._z(p), y)
        else:
            self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(np.asarray(p, float), y)
        return self

    def predict(self, p):
        if self.method == "platt":
            return self.model_.predict_proba(self._z(p))[:, 1]
        return self.model_.predict(np.asarray(p, float))


def crossfit_calibrate(p, y, method: str = "platt", n_splits: int = 5) -> np.ndarray:
    """Честная оценка калибровки: калибратор обучается на одних фолдах OOF-прогнозов, применяется к другим."""
    p, y = np.asarray(p, float), np.asarray(y)
    out = np.zeros_like(p)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    for tr, va in skf.split(p.reshape(-1, 1), y):
        out[va] = Calibrator(method).fit(p[tr], y[tr]).predict(p[va])
    return out
