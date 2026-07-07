"""Probability calibration for the Stage 1 classifier.

The classifier ranks well but its raw probabilities are overconfident
(walk-forward log loss worse than the class prior). Per the plan, the
calibration layer is separate from the classifier and is fit on
predictions from data the classifier never trained on — the harness's
calibrate segment — never on training or test data.

Method: Dirichlet calibration — multinomial logistic regression on the
log-probabilities. It generalizes temperature/vector scaling, handles
three classes with few parameters, and behaves well on the ~1-2k rows a
calibrate segment provides. It is also kept deliberately separate so the
plan's tiered retraining cadence can refresh it more often than the
classifier itself.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

_EPS = 1e-9


class DirichletCalibrator:
    def __init__(self):
        self._lr: LogisticRegression | None = None
        self._classes_seen: np.ndarray | None = None

    def fit(self, proba: np.ndarray, y: np.ndarray) -> "DirichletCalibrator":
        """proba: (n, k) raw classifier probabilities; y: (n,) class ids."""
        X = np.log(np.clip(proba, _EPS, 1.0))
        self._lr = LogisticRegression(max_iter=1000, C=1.0)
        self._lr.fit(X, y)
        self._classes_seen = self._lr.classes_
        self._n_classes = proba.shape[1]
        return self

    def transform(self, proba: np.ndarray) -> np.ndarray:
        """Calibrated probabilities, same shape as input, rows sum to 1.

        If the calibrate segment happened to miss a class entirely, the
        missing column is zero-filled and rows renormalized, so the output
        shape never silently changes.
        """
        X = np.log(np.clip(proba, _EPS, 1.0))
        raw = self._lr.predict_proba(X)
        if raw.shape[1] == self._n_classes:
            return raw
        full = np.zeros((raw.shape[0], self._n_classes))
        for j, cls in enumerate(self._classes_seen):
            full[:, int(cls)] = raw[:, j]
        full /= full.sum(axis=1, keepdims=True)
        return full
