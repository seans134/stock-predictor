"""Stage 1 classifier: pre-market snapshot -> session opportunity class.

LightGBM multiclass over the three labels, evaluated the way the plan
demands — as a filter and ranker, not a trading model: probability quality
(log loss, Brier), per-class precision/recall, and ranking quality of the
opportunity score against the mandatory simple baselines (largest gap,
highest pre-market relative volume, random). The learned scanner earns its
complexity only by repeatably beating those baselines.

Opportunity score, per the plan's starting weights (to be tuned per fold
later, never on test results):

    score = 0.40 * P(opening-only) + 1.00 * P(sustained)
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from stockpredictor.labeling.labeler import LABEL_NONE, LABEL_OPENING_ONLY, LABEL_SUSTAINED
from stockpredictor.stage1.features import STAGE1_FEATURES

CLASS_ORDER = [LABEL_NONE, LABEL_OPENING_ONLY, LABEL_SUSTAINED]
_CLASS_INDEX = {label: i for i, label in enumerate(CLASS_ORDER)}

OPPORTUNITY_WEIGHTS = {LABEL_OPENING_ONLY: 0.40, LABEL_SUSTAINED: 1.00}


@dataclass(frozen=True)
class Stage1ModelConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 50


class Stage1Classifier:
    def __init__(self, config: Stage1ModelConfig | None = None):
        self.config = config or Stage1ModelConfig()
        self._model: lgb.LGBMClassifier | None = None

    def fit(self, train: pd.DataFrame) -> "Stage1Classifier":
        rows = train.dropna(subset=["label"])
        self._model = lgb.LGBMClassifier(
            objective="multiclass",
            num_class=len(CLASS_ORDER),
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            verbose=-1,
        )
        self._model.fit(rows[STAGE1_FEATURES], rows["label"].map(_CLASS_INDEX))
        return self

    def predict_proba(self, frame: pd.DataFrame) -> pd.DataFrame:
        proba = self._model.predict_proba(frame[STAGE1_FEATURES])
        out = pd.DataFrame(
            proba, columns=[f"p_{c}" for c in CLASS_ORDER], index=frame.index
        )
        out["opportunity_score"] = sum(
            OPPORTUNITY_WEIGHTS[c] * out[f"p_{c}"] for c in OPPORTUNITY_WEIGHTS
        )
        return out


def ranking_metrics(
    frame: pd.DataFrame, score_col: str, k: int = 5
) -> tuple[float, float]:
    """(precision@k, capture@k) of a score, averaged over test days.

    A hit is any session whose true label is an opportunity (not 'none').
    capture@k averages only over days that had at least one opportunity.
    """
    precisions, captures = [], []
    for _, day_rows in frame.groupby("date"):
        ranked = day_rows.sort_values(score_col, ascending=False, na_position="last")
        top = ranked.head(k)
        hits = (top["label"] != LABEL_NONE).sum()
        precisions.append(hits / min(k, len(ranked)))
        n_opps = (day_rows["label"] != LABEL_NONE).sum()
        if n_opps > 0:
            captures.append(hits / min(n_opps, k))
    return (
        float(np.mean(precisions)) if precisions else float("nan"),
        float(np.mean(captures)) if captures else float("nan"),
    )


def evaluate_stage1(
    test: pd.DataFrame, proba: pd.DataFrame, k: int = 5, seed: int = 0
) -> dict:
    """Filter/ranker metrics plus the plan's baseline comparisons, all on
    identical rows."""
    rows = test.dropna(subset=["label"]).copy()
    proba = proba.loc[rows.index]
    y_true = rows["label"].map(_CLASS_INDEX).to_numpy()
    p = proba[[f"p_{c}" for c in CLASS_ORDER]].to_numpy()

    one_hot = np.eye(len(CLASS_ORDER))[y_true]
    predicted = p.argmax(axis=1)

    metrics: dict = {
        "n_rows": len(rows),
        "log_loss": float(log_loss(y_true, p, labels=list(range(len(CLASS_ORDER))))),
        "brier": float(np.mean(np.sum((p - one_hot) ** 2, axis=1))),
        "accuracy": float(np.mean(predicted == y_true)),
    }
    for i, label in enumerate(CLASS_ORDER):
        tp = np.sum((predicted == i) & (y_true == i))
        metrics[f"precision_{label}"] = float(tp / max(np.sum(predicted == i), 1))
        metrics[f"recall_{label}"] = float(tp / max(np.sum(y_true == i), 1))

    rows["opportunity_score"] = proba["opportunity_score"]
    rows["baseline_gap"] = rows["gap"].abs()
    rows["baseline_pm_vol"] = rows["pm_vol_ratio"]
    rng = np.random.default_rng(seed)
    rows["baseline_random"] = rng.random(len(rows))

    for name, col in [
        ("model", "opportunity_score"),
        ("gap", "baseline_gap"),
        ("pm_vol", "baseline_pm_vol"),
        ("random", "baseline_random"),
    ]:
        precision, capture = ranking_metrics(rows, col, k=k)
        metrics[f"p_at_{k}_{name}"] = precision
        metrics[f"capture_at_{k}_{name}"] = capture
    return metrics
