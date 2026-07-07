"""Stage 2 baseline: gradient-boosted quantile regression.

Three LightGBM regressors estimate the 10th, 50th, and 90th percentiles of
the 15-minute forward return. Predictions are sorted per row so quantiles
cannot cross. Metrics follow the plan's Stage 2 evaluation list: pinball
loss per quantile, prediction-interval coverage and width, median-return
error, and direction accuracy — always alongside the zero-return baseline,
because a model that cannot beat "predict nothing happens" has no edge.

Hyperparameters are starting values, selected per walk-forward fold later,
never from test results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from stockpredictor.stage2.features import FEATURE_COLUMNS, TARGET_COLUMN

QUANTILES = (0.10, 0.50, 0.90)


@dataclass(frozen=True)
class QuantileModelConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 200
    quantiles: tuple[float, ...] = QUANTILES


class Stage2QuantileModel:
    def __init__(self, config: QuantileModelConfig | None = None):
        self.config = config or QuantileModelConfig()
        self._models: dict[float, lgb.LGBMRegressor] = {}

    def fit(self, train: pd.DataFrame) -> "Stage2QuantileModel":
        rows = train.dropna(subset=[TARGET_COLUMN])
        X = rows[FEATURE_COLUMNS]
        y = rows[TARGET_COLUMN]
        for q in self.config.quantiles:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                n_estimators=self.config.n_estimators,
                learning_rate=self.config.learning_rate,
                num_leaves=self.config.num_leaves,
                min_child_samples=self.config.min_child_samples,
                verbose=-1,
            )
            model.fit(X, y)
            self._models[q] = model
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Quantile predictions, sorted per row so they never cross."""
        X = frame[FEATURE_COLUMNS]
        preds = np.column_stack(
            [self._models[q].predict(X) for q in self.config.quantiles]
        )
        preds.sort(axis=1)
        out = pd.DataFrame(
            preds,
            columns=[f"q{int(q * 100):02d}" for q in self.config.quantiles],
            index=frame.index,
        )
        return out


def pinball_loss(y: np.ndarray, pred: np.ndarray, q: float) -> float:
    diff = y - pred
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def evaluate(test: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    """Stage 2 forecast metrics on rows with a defined target, with the
    zero-return baseline computed on exactly the same rows."""
    rows = test.dropna(subset=[TARGET_COLUMN])
    preds = predictions.loc[rows.index]
    y = rows[TARGET_COLUMN].to_numpy()
    q10 = preds["q10"].to_numpy()
    q50 = preds["q50"].to_numpy()
    q90 = preds["q90"].to_numpy()

    nonzero = y != 0
    metrics = {
        "n_rows": len(rows),
        "pinball_q10": pinball_loss(y, q10, 0.10),
        "pinball_q50": pinball_loss(y, q50, 0.50),
        "pinball_q90": pinball_loss(y, q90, 0.90),
        "coverage_80": float(np.mean((y >= q10) & (y <= q90))),
        "interval_width": float(np.mean(q90 - q10)),
        "median_mae": float(np.mean(np.abs(y - q50))),
        "direction_accuracy": float(np.mean(np.sign(q50[nonzero]) == np.sign(y[nonzero])))
        if nonzero.any()
        else float("nan"),
        # Zero-return baseline: predicts 0 for every quantile.
        "baseline_pinball_q10": pinball_loss(y, np.zeros_like(y), 0.10),
        "baseline_pinball_q50": pinball_loss(y, np.zeros_like(y), 0.50),
        "baseline_pinball_q90": pinball_loss(y, np.zeros_like(y), 0.90),
    }
    metrics["pinball_q50_vs_baseline"] = (
        metrics["pinball_q50"] / metrics["baseline_pinball_q50"]
        if metrics["baseline_pinball_q50"] > 0
        else float("nan")
    )
    return metrics
