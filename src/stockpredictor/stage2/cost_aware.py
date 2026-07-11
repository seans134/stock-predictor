"""Cost-aware Stage 2 heads: probability of beating costs, directly.

The quantile model answers "what will the return be"; the decision rules
then ask "will it beat costs" — a repurposing that loses information.
These two binary heads train on the decision's own question, per the
plan's forecast outputs:

    p_long_win  = P(forward return  >  round_trip_cost)
    p_short_win = P(forward return  < -round_trip_cost)

They complement, not replace, the quantile model: quantiles still supply
the risk caps and position sizing; these supply the entry edge.
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from stockpredictor.stage2.features import FEATURE_COLUMNS, TARGET_COLUMN


@dataclass(frozen=True)
class CostAwareConfig:
    n_estimators: int = 300
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 200


class Stage2CostClassifier:
    def __init__(self, round_trip_cost: float, config: CostAwareConfig | None = None):
        self.round_trip_cost = round_trip_cost
        self.config = config or CostAwareConfig()
        self._long: lgb.LGBMClassifier | None = None
        self._short: lgb.LGBMClassifier | None = None

    def _make(self) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=self.config.min_child_samples,
            verbose=-1,
        )

    def fit(self, train: pd.DataFrame) -> "Stage2CostClassifier":
        rows = train.dropna(subset=[TARGET_COLUMN])
        X = rows[FEATURE_COLUMNS]
        y = rows[TARGET_COLUMN].to_numpy()
        self._long = self._make().fit(X, (y > self.round_trip_cost).astype(int))
        self._short = self._make().fit(X, (y < -self.round_trip_cost).astype(int))
        return self

    def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
        X = frame[FEATURE_COLUMNS]
        return pd.DataFrame(
            {
                "p_long_win": self._long.predict_proba(X)[:, 1],
                "p_short_win": self._short.predict_proba(X)[:, 1],
            },
            index=frame.index,
        )
