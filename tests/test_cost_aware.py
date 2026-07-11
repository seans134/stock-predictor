import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.data.synthetic import synthetic_bars
from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import HOLD, LONG, SHORT, DecisionConfig, decide
from stockpredictor.sessions import SessionCalendar
from stockpredictor.stage2.cost_aware import CostAwareConfig, Stage2CostClassifier
from stockpredictor.stage2.features import TARGET_COLUMN, build_features


@pytest.fixture(scope="module")
def features():
    cal = SessionCalendar(start="2024-01-01", end="2024-12-31")
    bars = synthetic_bars(cal, dt.date(2024, 3, 4), dt.date(2024, 4, 12), seed=31)
    return build_features({"AAA": bars}, cal)


def test_cost_classifier_roundtrip(features):
    cost = CostModel().round_trip_cost
    model = Stage2CostClassifier(cost, CostAwareConfig(n_estimators=20, min_child_samples=20))
    model.fit(features)
    preds = model.predict(features)

    assert list(preds.columns) == ["p_long_win", "p_short_win"]
    assert preds["p_long_win"].between(0, 1).all()
    assert preds["p_short_win"].between(0, 1).all()
    # Sanity: average predicted win probability should be in the vicinity
    # of the empirical base rate, not degenerate at 0 or 1.
    base = (features[TARGET_COLUMN].dropna() > cost).mean()
    assert abs(preds["p_long_win"].mean() - base) < 0.25


def _row(p_long=0.7, p_short=0.1, q50=0.0):
    frame = pd.DataFrame(
        [{
            "r30": 0.001,
            "minutes_to_close": 120.0,
            "minutes_since_open": 60.0,
            "news_articles_2h": 2.0,
            "news_sent_signed_2h": 1.0,
        }]
    )
    preds = pd.DataFrame(
        [{"q10": -0.002, "q50": q50, "q90": 0.008,
          "p_long_win": p_long, "p_short_win": p_short}]
    )
    return frame, preds


def test_p_win_replaces_q50_edge():
    config = DecisionConfig(min_p_win=0.55)
    # q50 is zero (would HOLD under quantile rules) but p_long_win is high.
    frame, preds = _row(p_long=0.70, q50=0.0)
    assert decide(frame, preds, CostModel(), config).iloc[0] == LONG

    frame, preds = _row(p_long=0.40, p_short=0.10, q50=0.005)
    # High q50 no longer matters; neither head clears the threshold.
    assert decide(frame, preds, CostModel(), config).iloc[0] == HOLD

    frame, preds = _row(p_long=0.10, p_short=0.70)
    result = decide(frame, preds, CostModel(), config)
    # Short edge fires, but the trend gate still applies (r30 positive
    # here is within tolerance).
    assert result.iloc[0] == SHORT

    # Gate off: falls back to quantile edge behaviour.
    frame, preds = _row(p_long=0.10, p_short=0.10, q50=0.003)
    assert decide(frame, preds, CostModel(), DecisionConfig()).iloc[0] == LONG


def test_risk_caps_still_apply_with_p_win():
    config = DecisionConfig(min_p_win=0.55)
    frame, preds = _row(p_long=0.90)
    preds["q10"] = -0.02  # downside beyond the risk cap
    assert decide(frame, preds, CostModel(), config).iloc[0] == HOLD
