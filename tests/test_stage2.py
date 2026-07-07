import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.data.synthetic import synthetic_bars
from stockpredictor.sessions import ET, SessionCalendar
from stockpredictor.stage2.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_features,
)
from stockpredictor.stage2.quantile import (
    QuantileModelConfig,
    Stage2QuantileModel,
    evaluate,
)


@pytest.fixture(scope="module")
def cal() -> SessionCalendar:
    return SessionCalendar(start="2024-01-01", end="2024-12-31")


@pytest.fixture(scope="module")
def two_ticker_features(cal) -> pd.DataFrame:
    bars_a = synthetic_bars(cal, dt.date(2024, 3, 4), dt.date(2024, 4, 12), seed=11)
    bars_b = synthetic_bars(cal, dt.date(2024, 3, 4), dt.date(2024, 4, 12), seed=22)
    return build_features({"AAA": bars_a, "BBB": bars_b}, cal)


def test_target_matches_hand_computation(cal):
    bars = synthetic_bars(cal, dt.date(2024, 3, 18), dt.date(2024, 3, 18), seed=3)
    features = build_features({"AAA": bars}, cal)

    t = 5  # arbitrary bar with full forward horizon
    expected = np.log(bars.iloc[t + 3]["close"] / bars.iloc[t + 1]["open"])
    assert features.iloc[t][TARGET_COLUMN] == pytest.approx(expected)

    # Horizon crosses the session end: last three bars have no target.
    assert features[TARGET_COLUMN].tail(3).isna().all()
    assert features[TARGET_COLUMN].iloc[:-3].notna().all()


def test_no_lookahead_truncating_future_preserves_features(cal):
    bars = synthetic_bars(cal, dt.date(2024, 3, 4), dt.date(2024, 3, 15), seed=4)
    spy = synthetic_bars(cal, dt.date(2024, 3, 4), dt.date(2024, 3, 15), seed=5)

    full = build_features({"AAA": bars, "SPY": spy}, cal)
    cutoff = pd.Timestamp(2024, 3, 12, 12, 0, tz=ET)
    truncated = build_features({"AAA": bars.loc[bars.index < cutoff], "SPY": spy.loc[spy.index < cutoff]}, cal)

    full_aaa = full[full["ticker"] == "AAA"].set_index("bar_start")
    trunc_aaa = truncated[truncated["ticker"] == "AAA"].set_index("bar_start")
    common = trunc_aaa.index
    pd.testing.assert_frame_equal(
        full_aaa.loc[common, FEATURE_COLUMNS], trunc_aaa[FEATURE_COLUMNS]
    )


def test_no_cross_day_leakage(cal):
    bars = synthetic_bars(cal, dt.date(2024, 3, 18), dt.date(2024, 3, 20), seed=6)
    features = build_features({"AAA": bars}, cal).set_index("bar_start")

    for day in (dt.date(2024, 3, 18), dt.date(2024, 3, 19), dt.date(2024, 3, 20)):
        first_bar = features.loc[features["date"] == day].iloc[0]
        assert np.isnan(first_bar["r5"])  # no return across the overnight gap
        last_bars = features.loc[features["date"] == day].tail(3)
        assert last_bars[TARGET_COLUMN].isna().all()  # target never crosses days

    # Overnight gap: NaN on the first stored day, defined afterwards.
    day1 = features.loc[features["date"] == dt.date(2024, 3, 18)]
    day2 = features.loc[features["date"] == dt.date(2024, 3, 19)]
    assert day1["overnight_gap"].isna().all()
    assert day2["overnight_gap"].notna().all()


def test_context_features_join_and_missingness(cal, two_ticker_features):
    # No SPY/QQQ supplied: context features exist and are explicitly NaN.
    assert two_ticker_features["spy_r15"].isna().all()
    assert two_ticker_features["qqq_r30"].isna().all()

    bars = synthetic_bars(cal, dt.date(2024, 3, 18), dt.date(2024, 3, 19), seed=7)
    spy = synthetic_bars(cal, dt.date(2024, 3, 18), dt.date(2024, 3, 19), seed=8)
    features = build_features({"AAA": bars, "SPY": spy}, cal)
    aaa = features[features["ticker"] == "AAA"]
    # SPY r15 needs 3 bars of history: NaN early in the day, defined later.
    assert aaa["spy_r15"].notna().sum() > 0


def test_model_fit_predict_evaluate(two_ticker_features):
    features = two_ticker_features
    split_day = dt.date(2024, 4, 1)
    train = features[features["date"] < split_day]
    test = features[features["date"] >= split_day]

    config = QuantileModelConfig(n_estimators=30, min_child_samples=20)
    model = Stage2QuantileModel(config).fit(train)
    preds = model.predict(test)

    assert (preds["q10"] <= preds["q50"]).all()
    assert (preds["q50"] <= preds["q90"]).all()

    metrics = evaluate(test, preds)
    assert metrics["n_rows"] > 0
    assert 0.0 <= metrics["coverage_80"] <= 1.0
    assert metrics["pinball_q50"] > 0
    assert np.isfinite(metrics["pinball_q50_vs_baseline"])
    # On a pure random walk the model should not wildly beat predicting zero.
    assert metrics["pinball_q50_vs_baseline"] < 1.5
