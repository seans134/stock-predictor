import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.data.synthetic import synthetic_bars
from stockpredictor.labeling.labeler import LABEL_NONE, LABEL_OPENING_ONLY, LABEL_SUSTAINED
from stockpredictor.sessions import ET, SessionCalendar
from stockpredictor.stage1.features import (
    STAGE1_FEATURES,
    _pm_usable_mask,
    build_stage1_features,
)
from stockpredictor.stage1.model import (
    Stage1Classifier,
    Stage1ModelConfig,
    evaluate_stage1,
    ranking_metrics,
)


@pytest.fixture(scope="module")
def cal() -> SessionCalendar:
    return SessionCalendar(start="2024-01-01", end="2024-12-31")


def _pm_bars(day: dt.date, times: list[str], base_price: float = 100.0) -> pd.DataFrame:
    """Fabricate pre-market bars at given ET times."""
    index = pd.DatetimeIndex(
        [pd.Timestamp(f"{day} {t}", tz=ET) for t in times]
    )
    n = len(index)
    prices = base_price + np.arange(n) * 0.1
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices + 0.05,
            "low": prices - 0.05,
            "close": prices + 0.02,
            "volume": np.full(n, 1000.0),
            "trade_count": np.full(n, 50),
            "vwap": prices,
        },
        index=index,
    )


def test_cutoff_excludes_bar_arriving_after_nine(cal):
    day = dt.date(2024, 3, 19)
    bars = _pm_bars(day, ["08:45", "08:50", "08:55"])
    mask = _pm_usable_mask(bars.index, dt.time(9, 0))
    # 08:45 completes 08:50, available 08:50:05 -> usable
    # 08:50 completes 08:55, available 08:55:05 -> usable
    # 08:55 completes 09:00, available 09:00:05 -> AFTER the cutoff
    assert mask.tolist() == [True, True, False]


def test_gap_uses_last_usable_pm_bar_and_prev_close(cal):
    day1, day2 = dt.date(2024, 3, 18), dt.date(2024, 3, 19)
    rth = synthetic_bars(cal, day1, day2, seed=9)
    pm = _pm_bars(day2, ["07:00", "08:30", "08:55"], base_price=105.0)
    bars = pd.concat([rth.drop(columns=["available_ts"]), pm]).sort_index()

    features = build_stage1_features({"AAA": bars}, cal).set_index("date")
    row = features.loc[day2]

    prev_close = rth.loc[rth.index.date == day1, "close"].iloc[-1]
    # Last USABLE pm bar is 08:30 (08:55 arrives after the cutoff).
    pm_last_close = pm.loc[pm.index.time == dt.time(8, 30), "close"].iloc[0]
    assert row["gap"] == pytest.approx(np.log(pm_last_close / prev_close))
    assert row["n_pm_bars"] == 2

    # Day with no pre-market bars: gap is explicitly missing, not zero.
    assert features.loc[day1, "n_pm_bars"] == 0
    assert np.isnan(features.loc[day1, "gap"])


def test_no_lookahead_dropping_future_days_preserves_rows(cal):
    start, mid, end = dt.date(2024, 3, 4), dt.date(2024, 3, 8), dt.date(2024, 3, 15)
    rth = synthetic_bars(cal, start, end, seed=10).drop(columns=["available_ts"])
    pm_frames = [
        _pm_bars(d, ["07:00", "08:00"], base_price=100.0)
        for d in cal.sessions_between(start, end)
    ]
    bars = pd.concat([rth] + pm_frames).sort_index()

    full = build_stage1_features({"AAA": bars}, cal).set_index("date")
    cutoff_ts = pd.Timestamp(f"{mid} 23:59", tz=ET)
    truncated = build_stage1_features(
        {"AAA": bars.loc[bars.index <= cutoff_ts]}, cal
    ).set_index("date")

    common = truncated.index
    pd.testing.assert_frame_equal(
        full.loc[common, STAGE1_FEATURES], truncated[STAGE1_FEATURES]
    )


def test_classifier_roundtrip():
    rng = np.random.default_rng(0)
    n = 600
    frame = pd.DataFrame(rng.normal(size=(n, len(STAGE1_FEATURES))), columns=STAGE1_FEATURES)
    frame["date"] = [dt.date(2024, 1, 1) + dt.timedelta(days=i // 30) for i in range(n)]
    frame["label"] = rng.choice([LABEL_NONE, LABEL_OPENING_ONLY, LABEL_SUSTAINED], size=n)
    frame["gap"] = frame[STAGE1_FEATURES[0]]
    frame["pm_vol_ratio"] = frame[STAGE1_FEATURES[5]]

    model = Stage1Classifier(Stage1ModelConfig(n_estimators=20, min_child_samples=10)).fit(frame)
    proba = model.predict_proba(frame)

    p_cols = ["p_none", "p_opening_only", "p_sustained"]
    assert np.allclose(proba[p_cols].sum(axis=1), 1.0)
    expected_score = 0.4 * proba["p_opening_only"] + 1.0 * proba["p_sustained"]
    assert np.allclose(proba["opportunity_score"], expected_score)

    metrics = evaluate_stage1(frame, proba, k=5)
    assert metrics["n_rows"] == n
    assert metrics["log_loss"] > 0
    assert 0 <= metrics["p_at_5_model"] <= 1
    assert 0 <= metrics["capture_at_5_random"] <= 1


def test_ranking_metrics_hand_case():
    day = dt.date(2024, 3, 18)
    frame = pd.DataFrame(
        {
            "date": [day] * 4,
            "label": [LABEL_SUSTAINED, LABEL_NONE, LABEL_OPENING_ONLY, LABEL_NONE],
            "score": [0.9, 0.8, 0.7, 0.1],
        }
    )
    # Top-2 by score: sustained (hit), none (miss) -> precision 0.5.
    # Two true opportunities, one captured in top-2 -> capture 0.5.
    precision, capture = ranking_metrics(frame, "score", k=2)
    assert precision == pytest.approx(0.5)
    assert capture == pytest.approx(0.5)
