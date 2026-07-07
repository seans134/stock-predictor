import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.data.asof import AsOfData
from stockpredictor.data.store import BarStore
from stockpredictor.data.synthetic import synthetic_bars
from stockpredictor.labeling.labeler import (
    LABEL_NONE,
    LABEL_OPENING_ONLY,
    LABEL_SUSTAINED,
    LabelConfig,
    classify_sessions,
    qualify_windows,
)
from stockpredictor.labeling.tod import TimeOfDayStats
from stockpredictor.labeling.windows import build_windows, session_windows
from stockpredictor.sessions import ET, SessionCalendar

DAY = dt.date(2024, 3, 18)


@pytest.fixture(scope="module")
def cal() -> SessionCalendar:
    return SessionCalendar(start="2024-01-01", end="2024-12-31")


def _day_bars(cal, day=DAY, seed=1):
    bars = synthetic_bars(cal, day, day, seed=seed)
    return bars.tz_convert("UTC") if bars.index.tz is not None else bars


def test_session_windows_full_day(cal):
    bars = synthetic_bars(cal, DAY, DAY, seed=1)
    windows = session_windows(bars.tz_convert("UTC"), "TEST", DAY, cal)
    assert len(windows) == 26  # 6.5h regular session / 15min
    assert windows["r_w"].notna().all()
    assert (windows["n_subbars"] == 3).all()
    assert (windows["active_subbar_fraction"] == 1.0).all()
    assert windows.iloc[0]["slot"] == dt.time(9, 30)
    assert windows.iloc[-1]["slot"] == dt.time(15, 45)


def test_session_windows_gap_is_explicit(cal):
    bars = synthetic_bars(cal, DAY, DAY, seed=1).tz_convert("UTC")
    # Remove the 10:00-10:15 sub-bars entirely.
    gap_start = pd.Timestamp(2024, 3, 18, 10, 0, tz=ET).tz_convert("UTC")
    gap_end = gap_start + pd.Timedelta("15min")
    bars = bars.loc[(bars.index < gap_start) | (bars.index >= gap_end)]

    windows = session_windows(bars, "TEST", DAY, cal)
    gap_row = windows.loc[windows["slot"] == dt.time(10, 0)].iloc[0]
    assert np.isnan(gap_row["r_w"])
    assert gap_row["active_subbar_fraction"] == 0.0
    assert windows.loc[windows["slot"] != dt.time(10, 0), "r_w"].notna().all()


def test_tod_stats_min_obs_gate(cal):
    bars = synthetic_bars(cal, DAY, dt.date(2024, 3, 28), seed=2)  # 9 sessions
    windows = session_windows(bars.tz_convert("UTC"), "TEST", DAY, cal)
    all_windows = pd.concat(
        [
            session_windows(
                bars.tz_convert("UTC")[bars.index.date == d], "TEST", d, cal
            )
            for d in cal.sessions_between(DAY, dt.date(2024, 3, 28))
        ],
        ignore_index=True,
    )
    # 9 observations per slot < min_obs 60: nothing is trusted.
    stats = TimeOfDayStats(min_obs=60).fit(all_windows)
    assert stats.n_slots == 0
    transformed = stats.transform(windows)
    assert transformed["z_move"].isna().all()

    # Lower the gate and the same data fits fine.
    stats = TimeOfDayStats(min_obs=5).fit(all_windows)
    assert stats.n_slots == 26
    transformed = stats.transform(windows)
    assert transformed["z_move"].notna().all()
    assert transformed["dollar_vol_pctile"].between(0, 1).all()


def _window_row(**overrides):
    row = {
        "ticker": "TEST",
        "date": DAY,
        "window_start": pd.Timestamp(2024, 3, 18, 10, 0, tz=ET),
        "slot": dt.time(10, 0),
        "r_w": 0.004,          # 40 bps: > 2x the 11 bps placeholder cost
        "dollar_vol": 1e7,
        "trade_count": 500,
        "active_subbar_fraction": 1.0,
        "n_subbars": 3,
        "z_move": 2.0,
        "dollar_vol_pctile": 0.9,
        "trades_pctile": 0.9,
    }
    row.update(overrides)
    return row


def test_qualify_all_criteria_must_hold():
    config = LabelConfig()
    frame = pd.DataFrame(
        [
            _window_row(),                              # qualifies
            _window_row(z_move=1.0),                    # fails abnormal movement
            _window_row(r_w=0.0015),                    # fails cost coverage (~1.4x)
            _window_row(dollar_vol_pctile=0.5),         # fails liquidity
            _window_row(trade_count=10),                # fails not-an-artifact
            _window_row(z_move=np.nan),                 # missing stats never qualify
        ]
    )
    result = qualify_windows(frame, config)
    assert result["qualifies"].tolist() == [True, False, False, False, False, False]


def test_classify_sessions_three_classes(cal):
    config = LabelConfig()  # opening period = first 90 min: before 11:00

    def rows(ticker, qualifying_times):
        out = []
        for hhmm in ["09:30", "10:00", "11:30", "13:00", "14:30"]:
            hour, minute = map(int, hhmm.split(":"))
            out.append(
                {
                    "ticker": ticker,
                    "date": DAY,
                    "window_start": pd.Timestamp(2024, 3, 18, hour, minute, tz=ET),
                    "qualifies": hhmm in qualifying_times,
                }
            )
        return out

    qualified = pd.DataFrame(
        rows("QUIET", set())
        + rows("OPEN", {"09:30", "10:00"})
        + rows("SUST", {"09:30", "11:30", "13:00"})
    )
    result = classify_sessions(qualified, cal, config).set_index("ticker")
    assert result.loc["QUIET", "label"] == LABEL_NONE
    assert result.loc["OPEN", "label"] == LABEL_OPENING_ONLY
    assert result.loc["SUST", "label"] == LABEL_SUSTAINED
    assert result.loc["SUST", "n_late"] == 2
    assert result.loc["OPEN", "n_opening"] == 2


def test_end_to_end_on_synthetic_store(cal, tmp_path):
    store = BarStore(tmp_path / "bars.sqlite")
    try:
        bars = synthetic_bars(cal, dt.date(2024, 2, 1), dt.date(2024, 4, 30), seed=7)
        store.insert_bars("TEST", bars, source="test")
        data = AsOfData(store)

        fit = build_windows(data, cal, ["TEST"], dt.date(2024, 2, 1), dt.date(2024, 3, 31))
        stats = TimeOfDayStats(min_obs=20).fit(fit)
        assert stats.n_slots == 26

        windows = build_windows(data, cal, ["TEST"], dt.date(2024, 4, 1), dt.date(2024, 4, 30))
        qualified = qualify_windows(stats.transform(windows), LabelConfig())
        sessions = classify_sessions(qualified, cal, LabelConfig())

        assert len(sessions) == len(cal.sessions_between(dt.date(2024, 4, 1), dt.date(2024, 4, 30)))
        assert sessions["label"].isin([LABEL_NONE, LABEL_OPENING_ONLY, LABEL_SUSTAINED]).all()
        # A homogeneous random walk should rarely qualify: sanity-bound the rate.
        assert qualified["qualifies"].mean() < 0.20
    finally:
        store.close()
