# tests/test_orb_strategy.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.orb.strategy import (
    ORBConfig,
    find_orb_signal,
    opening_range,
    prior_session_atr,
    stop_and_target,
)
from stockpredictor.sessions import ET

DAY = dt.date(2024, 3, 18)


def _bars(rows):
    """rows: list of (hh, mm, open, high, low, close, volume)."""
    idx, data = [], []
    for hh, mm, o, h, lo, c, v in rows:
        idx.append(pd.Timestamp(DAY.year, DAY.month, DAY.day, hh, mm, tz=ET))
        data.append({"open": o, "high": h, "low": lo, "close": c, "volume": v})
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


def _flat_session(open_ts_hh=9, open_ts_mm=30, n=12, price=100.0, vol=1000):
    rows = []
    for k in range(n):
        t = pd.Timestamp(DAY.year, DAY.month, DAY.day, 9, 30, tz=ET) + pd.Timedelta(minutes=5 * k)
        rows.append((t.hour, t.minute, price, price, price, price, vol))
    return _bars(rows)


def test_opening_range_first_15_minutes():
    # three 5-min bars 09:30/09:35/09:40 form the 15-min range; later bars ignored.
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 102, 98, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 105, 95, 100, 1000),  # outside the 15-min window
    ])
    assert opening_range(session, ORBConfig(range_minutes=15)) == (102.0, 98.0)


def test_opening_range_empty():
    assert opening_range(_bars([]), ORBConfig()) is None


def test_prior_session_atr_true_range():
    prior = _bars([
        (9, 30, 100, 102, 99, 101, 1000),   # TR = 3 (no prev close)
        (9, 35, 101, 103, 100, 102, 1000),  # TR = max(3, |103-101|, |100-101|) = 3
    ])
    atr = prior_session_atr(prior, ORBConfig())
    assert atr == pytest.approx(3.0)


def test_prior_session_atr_none_when_empty():
    assert prior_session_atr(None, ORBConfig()) is None
    assert prior_session_atr(_bars([]), ORBConfig()) is None


def test_long_breakout_confirmed():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),   # range hi=101, lo=99
        (9, 45, 100, 103, 100, 102.5, 5000),  # close 102.5 > 101 + 0.1*ATR, high volume
    ])
    sig = find_orb_signal(session, atr=2.0, cfg=ORBConfig(range_minutes=15, min_break_atr=0.1, vol_mult=1.0))
    assert sig is not None
    assert sig.direction == 1
    assert sig.signal_bar == pd.Timestamp(DAY.year, DAY.month, DAY.day, 9, 45, tz=ET)


def test_short_breakout_confirmed():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 100, 96, 96.5, 5000),   # close 96.5 < 99 - 0.1*ATR
    ])
    sig = find_orb_signal(session, atr=2.0, cfg=ORBConfig(range_minutes=15))
    assert sig is not None and sig.direction == -1


def test_break_too_small_rejected():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 101.1, 100, 101.05, 5000),  # only 0.05 over hi; needs 0.1*ATR=0.2
    ])
    assert find_orb_signal(session, atr=2.0, cfg=ORBConfig(min_break_atr=0.1)) is None


def test_low_volume_break_rejected():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 100),  # strong break, but volume far below median
    ])
    assert find_orb_signal(session, atr=2.0, cfg=ORBConfig(vol_mult=1.0)) is None


def test_no_break_returns_none():
    assert find_orb_signal(_flat_session(), atr=2.0, cfg=ORBConfig()) is None


def test_causality_truncating_future_bars():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 5000),  # signal forms here
        (9, 50, 102, 110, 90, 95, 9000),      # wild future bar
    ])
    cfg = ORBConfig()
    full = find_orb_signal(session, atr=2.0, cfg=cfg)
    truncated = find_orb_signal(session.iloc[:4], atr=2.0, cfg=cfg)
    assert full is not None and truncated is not None
    assert full.signal_bar == truncated.signal_bar
    assert full.direction == truncated.direction


def test_stop_and_target_atr_long():
    stop, target, risk = stop_and_target(
        entry=102.0, direction=1, range_high=101, range_low=99, atr=2.0,
        cfg=ORBConfig(stop_mode="atr", stop_atr_mult=1.0, rr=2.0),
    )
    assert stop == pytest.approx(100.0)      # 102 - 1*2
    assert risk == pytest.approx(2.0)
    assert target == pytest.approx(106.0)    # 102 + 2*2


def test_stop_and_target_boundary_long():
    stop, target, risk = stop_and_target(
        entry=102.0, direction=1, range_high=101, range_low=99, atr=2.0,
        cfg=ORBConfig(stop_mode="boundary", rr=1.0),
    )
    assert stop == pytest.approx(99.0)       # opposite range side
    assert target == pytest.approx(105.0)    # 102 + 1*(102-99)
