import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.orb.backtest import ORB_TRADE_COLUMNS, run_orb, simulate_orb_session
from stockpredictor.orb.strategy import ORBConfig
from stockpredictor.sessions import ET

DAY = dt.date(2024, 3, 18)
PRIOR = dt.date(2024, 3, 15)


def _session(day, rows):
    idx, data = [], []
    for hh, mm, o, h, lo, c, v in rows:
        idx.append(pd.Timestamp(day.year, day.month, day.day, hh, mm, tz=ET))
        data.append({"open": o, "high": h, "low": lo, "close": c, "volume": v})
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


# A long breakout at 09:45, then a bar that reaches the target.
LONG_DAY_ROWS = [
    (9, 30, 100, 101, 99, 100, 1000),
    (9, 35, 100, 101, 99, 100, 1000),
    (9, 40, 100, 101, 99, 100, 1000),   # range hi=101, lo=99
    (9, 45, 100, 103, 100, 102.5, 5000),  # confirm long (close>101+0.2)
    (9, 50, 103, 108, 102, 107, 5000),    # entry at 09:50 open=103; target hits
    (9, 55, 107, 108, 106, 107, 5000),
]


def test_entry_at_next_bar_open_and_target():
    session = _session(DAY, LONG_DAY_ROWS)
    # ATR=2, stop_atr_mult=1 -> entry 103, stop 101, risk 2, rr=1 -> target 105.
    trade = simulate_orb_session(session, atr=2.0, cfg=ORBConfig(rr=1.0), costs=CostModel(),
                                 ticker="AAA", day=DAY)
    assert trade is not None
    assert trade.entry == pytest.approx(103.0)          # 09:50 open
    assert trade.entry_bar == pd.Timestamp(2024, 3, 18, 9, 50, tz=ET)
    assert trade.exit_reason == "target"
    assert trade.exit == pytest.approx(105.0)
    assert trade.gross_ret == pytest.approx((105.0 - 103.0) / 103.0)
    assert trade.net_ret == pytest.approx(trade.gross_ret - CostModel().round_trip_cost)
    assert trade.q10_hint() == pytest.approx(-trade.risk_frac_return)  # see note below


def test_stop_fills_first_when_bar_spans_both():
    rows = [
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 5000),   # confirm long
        (9, 50, 103, 106, 100, 104, 5000),     # entry 103; bar spans stop 101 AND target 105
    ]
    trade = simulate_orb_session(_session(DAY, rows), atr=2.0, cfg=ORBConfig(rr=1.0),
                                 costs=CostModel(), ticker="AAA", day=DAY)
    assert trade.exit_reason == "stop"
    assert trade.exit == pytest.approx(101.0)


def test_time_stop_at_eod():
    rows = [
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 5000),   # confirm long, entry next bar
        (9, 50, 103, 103.5, 102, 103, 5000),   # neither stop(101) nor target(105) hit
        (9, 55, 103, 103.5, 102, 103.2, 5000),
    ]
    trade = simulate_orb_session(_session(DAY, rows), atr=2.0, cfg=ORBConfig(rr=1.0),
                                 costs=CostModel(), ticker="AAA", day=DAY)
    assert trade.exit_reason == "time"
    assert trade.exit == pytest.approx(103.2)              # last bar close
    assert trade.exit_ts == pd.Timestamp(2024, 3, 18, 9, 55, tz=ET)


def test_short_pnl_sign():
    rows = [
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 100, 96, 96.5, 5000),    # confirm short, entry next bar
        (9, 50, 96, 96, 90, 92, 5000),        # entry 96; target = 96 - (96-98)=... see ATR
    ]
    # ATR=2, stop_atr_mult=1 -> short entry 96, stop 98, risk 2, rr=1 -> target 94.
    trade = simulate_orb_session(_session(DAY, rows), atr=2.0, cfg=ORBConfig(rr=1.0),
                                 costs=CostModel(), ticker="AAA", day=DAY)
    assert trade.direction == -1
    assert trade.exit_reason == "target"
    assert trade.exit == pytest.approx(94.0)
    assert trade.gross_ret == pytest.approx((-1) * (94.0 - 96.0) / 96.0)  # positive


def test_no_signal_returns_none():
    flat = _session(DAY, [
        (9, 30, 100, 100, 100, 100, 1000),
        (9, 35, 100, 100, 100, 100, 1000),
        (9, 40, 100, 100, 100, 100, 1000),
        (9, 45, 100, 100, 100, 100, 1000),
    ])
    assert simulate_orb_session(flat, atr=2.0, cfg=ORBConfig(), costs=CostModel(),
                                ticker="AAA", day=DAY) is None


def test_nan_entry_open_does_not_trade():
    rows = [
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 5000),   # confirm long, entry is next bar
        (9, 50, np.nan, 106, 102, 105, 5000),   # entry bar has NaN open
    ]
    assert simulate_orb_session(_session(DAY, rows), atr=2.0, cfg=ORBConfig(rr=1.0),
                                costs=CostModel(), ticker="AAA", day=DAY) is None


def test_nan_final_close_time_stop_does_not_trade():
    rows = [
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 5000),   # confirm long, entry next bar
        (9, 50, 103, 103.5, 102, 103, 5000),   # neither stop(101) nor target(105) hit
        (9, 55, 103, 103.5, 102, np.nan, 5000),  # final bar NaN close -> can't close cleanly
    ]
    assert simulate_orb_session(_session(DAY, rows), atr=2.0, cfg=ORBConfig(rr=1.0),
                                costs=CostModel(), ticker="AAA", day=DAY) is None


def _prior_flat(day):
    return _session(day, [
        (9, 30, 100, 102, 99, 101, 1000),
        (9, 35, 101, 103, 100, 102, 1000),
    ])


def test_run_orb_schema_and_columns():
    bars = pd.concat([_prior_flat(PRIOR), _session(DAY, LONG_DAY_ROWS)])
    trades = run_orb({"AAA": bars}, cfg=ORBConfig(rr=1.0), costs=CostModel())
    assert list(trades.columns) == ORB_TRADE_COLUMNS
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["direction"] == LONG
    assert row["q10"] < 0                       # -risk_frac_return
    assert row["entry_ts"] == pd.Timestamp(2024, 3, 18, 9, 50, tz=ET)


def test_run_orb_universe_filter_gates_names():
    bars = pd.concat([_prior_flat(PRIOR), _session(DAY, LONG_DAY_ROWS)])
    both = {"AAA": bars, "BBB": bars}
    pure = run_orb(both, cfg=ORBConfig(rr=1.0), costs=CostModel())
    gated = run_orb(both, cfg=ORBConfig(rr=1.0), costs=CostModel(),
                    universe_filter={("AAA", DAY)})
    assert set(pure["ticker"]) == {"AAA", "BBB"}
    assert set(gated["ticker"]) == {"AAA"}


def test_run_orb_skips_first_session_no_prior_atr():
    # Only one session available -> no prior ATR -> no trades.
    trades = run_orb({"AAA": _session(DAY, LONG_DAY_ROWS)}, cfg=ORBConfig(), costs=CostModel())
    assert trades.empty
