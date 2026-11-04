"""Regression: run_portfolio must not flatten multi-day-hold positions (e.g.
pairs trades) at every day boundary. The old code force-flushed ALL open
positions via close_due(None) whenever the trade date changed, which broke
the cross-day concurrency cap, sized new trades on equity that already
included a still-open position's future P&L, and could build a
non-chronological equity curve. This pins the fix: close_due(entry_ts) each
iteration (no unconditional day-boundary flush), with day-change detection
used only to reset the daily counters, plus a chronologically sorted curve."""
import datetime as dt

import pandas as pd

from stockpredictor.execution.portfolio import RiskConfig, run_portfolio

ET = "America/New_York"

D1 = dt.date(2024, 3, 18)
D2 = dt.date(2024, 3, 19)
D5 = dt.date(2024, 3, 22)


def _ts(day, hh, mm):
    return pd.Timestamp(day.year, day.month, day.day, hh, mm, tz=ET)


def _pair_trade(entry_day, entry_ts, exit_ts, net_ret, ticker):
    return {
        "ticker": ticker, "date": entry_day, "direction": "long",
        "bar_start": entry_ts, "gross_ret": net_ret, "net_ret": net_ret,
        "q10": -0.02, "entry_ts": entry_ts, "exit_ts": exit_ts,
    }


def test_multiday_hold_blocks_concurrent_entry_across_day_boundary():
    # Trade A: entered day 1, held open across days 2-4, exits day 5.
    # Trade B: entered day 2, while A is still open.
    trades = pd.DataFrame([
        _pair_trade(D1, _ts(D1, 16, 0), _ts(D5, 16, 0), 0.01, "AAA/BBB"),
        _pair_trade(D2, _ts(D2, 16, 0), _ts(D2, 16, 30), 0.01, "CCC/DDD"),
    ])
    result = run_portfolio(trades, RiskConfig(max_positions=1))
    # Under the old code, the day-boundary close_due(None) would have flushed
    # A on day 2's first row, wrongly freeing a slot for B.
    assert result["n_taken"] == 1
    assert result["n_skipped_concurrency"] == 1


def test_multiday_hold_taken_with_headroom_and_curve_sorted():
    trades = pd.DataFrame([
        _pair_trade(D1, _ts(D1, 16, 0), _ts(D5, 16, 0), 0.01, "AAA/BBB"),
        _pair_trade(D2, _ts(D2, 16, 0), _ts(D2, 16, 30), 0.01, "CCC/DDD"),
    ])
    result = run_portfolio(trades, RiskConfig(max_positions=10))
    assert result["n_taken"] == 2
    idx = result["equity_curve"].index
    assert list(idx) == sorted(idx)
