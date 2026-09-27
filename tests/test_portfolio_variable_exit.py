# tests/test_portfolio_variable_exit.py
import datetime as dt

import pandas as pd
import pytest

from stockpredictor.execution.portfolio import RiskConfig, run_portfolio

DAY = dt.date(2024, 3, 18)
ET = "America/New_York"


def _ts(hh, mm, day=DAY):
    return pd.Timestamp(day.year, day.month, day.day, hh, mm, tz=ET)


def _orb_trade(signal_hh, signal_mm, entry, exit_, net_ret, q10=-0.02, ticker="AAA", day=DAY):
    return {
        "ticker": ticker, "date": day, "direction": "long",
        "bar_start": _ts(signal_hh, signal_mm, day),
        "gross_ret": net_ret, "net_ret": net_ret, "q10": q10,
        "entry_ts": entry, "exit_ts": exit_,
    }


def test_variable_exit_realizes_pnl_at_exit_ts():
    # Position held 09:50 -> 10:40 (much longer than the 15-min default).
    trades = pd.DataFrame([
        _orb_trade(9, 45, _ts(9, 50), _ts(10, 40), 0.02),
    ])
    result = run_portfolio(trades, RiskConfig(max_positions=10))
    assert result["n_taken"] == 1
    # Curve point lands at the ORB exit time, not at a fixed 15-min horizon.
    assert result["equity_curve"].index[-1] == _ts(10, 40)


def test_variable_exit_concurrency_counts_full_span():
    # Two ORB trades overlap 09:50-10:40 and 10:00-10:30; cap 1 blocks the second.
    trades = pd.DataFrame([
        _orb_trade(9, 45, _ts(9, 50), _ts(10, 40), 0.01, ticker="AAA"),
        _orb_trade(9, 55, _ts(10, 0), _ts(10, 30), 0.01, ticker="BBB"),
    ])
    result = run_portfolio(trades, RiskConfig(max_positions=1))
    assert result["n_taken"] == 1
    assert result["n_skipped_concurrency"] == 1


def test_fallback_without_ts_columns_unchanged():
    # No entry_ts/exit_ts -> the existing fixed 15-min behavior (regression pin).
    trades = pd.DataFrame([{
        "ticker": "AAA", "date": DAY, "direction": "long",
        "bar_start": _ts(10, 0), "gross_ret": 0.01, "net_ret": 0.01, "q10": -0.01,
    }])
    result = run_portfolio(trades, RiskConfig(max_positions=10))
    assert result["n_taken"] == 1
    # Fixed horizon exit = bar_start + 5*(HORIZON_BARS+1) = 10:00 + 20min = 10:20.
    assert result["equity_curve"].index[-1] == _ts(10, 20)
