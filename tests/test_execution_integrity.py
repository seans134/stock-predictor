import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.portfolio import RiskConfig, run_portfolio
from stockpredictor.execution.timing import barrier_exit
from stockpredictor.orb.backtest import simulate_orb_session
from stockpredictor.orb.strategy import ORBConfig
from stockpredictor.stage2.features import _forward_target

ET = "America/New_York"


@pytest.mark.parametrize("direction,opening,high,low,stop,target,expected", [
    (1, 98, 100, 97, 101, 105, (98, "stop", True)),
    (-1, 102, 104, 101, 100, 96, (102, "stop", True)),
    (1, 108, 110, 107, 101, 105, (105, "target", True)),
    (-1, 92, 94, 90, 100, 96, (96, "target", True)),
    (1, 103, 106, 100, 101, 105, (101, "stop", False)),
    (-1, 98, 101, 95, 100, 96, (100, "stop", False)),
])
def test_gap_and_ambiguous_bar_policy(direction, opening, high, low, stop, target, expected):
    assert barrier_exit({"open": opening, "high": high, "low": low}, direction, stop, target) == expected


def test_original_stop_gap_counterexample_now_loses_at_gap_open():
    rows = [(100,101,99,100,1000)] * 3 + [
        (100,103,100,102.5,5000), (103,103.5,102,103,5000),
        (103,103.5,102,103,5000), (98,100,97,99,5000),
        (104,106,103,105,5000)]
    bars = pd.DataFrame(rows, columns=["open","high","low","close","volume"],
                        index=pd.date_range("2024-03-18 09:30", periods=8, freq="5min", tz=ET))
    trade = simulate_orb_session(bars, 2, ORBConfig(), CostModel(), "AAA", dt.date(2024,3,18))
    assert trade.entry_bar == pd.Timestamp("2024-03-18 09:55", tz=ET)
    assert trade.exit_ts == pd.Timestamp("2024-03-18 10:00", tz=ET)
    assert trade.exit == 98 and trade.exit_reason == "stop"
    assert trade.gross_ret == pytest.approx(98 / 103 - 1)


def test_missing_bar_does_not_shorten_target_horizon():
    idx = pd.date_range("2024-03-18 10:00", periods=8, freq="5min", tz=ET)
    bars = pd.DataFrame({"open": np.arange(100,108), "close": np.arange(101,109)}, index=idx)
    assert _forward_target(bars).iloc[0] == pytest.approx(105 / 102 - 1)
    assert np.isnan(_forward_target(bars.drop(idx[1])).iloc[0])


def _trade(ticker, entry, exit_, ret, signal="2024-03-18 09:30"):
    return dict(ticker=ticker, bar_start=pd.Timestamp(signal, tz=ET),
                entry_ts=pd.Timestamp(entry, tz=ET), exit_ts=pd.Timestamp(exit_, tz=ET),
                q10=-.01, net_ret=ret, date=dt.date(2024,3,18))


def test_portfolio_uses_entry_order_not_signal_order_and_counts_initial_loss():
    trades = pd.DataFrame([
        _trade("A", "2024-03-18 11:00", "2024-03-18 12:00", .5),
        _trade("B", "2024-03-18 10:00", "2024-03-18 11:30", -.1, "2024-03-18 09:45"),
    ])
    result = run_portfolio(trades, RiskConfig(max_positions=1))
    assert result["accepted_trades"].ticker.tolist() == ["B"]
    assert result["max_drawdown"] == pytest.approx(-.02)


def test_duplicate_and_gross_cap_are_enforced():
    frame = pd.DataFrame([_trade(t, "2024-03-18 10:00", "2024-03-18 11:00", .01)
                          for t in ["A", "A", "B", "C", "D", "E", "F"]])
    result = run_portfolio(frame, RiskConfig(max_positions=20))
    assert result["n_taken"] == 5
    assert result["n_skipped_duplicate"] == 1
    assert result["n_skipped_capital"] == 1
    assert result["peak_gross_fraction"] <= 1


def test_prior_day_loss_does_not_halt_new_day():
    frame = pd.DataFrame([
        _trade("A", "2024-03-18 10:00", "2024-03-18 11:00", -.1),
        _trade("B", "2024-03-19 10:00", "2024-03-19 11:00", .01),
    ])
    assert run_portfolio(frame, RiskConfig(daily_loss_limit=.001))["n_taken"] == 2


def test_exit_on_new_day_counts_against_that_days_limit():
    frame = pd.DataFrame([
        _trade("A", "2024-03-18 10:00", "2024-03-19 10:00", -.1),
        _trade("B", "2024-03-19 10:05", "2024-03-19 11:00", .01),
    ])
    assert run_portfolio(frame, RiskConfig(daily_loss_limit=.001))["n_taken"] == 1


def test_unknown_downside_and_bad_timing_fail_closed():
    frame = pd.DataFrame([_trade("A", "2024-03-18 10:00", "2024-03-18 11:00", .01)])
    frame["q10"] = np.nan
    with pytest.raises(ValueError, match="unknown downside"):
        run_portfolio(frame, RiskConfig())
    frame["exit_ts"] = frame["entry_ts"]
    with pytest.raises(ValueError, match="entry < exit"):
        run_portfolio(frame, RiskConfig())
