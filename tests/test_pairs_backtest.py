import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.pairs.backtest import (
    PAIRS_TRADE_COLUMNS,
    run_pairs,
    simulate_pair,
    spread_zscore,
)
from stockpredictor.pairs.selection import PairModel, PairsConfig
from stockpredictor.sessions import ET

# beta=1, alpha=0, so spread = log A - log B; mu=0, sigma=0.01 for easy z arithmetic.
MODEL = PairModel(a="A", b="B", alpha=0.0, beta=1.0, mu=0.0, sigma=0.01, adf_t=-5.0)


def _dates(n):
    return [dt.date(2024, 3, 1) + dt.timedelta(days=k) for k in range(n)]


def _panel(a_prices, b_prices):
    idx = _dates(len(a_prices))
    return pd.DataFrame({"A": a_prices, "B": b_prices}, index=idx)


def _session_ts(dates):
    return {d: pd.Timestamp(d.year, d.month, d.day, 16, 0, tz=ET) for d in dates}


def test_spread_zscore_matches_hand_computation():
    closes = _panel([np.e ** 0.02, np.e ** 0.00], [1.0, 1.0])  # log A = 0.02, 0.0; log B = 0
    z = spread_zscore(MODEL, closes["A"], closes["B"])
    assert z.iloc[0] == pytest.approx(0.02 / 0.01)  # (0.02-0)/0.01 = 2.0
    assert z.iloc[1] == pytest.approx(0.0)


def _z_to_prices(z_path):
    # Build A,B so that (logA - logB - mu)/sigma == z; keep B=1 (logB=0), logA = z*sigma.
    a = [float(np.exp(z * MODEL.sigma)) for z in z_path]
    b = [1.0] * len(z_path)
    return _panel(a, b)


def test_long_a_short_b_on_negative_z_then_reversion_exit():
    # z: -3 (signal), fill next day, then revert to 0 (decision day) and fill the
    # exit the day AFTER that (delayed exactly like the entry). Pad tail so both
    # the entry fill and the delayed exit fill have room.
    closes = _z_to_prices([-3.0, -3.0, 0.0, 0.0, 0.0])
    cfg = PairsConfig(z_entry=2.0, z_exit=0.5, z_stop=4.0, max_hold=10)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1
    tr = trades[0]
    assert tr.direction == 1                                   # long A / short B
    assert tr.entry_date == closes.index[1]                    # filled the day AFTER the -3 signal
    assert tr.exit_reason == "reversion"
    assert tr.exit_date == closes.index[3]                     # reversion decided at idx2, filled at idx3
    # A rose from exp(-3*sigma) to exp(0)=1 while B flat -> long-A/short-B is profitable.
    assert tr.gross_ret > 0
    assert tr.net_ret == pytest.approx(tr.gross_ret - 2 * CostModel().round_trip_cost)


def test_short_a_long_b_on_positive_z():
    closes = _z_to_prices([3.0, 3.0, 0.0, 0.0, 0.0])
    cfg = PairsConfig(z_entry=2.0, z_exit=0.5, z_stop=4.0, max_hold=10)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1 and trades[0].direction == -1
    assert trades[0].exit_date == closes.index[3]  # reversion decided at idx2, filled at idx3
    assert trades[0].gross_ret > 0  # A fell to par, short-A/long-B profits


def test_divergence_stop():
    # enter at -2.5, then diverge to -4.5 (|z| >= z_stop); stop decided at idx2,
    # filled at idx3 (the delayed exit).
    closes = _z_to_prices([-2.5, -2.5, -4.5, -4.5, -4.5])
    cfg = PairsConfig(z_entry=2.0, z_exit=0.5, z_stop=4.0, max_hold=10)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1 and trades[0].exit_reason == "stop"
    assert trades[0].exit_date == closes.index[3]


def test_time_stop():
    # enter, never revert, never diverge past stop -> time-stop decided once
    # held >= max_hold, filled the session after that decision.
    z = [-2.5, -2.5] + [-1.5] * 6
    closes = _z_to_prices(z)
    cfg = PairsConfig(z_entry=2.0, z_exit=0.2, z_stop=5.0, max_hold=3)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1 and trades[0].exit_reason == "time"
    # entry fills at idx1; held>=3 first true at idx4 (4-1=3) -> decided at idx4,
    # filled at idx5.
    assert trades[0].exit_date == closes.index[5]


def test_window_end_force_flat():
    # enter near the end, no exit trigger -> forced flat on the last session.
    closes = _z_to_prices([0.0, -2.5, -2.5, -1.5])
    cfg = PairsConfig(z_entry=2.0, z_exit=0.1, z_stop=5.0, max_hold=50)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1 and trades[0].exit_reason == "window_end"
    assert trades[0].exit_date == closes.index[-1]


def test_run_pairs_schema_and_direction_string():
    closes = _z_to_prices([-3.0, -3.0, 0.0, 0.0, 0.0])
    df = run_pairs([MODEL], closes, PairsConfig(z_entry=2.0, z_exit=0.5),
                   CostModel(), _session_ts(closes.index))
    assert list(df.columns) == PAIRS_TRADE_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "A/B"
    assert df.iloc[0]["direction"] == LONG
    assert df.iloc[0]["q10"] < 0
    assert df.iloc[0]["exit_ts"] > df.iloc[0]["entry_ts"]


def test_run_pairs_skips_missing_columns():
    closes = _z_to_prices([-3.0, -3.0, 0.0])
    other = PairModel(a="X", b="Y", alpha=0.0, beta=1.0, mu=0.0, sigma=0.01, adf_t=-5.0)
    df = run_pairs([other], closes, PairsConfig(), CostModel(), _session_ts(closes.index))
    assert df.empty and list(df.columns) == PAIRS_TRADE_COLUMNS
