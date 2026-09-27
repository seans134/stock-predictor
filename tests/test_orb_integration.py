"""Two-arm ORB: the gated arm is a strict subset of the pure arm's ticker-days,
and both replay through the risk engine. Uses synthetic bars only (no DB)."""

import datetime as dt

import pandas as pd

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.portfolio import RiskConfig, run_portfolio
from stockpredictor.orb.backtest import run_orb
from stockpredictor.orb.strategy import ORBConfig
from stockpredictor.sessions import ET

PRIOR = dt.date(2024, 3, 15)
DAY = dt.date(2024, 3, 18)


def _session(day, rows):
    idx, data = [], []
    for hh, mm, o, h, lo, c, v in rows:
        idx.append(pd.Timestamp(day.year, day.month, day.day, hh, mm, tz=ET))
        data.append({"open": o, "high": h, "low": lo, "close": c, "volume": v})
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


_PRIOR = [(9, 30, 100, 102, 99, 101, 1000), (9, 35, 101, 103, 100, 102, 1000)]
_BREAKOUT = [
    (9, 30, 100, 101, 99, 100, 1000),
    (9, 35, 100, 101, 99, 100, 1000),
    (9, 40, 100, 101, 99, 100, 1000),
    (9, 45, 100, 103, 100, 102.5, 5000),
    (9, 50, 103, 106, 102, 105, 5000),
]


def test_gated_is_subset_of_pure_and_portfolio_runs():
    bars = pd.concat([_session(PRIOR, _PRIOR), _session(DAY, _BREAKOUT)])
    universe = {"AAA": bars, "BBB": bars, "CCC": bars}
    cfg = ORBConfig(rr=1.0)

    pure = run_orb(universe, cfg, CostModel())
    watchlist = {("AAA", DAY), ("BBB", DAY)}  # Stage 1 flagged only two names
    gated = run_orb(universe, cfg, CostModel(), universe_filter=watchlist)

    pure_pairs = set(zip(pure["ticker"], pure["date"]))
    gated_pairs = set(zip(gated["ticker"], gated["date"]))
    assert gated_pairs <= pure_pairs
    assert gated_pairs == {("AAA", DAY), ("BBB", DAY)}

    # Both arms replay through the extended risk engine without error.
    for arm in (pure, gated):
        result = run_portfolio(arm, RiskConfig(max_positions=10))
        assert result["n_taken"] >= 1
