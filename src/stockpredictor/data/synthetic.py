"""Synthetic 5-minute bars for tests and pipeline development.

Random-walk prices on the real NYSE session grid, with availability
timestamps assigned by the same convention as real ingestion. Not a market
simulator — just structurally valid data so the pipeline can be exercised
without vendor keys.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from stockpredictor.data.availability import bar_available_ts
from stockpredictor.sessions import SessionCalendar


def synthetic_bars(
    calendar: SessionCalendar,
    start: dt.date,
    end: dt.date,
    seed: int = 0,
    initial_price: float = 100.0,
    timeframe: str = "5min",
) -> pd.DataFrame:
    """Generate bars for every regular-session 5-min window in [start, end].

    Returns a DataFrame indexed by tz-aware event timestamp (bar start, ET)
    with OHLCV, trade_count, vwap, and available_ts columns — the exact shape
    BarStore.insert_bars expects.
    """
    rng = np.random.default_rng(seed)
    step = pd.Timedelta(timeframe)

    starts: list[pd.Timestamp] = []
    for day in calendar.sessions_between(start, end):
        open_ts = pd.Timestamp(calendar.session_open(day))
        close_ts = pd.Timestamp(calendar.session_close(day))
        starts.extend(pd.date_range(open_ts, close_ts - step, freq=step))
    index = pd.DatetimeIndex(starts)

    n = len(index)
    log_returns = rng.normal(0.0, 0.001, size=n)
    closes = initial_price * np.exp(np.cumsum(log_returns))
    opens = np.roll(closes, 1)
    opens[0] = initial_price
    spread = np.abs(rng.normal(0.0, 0.0005, size=n)) * closes
    highs = np.maximum(opens, closes) + spread
    lows = np.minimum(opens, closes) - spread
    volume = rng.integers(1_000, 50_000, size=n).astype(float)

    bars = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volume,
            "trade_count": rng.integers(10, 500, size=n),
            "vwap": (highs + lows + closes) / 3,
        },
        index=index,
    )
    bars["available_ts"] = bar_available_ts(index, timeframe)
    return bars
