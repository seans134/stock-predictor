"""15-minute window construction for the Stage 1 qualifying-opportunity label.

Each regular trading session is divided into non-overlapping 15-minute
windows aligned to the opening bell. Every window aggregates its three
5-minute sub-bars into the direction-neutral quantities the qualification
criteria need. Windows that would extend past an early close are dropped.

All bars reaching this module must already have passed through the as-of
accessor; this module never touches storage directly.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from stockpredictor.data.asof import AsOfData
from stockpredictor.sessions import ET, SessionCalendar

WINDOW = pd.Timedelta("15min")
SUBBAR = pd.Timedelta("5min")
SUBBARS_PER_WINDOW = int(WINDOW / SUBBAR)

# Labels describe a finished session; they are computed as of this long
# after the close so late-arriving bars are included, deterministically.
LABEL_AS_OF_DELAY = pd.Timedelta("30min")

WINDOW_COLUMNS = [
    "ticker",
    "date",
    "window_start",  # tz-aware ET
    "slot",          # ET wall-clock time of window start (time-of-day bucket key)
    "r_w",           # abs 15-min log displacement, NaN if window has no bars
    "dollar_vol",
    "trade_count",
    "active_subbar_fraction",
    "n_subbars",
]


def session_windows(
    bars: pd.DataFrame,
    ticker: str,
    day: dt.date,
    calendar: SessionCalendar,
) -> pd.DataFrame:
    """Aggregate one session's 5-min bars (UTC-indexed, as-of filtered)
    into 15-minute window rows. Empty windows are kept with NaN movement so
    downstream code sees gaps explicitly rather than silently."""
    open_ts = pd.Timestamp(calendar.session_open(day))
    close_ts = pd.Timestamp(calendar.session_close(day))

    if not bars.empty:
        bars = bars.copy()
        bars.index = bars.index.tz_convert(ET)
        bars = bars.loc[(bars.index >= open_ts) & (bars.index < close_ts)]

    rows = []
    for window_start in pd.date_range(open_ts, close_ts - WINDOW, freq=WINDOW):
        if bars.empty:
            sub = bars
        else:
            sub = bars.loc[
                (bars.index >= window_start) & (bars.index < window_start + WINDOW)
            ]
        n_subbars = len(sub)
        if n_subbars == 0:
            r_w = np.nan
            dollar_vol = 0.0
            trade_count = 0
            active_fraction = 0.0
        else:
            first_open = float(sub.iloc[0]["open"])
            last_close = float(sub.iloc[-1]["close"])
            r_w = abs(np.log(last_close / first_open))
            price_ref = sub["vwap"].fillna(sub["close"])
            dollar_vol = float((price_ref * sub["volume"]).sum())
            trade_count = int(sub["trade_count"].fillna(0).sum())
            active_fraction = float((sub["volume"] > 0).sum() / SUBBARS_PER_WINDOW)

        rows.append(
            {
                "ticker": ticker,
                "date": day,
                "window_start": window_start,
                "slot": window_start.time(),
                "r_w": r_w,
                "dollar_vol": dollar_vol,
                "trade_count": trade_count,
                "active_subbar_fraction": active_fraction,
                "n_subbars": n_subbars,
            }
        )
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def build_windows(
    data: AsOfData,
    calendar: SessionCalendar,
    tickers: list[str],
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Window rows for every ticker-session in [start, end], fetched
    through the as-of accessor at close + LABEL_AS_OF_DELAY."""
    frames = []
    for day in calendar.sessions_between(start, end):
        open_ts = pd.Timestamp(calendar.session_open(day))
        close_ts = pd.Timestamp(calendar.session_close(day))
        as_of = close_ts + LABEL_AS_OF_DELAY
        for ticker in tickers:
            # Lookback comfortably covers open..close from the as-of point.
            bars = data.get_bars(ticker, as_of=as_of, lookback=pd.Timedelta("9h"))
            bars = bars.loc[
                (bars.index >= open_ts.tz_convert("UTC"))
                & (bars.index < close_ts.tz_convert("UTC"))
            ]
            frames.append(session_windows(bars, ticker, day, calendar))
    if not frames:
        return pd.DataFrame(columns=WINDOW_COLUMNS)
    return pd.concat(frames, ignore_index=True)
