# src/stockpredictor/pairs/data.py
"""Daily RTH-close panel from the 5-minute store, for pairs formation/trading.

One column per ticker, one row per ET session, value = the last regular-session
5-minute bar's close. Fetched through the as-of accessor at the range end
(finished sessions). Missing ticker-days stay NaN and are never zero-filled.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.data.asof import AsOfData
from stockpredictor.labeling.windows import LABEL_AS_OF_DELAY
from stockpredictor.sessions import ET, SessionCalendar


def load_daily_closes(
    data: AsOfData, calendar: SessionCalendar, tickers: list[str],
    start: dt.date, end: dt.date,
) -> pd.DataFrame:
    sessions = calendar.sessions_between(start, end)
    if not sessions:
        return pd.DataFrame()
    session_set = set(sessions)
    range_open = pd.Timestamp(calendar.session_open(sessions[0]))
    range_close = pd.Timestamp(calendar.session_close(sessions[-1]))
    as_of = range_close + LABEL_AS_OF_DELAY
    opens = {d: pd.Timestamp(calendar.session_open(d)) for d in sessions}
    closes = {d: pd.Timestamp(calendar.session_close(d)) for d in sessions}

    series: dict[str, pd.Series] = {}
    for ticker in tickers:
        bars = data.get_bars(ticker, as_of=as_of, lookback=as_of - range_open)
        if bars.empty:
            continue
        et = bars.index.tz_convert(ET)
        day = pd.Series(et.date, index=bars.index)
        ts = pd.Series(et, index=bars.index)
        mask = day.isin(session_set) & (ts >= day.map(opens)) & (ts < day.map(closes))
        rth = bars.loc[mask.to_numpy()]
        if rth.empty:
            continue
        rth_day = pd.Series(rth.index.tz_convert(ET).date, index=rth.index)
        series[ticker] = rth.groupby(rth_day)["close"].last()
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series).sort_index()
    panel.index.name = "date"
    return panel
