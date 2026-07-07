"""Loading regular-session bars for Stage 2.

The store holds extended-hours bars too (Stage 1 needs pre-market), but
Stage 2 v1 predicts only inside the regular session, so bars are filtered
to each day's open..close using the session calendar.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.data.asof import AsOfData
from stockpredictor.labeling.windows import LABEL_AS_OF_DELAY
from stockpredictor.sessions import ET, SessionCalendar


def load_rth_bars(
    data: AsOfData,
    calendar: SessionCalendar,
    ticker: str,
    start: dt.date,
    end: dt.date,
) -> pd.DataFrame:
    """Regular-session bars for [start, end], fetched through the as-of
    accessor at the range end (training data for finished sessions)."""
    sessions = calendar.sessions_between(start, end)
    if not sessions:
        return pd.DataFrame()
    range_open = pd.Timestamp(calendar.session_open(sessions[0]))
    range_close = pd.Timestamp(calendar.session_close(sessions[-1]))
    as_of = range_close + LABEL_AS_OF_DELAY
    bars = data.get_bars(ticker, as_of=as_of, lookback=as_of - range_open)
    if bars.empty:
        return bars

    et_index = bars.index.tz_convert(ET)
    opens = {d: pd.Timestamp(calendar.session_open(d)) for d in sessions}
    closes = {d: pd.Timestamp(calendar.session_close(d)) for d in sessions}
    day = pd.Series(et_index.date, index=bars.index)
    open_ts = day.map(opens)
    close_ts = day.map(closes)
    mask = open_ts.notna() & (pd.Series(et_index, index=bars.index) >= open_ts) & (
        pd.Series(et_index, index=bars.index) < close_ts
    )
    return bars.loc[mask.to_numpy()]
