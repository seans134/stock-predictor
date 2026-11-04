"""The as-of accessor: the only door feature code may use to reach data.

Feature functions never read raw storage or vendor APIs directly. They ask
this accessor "what was knowable at time T", and the same accessor serves
backtest (replaying availability timestamps) and live (what has actually
arrived). Point-in-time correctness is enforced here, by construction.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.data.news_store import NewsStore
from stockpredictor.data.store import BarStore


class AsOfData:
    def __init__(self, store: BarStore, news: NewsStore | None = None):
        self._store = store
        self._news = news

    def get_bars(
        self,
        ticker: str,
        as_of: dt.datetime | pd.Timestamp,
        lookback: dt.timedelta | pd.Timedelta | None = None,
        timeframe: str = "5min",
    ) -> pd.DataFrame:
        """Bars for `ticker` visible at `as_of`, oldest first.

        `lookback` bounds how far back the window starts (measured from
        `as_of`); None returns all visible history. Bars whose availability
        timestamp is after `as_of` do not exist from the caller's point of
        view, even if their event time is earlier.
        """
        as_of = pd.Timestamp(as_of)
        start = as_of - pd.Timedelta(lookback) if lookback is not None else None
        return self._store.get_bars_asof(
            ticker, as_of=as_of, start=start, end=as_of, timeframe=timeframe
        )

    def get_news(
        self,
        ticker: str,
        as_of: dt.datetime | pd.Timestamp,
        lookback: dt.timedelta | pd.Timedelta | None = None,
    ) -> pd.DataFrame:
        """Articles tagged with `ticker` visible at `as_of`, oldest first.
        Articles whose availability timestamp is after `as_of` do not
        exist from the caller's point of view."""
        if self._news is None:
            raise ValueError("AsOfData was constructed without a NewsStore")
        as_of = pd.Timestamp(as_of)
        start = as_of - pd.Timedelta(lookback) if lookback is not None else None
        return self._news.get_news_asof(ticker, as_of=as_of, start=start)
