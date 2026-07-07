"""NYSE session calendar.

Single source of truth for trading days, open/close times, half-days,
and the pre-market cutoff. All returned datetimes are timezone-aware
US Eastern, per the project plan's timezone rule.
"""

from __future__ import annotations

import datetime as dt
from functools import lru_cache
from zoneinfo import ZoneInfo

import pandas as pd
import pandas_market_calendars as mcal

ET = ZoneInfo("America/New_York")

# Pre-market feature cutoff: all Stage 1 inputs must be available by this
# ET wall-clock time. Named constant per the quantitative-representation rule.
DEFAULT_PREMARKET_CUTOFF = dt.time(9, 0)


class SessionCalendar:
    """Wraps the exchange calendar behind the project's session semantics."""

    def __init__(self, start: str = "2015-01-01", end: str = "2030-12-31"):
        self._cal = mcal.get_calendar("NYSE")
        self._schedule = self._cal.schedule(start_date=start, end_date=end)
        # Normalize to ET once; pandas_market_calendars returns UTC.
        self._schedule = self._schedule.tz_convert(ET) if self._schedule.index.tz else pd.DataFrame(
            {
                col: self._schedule[col].dt.tz_convert(ET)
                for col in ("market_open", "market_close")
            },
            index=self._schedule.index,
        )

    def is_session(self, day: dt.date) -> bool:
        return pd.Timestamp(day) in self._schedule.index

    def sessions_between(self, start: dt.date, end: dt.date) -> list[dt.date]:
        idx = self._schedule.loc[str(start) : str(end)].index
        return [ts.date() for ts in idx]

    def session_open(self, day: dt.date) -> dt.datetime:
        return self._row(day)["market_open"].to_pydatetime()

    def session_close(self, day: dt.date) -> dt.datetime:
        """Handles half-days: early closes come from the exchange calendar."""
        return self._row(day)["market_close"].to_pydatetime()

    def premarket_cutoff(
        self, day: dt.date, cutoff: dt.time = DEFAULT_PREMARKET_CUTOFF
    ) -> dt.datetime:
        if not self.is_session(day):
            raise ValueError(f"{day} is not a trading session")
        return dt.datetime.combine(day, cutoff, tzinfo=ET)

    def next_session(self, day: dt.date) -> dt.date:
        after = self._schedule.index[self._schedule.index > pd.Timestamp(day)]
        if len(after) == 0:
            raise ValueError(f"no session after {day} in loaded calendar range")
        return after[0].date()

    def _row(self, day: dt.date) -> pd.Series:
        try:
            return self._schedule.loc[pd.Timestamp(day)]
        except KeyError:
            raise ValueError(f"{day} is not a trading session") from None


@lru_cache(maxsize=1)
def default_calendar() -> SessionCalendar:
    return SessionCalendar()
