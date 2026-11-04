"""Availability-time convention for ingested bars.

A bar that starts at event_ts is not knowable until it completes at
event_ts + timeframe, and in live operation it arrives some time after
that. Historical backfills must use the same conservative assumption so
backtest visibility never beats what live would have seen.
"""

from __future__ import annotations

import pandas as pd

# Assumed feed delay between a bar completing and the system being able to
# use it. Named constant per the quantitative-representation rule; revisit
# against measured live latency once the live feed runs in shadow mode.
BAR_INGEST_LATENCY = pd.Timedelta(seconds=5)

# Conservative delay between an article's claimed publish time and when the
# system could have used it. Historical news carries only published_utc (no
# first-seen), so backfill must assume a generous latency per the plan's
# timestamp-integrity rules. Live ingestion will record true first-seen
# times; until then this constant errs on the late side.
NEWS_INGEST_LATENCY = pd.Timedelta(minutes=15)

TIMEFRAME_DURATIONS = {
    "1min": pd.Timedelta(minutes=1),
    "5min": pd.Timedelta(minutes=5),
    "15min": pd.Timedelta(minutes=15),
    "30min": pd.Timedelta(minutes=30),
}


def bar_available_ts(event_index: pd.DatetimeIndex, timeframe: str) -> pd.Series:
    """Conservative availability timestamp for bars starting at event_index."""
    duration = TIMEFRAME_DURATIONS[timeframe]
    return pd.Series(event_index + duration + BAR_INGEST_LATENCY, index=event_index)
