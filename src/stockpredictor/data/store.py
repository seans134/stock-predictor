"""Bitemporal bar storage.

Every stored bar carries two timestamps, per the project plan:

- event_ts:     when the bar occurred (bar START time)
- available_ts: when the system could first have used it

Revisions of the same bar are stored as additional rows with a later
available_ts; nothing is ever overwritten. "As of T" queries return, for
each event_ts, the latest revision whose available_ts <= T. This makes
point-in-time visibility a filter, not a guess.

All timestamps are stored as integer epoch seconds (UTC). The API accepts
only timezone-aware datetimes so naive-timestamp bugs fail loudly.
"""

from __future__ import annotations

import datetime as dt
import sqlite3
from pathlib import Path

import pandas as pd

BAR_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "trade_count",
    "vwap",
]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    ticker       TEXT    NOT NULL,
    timeframe    TEXT    NOT NULL,
    event_ts     INTEGER NOT NULL,
    available_ts INTEGER NOT NULL,
    open         REAL    NOT NULL,
    high         REAL    NOT NULL,
    low          REAL    NOT NULL,
    close        REAL    NOT NULL,
    volume       REAL    NOT NULL,
    trade_count  INTEGER,
    vwap         REAL,
    ingested_ts  INTEGER NOT NULL,
    source       TEXT    NOT NULL,
    PRIMARY KEY (ticker, timeframe, event_ts, available_ts)
);
CREATE INDEX IF NOT EXISTS idx_bars_asof
    ON bars (ticker, timeframe, available_ts, event_ts);
"""


def _to_epoch(ts: dt.datetime | pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp {ts!r} must be timezone-aware")
    return int(ts.timestamp())


class BarStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def insert_bars(
        self,
        ticker: str,
        bars: pd.DataFrame,
        source: str,
        timeframe: str = "5min",
    ) -> int:
        """Idempotently insert bars.

        `bars` must be indexed by tz-aware event timestamp (bar start) and
        contain BAR_COLUMNS plus an `available_ts` column of tz-aware
        timestamps. Re-inserting identical rows is a no-op; a revised bar
        must carry a later available_ts and lands as a new revision row.
        """
        missing = [c for c in BAR_COLUMNS + ["available_ts"] if c not in bars.columns]
        if missing:
            raise ValueError(f"bars missing columns: {missing}")

        now = int(dt.datetime.now(dt.timezone.utc).timestamp())
        rows = [
            (
                ticker,
                timeframe,
                _to_epoch(event_ts),
                _to_epoch(row["available_ts"]),
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
                None if pd.isna(row["trade_count"]) else int(row["trade_count"]),
                None if pd.isna(row["vwap"]) else float(row["vwap"]),
                now,
                source,
            )
            for event_ts, row in bars.iterrows()
        ]
        with self._conn:
            cur = self._conn.executemany(
                "INSERT OR IGNORE INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                rows,
            )
        return cur.rowcount

    def get_bars_asof(
        self,
        ticker: str,
        as_of: dt.datetime | pd.Timestamp,
        start: dt.datetime | pd.Timestamp | None = None,
        end: dt.datetime | pd.Timestamp | None = None,
        timeframe: str = "5min",
    ) -> pd.DataFrame:
        """Bars visible at `as_of`: latest revision per event_ts with
        available_ts <= as_of. Returned index is the tz-aware (UTC) event
        timestamp; callers convert to ET at the feature layer.
        """
        params: list = [ticker, timeframe, _to_epoch(as_of)]
        clauses = ""
        if start is not None:
            clauses += " AND event_ts >= ?"
            params.append(_to_epoch(start))
        if end is not None:
            clauses += " AND event_ts <= ?"
            params.append(_to_epoch(end))

        query = f"""
            SELECT event_ts, {", ".join(BAR_COLUMNS)}, available_ts, source
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY event_ts ORDER BY available_ts DESC
                ) AS rn
                FROM bars
                WHERE ticker = ? AND timeframe = ? AND available_ts <= ?{clauses}
            )
            WHERE rn = 1
            ORDER BY event_ts
        """
        df = pd.read_sql_query(query, self._conn, params=params)
        for col in ("event_ts", "available_ts"):
            df[col] = pd.to_datetime(df[col], unit="s", utc=True)
        return df.set_index("event_ts")

    def tickers(self, timeframe: str = "5min") -> list[str]:
        cur = self._conn.execute(
            "SELECT DISTINCT ticker FROM bars WHERE timeframe = ? ORDER BY ticker",
            (timeframe,),
        )
        return [r[0] for r in cur.fetchall()]

    def coverage(self, ticker: str, timeframe: str = "5min") -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Earliest and latest stored event time for a ticker, or None."""
        cur = self._conn.execute(
            "SELECT MIN(event_ts), MAX(event_ts) FROM bars WHERE ticker = ? AND timeframe = ?",
            (ticker, timeframe),
        )
        lo, hi = cur.fetchone()
        if lo is None:
            return None
        return (
            pd.to_datetime(lo, unit="s", utc=True),
            pd.to_datetime(hi, unit="s", utc=True),
        )
