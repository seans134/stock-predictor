"""Bitemporal news storage.

Same discipline as the bar store: every article carries the publisher's
claimed publish time (event time) and a conservative availability time,
and "as of T" queries are a filter over availability — never a guess.

Two tables:
- articles:        one row per article (publisher, title, description,
                   keywords) keyed by the vendor's stable article id.
- article_tickers: one row per (article, tagged ticker) with the vendor's
                   entity-level sentiment when present. Sentiment is
                   nullable — absent sentiment is stored as NULL, never
                   coerced to neutral, per the missingness rule.

Timestamps are integer epoch seconds UTC; the API accepts only tz-aware
datetimes so naive-timestamp bugs fail loudly.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

import pandas as pd

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    article_id   TEXT    PRIMARY KEY,
    published_ts INTEGER NOT NULL,
    available_ts INTEGER NOT NULL,
    publisher    TEXT,
    title        TEXT,
    description  TEXT,
    keywords     TEXT,
    source       TEXT    NOT NULL,
    ingested_ts  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_articles_available ON articles (available_ts);

CREATE TABLE IF NOT EXISTS article_tickers (
    article_id          TEXT NOT NULL,
    ticker              TEXT NOT NULL,
    sentiment           TEXT,
    sentiment_reasoning TEXT,
    PRIMARY KEY (article_id, ticker)
);
CREATE INDEX IF NOT EXISTS idx_article_tickers ON article_tickers (ticker, article_id);
"""


def _to_epoch(ts: dt.datetime | pd.Timestamp) -> int:
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        raise ValueError(f"timestamp {ts!r} must be timezone-aware")
    return int(ts.timestamp())


class NewsStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._path)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def insert_articles(self, articles: list[dict], source: str) -> int:
        """Idempotently insert parsed articles.

        Each dict needs: article_id, published (tz-aware), available
        (tz-aware), publisher, title, description, keywords (list),
        tickers (list of {ticker, sentiment, sentiment_reasoning}).
        Returns the number of NEW article rows.
        """
        now = int(dt.datetime.now(dt.timezone.utc).timestamp())
        inserted = 0
        with self._conn:
            for article in articles:
                cur = self._conn.execute(
                    "INSERT OR IGNORE INTO articles VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        article["article_id"],
                        _to_epoch(article["published"]),
                        _to_epoch(article["available"]),
                        article.get("publisher"),
                        article.get("title"),
                        article.get("description"),
                        json.dumps(article.get("keywords") or []),
                        source,
                        now,
                    ),
                )
                if cur.rowcount:
                    inserted += 1
                self._conn.executemany(
                    "INSERT OR IGNORE INTO article_tickers VALUES (?,?,?,?)",
                    [
                        (
                            article["article_id"],
                            t["ticker"],
                            t.get("sentiment"),
                            t.get("sentiment_reasoning"),
                        )
                        for t in article.get("tickers", [])
                    ],
                )
        return inserted

    def get_news_asof(
        self,
        ticker: str,
        as_of: dt.datetime | pd.Timestamp,
        start: dt.datetime | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Articles tagged with `ticker` visible at `as_of`, oldest first.

        `start` bounds published time from below. Returns published /
        available as tz-aware UTC with publisher, title, sentiment.
        """
        params: list = [ticker, _to_epoch(as_of)]
        clause = ""
        if start is not None:
            clause = " AND a.published_ts >= ?"
            params.append(_to_epoch(start))
        query = f"""
            SELECT a.article_id, a.published_ts, a.available_ts, a.publisher,
                   a.title, a.description, a.keywords,
                   t.sentiment, t.sentiment_reasoning
            FROM articles a
            JOIN article_tickers t ON t.article_id = a.article_id
            WHERE t.ticker = ? AND a.available_ts <= ?{clause}
            ORDER BY a.published_ts
        """
        df = pd.read_sql_query(query, self._conn, params=params)
        for col in ("published_ts", "available_ts"):
            df[col] = pd.to_datetime(df[col], unit="s", utc=True)
        return df

    def all_articles(self) -> pd.DataFrame:
        """Every stored article (id, published/available, publisher, title,
        description), for global clustering and event classification.
        Sentiment stays per-ticker; join separately."""
        df = pd.read_sql_query(
            "SELECT article_id, published_ts, available_ts, publisher, title, description"
            " FROM articles",
            self._conn,
        )
        for col in ("published_ts", "available_ts"):
            df[col] = pd.to_datetime(df[col], unit="s", utc=True)
        return df

    def ticker_sentiments(self, ticker: str) -> pd.DataFrame:
        """(article_id, sentiment) rows for one ticker's tagged articles."""
        return pd.read_sql_query(
            "SELECT article_id, sentiment FROM article_tickers WHERE ticker = ?",
            self._conn,
            params=[ticker],
        )

    def coverage(self) -> tuple[int, pd.Timestamp | None, pd.Timestamp | None]:
        """(article count, earliest published, latest published)."""
        n, lo, hi = self._conn.execute(
            "SELECT COUNT(*), MIN(published_ts), MAX(published_ts) FROM articles"
        ).fetchone()
        to_ts = lambda v: pd.to_datetime(v, unit="s", utc=True) if v is not None else None
        return n, to_ts(lo), to_ts(hi)
