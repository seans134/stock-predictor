"""Historical news ingestion from Polygon's /v2/reference/news endpoint.

Fetches ALL market news in a date range (no ticker filter — one pass gets
every article, and multi-ticker articles arrive once instead of once per
tagged ticker), following pagination under the shared free-tier rate
limiter. ~185 articles/day market-wide, 1000 per page.

Availability discipline: historical articles carry only published_utc, so
each is assigned available = published + NEWS_INGEST_LATENCY, the plan's
conservative-availability rule for backfills. Entity-level sentiment from
the vendor's `insights` is stored per (article, ticker); articles tagged
without sentiment store NULL, never a synthetic neutral.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd

from stockpredictor.data.availability import NEWS_INGEST_LATENCY
from stockpredictor.data.news_store import NewsStore
from stockpredictor.ingest.polygon import BASE_URL, PolygonClient

log = logging.getLogger(__name__)

SOURCE = "polygon-news"
PAGE_LIMIT = 1000


def parse_articles(results: list[dict]) -> list[dict]:
    """Vendor payload -> NewsStore.insert_articles shape."""
    parsed = []
    for item in results:
        published = pd.Timestamp(item["published_utc"])
        if published.tzinfo is None:
            published = published.tz_localize("UTC")
        insights = {
            i["ticker"]: i for i in (item.get("insights") or []) if i.get("ticker")
        }
        tickers = [
            {
                "ticker": ticker,
                "sentiment": insights.get(ticker, {}).get("sentiment"),
                "sentiment_reasoning": insights.get(ticker, {}).get("sentiment_reasoning"),
            }
            for ticker in (item.get("tickers") or [])
        ]
        parsed.append(
            {
                "article_id": item["id"],
                "published": published,
                "available": published + NEWS_INGEST_LATENCY,
                "publisher": (item.get("publisher") or {}).get("name"),
                "title": item.get("title"),
                "description": item.get("description"),
                "keywords": item.get("keywords") or [],
                "tickers": tickers,
            }
        )
    return parsed


def ingest_news(
    store: NewsStore,
    client: PolygonClient,
    start: dt.date,
    end: dt.date,
) -> int:
    """Backfill all market news published in [start, end). Idempotent.
    Returns the number of new articles stored."""
    url = f"{BASE_URL}/v2/reference/news"
    params = {
        "published_utc.gte": str(start),
        "published_utc.lt": str(end),
        "order": "asc",
        "sort": "published_utc",
        "limit": PAGE_LIMIT,
    }
    inserted = 0
    page = 1
    payload = client.get_json(url, params)
    while True:
        articles = parse_articles(payload.get("results") or [])
        inserted += store.insert_articles(articles, source=SOURCE)
        log.info("news page %d: %d articles (%d new so far)", page, len(articles), inserted)
        next_url = payload.get("next_url")
        if not next_url:
            break
        page += 1
        payload = client.get_json(next_url)
    return inserted
