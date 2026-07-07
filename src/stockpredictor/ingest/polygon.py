"""Historical 5-minute bar ingestion from Polygon.io (rebranded Massive).

Free-tier notes, per the project plan's data-quality rules:
- Aggregates are consolidated across all US exchanges (SIP-based), so
  volume is representative — unlike IEX-only free feeds.
- Free tier: 5 API calls/minute, ~2 years of history. Backfills are
  rate-limited and just take a while; run them unattended.
- Bars are requested unadjusted (adjusted=false). Split/dividend adjustment
  rewrites history with future knowledge; corporate actions get their own
  point-in-time handling later.
- Aggregates include pre-market and after-hours bars. Everything is stored;
  session filtering belongs to the feature layer, and the plan needs
  pre-market bars for Stage 1.

Requires POLYGON_API_KEY in the environment or a .env file.
"""

from __future__ import annotations

import datetime as dt
import logging
import time

import pandas as pd
import requests

from stockpredictor.data.availability import bar_available_ts
from stockpredictor.data.store import BarStore

log = logging.getLogger(__name__)

SOURCE = "polygon"
BASE_URL = "https://api.polygon.io"
# Free tier allows 5 requests/minute; stay just under it.
FREE_TIER_SECONDS_PER_CALL = 12.5
MAX_BARS_PER_PAGE = 50_000


class PolygonClient:
    def __init__(
        self,
        api_key: str,
        seconds_per_call: float = FREE_TIER_SECONDS_PER_CALL,
        session: requests.Session | None = None,
    ):
        self._api_key = api_key
        self._seconds_per_call = seconds_per_call
        self._session = session or requests.Session()
        self._last_call = 0.0

    def _get(self, url: str, params: dict | None = None) -> dict:
        wait = self._last_call + self._seconds_per_call - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        params = dict(params or {})
        params["apiKey"] = self._api_key
        response = self._session.get(url, params=params, timeout=30)
        self._last_call = time.monotonic()
        if response.status_code == 429:
            # Rate-limited despite pacing: back off a full window and retry once.
            log.warning("rate limited; backing off 60s")
            time.sleep(60)
            response = self._session.get(url, params=params, timeout=30)
            self._last_call = time.monotonic()
        response.raise_for_status()
        return response.json()

    def fetch_5min_bars(
        self,
        ticker: str,
        start: dt.date,
        end: dt.date,
    ) -> pd.DataFrame:
        """Fetch raw 5-min aggregates, following pagination.

        Returns a DataFrame in BarStore.insert_bars shape: indexed by
        tz-aware UTC bar-start time with OHLCV, trade_count, vwap, and
        available_ts columns.
        """
        url = f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/5/minute/{start}/{end}"
        params = {"adjusted": "false", "sort": "asc", "limit": MAX_BARS_PER_PAGE}
        results: list[dict] = []
        page = 1
        payload = self._get(url, params)
        while True:
            results.extend(payload.get("results") or [])
            log.info("%s: page %d fetched, %d bars so far", ticker, page, len(results))
            next_url = payload.get("next_url")
            if not next_url:
                break
            log.info("%s: more pages remain; waiting out the rate limit...", ticker)
            page += 1
            payload = self._get(next_url)

        if not results:
            return pd.DataFrame()

        frame = pd.DataFrame(results)
        index = pd.to_datetime(frame["t"], unit="ms", utc=True)
        bars = pd.DataFrame(
            {
                "open": frame["o"].astype(float),
                "high": frame["h"].astype(float),
                "low": frame["l"].astype(float),
                "close": frame["c"].astype(float),
                "volume": frame["v"].astype(float),
                "trade_count": frame.get("n"),
                "vwap": frame.get("vw"),
            }
        ).set_index(pd.DatetimeIndex(index))
        bars["available_ts"] = bar_available_ts(bars.index, "5min")
        return bars


def ingest_history(
    store: BarStore,
    client: PolygonClient,
    tickers: list[str],
    start: dt.date,
    end: dt.date,
) -> dict[str, int]:
    """Backfill 5-min bars for each ticker; returns rows inserted per ticker."""
    inserted: dict[str, int] = {}
    for ticker in tickers:
        bars = client.fetch_5min_bars(ticker, start, end)
        if bars.empty:
            log.warning("%s: no bars returned for %s..%s", ticker, start, end)
            inserted[ticker] = 0
            continue
        inserted[ticker] = store.insert_bars(ticker, bars, source=SOURCE)
        log.info("%s: inserted %d new bars", ticker, inserted[ticker])
    return inserted
