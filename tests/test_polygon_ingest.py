import datetime as dt

import pandas as pd

from stockpredictor.data.availability import BAR_INGEST_LATENCY
from stockpredictor.ingest.polygon import PolygonClient


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload
        self.status_code = 200

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        pass


class FakeSession:
    """Serves two pages to exercise next_url pagination."""

    def __init__(self):
        base = int(pd.Timestamp("2024-03-18 13:30", tz="UTC").timestamp() * 1000)
        five_min_ms = 5 * 60 * 1000
        self.pages = {
            None: {
                "results": [
                    {"t": base, "o": 10.0, "h": 10.5, "l": 9.9, "c": 10.2, "v": 1000, "n": 42, "vw": 10.1},
                    {"t": base + five_min_ms, "o": 10.2, "h": 10.6, "l": 10.0, "c": 10.4, "v": 900, "n": 37, "vw": 10.3},
                ],
                "next_url": "https://api.polygon.io/next-page",
            },
            "https://api.polygon.io/next-page": {
                "results": [
                    {"t": base + 2 * five_min_ms, "o": 10.4, "h": 10.7, "l": 10.3, "c": 10.6, "v": 800, "n": 31, "vw": 10.5},
                ],
            },
        }
        self.calls: list[str] = []

    def get(self, url: str, params=None, timeout=None) -> FakeResponse:
        self.calls.append(url)
        key = url if url in self.pages else None
        return FakeResponse(self.pages[key])


def test_fetch_parses_and_paginates():
    session = FakeSession()
    client = PolygonClient("key", seconds_per_call=0, session=session)
    bars = client.fetch_5min_bars("TEST", dt.date(2024, 3, 18), dt.date(2024, 3, 18))

    assert len(bars) == 3  # both pages combined
    assert len(session.calls) == 2
    assert list(bars.columns) == ["open", "high", "low", "close", "volume", "trade_count", "vwap", "available_ts"]

    first = bars.index[0]
    assert first == pd.Timestamp("2024-03-18 13:30", tz="UTC")
    assert bars.loc[first, "close"] == 10.2
    assert bars.loc[first, "trade_count"] == 42
    # Availability follows the shared convention: bar end + ingest latency.
    assert bars.loc[first, "available_ts"] == first + pd.Timedelta("5min") + BAR_INGEST_LATENCY


def test_empty_result_returns_empty_frame():
    session = FakeSession()
    session.pages = {None: {"results": []}}
    client = PolygonClient("key", seconds_per_call=0, session=session)
    bars = client.fetch_5min_bars("TEST", dt.date(2024, 3, 18), dt.date(2024, 3, 18))
    assert bars.empty
