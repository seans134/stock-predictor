# tests/test_pairs_data.py
import datetime as dt

import pandas as pd

from stockpredictor.data.asof import AsOfData
from stockpredictor.data.store import BarStore
from stockpredictor.pairs.data import load_daily_closes
from stockpredictor.sessions import ET, default_calendar


def _insert_session(store, ticker, day, closes_by_hhmm):
    """Insert 5-min RTH bars for one session; available immediately."""
    rows = []
    for hhmm, close in closes_by_hhmm.items():
        h, m = map(int, hhmm.split(":"))
        ts = pd.Timestamp(day.year, day.month, day.day, h, m, tz=ET)
        rows.append((ts, close))
    idx = pd.DatetimeIndex([r[0] for r in rows]).tz_convert("UTC")
    df = pd.DataFrame(
        {
            "open": [r[1] for r in rows], "high": [r[1] for r in rows],
            "low": [r[1] for r in rows], "close": [r[1] for r in rows],
            "volume": [1000.0] * len(rows), "trade_count": [10] * len(rows),
            "vwap": [r[1] for r in rows],
            "available_ts": [ts + pd.Timedelta(minutes=6) for ts in idx],
        },
        index=idx,
    )
    store.insert_bars(ticker, df, source="test", timeframe="5min")


def test_load_daily_closes_last_rth_close_per_session(tmp_path):
    db = tmp_path / "bars.sqlite"
    store = BarStore(db)
    cal = default_calendar()
    # Two consecutive sessions for two tickers.
    d1, d2 = dt.date(2024, 3, 18), dt.date(2024, 3, 19)
    for day, last in [(d1, 101.0), (d2, 103.0)]:
        _insert_session(store, "AAA", day, {"09:35": 100.0, "15:55": last})
        _insert_session(store, "BBB", day, {"09:35": 50.0, "15:55": last / 2})
    store.close()

    store = BarStore(db)
    try:
        panel = load_daily_closes(AsOfData(store), cal, ["AAA", "BBB"], d1, d2)
    finally:
        store.close()
    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel.loc[d1, "AAA"] == 101.0   # last RTH close of the session
    assert panel.loc[d2, "AAA"] == 103.0
    assert panel.loc[d2, "BBB"] == 103.0 / 2
