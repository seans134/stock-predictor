import datetime as dt

import pandas as pd
import pytest

from stockpredictor.data.asof import AsOfData
from stockpredictor.data.availability import BAR_INGEST_LATENCY
from stockpredictor.data.store import BarStore
from stockpredictor.data.synthetic import synthetic_bars
from stockpredictor.sessions import SessionCalendar

TICKER = "TEST"


@pytest.fixture(scope="module")
def cal() -> SessionCalendar:
    return SessionCalendar(start="2024-01-01", end="2024-12-31")


@pytest.fixture(scope="module")
def bars(cal) -> pd.DataFrame:
    return synthetic_bars(cal, dt.date(2024, 3, 18), dt.date(2024, 3, 22), seed=42)


@pytest.fixture
def store(tmp_path) -> BarStore:
    s = BarStore(tmp_path / "bars.sqlite")
    yield s
    s.close()


def test_synthetic_bars_follow_session_grid(cal, bars):
    days = {ts.date() for ts in bars.index}
    assert days == set(cal.sessions_between(dt.date(2024, 3, 18), dt.date(2024, 3, 22)))
    first = bars.index[0]
    assert (first.hour, first.minute) == (9, 30)
    # Last bar of each day starts at 15:55 (close 16:00 minus one 5-min bar)
    last = max(ts for ts in bars.index if ts.date() == dt.date(2024, 3, 18))
    assert (last.hour, last.minute) == (15, 55)


def test_insert_is_idempotent(store, bars):
    assert store.insert_bars(TICKER, bars, source="test") == len(bars)
    assert store.insert_bars(TICKER, bars, source="test") == 0


def test_naive_timestamp_rejected(store, bars):
    naive = bars.copy()
    naive.index = naive.index.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        store.insert_bars(TICKER, naive, source="test")


def test_bar_not_visible_until_available(store, bars):
    store.insert_bars(TICKER, bars, source="test")
    data = AsOfData(store)
    event_ts = bars.index[10]
    bar_end = event_ts + pd.Timedelta("5min")

    # At bar end the bar exists in the world but has not arrived yet.
    visible = data.get_bars(TICKER, as_of=bar_end)
    assert event_ts not in visible.index

    # Once the ingest latency has elapsed it becomes visible.
    visible = data.get_bars(TICKER, as_of=bar_end + BAR_INGEST_LATENCY)
    assert event_ts in visible.index


def test_lookback_window(store, bars):
    store.insert_bars(TICKER, bars, source="test")
    data = AsOfData(store)
    as_of = bars.index[50] + pd.Timedelta("10min")
    window = data.get_bars(TICKER, as_of=as_of, lookback=pd.Timedelta("30min"))
    assert not window.empty
    assert window.index.min() >= as_of - pd.Timedelta("30min")
    assert (window["available_ts"] <= as_of).all()


def test_revision_visibility(store, bars):
    """A revised bar shows its original value before the revision arrived
    and the corrected value afterwards — history is never rewritten."""
    store.insert_bars(TICKER, bars, source="test")
    event_ts = bars.index[5]
    original_close = bars.loc[event_ts, "close"]

    revision = bars.loc[[event_ts]].copy()
    revision["close"] = original_close + 1.0
    revision["available_ts"] = revision["available_ts"] + pd.Timedelta(hours=2)
    store.insert_bars(TICKER, revision, source="test")

    data = AsOfData(store)
    before = data.get_bars(TICKER, as_of=event_ts + pd.Timedelta(minutes=30))
    after = data.get_bars(TICKER, as_of=event_ts + pd.Timedelta(hours=3))
    assert before.loc[event_ts, "close"] == pytest.approx(original_close)
    assert after.loc[event_ts, "close"] == pytest.approx(original_close + 1.0)


def test_lookahead_appending_future_does_not_change_past(store, bars):
    """The plan's look-ahead test: features computed as of T must be
    identical before and after future rows are appended."""
    midpoint = len(bars) // 2
    first_half, second_half = bars.iloc[:midpoint], bars.iloc[midpoint:]

    store.insert_bars(TICKER, first_half, source="test")
    data = AsOfData(store)
    as_of = first_half.index[-1] + pd.Timedelta("10min")
    snapshot_before = data.get_bars(TICKER, as_of=as_of)

    store.insert_bars(TICKER, second_half, source="test")
    snapshot_after = data.get_bars(TICKER, as_of=as_of)

    pd.testing.assert_frame_equal(snapshot_before, snapshot_after)
    assert not snapshot_before.empty
