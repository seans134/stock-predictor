import datetime as dt

import pandas as pd
import pytest

from stockpredictor.research.earnings_dates import (first_full_session, item_202_rows, summarize,
                                                    timing_bucket)
from stockpredictor.sessions import SessionCalendar


@pytest.fixture(scope="module")
def calendar():
    return SessionCalendar(start="2024-01-01", end="2024-12-31")


def test_item_filter_keeps_only_item_202_8ks_in_the_window():
    columns = dict(
        form=["8-K", "8-K", "10-Q", "8-K/A", "8-K"],
        items=["2.02,9.01", "5.02", "2.02", "2.02", "12.02"],
        acceptanceDateTime=["2024-02-01T16:05:00.000Z"] * 5,
        filingDate=["2024-02-01", "2024-02-01", "2024-02-01", "2024-02-02", "2024-02-01"],
        reportDate=["2024-02-01"] * 5,
        accessionNumber=["a", "b", "c", "d", "e"])
    assert [r["accession"] for r in item_202_rows(columns, "2024-01-01", "2024-12-31")] == ["a", "d"]
    assert item_202_rows(columns, "2024-03-01", "2024-12-31") == []


@pytest.mark.parametrize("accepted,session,bucket", [
    ("2024-03-14T12:00:00.000Z", dt.date(2024, 3, 14), "before_open"),      # 08:00 ET
    ("2024-03-14T15:00:00.000Z", dt.date(2024, 3, 15), "during_session"),   # 11:00 ET
    ("2024-03-14T20:05:00.000Z", dt.date(2024, 3, 15), "after_close"),      # 16:05 ET
    ("2024-03-15T21:00:00.000Z", dt.date(2024, 3, 18), "after_close"),      # Friday evening
    ("2024-03-16T14:00:00.000Z", dt.date(2024, 3, 18), "non_session_day"),  # Saturday
])
def test_first_full_session_never_precedes_the_filing(calendar, accepted, session, bucket):
    assert first_full_session(accepted, calendar) == session
    assert timing_bucket(accepted, calendar) == bucket


def test_summary_flags_gaps_and_close_pairs_and_counts_zero_tickers():
    events = pd.DataFrame([
        dict(ticker="AAA", form="8-K", filed="2024-01-30"),
        dict(ticker="AAA", form="8-K", filed="2024-02-20"),
        dict(ticker="AAA", form="8-K", filed="2024-08-01"),
        dict(ticker="AAA", form="8-K/A", filed="2024-08-02"),
    ])
    summary = summarize(events, ["AAA", "BBB"])
    assert summary["per_ticker"] == {"AAA": 3, "BBB": 0}
    assert summary["amendments"] == 1
    assert [g["days"] for g in summary["long_gaps"]] == [163]
    assert [g["days"] for g in summary["close_pairs"]] == [21]
