import datetime as dt

import pytest

from stockpredictor.sessions import ET, SessionCalendar


@pytest.fixture(scope="module")
def cal() -> SessionCalendar:
    return SessionCalendar(start="2024-01-01", end="2024-12-31")


def test_holiday_is_not_session(cal):
    assert not cal.is_session(dt.date(2024, 7, 4))  # Independence Day
    assert not cal.is_session(dt.date(2024, 7, 6))  # Saturday
    assert cal.is_session(dt.date(2024, 7, 5))


def test_regular_session_times(cal):
    day = dt.date(2024, 3, 20)
    assert cal.session_open(day) == dt.datetime(2024, 3, 20, 9, 30, tzinfo=ET)
    assert cal.session_close(day) == dt.datetime(2024, 3, 20, 16, 0, tzinfo=ET)


def test_half_day_early_close(cal):
    # July 3, 2024 closed at 13:00 ET
    day = dt.date(2024, 7, 3)
    close = cal.session_close(day)
    assert (close.hour, close.minute) == (13, 0)


def test_dst_transition_keeps_et_open_constant(cal):
    # 09:30 ET is 14:30 UTC in winter, 13:30 UTC in summer.
    winter = cal.session_open(dt.date(2024, 1, 16)).astimezone(dt.timezone.utc)
    summer = cal.session_open(dt.date(2024, 6, 17)).astimezone(dt.timezone.utc)
    assert winter.hour == 14
    assert summer.hour == 13


def test_premarket_cutoff(cal):
    cutoff = cal.premarket_cutoff(dt.date(2024, 3, 20))
    assert cutoff == dt.datetime(2024, 3, 20, 9, 0, tzinfo=ET)
    with pytest.raises(ValueError):
        cal.premarket_cutoff(dt.date(2024, 7, 4))


def test_next_session_skips_weekend_and_holiday(cal):
    assert cal.next_session(dt.date(2024, 7, 3)) == dt.date(2024, 7, 5)
    assert cal.next_session(dt.date(2024, 3, 22)) == dt.date(2024, 3, 25)  # Fri -> Mon
