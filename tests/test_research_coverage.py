import datetime as dt
import sqlite3

import pytest

from stockpredictor.research.coverage import audit
from stockpredictor.sessions import SessionCalendar
from stockpredictor.data.synthetic import synthetic_bars
from stockpredictor.research.baselines import require_complete_evaluation


def test_audit_distinguishes_missing_late_and_revised_on_early_close(tmp_path):
    db = tmp_path / "bars.db"
    cal = SessionCalendar("2024-11-29", "2024-11-29")
    day = dt.date(2024, 11, 29)
    start = int(cal.session_open(day).timestamp())
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE bars(ticker, timeframe, event_ts, available_ts)")
        # 42 bars on this early close. One absent, one only late; a late revision
        # must not hide the usable original version of another bar.
        rows = [("SPY", "5min", start+i*300, start+i*300+305) for i in range(42) if i not in (1, 2)]
        rows += [("SPY", "5min", start+600, start+1000), ("SPY", "5min", start, start+999)]
        conn.executemany("INSERT INTO bars VALUES(?,?,?,?)", rows)
    manifest = dict(development_start=str(day), development_end=str(day))
    sessions, ledger = audit(db, ["SPY"], day, day, manifest, cal)
    assert sessions.iloc[0][["expected", "timely", "absent", "no_timely_version"]].tolist() == [42, 40, 1, 1]
    assert ledger.reason.tolist() == ["absent", "no_timely_version"]
    assert not sessions.iloc[0].complete
    with sqlite3.connect(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM bars").fetchone()[0] == 42


def test_audit_protected_dates_before_database_access(tmp_path):
    manifest = dict(development_start="2024-01-02", development_end="2024-01-03")
    with pytest.raises(ValueError, match="protected"):
        audit(tmp_path / "absent.db", ["SPY"], dt.date(2024, 1, 4), dt.date(2024, 1, 5), manifest, None)
    assert not (tmp_path / "absent.db").exists()


@pytest.mark.parametrize("missing_day", [True, False])
def test_performance_gate_rejects_whole_day_and_single_gap(missing_day):
    cal = SessionCalendar("2024-11-27", "2024-11-29")
    days = cal.sessions_between(dt.date(2024, 11, 27), dt.date(2024, 11, 29))
    frame = synthetic_bars(cal, days[0], days[-1])
    require_complete_evaluation({"SPY": frame}, cal, days)
    if missing_day:
        frame = frame[frame.index.tz_convert("America/New_York").date != days[-1]]
    else:
        frame = frame.drop(frame.index[-1])
    with pytest.raises(ValueError, match="Incomplete evaluation coverage"):
        require_complete_evaluation({"SPY": frame}, cal, days)
