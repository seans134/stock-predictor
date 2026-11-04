"""Read-only, outcome-blind session coverage audit for a registered universe.

Run: python -m stockpredictor.research.coverage REGISTRATION --output DIRECTORY
No prices or returns are read. Missing intervals are not presumed halts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sqlite3

import pandas as pd

from stockpredictor.research.baselines import INSTRUMENT_TYPES
from stockpredictor.research.governance import digest, fingerprint, require_development, write_new
from stockpredictor.sessions import default_calendar


def audit(db, tickers, start, end, manifest, calendar):
    require_development(manifest, str(start), str(end))
    if not tickers or len(set(tickers)) != len(tickers):
        raise ValueError("Expected a nonempty unique universe")
    if any(t not in INSTRUMENT_TYPES for t in tickers):
        raise ValueError("Unsupported instrument")
    days = calendar.sessions_between(start, end)
    if not days:
        raise ValueError("No sessions in interval")
    grids = {day: [int(t.timestamp()) for t in pd.date_range(
        calendar.session_open(day), calendar.session_close(day), freq="5min", inclusive="left")]
        for day in days}
    lo, hi = min(grids[days[0]]), max(grids[days[-1]]) + 300
    ledger, sessions = [], []
    with sqlite3.connect(Path(db).resolve().as_uri()+"?mode=ro", uri=True) as conn:
        conn.execute("BEGIN")  # one consistent read snapshot
        for ticker in tickers:
            rows = conn.execute("""SELECT event_ts, MIN(available_ts),
                MAX(CASE WHEN available_ts BETWEEN event_ts+300 AND event_ts+305 THEN 1 ELSE 0 END)
                FROM bars WHERE ticker=? AND timeframe='5min' AND event_ts>=? AND event_ts<?
                GROUP BY event_ts""", (ticker, lo, hi)).fetchall()
            present = {ts: (first, timely) for ts, first, timely in rows}
            for day, expected in grids.items():
                missing, unavailable = 0, 0
                for ts in expected:
                    record = present.get(ts)
                    if record and record[1]:
                        continue
                    reason = "absent" if record is None else "no_timely_version"
                    missing += record is None
                    unavailable += record is not None
                    ledger.append(dict(ticker=ticker, date=str(day), event_ts=ts,
                        bar_start_et=str(pd.Timestamp(ts, unit="s", tz="UTC").tz_convert("America/New_York")),
                        reason=reason, first_available_ts=record[0] if record else None))
                sessions.append(dict(ticker=ticker, date=str(day), expected=len(expected),
                    absent=missing, no_timely_version=unavailable,
                    timely=len(expected)-missing-unavailable,
                    complete=not (missing or unavailable)))
    return pd.DataFrame(sessions), pd.DataFrame(ledger, columns=[
        "ticker", "date", "event_ts", "bar_start_et", "reason", "first_available_ts"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    record = json.loads(args.registration.read_text())
    if fingerprint({k: v for k, v in record.items() if k != "registration_sha256"}) != record["registration_sha256"]:
        raise ValueError("Registration modified")
    spec = record["spec"]
    manifest = json.loads((args.registration.parent.parent / "split_manifest.json").read_text())
    if fingerprint(manifest) != record["manifest_sha256"]:
        raise ValueError("Manifest changed")
    start = max(dt.date.fromisoformat(manifest["development_start"]),
                dt.date.fromisoformat(spec["start"])-dt.timedelta(days=45))
    end = dt.date.fromisoformat(spec["end"])
    db = args.root / "data/bars.sqlite"
    before = digest(db)
    if before != spec["data_sha256"]:
        raise ValueError("Registered database changed; audit requires original input")
    session_rows, ledger = audit(db, spec["tickers"], start, end, manifest, default_calendar())
    if digest(db) != before:
        raise ValueError("Database changed during audit")
    args.output.mkdir(parents=True, exist_ok=False)
    session_rows.to_csv(args.output / "sessions.csv", index=False)
    ledger.to_csv(args.output / "missing_bars.csv", index=False)
    totals = session_rows.groupby("ticker")[["expected", "timely", "absent", "no_timely_version", "complete"]].sum()
    totals.to_csv(args.output / "by_ticker.csv")
    summary = dict(registration_id=spec["id"], registration_sha256=record["registration_sha256"],
        database_sha256=before, audit_source_sha256=digest(Path(__file__)),
        created_at=dt.datetime.now(dt.timezone.utc).isoformat(), start=str(start), end=str(end),
        evaluation_start=spec["start"], tickers=len(spec["tickers"]),
        ticker_sessions=len(session_rows), incomplete_sessions=int((~session_rows.complete).sum()),
        absent_bars=int(session_rows.absent.sum()), no_timely_version=int(session_rows.no_timely_version.sum()),
        interpretation="Coverage only, including warmup. No returns inspected. Missing bars require source investigation; no exclusion or imputation authorized by this report.")
    write_new(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
