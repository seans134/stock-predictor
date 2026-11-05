"""Outcome-blind SEC 8-K Item 2.02 dates for a registered universe; reads no prices.

Run: python -m stockpredictor.research.earnings_dates REGISTRATION --start DATE --end DATE
     --output FILE
Requires SEC_USER_AGENT (a name and contact, as the SEC's access rules ask) in the
environment or .env, so that the contact never enters the repository.
Item 2.02 means "Results of Operations and Financial Condition". It is a proxy for
earnings reports: a filing can also be a preannouncement, and it can follow the
company's press release. Acceptance time is when the SEC accepted the filing, not
when the market first knew.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import pandas as pd
import requests

from stockpredictor.research.governance import write_new
from stockpredictor.sessions import ET, default_calendar

ITEM = "2.02"
# The SEC limits clients to 10 requests per second; stay well below that.
REQUEST_SPACING_SECONDS = 0.25
ROOT = Path(__file__).resolve().parents[3]
MARKET_OPEN = dt.time(9, 30)


def item_202_rows(columns: dict, start: str, end: str) -> list[dict]:
    """Filter one columnar EDGAR filings block to original and amended Item 2.02 8-Ks."""
    rows = []
    for form, items, accepted, filed, report, accession in zip(
            columns["form"], columns["items"], columns["acceptanceDateTime"],
            columns["filingDate"], columns["reportDate"], columns["accessionNumber"]):
        if form not in ("8-K", "8-K/A") or ITEM not in str(items).split(","):
            continue
        if start <= filed <= end:
            rows.append(dict(form=form, accepted=accepted, filed=filed, report=report,
                             accession=accession, items=items))
    return rows


def spans_window(start_of_block: str, start: str) -> bool:
    """True if an older-filings page can still hold filings inside the window."""
    return start_of_block >= start


def first_full_session(accepted: str, calendar) -> dt.date:
    """First session that opens after acceptance; a pre-open filing uses that same day."""
    stamp = pd.Timestamp(accepted).tz_convert(ET)
    day = stamp.date()
    if calendar.is_session(day) and stamp.time() < MARKET_OPEN:
        return day
    return calendar.sessions_between(day + dt.timedelta(days=1), day + dt.timedelta(days=10))[0]


def timing_bucket(accepted: str, calendar) -> str:
    stamp = pd.Timestamp(accepted).tz_convert(ET)
    day = stamp.date()
    if not calendar.is_session(day):
        return "non_session_day"
    if stamp.time() < MARKET_OPEN:
        return "before_open"
    if stamp < pd.Timestamp(calendar.session_close(day)):
        return "during_session"
    return "after_close"


def summarize(events: pd.DataFrame, tickers, long_gap_days: int = 110, short_gap_days: int = 60) -> dict:
    """Coverage by ticker. Gaps flag possible misses; close pairs flag non-earnings filings."""
    per_ticker, long_gaps, close_pairs = {}, [], []
    originals = events[events.form == "8-K"]
    for ticker in tickers:
        filed = sorted(originals[originals.ticker == ticker].filed)
        per_ticker[ticker] = len(filed)
        for earlier, later in zip(filed, filed[1:]):
            days = (dt.date.fromisoformat(later) - dt.date.fromisoformat(earlier)).days
            if days > long_gap_days:
                long_gaps.append(dict(ticker=ticker, previous=earlier, next=later, days=days))
            elif days < short_gap_days:
                close_pairs.append(dict(ticker=ticker, previous=earlier, next=later, days=days))
    counts = pd.Series(per_ticker)
    return dict(tickers=len(per_ticker), originals=int(len(originals)), amendments=int((events.form == "8-K/A").sum()),
                per_ticker=per_ticker, min_per_ticker=int(counts.min()), median_per_ticker=float(counts.median()),
                max_per_ticker=int(counts.max()), long_gaps=long_gaps, close_pairs=close_pairs)


def fetch_json(url: str, user_agent: str, get=requests.get, sleep=time.sleep):
    sleep(REQUEST_SPACING_SECONDS)
    response = get(url, headers={"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"}, timeout=30)
    if response.status_code != 200:
        raise ValueError(f"SEC request rejected: HTTP {response.status_code}")
    return response.json()


def collect(tickers, start, end, user_agent, get=requests.get, sleep=time.sleep):
    directory = fetch_json("https://www.sec.gov/files/company_tickers.json", user_agent, get, sleep)
    ciks = {row["ticker"]: int(row["cik_str"]) for row in directory.values()}
    rows, failures, identities = [], {}, {}
    for ticker in tickers:
        if ticker not in ciks:
            failures[ticker] = "ticker_not_in_sec_directory"
            continue
        try:
            data = fetch_json(f"https://data.sec.gov/submissions/CIK{ciks[ticker]:010d}.json", user_agent, get, sleep)
            identities[ticker] = dict(cik=ciks[ticker], name=data.get("name"))
            found = item_202_rows(data["filings"]["recent"], start, end)
            for page in data["filings"].get("files", []):
                if spans_window(page["filingTo"], start):
                    older = fetch_json(f"https://data.sec.gov/submissions/{page['name']}", user_agent, get, sleep)
                    found += item_202_rows(older, start, end)
        except (requests.RequestException, ValueError, KeyError) as exc:
            failures[ticker] = type(exc).__name__
            continue
        rows += [dict(ticker=ticker, **row) for row in found]
    return rows, failures, identities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Evidence file already exists")
    load_dotenv(ROOT / ".env")
    user_agent = os.getenv("SEC_USER_AGENT")
    if not user_agent:
        raise ValueError("Set SEC_USER_AGENT to a name and contact before calling the SEC")
    tickers = json.loads(args.registration.read_text())["spec"]["tickers"]
    observed = dt.datetime.now(dt.timezone.utc).isoformat()
    rows, failures, identities = collect(tickers, args.start, args.end, user_agent)
    calendar = default_calendar()
    for row in rows:
        row["first_full_session"] = str(first_full_session(row["accepted"], calendar))
        row["timing"] = timing_bucket(row["accepted"], calendar)
    write_new(args.output, dict(observed_at=observed, start=args.start, end=args.end, tickers=tickers,
                                failures=failures, complete=not failures, identities=identities,
                                events=sorted(rows, key=lambda r: (r["ticker"], r["accepted"]))))
    print(json.dumps({"events": len(rows), "failures": failures}, indent=1))


if __name__ == "__main__":
    main()
