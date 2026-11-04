"""Outcome-blind split evidence for a registered universe; reads no prices.

Run: python -m stockpredictor.research.corporate_actions REGISTRATION --output FILE
Rate limited to the free tier. Any failed or truncated response makes the
evidence incomplete, and an incomplete file cannot back a registration.
Splits only: spin-offs and other corporate actions are not covered.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time

from dotenv import load_dotenv
import requests

from stockpredictor.research.governance import write_new

ROOT = Path(__file__).resolve().parents[3]
REQUEST_SPACING_SECONDS = 12.5


def collect(tickers, start, end, key, get=requests.get, sleep=time.sleep):
    results, failures = {}, {}
    for number, ticker in enumerate(tickers):
        if number:
            sleep(REQUEST_SPACING_SECONDS)
        try:
            response = get("https://api.polygon.io/v3/reference/splits",
                           params={"ticker": ticker, "execution_date.gte": start,
                                   "execution_date.lte": end, "limit": 1000},
                           headers={"Authorization": "Bearer " + key}, timeout=30)
            if response.status_code != 200:
                failures[ticker] = f"http_{response.status_code}"
                continue
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            # Exception text can carry request details; retain the type only.
            failures[ticker] = type(exc).__name__
            continue
        if payload.get("next_url"):
            failures[ticker] = "truncated_response"
            continue
        results[ticker] = [{"execution_date": row["execution_date"],
                            "split_from": row["split_from"], "split_to": row["split_to"]}
                           for row in payload.get("results", [])]
    return results, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("registration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", required=True, help="Earliest execution date, including warmup")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Evidence file already exists")
    spec = json.loads(args.registration.read_text())["spec"]
    load_dotenv(ROOT / ".env")
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        raise ValueError("Provider credential not configured")
    observed = dt.datetime.now(dt.timezone.utc).isoformat()
    splits, failures = collect(spec["tickers"], args.start, spec["end"], key)
    write_new(args.output, dict(observed_at=observed, start=args.start, end=spec["end"],
                                tickers=spec["tickers"], splits=splits, failures=failures,
                                complete=not failures, scope="splits only"))
    print(json.dumps({"complete": not failures, "failures": failures,
                      "splits": {t: s for t, s in splits.items() if s}}, indent=1))


if __name__ == "__main__":
    main()
