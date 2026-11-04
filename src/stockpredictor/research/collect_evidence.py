"""Bounded issuer-source capture and draft event seeding; no unattended service."""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlparse

import requests

from stockpredictor.news.evidence import EvidenceStore, utc_now

MAX_BYTES = 4 * 1024 * 1024


def collect(store, sources, rounds=1, interval=15):
    if not 1 <= rounds <= 3 or not 15 <= interval <= 60 or not 1 <= len(sources) <= 20:
        raise ValueError("Collection must be bounded: 1-3 rounds, 1-20 sources, 15-60 seconds between rounds")
    for source in sources:
        url = urlparse(source["url"])
        if url.scheme != "https" or url.username or url.password or url.query or url.fragment:
            raise ValueError("Use credential-free canonical HTTPS issuer URLs")
    results = []
    for round_number in range(rounds):
        if round_number:
            time.sleep(interval)
        for source in sources:
            sid = store.add_source(source["url"], source["issuer"], source["ticker"])
            started = utc_now()
            status, body, outcome = None, None, "request_failed"
            try:
                with requests.get(source["url"], timeout=20, stream=True, allow_redirects=False,
                                  headers={"User-Agent": "StockPredictorResearch/0.1"}) as response:
                    status = response.status_code
                    if status == 200 and "text/html" in response.headers.get("Content-Type", "").lower():
                        chunks, size = [], 0
                        for chunk in response.iter_content(65536):
                            size += len(chunk)
                            if size > MAX_BYTES:
                                raise ValueError("source_size_limit")
                            chunks.append(chunk)
                        body = b"".join(chunks)
                        if not body:
                            raise ValueError("empty_source")
                        outcome = "ok"
                    else:
                        outcome = "http_or_content_type_rejected"
            except (requests.RequestException, ValueError) as exc:
                outcome, body = type(exc).__name__, None
            received = utc_now()
            vid = store.observe(sid, started, received, status, outcome, body)
            result = dict(round=round_number+1, source_id=sid, version_id=vid, outcome=outcome, http_status=status)
            results.append(result)
            print(json.dumps(result), flush=True)
    return results


def seed(store, example):
    payload = dict(ticker=example["ticker"], issuer=example["issuer"], event_type="revision",
        fiscal_period=example["period"], metric=example["metric"], units=example["units"],
        prior_range=example["prior_range"], new_range=example["new_range"],
        publication_claim=example["new_publication_date"],
        cluster_id=f"draft:{example['ticker']}:{example['new_publication_date']}:{example['period']}",
        evidence_basis="Human-entered claim from earlier primary-source audit; raw versions newly archived; exact timing and comparability unverified")
    for side in ("prior", "new"):
        row = store.conn.execute("SELECT v.id FROM versions v JOIN sources s ON s.id=v.source_id "
            "WHERE s.url=? ORDER BY v.first_observed_at DESC LIMIT 1", (example[side+"_url"],)).fetchone()
        if row is None:
            raise ValueError("Cannot seed event without both archived issuer sources")
        payload[side+"_version_id"] = row["id"]
    identifier = store.add_event(payload)
    store.review(identifier, "Codex source-audit draft", "pending", {},
        "Issuer releases support revision claim. Exact release timing, latest prior guidance, accounting comparability and novelty require verification.")
    result = store.readiness(identifier, utc_now())
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("collect")
    p.add_argument("--sources", type=Path, required=True)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--interval", type=int, default=15)
    p = sub.add_parser("seed")
    p.add_argument("--example", type=Path, required=True)
    args = parser.parse_args()
    store = EvidenceStore(args.store)
    try:
        if args.command == "collect":
            collect(store, json.loads(args.sources.read_text()), args.rounds, args.interval)
        else:
            seed(store, json.loads(args.example.read_text()))
    finally:
        store.close()


if __name__ == "__main__":
    main()
