"""Recover audited missing bars to immutable source evidence, never BarStore.

Commands: plan DIRECTORY; fetch DIRECTORY; seal DIRECTORY.
The plan is frozen before requests; downloads are resumable and rate limited.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import time
import math

from dotenv import load_dotenv
import pandas as pd
import requests

from stockpredictor.research.governance import digest, fingerprint, require_development, write_new

ROOT = Path(__file__).resolve().parents[3]
POLICY = {
    "version": "reconstruction-strict-v1",
    "purpose": "exploratory historical reconstruction only; never confirmation or live input",
    "merge": "add only audited absent timestamps; preserve every original bar",
    "availability": "record retrieval time separately; bar-end+5s is an explicit hypothetical decision assumption, not observed availability",
    "missing": "no imputation, no removed sessions or trades; any unresolved evaluation bar blocks performance",
    "execution": "unchanged strategy, stop and cost rules; unknown intrabar stop paths cannot be bounded from adjacent candles",
}


def read_plan(folder):
    plan = json.loads((folder / "plan.json").read_text())
    if plan["sha256"] != fingerprint({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("Recovery plan modified")
    return plan


def plan(folder, root=ROOT):
    workspace = root / "data/edge_research"
    manifest = json.loads((workspace / "split_manifest.json").read_text())
    ledgers = [workspace / "audits" / name / "missing_bars.csv"
               for name in ("coverage-20261021", "spy-coverage-20261021")]
    missing = pd.concat([pd.read_csv(p) for p in ledgers], ignore_index=True)
    database_sha = digest(root / "data/bars.sqlite")
    for ledger in ledgers:
        audit_summary = json.loads((ledger.parent / "summary.json").read_text())
        if audit_summary["database_sha256"] != database_sha:
            raise ValueError("Audited database changed; repeat the coverage audit")
    if set(missing.reason) != {"absent"}:
        raise ValueError("This recovery plan handles absent bars only")
    missing = missing.drop_duplicates(["ticker", "event_ts"]).sort_values(["ticker", "event_ts"])
    jobs = []
    for ticker, rows in missing.groupby("ticker"):
        dates = sorted(set(rows.date))
        # Group only short requests so the base-minute scan limit cannot truncate.
        chunks = []
        for day in dates:
            if not chunks or (dt.date.fromisoformat(day)-dt.date.fromisoformat(chunks[-1][0])).days > 7:
                chunks.append([day])
            else:
                chunks[-1].append(day)
        for chunk in chunks:
            require_development(manifest, chunk[0], chunk[-1])
            jobs.append(dict(ticker=ticker, start=chunk[0], end=chunk[-1],
                             file=f"{ticker}-{chunk[0]}-{chunk[-1]}.json"))
    value = dict(created_at=dt.datetime.now(dt.timezone.utc).isoformat(), policy=POLICY,
                 database_sha256=database_sha,
                 split_manifest_sha256=fingerprint(manifest),
                 audit_hashes={str(p.relative_to(root)): digest(p) for p in ledgers},
                 jobs=jobs, missing=missing[["ticker", "event_ts"]].to_dict("records"))
    value["sha256"] = fingerprint(value)
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / "plan.json", value)
    print(f"Frozen {len(jobs)} requests for {len(missing)} absent bars", flush=True)


def fetch(folder):
    value = read_plan(folder)
    load_dotenv(ROOT / ".env")
    key = os.getenv("POLYGON_API_KEY")
    if not key:
        raise ValueError("Provider credential not configured")
    last = time.monotonic()  # respect pacing even on restart
    for number, job in enumerate(value["jobs"], 1):
        path = folder / "responses" / job["file"]
        if path.exists():
            continue
        time.sleep(max(0, 12.5-(time.monotonic()-last)))
        observed = dt.datetime.now(dt.timezone.utc).isoformat()
        try:
            response = requests.get(
                f"https://api.polygon.io/v2/aggs/ticker/{job['ticker']}/range/5/minute/{job['start']}/{job['end']}",
                params={"adjusted": "false", "sort": "asc", "limit": 50000},
                headers={"Authorization": "Bearer " + key}, timeout=30)
            last = time.monotonic()
            if response.status_code != 200:
                print(f"Stopped: HTTP {response.status_code}; completed evidence preserved", flush=True)
                return
            payload = response.json()
            if payload.get("next_url") or payload.get("status") not in ("OK", "DELAYED"):
                print("Stopped: incomplete or unsuccessful source response", flush=True)
                return
        except (requests.RequestException, ValueError) as exc:
            print(f"Stopped: {type(exc).__name__}; no credential-bearing error text logged", flush=True)
            return
        write_new(path, dict(job=job, observed_at=observed, received_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                            historical_availability="unverified", adjusted=False,
                            bars=payload.get("results") or []))
        print(f"{number}/{len(value['jobs'])} saved {job['ticker']} {job['start']}..{job['end']}", flush=True)


def additions(folder, manifest):
    """Validate all frozen evidence and return only planned missing timestamps."""
    plan_value = read_plan(folder)
    if digest(folder / "plan.json") != manifest["plan_file_sha256"]:
        raise ValueError("Recovery plan hash mismatch")
    wanted = {(row["ticker"], row["event_ts"]) for row in plan_value["missing"]}
    recovered = {}
    for job in plan_value["jobs"]:
        path = folder / "responses" / job["file"]
        if digest(path) != manifest["responses"][job["file"]]:
            raise ValueError("Recovered evidence changed")
        evidence = json.loads(path.read_text())
        if evidence["job"] != job or evidence["adjusted"] is not False:
            raise ValueError("Response provenance mismatch")
        observed = pd.Timestamp(evidence["received_at"])
        if observed.tzinfo is None:
            raise ValueError("Observation time must be timezone aware")
        for bar in evidence["bars"]:
            if bar["t"] % 300000:
                raise ValueError("Unaligned source bar")
            ts = bar["t"] // 1000
            key = (job["ticker"], ts)
            if key not in wanted:
                continue
            if key in recovered:
                raise ValueError("Duplicate recovered timestamp")
            prices = [bar[k] for k in ("o", "h", "l", "c")]
            if (not all(math.isfinite(p) and p > 0 for p in prices)
                or not math.isfinite(bar["v"]) or bar["v"] < 0
                or bar["h"] < max(prices) or bar["l"] > min(prices)):
                raise ValueError("Invalid recovered OHLCV")
            if observed.timestamp() < ts+300:
                raise ValueError("Recovery observation predates bar completion")
            recovered[key] = dict(ticker=job["ticker"], event_ts=ts,
                open=bar["o"], high=bar["h"], low=bar["l"], close=bar["c"], volume=bar["v"],
                recovery_observed_at=observed, source_file=job["file"])
    return list(recovered.values()), sorted(wanted-set(recovered))


def seal(folder):
    value = read_plan(folder)
    manifest = dict(policy=value["policy"], database_sha256=value["database_sha256"],
                    plan_file_sha256=digest(folder / "plan.json"),
                    responses={job["file"]: digest(folder / "responses" / job["file"]) for job in value["jobs"]})
    recovered, missing = additions(folder, manifest)
    manifest.update(recovered_bars=len(recovered), unresolved_bars=len(missing),
                    unresolved=[dict(ticker=t, event_ts=s) for t, s in missing])
    write_new(folder / "manifest.json", manifest)
    print(f"Sealed {len(recovered)} recovered bars; {len(missing)} unresolved")


def overlay(data, folder, manifest_sha, database_sha):
    if digest(folder / "manifest.json") != manifest_sha:
        raise ValueError("Reconstruction manifest changed")
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["database_sha256"] != database_sha or manifest["policy"] != POLICY:
        raise ValueError("Reconstruction database/policy mismatch")
    recovered, _ = additions(folder, manifest)
    for ticker, frame in data.items():
        # Do not present synthetic or reconstructed availability as observed.
        frame = frame.rename(columns={"available_ts": "original_assumed_available_ts"})
        rows = [r for r in recovered if r["ticker"] == ticker]
        if rows:
            extra = pd.DataFrame(rows)
            extra["event_ts"] = pd.to_datetime(extra.event_ts, unit="s", utc=True)
            extra = extra.set_index("event_ts")
            if extra.index.isin(frame.index).any():
                raise ValueError("Recovery attempted to replace original bars")
            frame = pd.concat([frame, extra]).sort_index()
        frame["modeled_available_ts"] = frame.index + pd.Timedelta(seconds=305)
        data[ticker] = frame
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "fetch", "seal"])
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    {"plan": plan, "fetch": fetch, "seal": seal}[args.command](args.directory)


if __name__ == "__main__":
    main()
