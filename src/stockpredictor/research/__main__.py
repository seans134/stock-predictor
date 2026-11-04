"""python -m stockpredictor.research prepare|run: development-only research CLI."""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import sqlite3
import subprocess
import uuid

import pandas as pd

from stockpredictor.execution.portfolio import RiskConfig
from stockpredictor.research.baselines import (BaselineConfig, STOCKS, INSTRUMENT_TYPES,
                                               opening_volume, closing_momentum, require_complete_evaluation)
from stockpredictor.research.governance import (digest, fingerprint, inventory, register,
                                               require_development, verify_registration, write_new)
from stockpredictor.research.reporting import compare
from stockpredictor.sessions import default_calendar

DEFAULT_ROOT = Path(__file__).resolve().parents[3]


def load_development(db: Path, tickers, start, end, manifest):
    """Read only permitted dates; discard revisions unavailable at bar completion.

    Historical first-version correctness remains limited by vendor backfill:
    a synthetic availability timestamp cannot prove original publication.
    """
    require_development(manifest, str(start), str(end))
    for ticker in tickers:
        if ticker not in INSTRUMENT_TYPES:
            raise ValueError(f"Unsupported instrument {ticker}")
    lo = int(pd.Timestamp(start, tz="America/New_York").timestamp())
    hi = int((pd.Timestamp(end, tz="America/New_York") + pd.Timedelta(days=1)).timestamp())
    loaded = {}
    with sqlite3.connect(db.resolve().as_uri()+"?mode=ro", uri=True) as conn:
        for ticker in tickers:
            frame = pd.read_sql_query("""
                SELECT event_ts, open, high, low, close, volume, available_ts FROM (
                    SELECT *, ROW_NUMBER() OVER (PARTITION BY event_ts ORDER BY available_ts DESC) rn
                    FROM bars WHERE ticker=? AND timeframe='5min'
                    AND event_ts>=? AND event_ts<?
                    AND available_ts>=event_ts+300 AND available_ts<=event_ts+305
                ) WHERE rn=1 ORDER BY event_ts
            """, conn, params=(ticker, lo, hi))
            if frame.empty:
                raise ValueError(f"No timely bars for registered ticker {ticker}")
            frame["event_ts"] = pd.to_datetime(frame.event_ts, unit="s", utc=True)
            frame["available_ts"] = pd.to_datetime(frame.available_ts, unit="s", utc=True)
            loaded[ticker] = frame.set_index("event_ts")
    return loaded


def prepare(root: Path, workspace: Path, start: str, end: str, families=("H1", "H3"), reconstruction=None):
    manifest_path = workspace / "split_manifest.json"
    if not manifest_path.exists():
        manifest = inventory(root / "data/experiments")
        manifest.update(schema_version=1, confirmation_access="disabled",
                        validation_dates=[], final_confirmation_dates=[],
                        reason="No audited pristine historical holdout. New runner only accesses development. Prospective confirmation must be separately registered.")
        write_new(manifest_path, manifest)
    manifest = json.loads(manifest_path.read_text())
    require_development(manifest, start, end)
    # Full DB hash establishes lineage, not access to price outcomes.
    db = root / "data/bars.sqlite"
    data_sha = digest(db)
    reconstruction_spec = None
    if reconstruction is not None:
        from stockpredictor.research.recovery import POLICY, additions
        reconstruction = reconstruction.resolve()
        recovery_manifest = json.loads((reconstruction / "manifest.json").read_text())
        if recovery_manifest["database_sha256"] != data_sha or recovery_manifest["policy"] != POLICY:
            raise ValueError("Reconstruction database/policy mismatch")
        additions(reconstruction, recovery_manifest)
        reconstruction_spec = {"directory": str(reconstruction.relative_to(root.resolve())),
                               "manifest_sha256": digest(reconstruction / "manifest.json"), "policy": POLICY}
    # Eligibility by coverage is assessed only in the pre-evaluation warmup.
    # This avoids selecting names because they have convenient future outcomes.
    cutoff = int(pd.Timestamp(start, tz="America/New_York").timestamp())
    warmup_start = max(dt.date.fromisoformat(manifest["development_start"]),
                       dt.date.fromisoformat(start)-dt.timedelta(days=45))
    warmup_epoch = int(pd.Timestamp(warmup_start, tz="America/New_York").timestamp())
    with sqlite3.connect(db.resolve().as_uri()+"?mode=ro", uri=True) as conn:
        covered = {row[0] for row in conn.execute(
            "SELECT DISTINCT ticker FROM bars WHERE timeframe='5min' AND event_ts>=? AND event_ts<? "
            "AND available_ts>=event_ts+300 AND available_ts<=event_ts+305",
            (warmup_epoch, cutoff))}
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True, capture_output=True, check=True).stdout.strip()
    from dataclasses import asdict
    for family in families:
        identifier = family + "-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
        spec = {
            "id": identifier, "family": family, "status": "registered_development_only",
            "start": start, "end": end,
            "tickers": [ticker for ticker in STOCKS if ticker in covered] if family == "H1" else ["SPY"],
            "coverage_exclusions": [ticker for ticker in STOCKS if ticker not in covered] if family == "H1" else [],
            "coverage_rule": "explicit instrument allowlist intersected with timely bars in pre-evaluation warmup; no outcome-based exclusion",
            "parameters": asdict(BaselineConfig()),
            "risk": asdict(RiskConfig(max_name_fraction=.02, max_positions=20,
                                     max_gross_fraction=.40, fixed_name_fraction=.02)),
            "primary_endpoint": "mean full-session daily net portfolio return",
            "control": "ordinary opening activity rvol<=1" if family == "H1" else "unconditional long same closing window",
            "hypothesis": "positive long continuation after abnormal opening volume" if family == "H1" else "positive first-half-hour return predicts closing-window continuation",
            "exit": "10% prior daily ATR stop or scheduled close" if family == "H1" else "scheduled close",
            "cost_scenarios_bps": [0,2,5,11,20],
            "data_sha256": data_sha, "git_commit": source_commit,
            "confirmation": "prohibited; no adjusted inference or operational validation yet",
            "review": "once after registered run; no selection within run",
            "limitations": ["static surviving-name allowlist", "synthetic historical availability",
                           "no observed quotes/impact", "scheduled close proxy, not auction fills",
                           "no corporate-action event feed; prior raw-price scale may be affected",
                           "realized drawdown excludes intraday unrealized losses"],
        }
        if reconstruction_spec:
            spec["reconstruction"] = reconstruction_spec
            spec["limitations"].append("Reconstructed historical bars: decision availability is hypothetical, retrieval provenance is recorded; exploratory only")
        register(workspace / "registry" / f"{identifier}.json", spec, manifest, root)
        print(identifier)


def run(root: Path, workspace: Path, identifier: str):
    if Path(identifier).name != identifier or "/" in identifier or "\\" in identifier:
        raise ValueError("Invalid experiment identifier")
    manifest = json.loads((workspace / "split_manifest.json").read_text())
    registration = json.loads((workspace / "registry" / f"{identifier}.json").read_text())
    spec = verify_registration(registration, manifest, root)
    db = root / "data/bars.sqlite"
    if digest(db) != spec["data_sha256"]:
        raise ValueError("Input database changed since registration")
    calendar = default_calendar()
    start, end = dt.date.fromisoformat(spec["start"]), dt.date.fromisoformat(spec["end"])
    warmup = max(dt.date.fromisoformat(manifest["development_start"]), start-dt.timedelta(days=45))
    days = calendar.sessions_between(start, end)
    if not days:
        raise ValueError("No evaluation sessions")
    output = workspace / "runs" / identifier
    output.mkdir(parents=True, exist_ok=False)
    write_new(output / "started.json", {"registration_sha256": registration["registration_sha256"],
                                      "started_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    try:
        data = load_development(db, spec["tickers"], warmup, end, manifest)
        if spec.get("reconstruction"):
            from stockpredictor.research.recovery import overlay
            reconstruction = spec["reconstruction"]
            folder = (root / reconstruction["directory"]).resolve()
            folder.relative_to(root.resolve())
            data = overlay(data, folder, reconstruction["manifest_sha256"], spec["data_sha256"])
            # Even a valid reconstruction cannot expand the registered interval.
            for ticker, frame in data.items():
                dates = frame.index.tz_convert("America/New_York").date
                data[ticker] = frame[(dates >= warmup) & (dates <= end)]
        require_complete_evaluation(data, calendar, days)
        cfg = BaselineConfig(**spec["parameters"])
        if spec["family"] == "H1":
            primary, selected = opening_volume(data, calendar, days, cfg)
            control, control_selected = opening_volume(data, calendar, days, cfg, ordinary=True)
            selected.to_csv(output / "selections.csv", index=False)
            control_selected.to_csv(output / "control_selections.csv", index=False)
        else:
            primary = closing_momentum(data["SPY"], calendar, days, cfg)
            control = closing_momentum(data["SPY"], calendar, days, cfg, unconditional=True)
        report, daily, accepted, control_accepted = compare(primary, control, days,
                                                           RiskConfig(**spec["risk"]), spec["cost_scenarios_bps"])
        coverage = {ticker: {"bars": len(frame), "first_bar": str(frame.index.min()),
                             "last_bar": str(frame.index.max()),
                             "evaluation_dates_present": len(set(frame.index.tz_convert("America/New_York").date).intersection(days))}
                    for ticker, frame in data.items()}
        report.update(experiment_id=identifier, registration_sha256=registration["registration_sha256"],
                      limitations=spec["limitations"], coverage=coverage,
                      status="completed_development_only")
        if spec.get("reconstruction"):
            report["reconstruction"] = spec["reconstruction"]
        primary.to_csv(output / "signals.csv", index=False)
        control.to_csv(output / "control_signals.csv", index=False)
        accepted.to_csv(output / "accepted_trades.csv", index=False)
        control_accepted.to_csv(output / "control_accepted_trades.csv", index=False)
        daily.to_csv(output / "daily.csv")
        write_new(output / "report.json", report)
        print(json.dumps({"output": str(output), "decision": report["decision"], "primary": report["primary"]}, indent=2))
    except Exception as exc:
        blocked = str(exc).startswith(("Incomplete evaluation coverage:", "No timely bars for registered ticker"))
        write_new(output / "failed.json", {"status": "blocked_by_data" if blocked else "failed", "error": str(exc),
                                          "registration_sha256": registration["registration_sha256"]})
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workspace", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--family", choices=("H1", "H3"), default=None)
    p.add_argument("--reconstruction", type=Path, default=None)
    p = sub.add_parser("run")
    p.add_argument("identifier")
    args = parser.parse_args(argv)
    workspace = args.workspace or args.root / "data/edge_research"
    if args.command == "prepare":
        prepare(args.root, workspace, args.start, args.end,
                (args.family,) if args.family else ("H1", "H3"), args.reconstruction)
    else:
        run(args.root, workspace, args.identifier)


if __name__ == "__main__":
    main()
