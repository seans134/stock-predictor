"""Immutable-on-write registration and shared date restrictions for new research."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

from stockpredictor.execution.timing import EXECUTION_VERSION


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_new(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as out:
        json.dump(value, out, indent=2, allow_nan=False)
        out.write("\n")


def source_hash(root: Path) -> str:
    return fingerprint({str(p.relative_to(root)): digest(p)
                        for folder in ("src", "tests")
                        for p in sorted((root / folder).rglob("*.py"))})


def inventory(experiment_dir: Path) -> dict:
    records, starts, ends = [], [], []
    for path in sorted(experiment_dir.glob("*.json")):
        data = json.loads(path.read_text())
        segments = []
        for fold in data.get("folds", []):
            for kind in ("train", "calibrate", "test"):
                span = fold.get(kind)
                if span:
                    lo, hi = map(dt.date.fromisoformat, span)
                    if lo > hi:
                        raise ValueError(f"Invalid date interval: {path}")
                    starts.append(lo)
                    ends.append(hi)
                    segments.append({"kind": kind, "start": str(lo), "end": str(hi)})
        records.append({"file": path.name, "sha256": digest(path), "segments": segments})
    if not starts:
        raise ValueError("No historical experiment date evidence")
    return {"records": records, "development_start": str(min(starts)),
            "development_end": str(max(ends)),
            "unknown_exposure_policy": "All dates through the latest recorded exposure are development; later historical dates are locked, not presumed pristine."}


def require_development(manifest: dict, start: str, end: str) -> None:
    lo, hi = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    permitted_lo = dt.date.fromisoformat(manifest["development_start"])
    permitted_hi = dt.date.fromisoformat(manifest["development_end"])
    if not permitted_lo <= lo <= hi <= permitted_hi:
        raise ValueError("Requested dates are outside the global development interval; protected data cannot be opened by this runner")


def register(path: Path, spec: dict, manifest: dict, root: Path) -> dict:
    require_development(manifest, spec["start"], spec["end"])
    record = {"spec": spec, "manifest_sha256": fingerprint(manifest),
              "source_sha256": source_hash(root), "execution_version": EXECUTION_VERSION,
              "registered_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    record["registration_sha256"] = fingerprint(record)
    write_new(path, record)
    return record


def verify_registration(record: dict, manifest: dict, root: Path) -> dict:
    unsigned = {k: v for k, v in record.items() if k != "registration_sha256"}
    if record["registration_sha256"] != fingerprint(unsigned):
        raise ValueError("Registration content was modified")
    if record["manifest_sha256"] != fingerprint(manifest):
        raise ValueError("Global date manifest changed; register a new experiment")
    if record["source_sha256"] != source_hash(root):
        raise ValueError("Source/tests changed since registration; register a new experiment")
    spec = record["spec"]
    require_development(manifest, spec["start"], spec["end"])
    return spec
