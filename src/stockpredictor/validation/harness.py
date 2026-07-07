"""Walk-forward experiment runner.

Runs a caller-supplied fold function over generated folds and records the
whole experiment — configuration, fold boundaries, per-fold metrics — as an
append-only JSON file. The plan's experiment-discipline rule: every
configuration tried is recorded, so the size of the search is known when
results are judged.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
from pathlib import Path
from typing import Callable

from stockpredictor.validation.folds import Fold

DEFAULT_EXPERIMENT_DIR = Path("data") / "experiments"


def run_walkforward(
    folds: list[Fold],
    run_fold: Callable[[Fold], dict],
    experiment_name: str,
    config_record: dict,
    out_dir: str | Path = DEFAULT_EXPERIMENT_DIR,
) -> dict:
    """Execute run_fold per fold, persist the experiment, return the record.

    run_fold receives a Fold and returns a flat dict of numeric metrics.
    """
    fold_results = []
    for fold in folds:
        metrics = run_fold(fold)
        fold_results.append({**fold.summary(), "metrics": metrics})

    record = {
        "experiment": experiment_name,
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": _jsonable(config_record),
        "n_folds": len(folds),
        "folds": fold_results,
    }

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{stamp}_{experiment_name}.json"
    path.write_text(json.dumps(record, indent=2))
    record["path"] = str(path)
    return record


def _jsonable(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    return value
