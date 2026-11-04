"""Read-only audit of existing experiments; never loads holdout price outcomes.

Run from the repository root with .venv/Scripts/python.exe
docs/research/audit_saved_results.py. Outputs descriptive, post-selection
diagnostics, not fresh out-of-sample evidence.
"""
from pathlib import Path
import datetime as dt
import json
import math
import sqlite3

import numpy as np
import pandas as pd

from stockpredictor.orb.backtest import simulate_orb_session
from stockpredictor.orb.strategy import ORBConfig
from stockpredictor.execution.costs import CostModel

ROOT = Path(__file__).resolve().parents[2]
result = {"audit_date": "2026-10-20", "experiments": [], "blotters": {}}
for path in sorted((ROOT / "data/experiments").glob("*.json")):
    record = json.loads(path.read_text())
    folds = record["folds"]
    for prefix in ("", "pure_", "gated_", "all_", "cand_"):
        metrics = [f.get("metrics", {}) for f in folds]
        count_key = prefix + "n_trades"
        n = sum(m.get(count_key, 0) for m in metrics)
        if not n:
            continue
        row = {"file": path.name, "variant": prefix.rstrip("_") or "default",
               "n_trades": n, "test_start": folds[0]["test"][0],
               "test_end": folds[-1]["test"][-1]}
        for metric in ("avg_gross_bps", "avg_net_bps"):
            key = prefix + metric
            valid = [m for m in metrics if m.get(count_key, 0)]
            assert all(math.isfinite(m[key]) for m in valid)
            row[metric] = sum(m[count_key] * m[key] for m in valid) / n
        result["experiments"].append(row)

rng = np.random.default_rng(20261020)
for name in ("orb-full_pure", "orb-full_gated", "backtest-morning-window",
             "backtest-cost-aware-morning"):
    path = ROOT / "data" / f"trades_{name}.csv"
    frame = pd.read_csv(path)
    # Preserve cross-sectional dependence by resampling whole trading days.
    # Circular blocks of five active dates approximate short serial dependence.
    daily = frame.groupby("date")["gross_ret"].agg(["sum", "count"])
    n_days = len(daily)
    starts = rng.integers(0, n_days, size=(10000, math.ceil(n_days / 5)))
    indices = ((starts[..., None] + np.arange(5)) % n_days).reshape(10000, -1)[:, :n_days]
    means = (daily["sum"].to_numpy()[indices].sum(axis=1)
             / daily["count"].to_numpy()[indices].sum(axis=1)) * 10000
    stats = {"trades": len(frame), "active_dates": n_days,
             "gross_bps": frame.gross_ret.mean() * 10000,
             "net_bps": frame.net_ret.mean() * 10000,
             "gross_95pct_block_bootstrap_interval_bps": np.quantile(means, [.025, .975]).tolist(),
             "interval_warning": "Descriptive, conditional on selected strategy; no multiple-testing correction. Blocks cover active dates only, not a portfolio return series.",
             "net_bps_at_round_trip_cost": {str(cost): frame.gross_ret.mean() * 10000 - cost
                                             for cost in (0, 2, 5, 11, 20)},
             "direction_gross_bps": (frame.groupby("direction").gross_ret.mean() * 10000).to_dict()}
    result["blotters"][name] = stats

uri = (ROOT / "data/news.sqlite").as_uri() + "?mode=ro"
with sqlite3.connect(uri, uri=True) as conn:
    count, minimum, maximum = conn.execute(
        "SELECT COUNT(*), MIN(available_ts-published_ts), MAX(available_ts-published_ts) FROM articles"
    ).fetchone()
    result["news_metadata"] = {"articles": count, "min_assumed_delay_seconds": minimum,
                               "max_assumed_delay_seconds": maximum,
                               "scope": "Metadata only; no news-conditioned return analysis."}

# Synthetic counterexample: the bar after entry gaps entirely below the stop,
# then price recovers. The corrected simulator must exit at the gap opening.
rows = [(100,101,99,100,1000)] * 3 + [
    (100,103,100,102.5,5000), (103,103.5,102,103,5000),
    (103,103.5,102,103,5000), (98,100,97,99,5000), (104,106,103,105,5000)]
session = pd.DataFrame(rows, columns=["open","high","low","close","volume"],
                       index=pd.date_range("2024-03-18 09:30", periods=len(rows), freq="5min", tz="America/New_York"))
trade = simulate_orb_session(session, 2.0, ORBConfig(), CostModel(), "SYNTHETIC", dt.date(2024,3,18))
assert trade.entry == 103 and trade.stop == 101
assert trade.exit_reason == "stop" and trade.exit == 98
result["synthetic_stop_gap"] = {"entry": trade.entry, "stop": trade.stop,
                                "next_bar_open": 98, "next_bar_high": 100,
                                "simulator_exit": trade.exit, "simulator_reason": trade.exit_reason,
                                "finding": "Corrected simulator exits at the gap opening reference; actual execution still requires quotes/liquidity."}
out = ROOT / "docs/research/2026-10-20-audit-current.json"
out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
print(json.dumps({"output": str(out), "blotters": result["blotters"],
                  "news_metadata": result["news_metadata"],
                  "synthetic_stop_gap": result["synthetic_stop_gap"]}, indent=2))
