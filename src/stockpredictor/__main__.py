"""CLI entry point: python -m stockpredictor <command>."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_DB = Path("data") / "bars.sqlite"


def cmd_ingest(args: argparse.Namespace) -> int:
    from stockpredictor.data.store import BarStore
    from stockpredictor.ingest.polygon import PolygonClient, ingest_history

    load_dotenv()
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print(
            "Missing POLYGON_API_KEY.\n"
            "Create a free account at https://polygon.io (Massive), copy your API key,\n"
            "and put it in a .env file (see .env.example).",
            file=sys.stderr,
        )
        return 1

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]

    print(
        f"Backfilling {len(tickers)} tickers ({args.start}..{args.end}). "
        "Free tier is rate-limited to ~5 calls/min; this may take a while."
    )
    store = BarStore(args.db)
    try:
        inserted = ingest_history(store, PolygonClient(api_key), tickers, start, end)
    finally:
        store.close()
    total = sum(inserted.values())
    print(f"Inserted {total} new bars across {len(tickers)} tickers into {args.db}")
    for ticker, count in inserted.items():
        print(f"  {ticker}: {count}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    from stockpredictor.data.asof import AsOfData
    from stockpredictor.data.store import BarStore
    from stockpredictor.labeling.labeler import LabelConfig, classify_sessions, qualify_windows
    from stockpredictor.labeling.tod import TimeOfDayStats
    from stockpredictor.labeling.windows import build_windows
    from stockpredictor.sessions import default_calendar

    fit_start = dt.date.fromisoformat(args.fit_start)
    fit_end = dt.date.fromisoformat(args.fit_end)
    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if fit_end >= start:
        print("--fit-end must be before --start (fold isolation)", file=sys.stderr)
        return 1

    store = BarStore(args.db)
    try:
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if args.tickers
            else store.tickers()
        )
        data = AsOfData(store)
        calendar = default_calendar()
        config = LabelConfig()

        print(f"Fitting time-of-day stats on {fit_start}..{fit_end} ({len(tickers)} tickers)...")
        stats = TimeOfDayStats().fit(build_windows(data, calendar, tickers, fit_start, fit_end))
        print(f"  {stats.n_slots} (ticker, slot) buckets with trusted statistics")

        print(f"Labeling {start}..{end}...")
        windows = build_windows(data, calendar, tickers, start, end)
        qualified = qualify_windows(stats.transform(windows), config)
        sessions = classify_sessions(qualified, calendar, config)
    finally:
        store.close()

    sessions.to_csv(args.out, index=False)
    print(f"\nWrote {len(sessions)} session labels to {args.out}\n")
    print("Class distribution:")
    overall = sessions["label"].value_counts()
    for label, count in overall.items():
        print(f"  {label:>13}: {count:5d}  ({count / len(sessions):.1%})")
    print("\nPer ticker:")
    table = sessions.pivot_table(index="ticker", columns="label", aggfunc="size", fill_value=0)
    print(table.to_string())
    rate = qualified["qualifies"].mean()
    print(f"\nQualifying-window rate: {rate:.2%} of {len(qualified)} windows")
    return 0


def cmd_walkforward(args: argparse.Namespace) -> int:
    from stockpredictor.data.asof import AsOfData
    from stockpredictor.data.store import BarStore
    from stockpredictor.labeling.labeler import (
        LABEL_NONE,
        LABEL_OPENING_ONLY,
        LABEL_SUSTAINED,
        LabelConfig,
        classify_sessions,
        qualify_windows,
    )
    from stockpredictor.labeling.tod import TimeOfDayStats
    from stockpredictor.labeling.windows import build_windows
    from stockpredictor.sessions import default_calendar
    from stockpredictor.validation.folds import WalkForwardConfig, generate_folds
    from stockpredictor.validation.harness import run_walkforward

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    wf_config = WalkForwardConfig(
        train_sessions=args.train,
        test_sessions=args.test,
        step_sessions=args.step,
        calibrate_sessions=args.calibrate,
        embargo_sessions=args.embargo,
        expanding=args.expanding,
        holdout_sessions=args.holdout,
    )
    calendar = default_calendar()
    sessions = calendar.sessions_between(start, end)
    folds = generate_folds(sessions, wf_config)
    if not folds:
        print(
            f"No folds fit: {len(sessions)} sessions available but one fold needs "
            f"{wf_config.train_sessions + wf_config.calibrate_sessions + wf_config.test_sessions} "
            "plus embargoes and holdout.",
            file=sys.stderr,
        )
        return 1
    print(f"{len(folds)} folds over {len(sessions)} sessions ({args.holdout} held out)")

    store = BarStore(args.db)
    try:
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if args.tickers
            else store.tickers()
        )
        print(f"Building windows for {len(tickers)} tickers {start}..{end} (one-time pass)...")
        windows = build_windows(AsOfData(store), calendar, tickers, start, end)
    finally:
        store.close()

    label_config = LabelConfig()

    def run_fold(fold):
        train_w = windows[windows["date"].isin(set(fold.train_days))]
        test_w = windows[windows["date"].isin(set(fold.test_days))]
        stats = TimeOfDayStats().fit(train_w)
        qualified = qualify_windows(stats.transform(test_w), label_config)
        labeled = classify_sessions(qualified, calendar, label_config)
        n = max(len(labeled), 1)
        counts = labeled["label"].value_counts()
        return {
            "n_sessions": len(labeled),
            "trusted_slots": stats.n_slots,
            "frac_none": counts.get(LABEL_NONE, 0) / n,
            "frac_opening_only": counts.get(LABEL_OPENING_ONLY, 0) / n,
            "frac_sustained": counts.get(LABEL_SUSTAINED, 0) / n,
            "qualify_rate": float(qualified["qualifies"].mean()),
        }

    record = run_walkforward(
        folds,
        run_fold,
        experiment_name=args.name,
        config_record={
            "walkforward": wf_config,
            "label": label_config,
            "tickers": tickers,
            "range": [str(start), str(end)],
        },
    )

    print(f"\n{'fold':>4} {'test period':^23} {'none':>6} {'open':>6} {'sust':>6} {'qrate':>6}")
    for fr in record["folds"]:
        m = fr["metrics"]
        print(
            f"{fr['fold_id']:>4} {fr['test'][0]} .. {fr['test'][1]} "
            f"{m['frac_none']:>6.1%} {m['frac_opening_only']:>6.1%} "
            f"{m['frac_sustained']:>6.1%} {m['qualify_rate']:>6.2%}"
        )
    import statistics

    print("\nAcross folds (mean +/- stdev):")
    for key in ("frac_none", "frac_opening_only", "frac_sustained", "qualify_rate"):
        values = [fr["metrics"][key] for fr in record["folds"]]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {key:>18}: {mean:.1%} +/- {sd:.1%}")
    print(f"\nExperiment record: {record['path']}")
    return 0


def cmd_stage2(args: argparse.Namespace) -> int:
    from stockpredictor.data.asof import AsOfData
    from stockpredictor.data.store import BarStore
    from stockpredictor.sessions import default_calendar
    from stockpredictor.stage2.data import load_rth_bars
    from stockpredictor.stage2.features import TARGET_COLUMN, build_features
    from stockpredictor.stage2.quantile import (
        QuantileModelConfig,
        Stage2QuantileModel,
        evaluate,
    )
    from stockpredictor.validation.folds import WalkForwardConfig, generate_folds
    from stockpredictor.validation.harness import run_walkforward

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    wf_config = WalkForwardConfig(
        train_sessions=args.train,
        test_sessions=args.test,
        step_sessions=args.step,
        embargo_sessions=args.embargo,
        holdout_sessions=args.holdout,
    )
    calendar = default_calendar()
    sessions = calendar.sessions_between(start, end)
    folds = generate_folds(sessions, wf_config)
    if not folds:
        print("No folds fit in the requested range.", file=sys.stderr)
        return 1
    print(f"{len(folds)} folds over {len(sessions)} sessions ({args.holdout} held out)")

    store = BarStore(args.db)
    try:
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if args.tickers
            else store.tickers()
        )
        data = AsOfData(store)
        print(f"Loading bars and building features for {len(tickers)} tickers...")
        bars_by_ticker = {
            t: load_rth_bars(data, calendar, t, start, end) for t in tickers
        }
    finally:
        store.close()
    features = build_features(bars_by_ticker, calendar)
    print(f"  {len(features)} feature rows")

    model_config = QuantileModelConfig()

    def run_fold(fold):
        train = features[features["date"].isin(set(fold.train_days))]
        test = features[features["date"].isin(set(fold.test_days))]
        model = Stage2QuantileModel(model_config).fit(train)
        preds = model.predict(test)
        metrics = evaluate(test, preds)
        print(f"  fold {fold.fold_id}: done ({metrics['n_rows']} test rows)")
        return metrics

    record = run_walkforward(
        folds,
        run_fold,
        experiment_name=args.name,
        config_record={
            "walkforward": wf_config,
            "model": model_config,
            "tickers": tickers,
            "range": [str(start), str(end)],
        },
    )

    print(f"\n{'fold':>4} {'test period':^23} {'cov80':>6} {'width':>7} {'medMAE':>7} {'dirAcc':>6} {'vs0':>5}")
    for fr in record["folds"]:
        m = fr["metrics"]
        print(
            f"{fr['fold_id']:>4} {fr['test'][0]} .. {fr['test'][1]} "
            f"{m['coverage_80']:>6.1%} {m['interval_width'] * 1e4:>6.1f}bp "
            f"{m['median_mae'] * 1e4:>6.2f}bp {m['direction_accuracy']:>6.1%} "
            f"{m['pinball_q50_vs_baseline']:>5.3f}"
        )
    import statistics

    print("\nAcross folds (mean +/- stdev):")
    for key in ("coverage_80", "interval_width", "median_mae", "direction_accuracy", "pinball_q50_vs_baseline"):
        values = [fr["metrics"][key] for fr in record["folds"]]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {key:>24}: {mean:.4f} +/- {sd:.4f}")
    print(f"\nExperiment record: {record['path']}")
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    from stockpredictor.data.store import BarStore

    store = BarStore(args.db)
    try:
        tickers = store.tickers()
        if not tickers:
            print(f"No data in {args.db}")
            return 0
        for ticker in tickers:
            lo, hi = store.coverage(ticker)
            print(f"{ticker}: {lo} .. {hi}")
    finally:
        store.close()
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="stockpredictor")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="backfill historical 5-min bars from Polygon")
    p_ingest.add_argument("--tickers", required=True, help="comma-separated, e.g. AAPL,MSFT,NVDA")
    p_ingest.add_argument("--start", required=True, help="ISO date, e.g. 2024-01-01")
    p_ingest.add_argument("--end", required=True, help="ISO date, e.g. 2024-06-30")
    p_ingest.add_argument("--db", default=str(DEFAULT_DB))
    p_ingest.set_defaults(func=cmd_ingest)

    p_label = sub.add_parser("label", help="compute Stage 1 session labels")
    p_label.add_argument("--fit-start", required=True, help="time-of-day stats fit range start")
    p_label.add_argument("--fit-end", required=True, help="fit range end (must precede --start)")
    p_label.add_argument("--start", required=True, help="labeling range start")
    p_label.add_argument("--end", required=True, help="labeling range end")
    p_label.add_argument("--tickers", default=None, help="comma-separated; default: all stored")
    p_label.add_argument("--db", default=str(DEFAULT_DB))
    p_label.add_argument("--out", default=str(Path("data") / "session_labels.csv"))
    p_label.set_defaults(func=cmd_label)

    p_wf = sub.add_parser("walkforward", help="run the labeler across walk-forward folds")
    p_wf.add_argument("--start", required=True)
    p_wf.add_argument("--end", required=True)
    p_wf.add_argument("--train", type=int, default=250, help="train sessions per fold")
    p_wf.add_argument("--test", type=int, default=21, help="test sessions per fold")
    p_wf.add_argument("--step", type=int, default=21, help="sessions to advance per fold")
    p_wf.add_argument("--calibrate", type=int, default=0)
    p_wf.add_argument("--embargo", type=int, default=1)
    p_wf.add_argument("--holdout", type=int, default=42, help="final sessions never folded")
    p_wf.add_argument("--expanding", action="store_true")
    p_wf.add_argument("--tickers", default=None)
    p_wf.add_argument("--name", default="label-stability")
    p_wf.add_argument("--db", default=str(DEFAULT_DB))
    p_wf.set_defaults(func=cmd_walkforward)

    p_s2 = sub.add_parser("stage2", help="walk-forward Stage 2 quantile model")
    p_s2.add_argument("--start", required=True)
    p_s2.add_argument("--end", required=True)
    p_s2.add_argument("--train", type=int, default=250)
    p_s2.add_argument("--test", type=int, default=21)
    p_s2.add_argument("--step", type=int, default=21)
    p_s2.add_argument("--embargo", type=int, default=1)
    p_s2.add_argument("--holdout", type=int, default=42)
    p_s2.add_argument("--tickers", default=None)
    p_s2.add_argument("--name", default="stage2-quantile-baseline")
    p_s2.add_argument("--db", default=str(DEFAULT_DB))
    p_s2.set_defaults(func=cmd_stage2)

    p_cov = sub.add_parser("coverage", help="show stored date range per ticker")
    p_cov.add_argument("--db", default=str(DEFAULT_DB))
    p_cov.set_defaults(func=cmd_coverage)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
