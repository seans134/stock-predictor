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


def cmd_stage1(args: argparse.Namespace) -> int:
    import pandas as pd

    from stockpredictor.data.asof import AsOfData
    from stockpredictor.data.store import BarStore
    from stockpredictor.labeling.labeler import LabelConfig, classify_sessions, qualify_windows
    from stockpredictor.labeling.tod import TimeOfDayStats
    from stockpredictor.labeling.windows import LABEL_AS_OF_DELAY, session_windows
    from stockpredictor.sessions import ET, default_calendar
    from stockpredictor.stage1.calibration import DirichletCalibrator
    from stockpredictor.stage1.features import build_stage1_features
    from stockpredictor.stage1.model import (
        CLASS_ORDER,
        Stage1Classifier,
        Stage1ModelConfig,
        evaluate_stage1,
        proba_frame,
    )
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
        holdout_sessions=args.holdout,
    )
    calendar = default_calendar()
    sessions = calendar.sessions_between(start, end)
    session_set = set(sessions)
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
        range_close = pd.Timestamp(calendar.session_close(sessions[-1]))
        as_of = range_close + LABEL_AS_OF_DELAY
        range_open = pd.Timestamp(calendar.session_open(sessions[0])) - pd.Timedelta("6h")
        print(f"Loading extended-hours bars for {len(tickers)} tickers...")
        bars_by_ticker = {
            t: data.get_bars(t, as_of=as_of, lookback=as_of - range_open) for t in tickers
        }
    finally:
        store.close()

    print("Building pre-market features...")
    features = build_stage1_features(bars_by_ticker, calendar)
    print(f"  {len(features)} ticker-session rows")

    print("Building 15-min windows for labels...")
    window_frames = []
    for ticker, bars in bars_by_ticker.items():
        if bars.empty:
            continue
        et_dates = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
        for day, day_bars in bars.groupby(et_dates):
            if day in session_set:
                window_frames.append(session_windows(day_bars, ticker, day, calendar))
    windows = pd.concat(window_frames, ignore_index=True)

    label_config = LabelConfig()
    model_config = Stage1ModelConfig()

    import numpy as np

    class_index = {label: i for i, label in enumerate(CLASS_ORDER)}

    def label_rows(day_set, stats):
        segment = windows[windows["date"].isin(day_set)]
        labels = classify_sessions(
            qualify_windows(stats.transform(segment), label_config), calendar, label_config
        )
        return features.merge(labels[["ticker", "date", "label"]], on=["ticker", "date"])

    def run_fold(fold):
        # Label thresholds and time-of-day stats come from the train
        # segment only; train, calibrate, and test labels all use them.
        stats = TimeOfDayStats().fit(windows[windows["date"].isin(set(fold.train_days))])
        train_rows = label_rows(set(fold.train_days), stats)
        calib_rows = label_rows(set(fold.calibrate_days), stats)
        test_rows = label_rows(set(fold.test_days), stats)

        model = Stage1Classifier(model_config).fit(train_rows)

        # Calibrator fits on predictions for data the classifier never saw.
        p_cols = [f"p_{c}" for c in CLASS_ORDER]
        calib_proba = model.predict_proba(calib_rows)[p_cols].to_numpy()
        calib_y = calib_rows["label"].map(class_index).to_numpy()
        calibrator = DirichletCalibrator().fit(calib_proba, calib_y)

        raw_proba = model.predict_proba(test_rows)
        cal_proba = proba_frame(
            calibrator.transform(raw_proba[p_cols].to_numpy()), test_rows.index
        )

        prior = train_rows["label"].map(class_index).value_counts(normalize=True)
        prior = prior.reindex(range(len(CLASS_ORDER)), fill_value=0.0).to_numpy()

        metrics = evaluate_stage1(test_rows, cal_proba, k=args.top_k, prior=prior)
        raw_metrics = evaluate_stage1(test_rows, raw_proba, k=args.top_k)
        metrics["log_loss_raw"] = raw_metrics["log_loss"]
        metrics["brier_raw"] = raw_metrics["brier"]
        print(f"  fold {fold.fold_id}: done ({metrics['n_rows']} test rows)")
        return metrics

    record = run_walkforward(
        folds,
        run_fold,
        experiment_name=args.name,
        config_record={
            "walkforward": wf_config,
            "label": label_config,
            "model": model_config,
            "tickers": tickers,
            "top_k": args.top_k,
            "range": [str(start), str(end)],
        },
    )

    k = args.top_k
    print(f"\n{'fold':>4} {'test period':^23} {'ll_raw':>7} {'ll_cal':>7} {'ll_prior':>8} "
          f"{'p@' + str(k) + ' model':>9} {'gap':>6} {'pmvol':>6} {'rand':>6}")
    for fr in record["folds"]:
        m = fr["metrics"]
        print(
            f"{fr['fold_id']:>4} {fr['test'][0]} .. {fr['test'][1]} "
            f"{m['log_loss_raw']:>7.4f} {m['log_loss']:>7.4f} {m['log_loss_prior']:>8.4f} "
            f"{m[f'p_at_{k}_model']:>9.1%} {m[f'p_at_{k}_gap']:>6.1%} "
            f"{m[f'p_at_{k}_pm_vol']:>6.1%} {m[f'p_at_{k}_random']:>6.1%}"
        )
    import statistics

    print("\nAcross folds (mean +/- stdev):")
    keys = ["log_loss_raw", "log_loss", "log_loss_prior", "brier_raw", "brier",
            "accuracy", f"p_at_{k}_model", f"p_at_{k}_gap",
            f"p_at_{k}_pm_vol", f"p_at_{k}_random", f"capture_at_{k}_model",
            f"capture_at_{k}_gap", "recall_sustained", "precision_sustained"]
    for key in keys:
        values = [fr["metrics"][key] for fr in record["folds"]]
        mean = statistics.mean(values)
        sd = statistics.stdev(values) if len(values) > 1 else 0.0
        print(f"  {key:>22}: {mean:.4f} +/- {sd:.4f}")
    print(f"\nExperiment record: {record['path']}")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    import numpy as np
    import pandas as pd

    from stockpredictor.data.asof import AsOfData
    from stockpredictor.data.store import BarStore
    from stockpredictor.execution.costs import CostModel
    from stockpredictor.execution.decisions import DecisionConfig, decide
    from stockpredictor.execution.simulator import simulate, summarize
    from stockpredictor.labeling.labeler import LabelConfig, classify_sessions, qualify_windows
    from stockpredictor.labeling.tod import TimeOfDayStats
    from stockpredictor.labeling.windows import LABEL_AS_OF_DELAY, session_windows
    from stockpredictor.sessions import ET, default_calendar
    from stockpredictor.stage1.calibration import DirichletCalibrator
    from stockpredictor.stage1.features import build_stage1_features
    from stockpredictor.stage1.model import (
        CLASS_ORDER,
        Stage1Classifier,
        Stage1ModelConfig,
        proba_frame,
    )
    from stockpredictor.stage2.features import TARGET_COLUMN, build_features
    from stockpredictor.stage2.quantile import QuantileModelConfig, Stage2QuantileModel
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
        holdout_sessions=args.holdout,
    )
    calendar = default_calendar()
    sessions = calendar.sessions_between(start, end)
    session_set = set(sessions)
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
        range_close = pd.Timestamp(calendar.session_close(sessions[-1]))
        as_of = range_close + LABEL_AS_OF_DELAY
        range_open = pd.Timestamp(calendar.session_open(sessions[0])) - pd.Timedelta("6h")
        print(f"Loading extended-hours bars for {len(tickers)} tickers...")
        bars_by_ticker = {
            t: data.get_bars(t, as_of=as_of, lookback=as_of - range_open) for t in tickers
        }
    finally:
        store.close()

    opens = {d: pd.Timestamp(calendar.session_open(d)) for d in sessions}
    closes = {d: pd.Timestamp(calendar.session_close(d)) for d in sessions}

    def rth_only(bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return bars
        day = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
        ts = pd.Series(bars.index.tz_convert(ET), index=bars.index)
        mask = day.isin(session_set) & (ts >= day.map(opens)) & (ts < day.map(closes))
        return bars.loc[mask.to_numpy()]

    rth_by_ticker = {t: rth_only(b) for t, b in bars_by_ticker.items()}

    print("Building Stage 1 pre-market features...")
    s1_features = build_stage1_features(bars_by_ticker, calendar)
    print("Building Stage 2 intraday features...")
    s2_features = build_features(rth_by_ticker, calendar)
    print("Building 15-min windows for labels...")
    window_frames = []
    for ticker, bars in rth_by_ticker.items():
        if bars.empty:
            continue
        et_dates = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
        for day, day_bars in bars.groupby(et_dates):
            window_frames.append(session_windows(day_bars, ticker, day, calendar))
    windows = pd.concat(window_frames, ignore_index=True)

    label_config = LabelConfig()
    s1_config = Stage1ModelConfig()
    s2_config = QuantileModelConfig()
    costs = CostModel()
    decision_config = DecisionConfig()
    class_index = {label: i for i, label in enumerate(CLASS_ORDER)}
    p_cols = [f"p_{c}" for c in CLASS_ORDER]

    def label_rows(day_set, stats):
        segment = windows[windows["date"].isin(day_set)]
        labels = classify_sessions(
            qualify_windows(stats.transform(segment), label_config), calendar, label_config
        )
        return s1_features.merge(labels[["ticker", "date", "label"]], on=["ticker", "date"])

    def run_fold(fold):
        train_days = set(fold.train_days)
        test_days = set(fold.test_days)

        # Stage 1: train, calibrate, score test-day pre-market snapshots.
        stats = TimeOfDayStats().fit(windows[windows["date"].isin(train_days)])
        s1_model = Stage1Classifier(s1_config).fit(label_rows(train_days, stats))
        calib_rows = label_rows(set(fold.calibrate_days), stats)
        calibrator = DirichletCalibrator().fit(
            s1_model.predict_proba(calib_rows)[p_cols].to_numpy(),
            calib_rows["label"].map(class_index).to_numpy(),
        )
        test_snapshots = s1_features[s1_features["date"].isin(test_days)]
        scores = proba_frame(
            calibrator.transform(
                s1_model.predict_proba(test_snapshots)[p_cols].to_numpy()
            ),
            test_snapshots.index,
        )
        snap = test_snapshots[["ticker", "date"]].copy()
        snap["score"] = scores["opportunity_score"]

        # Candidate list per day: score floor plus max count.
        candidates: set[tuple[str, dt.date]] = set()
        for day, day_rows in snap.groupby("date"):
            qualified = day_rows[day_rows["score"] >= args.min_score]
            top = qualified.nlargest(args.max_candidates, "score")
            candidates.update(zip(top["ticker"], top["date"]))

        # Stage 2 on candidates only.
        s2_train = s2_features[s2_features["date"].isin(train_days)]
        s2_model = Stage2QuantileModel(s2_config).fit(s2_train)
        pair_index = pd.MultiIndex.from_frame(s2_features[["ticker", "date"]])
        s2_test = s2_features.loc[pair_index.isin(candidates)]

        if s2_test.empty:
            metrics = summarize(pd.DataFrame(), 0, len(test_days))
            metrics["n_candidates"] = 0
            metrics["candidate_days"] = 0
            return metrics

        preds = s2_model.predict(s2_test)
        sim_frame = s2_test.copy()
        sim_frame["decision"] = decide(s2_test, preds, costs, decision_config)
        trades = simulate(sim_frame, costs)
        metrics = summarize(trades, len(s2_test), len(test_days))
        metrics["n_candidates"] = len(candidates)
        metrics["candidate_days"] = int(snap[snap["score"] >= args.min_score]["date"].nunique())
        print(
            f"  fold {fold.fold_id}: {metrics['n_candidates']} candidates, "
            f"{metrics['n_trades']} trades"
        )
        return metrics

    record = run_walkforward(
        folds,
        run_fold,
        experiment_name=args.name,
        config_record={
            "walkforward": wf_config,
            "label": label_config,
            "stage1": s1_config,
            "stage2": s2_config,
            "costs": costs,
            "decisions": decision_config,
            "min_score": args.min_score,
            "max_candidates": args.max_candidates,
            "tickers": tickers,
            "range": [str(start), str(end)],
        },
    )

    print(f"\n{'fold':>4} {'test period':^23} {'cands':>5} {'trades':>6} {'t/day':>5} "
          f"{'hit':>6} {'gross':>7} {'net':>7} {'total':>7}")
    for fr in record["folds"]:
        m = fr["metrics"]
        hit = f"{m['hit_rate']:.1%}" if m["n_trades"] else "  -"
        gross = f"{m['avg_gross_bps']:.1f}bp" if m["n_trades"] else "  -"
        net = f"{m['avg_net_bps']:.1f}bp" if m["n_trades"] else "  -"
        print(
            f"{fr['fold_id']:>4} {fr['test'][0]} .. {fr['test'][1]} "
            f"{m['n_candidates']:>5} {m['n_trades']:>6} {m['trades_per_day']:>5.1f} "
            f"{hit:>6} {gross:>7} {net:>7} {m['total_net_return']:>7.4f}"
        )
    import statistics

    print("\nAcross folds:")
    totals = [fr["metrics"]["total_net_return"] for fr in record["folds"]]
    n_trades = sum(fr["metrics"]["n_trades"] for fr in record["folds"])
    all_net = [
        fr["metrics"]["avg_net_bps"]
        for fr in record["folds"]
        if fr["metrics"]["n_trades"] > 0
    ]
    print(f"  total trades: {n_trades}")
    print(f"  sum of net returns: {sum(totals):.4f} ({sum(totals) * 1e4:.0f} bps)")
    if all_net:
        print(f"  avg net per trade: {statistics.mean(all_net):.2f} bps")
    print(f"\nExperiment record: {record['path']}")
    return 0


def cmd_ablation(args: argparse.Namespace) -> int:
    """The plan's downstream-value test: the same Stage 2 model evaluated
    on ALL test ticker-days vs ONLY Stage 1 candidates. If Stage 2 is no
    better on candidates, Stage 1 adds compute savings but no signal."""
    import numpy as np
    import pandas as pd

    from stockpredictor.data.asof import AsOfData
    from stockpredictor.data.store import BarStore
    from stockpredictor.execution.costs import CostModel
    from stockpredictor.execution.decisions import DecisionConfig, decide
    from stockpredictor.execution.simulator import simulate, summarize
    from stockpredictor.labeling.labeler import LabelConfig, classify_sessions, qualify_windows
    from stockpredictor.labeling.tod import TimeOfDayStats
    from stockpredictor.labeling.windows import LABEL_AS_OF_DELAY, session_windows
    from stockpredictor.sessions import ET, default_calendar
    from stockpredictor.stage1.calibration import DirichletCalibrator
    from stockpredictor.stage1.features import build_stage1_features
    from stockpredictor.stage1.model import (
        CLASS_ORDER,
        Stage1Classifier,
        Stage1ModelConfig,
        proba_frame,
    )
    from stockpredictor.stage2.features import build_features
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
        calibrate_sessions=args.calibrate,
        embargo_sessions=args.embargo,
        holdout_sessions=args.holdout,
    )
    calendar = default_calendar()
    sessions = calendar.sessions_between(start, end)
    session_set = set(sessions)
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
        range_close = pd.Timestamp(calendar.session_close(sessions[-1]))
        as_of = range_close + LABEL_AS_OF_DELAY
        range_open = pd.Timestamp(calendar.session_open(sessions[0])) - pd.Timedelta("6h")
        print(f"Loading extended-hours bars for {len(tickers)} tickers...")
        bars_by_ticker = {
            t: data.get_bars(t, as_of=as_of, lookback=as_of - range_open) for t in tickers
        }
    finally:
        store.close()

    opens = {d: pd.Timestamp(calendar.session_open(d)) for d in sessions}
    closes = {d: pd.Timestamp(calendar.session_close(d)) for d in sessions}

    def rth_only(bars: pd.DataFrame) -> pd.DataFrame:
        if bars.empty:
            return bars
        day = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
        ts = pd.Series(bars.index.tz_convert(ET), index=bars.index)
        mask = day.isin(session_set) & (ts >= day.map(opens)) & (ts < day.map(closes))
        return bars.loc[mask.to_numpy()]

    rth_by_ticker = {t: rth_only(b) for t, b in bars_by_ticker.items()}

    news_builder = None
    if args.news_db and Path(args.news_db).exists():
        from stockpredictor.data.news_store import NewsStore
        from stockpredictor.news.features import NewsFeatureBuilder

        print(f"Loading and clustering news from {args.news_db}...")
        news_store = NewsStore(args.news_db)
        news_builder = NewsFeatureBuilder(news_store)
    else:
        print("No news store found; running without news features.")

    print("Building Stage 1 pre-market features...")
    s1_features = build_stage1_features(bars_by_ticker, calendar, news=news_builder)
    print("Building Stage 2 intraday features...")
    s2_features = build_features(rth_by_ticker, calendar, news=news_builder)
    print("Building 15-min windows for labels...")
    window_frames = []
    for ticker, bars in rth_by_ticker.items():
        if bars.empty:
            continue
        et_dates = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
        for day, day_bars in bars.groupby(et_dates):
            window_frames.append(session_windows(day_bars, ticker, day, calendar))
    windows = pd.concat(window_frames, ignore_index=True)

    label_config = LabelConfig()
    s1_config = Stage1ModelConfig()
    s2_config = QuantileModelConfig()
    costs = CostModel()
    decision_config = DecisionConfig(
        min_news_articles_2h=args.min_recent_news,
        align_news_sentiment=args.align_sentiment,
    )
    class_index = {label: i for i, label in enumerate(CLASS_ORDER)}
    p_cols = [f"p_{c}" for c in CLASS_ORDER]

    def label_rows(day_set, stats):
        segment = windows[windows["date"].isin(day_set)]
        labels = classify_sessions(
            qualify_windows(stats.transform(segment), label_config), calendar, label_config
        )
        return s1_features.merge(labels[["ticker", "date", "label"]], on=["ticker", "date"])

    def slice_metrics(rows, model, prefix):
        """Forecast + after-cost trading metrics for one evaluation slice."""
        if rows.empty:
            return {f"{prefix}_n_rows": 0}
        preds = model.predict(rows)
        forecast = evaluate(rows, preds)
        sim_frame = rows.copy()
        sim_frame["decision"] = decide(rows, preds, costs, decision_config)
        trades = simulate(sim_frame, costs)
        trading = summarize(trades, len(rows), rows["date"].nunique())
        return {
            f"{prefix}_n_rows": forecast["n_rows"],
            f"{prefix}_direction_accuracy": forecast["direction_accuracy"],
            f"{prefix}_pinball_q50_vs_baseline": forecast["pinball_q50_vs_baseline"],
            f"{prefix}_coverage_80": forecast["coverage_80"],
            f"{prefix}_n_trades": trading["n_trades"],
            f"{prefix}_avg_net_bps": trading["avg_net_bps"],
            f"{prefix}_avg_gross_bps": trading["avg_gross_bps"],
            f"{prefix}_hit_rate": trading["hit_rate"],
        }

    def run_fold(fold):
        train_days = set(fold.train_days)
        test_days = set(fold.test_days)

        stats = TimeOfDayStats().fit(windows[windows["date"].isin(train_days)])
        s1_model = Stage1Classifier(s1_config).fit(label_rows(train_days, stats))
        calib_rows = label_rows(set(fold.calibrate_days), stats)
        calibrator = DirichletCalibrator().fit(
            s1_model.predict_proba(calib_rows)[p_cols].to_numpy(),
            calib_rows["label"].map(class_index).to_numpy(),
        )
        test_snapshots = s1_features[s1_features["date"].isin(test_days)]
        scores = proba_frame(
            calibrator.transform(
                s1_model.predict_proba(test_snapshots)[p_cols].to_numpy()
            ),
            test_snapshots.index,
        )
        snap = test_snapshots[["ticker", "date"]].copy()
        snap["score"] = scores["opportunity_score"]
        candidates: set[tuple[str, dt.date]] = set()
        for day, day_rows in snap.groupby("date"):
            qualified = day_rows[day_rows["score"] >= args.min_score]
            top = qualified.nlargest(args.max_candidates, "score")
            candidates.update(zip(top["ticker"], top["date"]))

        s2_model = Stage2QuantileModel(s2_config).fit(
            s2_features[s2_features["date"].isin(train_days)]
        )
        test_all = s2_features[s2_features["date"].isin(test_days)]
        pair_index = pd.MultiIndex.from_frame(test_all[["ticker", "date"]])
        test_cand = test_all.loc[pair_index.isin(candidates)]

        metrics = {"n_candidates": len(candidates)}
        metrics.update(slice_metrics(test_all, s2_model, "all"))
        metrics.update(slice_metrics(test_cand, s2_model, "cand"))
        print(
            f"  fold {fold.fold_id}: {len(test_all)} universe rows, "
            f"{len(test_cand)} candidate rows"
        )
        return metrics

    record = run_walkforward(
        folds,
        run_fold,
        experiment_name=args.name,
        config_record={
            "walkforward": wf_config,
            "label": label_config,
            "stage1": s1_config,
            "stage2": s2_config,
            "costs": costs,
            "decisions": decision_config,
            "min_score": args.min_score,
            "max_candidates": args.max_candidates,
            "tickers": tickers,
            "news": bool(news_builder),
            "range": [str(start), str(end)],
        },
    )

    print(f"\n{'fold':>4} {'test period':^23} {'dirAcc all':>10} {'cand':>6} "
          f"{'net all':>8} {'cand':>8} {'trades all':>10} {'cand':>5}")
    for fr in record["folds"]:
        m = fr["metrics"]
        if m.get("cand_n_rows", 0) == 0:
            print(f"{fr['fold_id']:>4} {fr['test'][0]} .. {fr['test'][1]}  (no candidates)")
            continue
        net_all = f"{m['all_avg_net_bps']:.1f}" if m["all_n_trades"] else "-"
        net_cand = f"{m['cand_avg_net_bps']:.1f}" if m["cand_n_trades"] else "-"
        print(
            f"{fr['fold_id']:>4} {fr['test'][0]} .. {fr['test'][1]} "
            f"{m['all_direction_accuracy']:>10.1%} {m['cand_direction_accuracy']:>6.1%} "
            f"{net_all:>8} {net_cand:>8} {m['all_n_trades']:>10} {m['cand_n_trades']:>5}"
        )
    import statistics

    print("\nAcross folds (mean +/- stdev):")
    for key in (
        "all_direction_accuracy", "cand_direction_accuracy",
        "all_pinball_q50_vs_baseline", "cand_pinball_q50_vs_baseline",
        "all_coverage_80", "cand_coverage_80",
    ):
        values = [fr["metrics"][key] for fr in record["folds"] if key in fr["metrics"]]
        if not values:
            continue
        clean = [v for v in values if v == v]  # drop NaN
        mean = statistics.mean(clean)
        sd = statistics.stdev(clean) if len(clean) > 1 else 0.0
        print(f"  {key:>30}: {mean:.4f} +/- {sd:.4f}")
    print(f"\nExperiment record: {record['path']}")
    return 0


def cmd_ingest_news(args: argparse.Namespace) -> int:
    from stockpredictor.data.news_store import NewsStore
    from stockpredictor.ingest.polygon import PolygonClient
    from stockpredictor.ingest.polygon_news import ingest_news

    load_dotenv()
    api_key = os.environ.get("POLYGON_API_KEY")
    if not api_key:
        print("Missing POLYGON_API_KEY (see .env.example).", file=sys.stderr)
        return 1

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    print(f"Backfilling market news {start}..{end} (rate-limited, ~1000 articles/call)...")
    store = NewsStore(args.db)
    try:
        inserted = ingest_news(store, PolygonClient(api_key), start, end)
        n, lo, hi = store.coverage()
    finally:
        store.close()
    print(f"Inserted {inserted} new articles into {args.db}")
    print(f"Store now holds {n} articles, {lo} .. {hi}")
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

    p_s1 = sub.add_parser("stage1", help="walk-forward Stage 1 pre-market classifier")
    p_s1.add_argument("--start", required=True)
    p_s1.add_argument("--end", required=True)
    p_s1.add_argument("--train", type=int, default=250)
    p_s1.add_argument("--test", type=int, default=21)
    p_s1.add_argument("--step", type=int, default=21)
    p_s1.add_argument("--calibrate", type=int, default=40)
    p_s1.add_argument("--embargo", type=int, default=1)
    p_s1.add_argument("--holdout", type=int, default=42)
    p_s1.add_argument("--top-k", type=int, default=5)
    p_s1.add_argument("--tickers", default=None)
    p_s1.add_argument("--name", default="stage1-classifier-baseline")
    p_s1.add_argument("--db", default=str(DEFAULT_DB))
    p_s1.set_defaults(func=cmd_stage1)

    p_bt = sub.add_parser("backtest", help="after-cost backtest: Stage 1 -> Stage 2 -> decisions")
    p_bt.add_argument("--start", required=True)
    p_bt.add_argument("--end", required=True)
    p_bt.add_argument("--train", type=int, default=250)
    p_bt.add_argument("--test", type=int, default=21)
    p_bt.add_argument("--step", type=int, default=21)
    p_bt.add_argument("--calibrate", type=int, default=40)
    p_bt.add_argument("--embargo", type=int, default=1)
    p_bt.add_argument("--holdout", type=int, default=42)
    p_bt.add_argument("--max-candidates", type=int, default=5)
    p_bt.add_argument("--min-score", type=float, default=0.5)
    p_bt.add_argument("--tickers", default=None)
    p_bt.add_argument("--name", default="after-cost-backtest")
    p_bt.add_argument("--db", default=str(DEFAULT_DB))
    p_bt.set_defaults(func=cmd_backtest)

    p_ab = sub.add_parser("ablation", help="Stage 2 on candidates vs full universe")
    p_ab.add_argument("--start", required=True)
    p_ab.add_argument("--end", required=True)
    p_ab.add_argument("--train", type=int, default=250)
    p_ab.add_argument("--test", type=int, default=21)
    p_ab.add_argument("--step", type=int, default=21)
    p_ab.add_argument("--calibrate", type=int, default=40)
    p_ab.add_argument("--embargo", type=int, default=1)
    p_ab.add_argument("--holdout", type=int, default=42)
    p_ab.add_argument("--max-candidates", type=int, default=5)
    p_ab.add_argument("--min-score", type=float, default=0.5)
    p_ab.add_argument("--news-db", default=str(Path("data") / "news.sqlite"))
    p_ab.add_argument("--min-recent-news", type=float, default=0.0,
                      help="require this many articles in the last 2h to trade (0 = off)")
    p_ab.add_argument("--align-sentiment", action="store_true",
                      help="longs need positive 2h signed sentiment, shorts negative")
    p_ab.add_argument("--tickers", default=None)
    p_ab.add_argument("--name", default="downstream-value-ablation")
    p_ab.add_argument("--db", default=str(DEFAULT_DB))
    p_ab.set_defaults(func=cmd_ablation)

    p_news = sub.add_parser("ingest-news", help="backfill market news from Polygon")
    p_news.add_argument("--start", required=True, help="ISO date")
    p_news.add_argument("--end", required=True, help="ISO date (exclusive)")
    p_news.add_argument("--db", default=str(Path("data") / "news.sqlite"))
    p_news.set_defaults(func=cmd_ingest_news)

    p_cov = sub.add_parser("coverage", help="show stored date range per ticker")
    p_cov.add_argument("--db", default=str(DEFAULT_DB))
    p_cov.set_defaults(func=cmd_coverage)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
