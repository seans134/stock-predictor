"""Comparable daily reports, with explicitly descriptive dependence intervals."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.execution.portfolio import RiskConfig, run_portfolio
from stockpredictor.sessions import ET


def daily_portfolio(trades, days, risk: RiskConfig):
    replay = run_portfolio(trades, risk)
    curve = replay["equity_curve"]
    values = curve.groupby(curve.index.tz_convert(ET).date).last() if len(curve) else pd.Series(dtype=float)
    equity = values.reindex(days).ffill().fillna(risk.initial_equity)
    previous = equity.shift(1).fillna(risk.initial_equity)
    daily = pd.DataFrame({"equity": equity, "return": equity / previous - 1.0})
    daily.index.name = "date"
    metrics = {k: v for k, v in replay.items() if k not in ("equity_curve", "accepted_trades")}
    accepted = replay["accepted_trades"]
    metrics["avg_gross_bps"] = float(accepted.gross_ret.mean()*1e4) if len(accepted) else None
    metrics["avg_net_bps"] = float(accepted.net_ret.mean()*1e4) if len(accepted) else None
    metrics["active_dates"] = int(accepted.entry_ts.dt.tz_convert(ET).dt.date.nunique()) if len(accepted) else 0
    metrics["daily_mean_bps"] = float(daily["return"].mean()*1e4)
    metrics["worst_day"] = float(daily["return"].min())
    metrics["daily_5pct_quantile"] = float(daily["return"].quantile(.05))
    metrics["traded_notional_over_initial_equity"] = float(accepted.notional.sum()*2/risk.initial_equity) if len(accepted) else 0.0
    metrics["ticker_pnl"] = (accepted.assign(pnl=accepted.notional*accepted.net_ret)
                             .groupby("ticker").pnl.sum().to_dict()) if len(accepted) else {}
    return daily, metrics, accepted


def block_interval(values, block=5, repetitions=5000):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    rng = np.random.default_rng(20261020)
    starts = rng.integers(0, len(values), (repetitions, int(np.ceil(len(values)/block))))
    indices = ((starts[..., None] + np.arange(block)) % len(values)).reshape(repetitions, -1)[:, :len(values)]
    return (np.quantile(values[indices].mean(axis=1), [.025, .975])*1e4).tolist()


def compare(primary, control, days, risk, cost_grid=(0, 2, 5, 11, 20)):
    daily, metrics, accepted = daily_portfolio(primary, days, risk)
    benchmark, bm, bm_accepted = daily_portfolio(control, days, risk)
    paired = daily.join(benchmark.add_prefix("control_"))
    paired["incremental_return"] = paired["return"] - paired.control_return
    scenarios = {}
    for cost in cost_grid:
        priced = primary.copy()
        priced["net_ret"] = priced.gross_ret - cost / 1e4
        _, summary, _ = daily_portfolio(priced, days, risk)
        scenarios[str(cost)] = summary
    ci = block_interval(paired["return"])
    metrics["daily_mean_95pct_interval_bps"] = ci
    metrics["incremental_daily_mean_95pct_interval_bps"] = block_interval(paired.incremental_return)
    return {
        "primary": metrics, "control": bm, "cost_scenarios_bps": scenarios,
        "decision": "fail" if metrics["avg_gross_bps"] is not None and metrics["avg_gross_bps"] <= 0 else "inconclusive",
        "decision_basis": "Development-only signal triage; negative gross expectancy fails this implementation. Otherwise insufficient confirmatory evidence.",
        "inference": "Five-session circular block bootstrap of full daily returns; descriptive and post-selection, not multiplicity-adjusted. No annualized Sharpe or promotion claim.",
    }, paired, accepted, bm_accepted
