"""Event-driven backtest over 5m bars: turn Signals into fills, exits, and an equity curve.

Rules:
- Intraday only: a position is opened and must close the same session (time-stop at session end).
- Limit entry fills when a later bar's range touches the entry price.
- If a bar spans both stop and target, assume the STOP fills first (conservative).
- Position size risks 1% of current equity; size = risk$ / (entry-stop distance).
- One-way transaction cost `cost_per_unit` applied to entry and to exit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .strategy import Signal, StrategyConfig, find_session_signal


@dataclass
class Trade:
    date: pd.Timestamp
    direction: int
    entry: float
    stop: float
    target: float
    exit: float
    exit_reason: str        # "target" | "stop" | "time"
    r_multiple: float
    pnl: float
    used_fvg: bool


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.Series | None = None
    start_equity: float = 10_000.0


def _prior_day_levels(daily: pd.DataFrame, day: pd.Timestamp):
    """PDH/PDL = high/low of the most recent daily bar strictly before `day`."""
    if day.tzinfo is not None:
        day = day.tz_localize(None)  # daily index is tz-naive
    prev = daily[daily.index < day]
    if prev.empty:
        return None
    row = prev.iloc[-1]
    return float(row["High"]), float(row["Low"])


def _simulate_trade(sig: Signal, session: pd.DataFrame, cost_per_unit: float,
                    equity: float, risk_frac: float):
    """From the entry bar forward, fill the limit then resolve stop/target/time-stop."""
    after = session[session.index > sig.entry_time]
    if after.empty:
        return None

    filled = False
    for ts, bar in after.iterrows():
        lo, hi = bar["Low"], bar["High"]
        if not filled:
            if lo <= sig.entry <= hi:  # limit touched
                filled = True
                fill_from = ts
            else:
                continue
        if filled:
            # On the fill bar and every later bar, check exits.
            hit_stop = lo <= sig.stop <= hi
            hit_tgt = lo <= sig.target <= hi
            if hit_stop and hit_tgt:
                exit_px, reason = sig.stop, "stop"   # conservative
            elif hit_stop:
                exit_px, reason = sig.stop, "stop"
            elif hit_tgt:
                exit_px, reason = sig.target, "target"
            else:
                continue
            return _close(sig, exit_px, reason, cost_per_unit, equity, risk_frac)

    if filled:  # time-stop at last session bar
        exit_px = float(after.iloc[-1]["Close"])
        return _close(sig, exit_px, "time", cost_per_unit, equity, risk_frac)
    return None


def _close(sig: Signal, exit_px: float, reason: str, cost_per_unit: float,
           equity: float, risk_frac: float):
    stop_dist = abs(sig.entry - sig.stop)
    if stop_dist <= 0:
        return None
    risk_dollars = risk_frac * equity
    size = risk_dollars / stop_dist
    gross = sig.direction * (exit_px - sig.entry) * size
    costs = cost_per_unit * size * 2  # entry + exit
    pnl = gross - costs
    r_multiple = pnl / risk_dollars if risk_dollars else 0.0
    return exit_px, reason, pnl, r_multiple, size


def run_backtest(intraday: pd.DataFrame, daily: pd.DataFrame, cfg: StrategyConfig,
                 cost_per_unit: float, start_equity: float = 10_000.0,
                 risk_frac: float = 0.01) -> BacktestResult:
    res = BacktestResult(start_equity=start_equity)
    equity = start_equity
    curve_times, curve_vals = [], []

    # Group by ET calendar day once (normalize() is computed a single time), instead of
    # re-scanning the whole intraday frame for every day.
    for day, session in intraday.groupby(intraday.index.normalize(), sort=True):
        if session.empty:
            continue
        levels = _prior_day_levels(daily, day)
        if levels is None:
            continue
        pdh, pdl = levels
        sig = find_session_signal(session, pdh, pdl, cfg)
        if sig is None:
            continue
        outcome = _simulate_trade(sig, session, cost_per_unit, equity, risk_frac)
        if outcome is None:
            continue
        exit_px, reason, pnl, r_multiple, _ = outcome
        equity += pnl
        res.trades.append(Trade(
            date=day, direction=sig.direction, entry=sig.entry, stop=sig.stop,
            target=sig.target, exit=exit_px, exit_reason=reason,
            r_multiple=r_multiple, pnl=pnl, used_fvg=sig.used_fvg))
        curve_times.append(day)
        curve_vals.append(equity)

    if curve_vals:
        res.equity_curve = pd.Series(curve_vals, index=pd.DatetimeIndex(curve_times))
    return res
