"""Daily z-score reversion backtest for cointegrated pairs.

Decision at a session's close, fill at the NEXT session's close (no look-ahead).
Dollar-neutral legs: a pair trade's return is direction*(r_A - r_B) on one unit
of per-leg capital; both legs pay a round trip, so cost = 2 * round_trip_cost.
The hedge ratio drives the z-signal, not leg sizing. Emits portfolio-schema
trades so the existing risk engine can replay them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.pairs.selection import PairModel, PairsConfig

PAIRS_TRADE_COLUMNS = [
    "ticker", "date", "bar_start", "direction",
    "gross_ret", "net_ret", "q10", "entry_ts", "exit_ts",
]


@dataclass(frozen=True)
class PairTrade:
    pair: str
    entry_date: object
    exit_date: object
    direction: int              # +1 long-A/short-B, -1 short-A/long-B
    entry_z: float
    exit_z: float
    exit_reason: str            # "reversion" | "stop" | "time" | "window_end"
    gross_ret: float
    net_ret: float
    risk_frac_return: float


def spread_zscore(model: PairModel, closes_a: pd.Series, closes_b: pd.Series) -> pd.Series:
    spread = np.log(closes_a) - model.alpha - model.beta * np.log(closes_b)
    return (spread - model.mu) / model.sigma


def simulate_pair(
    model: PairModel, test_closes: pd.DataFrame, cfg: PairsConfig,
    costs: CostModel, session_ts: dict,
) -> list[PairTrade]:
    a, b = model.a, model.b
    if a not in test_closes.columns or b not in test_closes.columns:
        return []
    df = test_closes[[a, b]].dropna()
    if len(df) < 2:
        return []
    z = spread_zscore(model, df[a], df[b])
    dates = list(df.index)
    prices_a = df[a].to_numpy()
    prices_b = df[b].to_numpy()
    cost2 = 2.0 * costs.round_trip_cost
    risk_frac = max(cfg.z_stop - cfg.z_entry, 0.0) * model.sigma

    trades: list[PairTrade] = []
    pos = None              # dict: direction, entry_idx, entry_a, entry_b, entry_z
    pending = None          # (direction, decision_z) awaiting next-session entry fill
    pending_exit = None     # reason string awaiting next-session exit fill
    last = len(dates) - 1

    def close_position(exit_idx: int, reason: str) -> None:
        nonlocal pos
        r_a = prices_a[exit_idx] / pos["entry_a"] - 1.0
        r_b = prices_b[exit_idx] / pos["entry_b"] - 1.0
        gross = pos["direction"] * (r_a - r_b)
        trades.append(PairTrade(
            pair=f"{a}/{b}", entry_date=dates[pos["entry_idx"]], exit_date=dates[exit_idx],
            direction=pos["direction"], entry_z=pos["entry_z"], exit_z=float(z.iloc[exit_idx]),
            exit_reason=reason, gross_ret=float(gross), net_ret=float(gross - cost2),
            risk_frac_return=float(risk_frac),
        ))
        pos = None

    for idx in range(len(dates)):
        zt = float(z.iloc[idx])
        # 1) fill a pending exit at THIS close (the session after the exit signal),
        # symmetric with the entry fill delay below -> no look-ahead on either leg.
        if pending_exit is not None:
            close_position(idx, pending_exit)
            pending_exit = None
            continue
        # 2) fill a pending entry at THIS close (the session after the signal)
        if pending is not None and pos is None:
            direction, entry_z = pending
            pos = {"direction": direction, "entry_idx": idx,
                   "entry_a": prices_a[idx], "entry_b": prices_b[idx], "entry_z": entry_z}
            pending = None
            continue
        # 3) manage an open position: detect an exit condition here, but fill it
        # next session. The last session has no "next" day to fill on, so force
        # flat immediately there instead of queuing a pending exit.
        if pos is not None:
            held = idx - pos["entry_idx"]
            if idx == last:
                close_position(idx, "window_end")
            elif abs(zt) <= cfg.z_exit:
                pending_exit = "reversion"
            elif abs(zt) >= cfg.z_stop:
                pending_exit = "stop"
            elif held >= cfg.max_hold:
                pending_exit = "time"
            continue
        # 4) flat: look for an entry signal, but only if a fill day AND an exit day remain
        if idx < last - 1:
            if zt <= -cfg.z_entry:
                pending = (1, zt)
            elif zt >= cfg.z_entry:
                pending = (-1, zt)

    if pos is not None:  # safety: force flat on the last session
        close_position(last, "window_end")
    return trades


def run_pairs(
    models: list[PairModel], test_closes: pd.DataFrame, cfg: PairsConfig,
    costs: CostModel, session_ts: dict,
) -> pd.DataFrame:
    rows = []
    for model in models:
        for tr in simulate_pair(model, test_closes, cfg, costs, session_ts):
            rows.append({
                "ticker": tr.pair,
                "date": tr.entry_date,
                "bar_start": session_ts[tr.entry_date],
                "direction": LONG if tr.direction == 1 else SHORT,
                "gross_ret": tr.gross_ret,
                "net_ret": tr.net_ret,
                "q10": -tr.risk_frac_return,
                "entry_ts": session_ts[tr.entry_date],
                "exit_ts": session_ts[tr.exit_date],
            })
    return pd.DataFrame(rows, columns=PAIRS_TRADE_COLUMNS)
