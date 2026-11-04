"""Risk engine v1: position sizing and portfolio-level controls.

The last gate before capital is committed. Stage 2 says what it expects;
this layer decides how much, if any, capital is exposed to that
expectation, and its limits protect capital even when the models are
wrong. Every constant is named in RiskConfig — starting values to be
selected per walk-forward fold, never from test results.

v1 scope (per the plan's execution-and-risk-engine section):
- Risk-based sizing: a trade's downside (|q10|) maps to a fixed fraction
  of current equity at risk, capped per name.
- Max concurrent positions across the book.
- Daily loss limit: once the day's realized P&L breaches it, no new
  entries for that session.
Deferred: sector/correlation concentration, drawdown de-risking tiers,
borrow costs, data-quality kill-switch (needs a live feed to kill).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpredictor.execution.timing import entry_time, exit_time
from stockpredictor.sessions import ET


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 100_000.0
    risk_per_trade: float = 0.005      # fraction of equity at risk per trade
    q10_floor: float = 0.0010          # 10 bps: min assumed downside for sizing
    max_name_fraction: float = 0.20    # notional cap per position vs equity
    max_positions: int = 5             # concurrent positions across the book
    daily_loss_limit: float = 0.02     # fraction of day-start equity
    max_gross_fraction: float = 1.0   # reserved gross notional / realized equity
    fixed_name_fraction: float | None = None  # transparent research sizing

    def __post_init__(self):
        values = (self.initial_equity, self.risk_per_trade, self.q10_floor,
                  self.max_name_fraction, self.daily_loss_limit, self.max_gross_fraction)
        if any(not np.isfinite(v) or v <= 0 for v in values) or self.max_positions < 1:
            raise ValueError("Risk limits must be finite and positive")
        if self.fixed_name_fraction is not None and not (0 < self.fixed_name_fraction <= self.max_name_fraction):
            raise ValueError("Fixed allocation must be positive and within the name cap")


def size_position(equity: float, q10: float, config: RiskConfig) -> float:
    """Notional such that the forecast downside risks risk_per_trade of
    equity, capped at max_name_fraction. Larger forecast risk -> smaller
    position."""
    if config.fixed_name_fraction is not None:
        return equity * config.fixed_name_fraction
    if not np.isfinite(q10):
        raise ValueError("Cannot size a position with unknown downside")
    downside = max(abs(q10), config.q10_floor)
    notional = equity * config.risk_per_trade / downside
    return min(notional, equity * config.max_name_fraction)


def run_portfolio(trades: pd.DataFrame, config: RiskConfig) -> dict:
    """Chronological entry/exit replay with reserved gross capital.

    Equity/drawdown are REALIZED, not intraday marked-to-market. Multi-day
    strategies require marks before these risk statistics are deployable.
    Exit events precede same-timestamp entries. Stable ticker ordering breaks
    ties; it never selects trades using future returns.
    """
    counters = {k: 0 for k in ("n_taken", "n_skipped_concurrency", "n_skipped_daily_halt",
                               "n_skipped_duplicate", "n_skipped_capital")}
    equity = config.initial_equity
    positions = {}
    points, accepted, exposure = [], [], []
    day, day_start, day_pnl, halted = None, equity, 0.0, False
    events = []
    if not trades.empty:
        explicit = {"entry_ts", "exit_ts"}.issubset(trades.columns)
        for i, row in enumerate(trades.to_dict("records")):
            start = pd.Timestamp(row["entry_ts"] if explicit else entry_time(row["bar_start"]))
            end = pd.Timestamp(row["exit_ts"] if explicit else exit_time(row["bar_start"]))
            if pd.isna(start) or pd.isna(end) or start.tzinfo is None or end.tzinfo is None or end <= start:
                raise ValueError("Trades require timezone-aware entry < exit timestamps")
            if "decision_ts" in row and pd.Timestamp(row["decision_ts"]) > start:
                raise ValueError("Entry precedes decision")
            if not np.isfinite(row["net_ret"]):
                raise ValueError("Nonfinite trade return")
            multiplier = float(row.get("gross_multiplier", 2.0 if "/" in row["ticker"] else 1.0))
            if not np.isfinite(multiplier) or multiplier < 1:
                raise ValueError("Invalid gross capital multiplier")
            row.update(entry_ts=start, exit_ts=end, gross_multiplier=multiplier)
            events.extend([(start, 1, row["ticker"], i, row), (end, 0, row["ticker"], i, row)])
    for timestamp, kind, ticker, i, row in sorted(events, key=lambda e: e[:4]):
        event_day = timestamp.tz_convert(ET).date()
        if event_day != day:
            day, day_start, day_pnl, halted = event_day, equity, 0.0, False
        if kind == 0:
            if i not in positions:
                continue
            position = positions.pop(i)
            pnl = position["notional"] * row["net_ret"]
            equity += pnl
            day_pnl += pnl
            points.append((timestamp, equity))
            halted |= day_pnl <= -config.daily_loss_limit * day_start
        else:
            if halted or equity <= 0:
                counters["n_skipped_daily_halt"] += 1
                continue
            if len(positions) >= config.max_positions:
                counters["n_skipped_concurrency"] += 1
                continue
            # Prevent overlap both for single names and constituents of pairs.
            constituents = set(ticker.split("/"))
            if any(constituents.intersection(p["ticker"].split("/")) for p in positions.values()):
                counters["n_skipped_duplicate"] += 1
                continue
            notional = size_position(equity, row["q10"], config)
            reserved = sum(p["notional"] * p["gross_multiplier"] for p in positions.values())
            if reserved + notional * row["gross_multiplier"] > equity * config.max_gross_fraction + 1e-8:
                counters["n_skipped_capital"] += 1
                continue
            position = {**row, "notional": notional}
            positions[i] = position
            accepted.append(position)
            counters["n_taken"] += 1
        reserved = sum(p["notional"] * p["gross_multiplier"] for p in positions.values())
        exposure.append((timestamp, reserved / equity if equity > 0 else float("inf")))
    curve = pd.Series([p[1] for p in points], index=pd.DatetimeIndex([p[0] for p in points]), dtype=float)
    peaks = curve.cummax().clip(lower=config.initial_equity)
    drawdown = float(((curve - peaks) / peaks).min()) if len(curve) else 0.0
    return {
        "final_equity": float(equity),
        "total_return": float(equity / config.initial_equity - 1),
        "max_drawdown": drawdown,
        "drawdown_basis": "realized equity; excludes unrealized intratrade losses",
        **counters,
        "equity_curve": curve,
        "accepted_trades": pd.DataFrame(accepted),
        "peak_gross_fraction": max((p[1] for p in exposure), default=0.0),
    }
