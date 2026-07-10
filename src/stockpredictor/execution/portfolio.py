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

from stockpredictor.execution.simulator import HORIZON_BARS


@dataclass(frozen=True)
class RiskConfig:
    initial_equity: float = 100_000.0
    risk_per_trade: float = 0.005      # fraction of equity at risk per trade
    q10_floor: float = 0.0010          # 10 bps: min assumed downside for sizing
    max_name_fraction: float = 0.20    # notional cap per position vs equity
    max_positions: int = 5             # concurrent positions across the book
    daily_loss_limit: float = 0.02     # fraction of day-start equity


def size_position(equity: float, q10: float, config: RiskConfig) -> float:
    """Notional such that the forecast downside risks risk_per_trade of
    equity, capped at max_name_fraction. Larger forecast risk -> smaller
    position."""
    downside = max(abs(q10), config.q10_floor)
    notional = equity * config.risk_per_trade / downside
    return min(notional, equity * config.max_name_fraction)


def run_portfolio(trades: pd.DataFrame, config: RiskConfig) -> dict:
    """Replay simulator trades chronologically under portfolio controls.

    `trades` needs bar_start (tz-aware), date, ticker, net_ret, and the
    q10 forecast that motivated the trade. Entries blocked by a control
    are skipped (never queued). P&L realizes at exit (entry + 15 min).
    Returns portfolio metrics plus the equity curve.
    """
    if trades.empty:
        return {
            "final_equity": config.initial_equity,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "n_taken": 0,
            "n_skipped_concurrency": 0,
            "n_skipped_daily_halt": 0,
            "equity_curve": pd.Series(dtype=float),
        }

    ordered = trades.sort_values("bar_start")
    equity = config.initial_equity
    open_positions: list[tuple[pd.Timestamp, float, float]] = []  # (exit, notional, ret)
    equity_points: list[tuple[pd.Timestamp, float]] = []
    day_start_equity = equity
    day_pnl = 0.0
    current_day = None
    halted = False
    n_taken = n_skip_conc = n_skip_halt = 0

    def close_due(now: pd.Timestamp | None):
        nonlocal equity, day_pnl
        nonlocal open_positions
        due = [p for p in open_positions if now is None or p[0] <= now]
        open_positions = [p for p in open_positions if not (now is None or p[0] <= now)]
        for exit_ts, notional, ret in sorted(due, key=lambda p: p[0]):
            pnl = notional * ret
            equity += pnl
            day_pnl += pnl
            equity_points.append((exit_ts, equity))

    for row in ordered.itertuples():
        entry_ts = row.bar_start + pd.Timedelta(minutes=5)  # next bar open
        exit_ts = row.bar_start + pd.Timedelta(minutes=5 * (HORIZON_BARS + 1))

        if row.date != current_day:
            close_due(None)  # flatten anything left from the prior day
            current_day = row.date
            day_start_equity = equity
            day_pnl = 0.0
            halted = False
        else:
            close_due(entry_ts)

        if halted or day_pnl < -config.daily_loss_limit * day_start_equity:
            halted = True
            n_skip_halt += 1
            continue
        if len(open_positions) >= config.max_positions:
            n_skip_conc += 1
            continue

        notional = size_position(equity, row.q10, config)
        open_positions.append((exit_ts, notional, row.net_ret))
        n_taken += 1

    close_due(None)
    curve = pd.Series(
        [e for _, e in equity_points],
        index=pd.DatetimeIndex([t for t, _ in equity_points]),
    )
    running_max = curve.cummax() if not curve.empty else curve
    drawdown = ((curve - running_max) / running_max).min() if not curve.empty else 0.0

    return {
        "final_equity": float(equity),
        "total_return": float(equity / config.initial_equity - 1.0),
        "max_drawdown": float(drawdown) if drawdown == drawdown else 0.0,
        "n_taken": n_taken,
        "n_skipped_concurrency": n_skip_conc,
        "n_skipped_daily_halt": n_skip_halt,
        "equity_curve": curve,
    }
