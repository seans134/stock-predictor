"""Trade simulator: decisions in, after-cost trades out.

Execution follows the Stage 2 timing rules exactly: a signal at bar t
enters at bar t+1's open and exits at bar t+3's close — fwd_ret_15m is
that same entry-to-exit return, so the simulator never assumes a fill at
the price the prediction was made from. Round-trip cost is subtracted
from every trade.

One position per ticker at a time: while a trade is open (bars t+1..t+3),
new signals in that ticker are skipped, not queued. Equal notional per
trade; risk-based sizing and portfolio-level caps belong to the risk
engine milestone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import HOLD, LONG, SHORT
from stockpredictor.stage2.features import TARGET_COLUMN

HORIZON_BARS = 3  # entry bar t+1 through exit at t+3's close

TRADE_COLUMNS = ["ticker", "date", "bar_start", "direction", "gross_ret", "net_ret", "q10"]


def simulate(frame: pd.DataFrame, costs: CostModel) -> pd.DataFrame:
    """Execute non-overlapping trades from decision rows.

    `frame` needs ticker/date/bar_start, a `decision` column, and the
    forward return. Rows whose forward return is undefined (horizon
    crosses the close) never trade regardless of decision.
    """
    cost = costs.round_trip_cost
    trades = []
    if frame.empty:
        return pd.DataFrame(trades, columns=TRADE_COLUMNS)
    for (ticker, day), group in frame.groupby(["ticker", "date"]):
        group = group.sort_values("bar_start")
        blocked_until = None
        for row in group.itertuples():
            decision = row.decision
            fwd = getattr(row, TARGET_COLUMN)
            if decision == HOLD or pd.isna(fwd):
                continue
            if blocked_until is not None and row.bar_start < blocked_until:
                continue
            direction = 1.0 if decision == LONG else -1.0
            gross = direction * fwd
            trades.append(
                {
                    "ticker": ticker,
                    "date": day,
                    "bar_start": row.bar_start,
                    "direction": decision,
                    "gross_ret": gross,
                    "net_ret": gross - cost,
                    # Forecast downside at decision time, for risk sizing.
                    "q10": getattr(row, "q10", float("nan")),
                }
            )
            # Position spans bars t+1..t+3; next signal usable at t+3.
            blocked_until = row.bar_start + pd.Timedelta(minutes=5 * HORIZON_BARS)
    return pd.DataFrame(trades, columns=TRADE_COLUMNS)


def summarize(trades: pd.DataFrame, n_decision_rows: int, n_days: int) -> dict:
    """Flat metrics for the experiment record."""
    if trades.empty:
        return {
            "n_trades": 0,
            "n_long": 0,
            "n_short": 0,
            "trades_per_day": 0.0,
            "hold_fraction": 1.0,
            "hit_rate": float("nan"),
            "avg_net_bps": float("nan"),
            "median_net_bps": float("nan"),
            "total_net_return": 0.0,
            "avg_gross_bps": float("nan"),
        }
    net = trades["net_ret"].to_numpy()
    return {
        "n_trades": int(len(trades)),
        "n_long": int((trades["direction"] == LONG).sum()),
        "n_short": int((trades["direction"] == SHORT).sum()),
        "trades_per_day": float(len(trades) / max(n_days, 1)),
        "hold_fraction": float(1.0 - len(trades) / max(n_decision_rows, 1)),
        "hit_rate": float(np.mean(net > 0)),
        "avg_net_bps": float(np.mean(net) * 1e4),
        "median_net_bps": float(np.median(net) * 1e4),
        "total_net_return": float(np.sum(net)),
        "avg_gross_bps": float(np.mean(trades["gross_ret"]) * 1e4),
    }
