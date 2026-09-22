"""Performance statistics from a list of Trades."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_stats(trades, start_equity: float, equity_curve: pd.Series | None) -> dict:
    n = len(trades)
    if n == 0:
        return {"trades": 0}

    pnls = np.array([t.pnl for t in trades])
    rs = np.array([t.r_multiple for t in trades])
    wins = pnls > 0
    losses = pnls < 0
    gross_win = pnls[wins].sum()
    gross_loss = -pnls[losses].sum()

    end_equity = start_equity + pnls.sum()

    # Max drawdown on the realized equity curve (prepend the starting point).
    if equity_curve is not None and len(equity_curve):
        eq = np.concatenate([[start_equity], equity_curve.to_numpy()])
    else:
        eq = np.array([start_equity, end_equity])
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    max_dd = dd.min()

    return {
        "trades": n,
        "wins": int(wins.sum()),
        "losses": int(losses.sum()),
        "win_rate": wins.mean(),
        "avg_win_r": rs[wins].mean() if wins.any() else 0.0,
        "avg_loss_r": rs[losses].mean() if losses.any() else 0.0,
        "expectancy_r": rs.mean(),
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else float("inf"),
        "total_pnl": pnls.sum(),
        "total_return_pct": (end_equity / start_equity - 1.0) * 100,
        "end_equity": end_equity,
        "max_drawdown_pct": max_dd * 100,
        "fvg_share": np.mean([t.used_fvg for t in trades]),
        "target_hits": int(sum(t.exit_reason == "target" for t in trades)),
        "stop_hits": int(sum(t.exit_reason == "stop" for t in trades)),
        "time_exits": int(sum(t.exit_reason == "time" for t in trades)),
    }


def buy_and_hold_return_pct(intraday: pd.DataFrame) -> float:
    first = float(intraday.iloc[0]["Close"])
    last = float(intraday.iloc[-1]["Close"])
    return (last / first - 1.0) * 100
