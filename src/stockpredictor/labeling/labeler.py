"""Stage 1 qualifying-opportunity label.

Implements the plan's five window-qualification criteria and the
session-level three-class target (none / opening-only / sustained). Every
constant is named in LabelConfig; the defaults are starting assumptions
that the walk-forward harness will later select per fold — nothing here
is tuned to test results.

Free-data placeholders, to be replaced when historical quotes exist:
- spread is an assumed constant (no bid/ask on the free tier), so the
  execution-quality spread check is against that assumption, and
- estimated market impact is zero (no order size or depth model yet).
Cost-coverage numbers are therefore plumbing-checks, not evidence.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from stockpredictor.sessions import SessionCalendar

LABEL_NONE = "none"
LABEL_OPENING_ONLY = "opening_only"
LABEL_SUSTAINED = "sustained"


@dataclass(frozen=True)
class LabelConfig:
    # Abnormal movement: z_move >= k_move
    k_move: float = 1.5
    # Cost coverage: r_w / round_trip_cost >= m_cost
    m_cost: float = 2.0
    # Liquidity: time-of-day percentile floors
    p_vol: float = 0.70
    p_trades: float = 0.70
    # Execution quality caps (bps)
    s_max_bps: float = 20.0
    i_max_bps: float = 10.0
    # Not-an-artifact floors
    n_min_trades: int = 50
    f_min_active: float = 0.67  # at least 2 of 3 sub-bars traded
    # Placeholder cost model (return units are bps/1e4)
    assumed_spread_bps: float = 5.0
    fees_bps: float = 1.0
    est_impact_bps: float = 0.0  # placeholder until a depth/size model exists
    # Session classification
    opening_minutes: int = 90
    min_late_windows: int = 2
    eps: float = 1e-6

    @property
    def round_trip_cost(self) -> float:
        """Round-trip cost in return units: spread + 2*slippage + fees.
        Placeholder slippage is a half-spread per side."""
        slippage_bps = self.assumed_spread_bps / 2.0
        return (self.assumed_spread_bps + 2.0 * slippage_bps + self.fees_bps) / 1e4


def qualify_windows(windows: pd.DataFrame, config: LabelConfig) -> pd.DataFrame:
    """Apply the five criteria to transformed window rows (must already
    carry z_move / dollar_vol_pctile / trades_pctile). Adds one boolean
    column per criterion plus `qualifies`; NaN inputs never qualify."""
    out = windows.copy()
    cost = config.round_trip_cost

    out["ok_move"] = out["z_move"] >= config.k_move
    out["cost_multiple"] = out["r_w"] / cost
    out["ok_cost"] = out["cost_multiple"] >= config.m_cost
    out["ok_liquidity"] = (out["dollar_vol_pctile"] >= config.p_vol) & (
        out["trades_pctile"] >= config.p_trades
    )
    # Placeholder spread/impact: constant inputs, real formula shape.
    out["ok_execution"] = (config.assumed_spread_bps <= config.s_max_bps) & (
        config.est_impact_bps <= config.i_max_bps
    )
    out["ok_real"] = (out["trade_count"] >= config.n_min_trades) & (
        out["active_subbar_fraction"] >= config.f_min_active
    )
    out["qualifies"] = (
        out["ok_move"]
        & out["ok_cost"]
        & out["ok_liquidity"]
        & out["ok_execution"]
        & out["ok_real"]
        & out["z_move"].notna()
    )
    return out


def classify_sessions(
    qualified: pd.DataFrame,
    calendar: SessionCalendar,
    config: LabelConfig,
) -> pd.DataFrame:
    """Collapse qualified window rows into one labeled row per
    (ticker, date): none / opening_only / sustained.

    - none:         no qualifying windows at all
    - sustained:    at least min_late_windows qualifying windows starting
                    after the opening period
    - opening_only: qualifying activity exists but does not persist beyond
                    the opening period
    """
    rows = []
    for (ticker, day), group in qualified.groupby(["ticker", "date"]):
        open_ts = pd.Timestamp(calendar.session_open(day))
        opening_end = open_ts + pd.Timedelta(minutes=config.opening_minutes)
        hits = group.loc[group["qualifies"]]
        n_total = len(hits)
        n_late = int((hits["window_start"] >= opening_end).sum())
        n_opening = n_total - n_late

        if n_total == 0:
            label = LABEL_NONE
        elif n_late >= config.min_late_windows:
            label = LABEL_SUSTAINED
        else:
            label = LABEL_OPENING_ONLY

        rows.append(
            {
                "ticker": ticker,
                "date": day,
                "label": label,
                "n_qualifying": n_total,
                "n_opening": n_opening,
                "n_late": n_late,
                "n_windows": len(group),
            }
        )
    return pd.DataFrame(
        rows,
        columns=["ticker", "date", "label", "n_qualifying", "n_opening", "n_late", "n_windows"],
    )
