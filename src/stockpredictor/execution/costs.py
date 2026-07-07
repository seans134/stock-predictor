"""The cost model: round-trip execution cost in return units.

    round_trip_cost = spread_cost + market_impact + fees (+ borrow, shorts)

Per the plan there is ONE cost model, used identically by the Stage 1
labeler's cost-coverage criterion, the Stage 2 decision rules, and the
backtest, so every stage prices execution the same way.

Free-data placeholders (no historical quotes yet): spread is an assumed
constant, slippage a half-spread per side, impact zero. These match the
labeler's LabelConfig constants; both are replaced together when real
bid/ask data arrives. Every number here is a named starting value to be
selected per walk-forward fold, never from test results. Short borrow
costs are deferred until shorts carry real borrow data.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostModel:
    spread_bps: float = 5.0
    fees_bps: float = 1.0
    impact_bps: float = 0.0  # placeholder until an order-size/depth model exists

    @property
    def round_trip_cost(self) -> float:
        """Round-trip cost in return units. Placeholder slippage is a
        half-spread per side, matching the Stage 1 labeler."""
        slippage_bps = self.spread_bps / 2.0
        total_bps = self.spread_bps + 2.0 * slippage_bps + self.fees_bps + 2.0 * self.impact_bps
        return total_bps / 1e4
