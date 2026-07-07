"""Per-stock, per-time-of-day reference statistics.

Everything the qualification criteria compare against — the movement
median/IQR and the dollar-volume and trade-count percentile ranks — is
normalized per ticker and per 15-minute time-of-day slot, and is estimated
from an explicit training date range only. Fold isolation lives here: the
caller decides the fit range; transform never updates state.

A (ticker, slot) pair with fewer than min_obs training observations gets
NaN statistics, so its windows can never qualify. Missing data is explicit
and conservative, never zero-filled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Minimum training observations per (ticker, slot) before its statistics
# are trusted. Named constant per the minimum-sample-requirements rule.
DEFAULT_MIN_OBS = 60


@dataclass
class _SlotStats:
    median_rw: float
    iqr_rw: float
    dollar_vol_sorted: np.ndarray
    trades_sorted: np.ndarray


class TimeOfDayStats:
    def __init__(self, min_obs: int = DEFAULT_MIN_OBS):
        self._min_obs = min_obs
        self._stats: dict[tuple[str, object], _SlotStats] = {}

    def fit(self, windows: pd.DataFrame) -> "TimeOfDayStats":
        """Fit from training-range window rows (build_windows output)."""
        self._stats.clear()
        valid = windows.dropna(subset=["r_w"])
        for (ticker, slot), group in valid.groupby(["ticker", "slot"]):
            if len(group) < self._min_obs:
                continue
            r_w = group["r_w"].to_numpy()
            q75, q25 = np.percentile(r_w, [75, 25])
            self._stats[(ticker, slot)] = _SlotStats(
                median_rw=float(np.median(r_w)),
                iqr_rw=float(q75 - q25),
                dollar_vol_sorted=np.sort(group["dollar_vol"].to_numpy()),
                trades_sorted=np.sort(group["trade_count"].to_numpy()),
            )
        return self

    def transform(self, windows: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
        """Add z_move, dollar_vol_pctile, trades_pctile columns.

        Rows whose (ticker, slot) has no trusted training statistics get
        NaN in all three, which downstream disqualifies.
        """
        out = windows.copy()
        z = np.full(len(out), np.nan)
        dv_pct = np.full(len(out), np.nan)
        tc_pct = np.full(len(out), np.nan)

        for i, row in enumerate(out.itertuples(index=False)):
            stats = self._stats.get((row.ticker, row.slot))
            if stats is None or pd.isna(row.r_w):
                continue
            z[i] = (row.r_w - stats.median_rw) / (stats.iqr_rw + eps)
            n = len(stats.dollar_vol_sorted)
            dv_pct[i] = np.searchsorted(stats.dollar_vol_sorted, row.dollar_vol, side="right") / n
            tc_pct[i] = np.searchsorted(stats.trades_sorted, row.trade_count, side="right") / n

        out["z_move"] = z
        out["dollar_vol_pctile"] = dv_pct
        out["trades_pctile"] = tc_pct
        return out

    @property
    def n_slots(self) -> int:
        return len(self._stats)
