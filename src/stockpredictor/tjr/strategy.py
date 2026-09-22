"""The mechanical TJR liquidity-sweep reversal engine.

Per trading day, within the NY AM killzone, we look for:
  1. a SWEEP of the prior-day high (PDH) or low (PDL) -- price pokes beyond then closes back inside;
  2. a MARKET STRUCTURE SHIFT (MSS) -- a close back through the most recent opposing swing;
  3. an ENTRY into the Fair Value Gap left by the MSS impulse (fallback: market at MSS close).

It emits at most one Signal per session (entry price, stop, target, direction).
Trade *management* (fills, stop/target, P&L) lives in backtest.py; this module only decides setups.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .indicators import find_fvgs, swing_highs, swing_lows


@dataclass
class Signal:
    direction: int          # +1 long, -1 short
    entry: float            # limit entry price (FVG edge) or market price
    stop: float
    target: float
    entry_time: pd.Timestamp
    used_fvg: bool
    target_mode: str


@dataclass
class StrategyConfig:
    killzone_start: str = "09:30"
    killzone_end: str = "12:00"
    sweep_lookback: int = 6      # bars allowed between the poke and the close-back-inside
    wing: int = 2                # fractal wing size for swing detection
    rr: float = 2.0              # fixed R multiple for target_mode="fixed_rr"
    stop_buffer_frac: float = 0.05  # extra stop beyond sweep extreme, as frac of sweep-to-entry
    require_fvg: bool = False    # if True, skip setups with no qualifying FVG
    target_mode: str = "fixed_rr"  # "fixed_rr" or "liquidity"


def _in_killzone(ts: pd.Timestamp, cfg: StrategyConfig) -> bool:
    t = ts.strftime("%H:%M")
    return cfg.killzone_start <= t <= cfg.killzone_end


def find_session_signal(session: pd.DataFrame, pdh: float, pdl: float,
                        cfg: StrategyConfig) -> Signal | None:
    """Scan one session's 5m bars for a single TJR setup. `session` is time-ordered, ET-indexed.

    Returns the first valid Signal or None. Bars are processed causally: at each candidate MSS bar
    we only use information available up to and including that bar.
    """
    kz = session[[_in_killzone(ts, cfg) for ts in session.index]]
    if len(kz) < 2 * cfg.wing + 3:
        return None

    highs = kz["High"].to_numpy()
    lows = kz["Low"].to_numpy()
    closes = kz["Close"].to_numpy()
    times = kz.index

    # Collect every fresh sweep of PDL (bullish) or PDH (bearish); try each in turn, take the
    # first that completes into a full setup. A "fresh" sweep is a poke beyond the level whose
    # immediately-preceding bar was inside it (so we don't re-fire on every bar of one excursion).
    candidates = []  # (direction, poke_idx, extreme)
    for i in range(len(kz)):
        below_prev = i == 0 or lows[i - 1] >= pdl
        above_prev = i == 0 or highs[i - 1] <= pdh
        if lows[i] < pdl and below_prev:
            candidates.append((+1, i, lows[i]))
        elif highs[i] > pdh and above_prev:
            candidates.append((-1, i, highs[i]))

    for direction, poke_idx, extreme in candidates:
        sig = _build_from_sweep(highs, lows, closes, times, direction, poke_idx, extreme,
                                pdh, pdl, cfg)
        if sig is not None:
            return sig
    return None


def _build_from_sweep(highs, lows, closes, times, direction, poke_idx, extreme,
                      pdh, pdl, cfg: StrategyConfig) -> Signal | None:
    """Given a detected sweep, require reclaim -> MSS -> entry, and size stop/target."""
    n = len(highs)
    # Reclaim: a close back inside the swept level within `sweep_lookback` bars.
    reclaim_idx = None
    for j in range(poke_idx, min(poke_idx + cfg.sweep_lookback + 1, n)):
        if direction == +1 and closes[j] > pdl:
            reclaim_idx = j
            break
        if direction == -1 and closes[j] < pdh:
            reclaim_idx = j
            break
    if reclaim_idx is None:
        return None

    # MSS: after the reclaim, a close breaks the most recent opposing swing formed since the sweep.
    mss_idx = None
    ref_swing = None
    for k in range(reclaim_idx + 1, n):
        recompute = (swing_highs(highs[:k].tolist(), cfg.wing) if direction == +1
                     else swing_lows(lows[:k].tolist(), cfg.wing))
        recompute = [s for s in recompute if poke_idx <= s < k]
        if not recompute:
            continue
        ref = recompute[-1]
        level = highs[ref] if direction == +1 else lows[ref]
        if direction == +1 and closes[k] > level:
            mss_idx, ref_swing = k, ref
            break
        if direction == -1 and closes[k] < level:
            mss_idx, ref_swing = k, ref
            break
    if mss_idx is None:
        return None

    # Entry: FVG left by the MSS impulse (fallback: market at MSS close).
    imp_h = highs[ref_swing: mss_idx + 1].tolist()
    imp_l = lows[ref_swing: mss_idx + 1].tolist()
    fvgs = [f for f in find_fvgs(imp_h, imp_l)
            if (f.kind == "bull" and direction == +1) or (f.kind == "bear" and direction == -1)]
    used_fvg = bool(fvgs)
    if used_fvg:
        f = fvgs[-1]
        entry = f.bottom if direction == +1 else f.top
    else:
        if cfg.require_fvg:
            return None
        entry = closes[mss_idx]

    risk = abs(entry - extreme)
    if risk <= 0:
        return None
    buffer = cfg.stop_buffer_frac * risk
    if direction == +1:
        stop = extreme - buffer
        target = entry + cfg.rr * (entry - stop) if cfg.target_mode == "fixed_rr" else pdh
    else:
        stop = extreme + buffer
        target = entry - cfg.rr * (stop - entry) if cfg.target_mode == "fixed_rr" else pdl

    if (direction == +1 and target <= entry) or (direction == -1 and target >= entry):
        return None

    return Signal(direction=direction, entry=float(entry), stop=float(stop),
                  target=float(target), entry_time=times[mss_idx],
                  used_fvg=used_fvg, target_mode=cfg.target_mode)
