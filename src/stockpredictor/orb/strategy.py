"""Opening-Range Breakout signal logic (pure, causal, no I/O).

The opening range is the high/low of the first `range_minutes` of the regular
session. A later bar whose CLOSE clears a boundary by at least
`min_break_atr x atr`, on volume >= `vol_mult x` the running median, fires a
signal (long above the high, short below the low). ATR is a prior-session scale
supplied by the caller, so nothing here can look ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

_SESSION_OPEN = pd.Timedelta(hours=9, minutes=30)  # NYSE regular open (ET)


@dataclass(frozen=True)
class ORBConfig:
    range_minutes: int = 15
    atr_lookback_sessions: int = 1
    stop_atr_mult: float = 1.0
    rr: float = 1.0
    min_break_atr: float = 0.10
    vol_mult: float = 1.0
    stop_mode: str = "atr"  # "atr" (default) or "boundary"


@dataclass(frozen=True)
class ORBSignal:
    direction: int          # +1 long, -1 short
    signal_bar: pd.Timestamp
    range_high: float
    range_low: float
    atr: float
    entry: float            # confirming-bar close (decision-time reference)
    stop: float
    target: float


def _session_open(session: pd.DataFrame) -> pd.Timestamp:
    return session.index[0].normalize() + _SESSION_OPEN


def opening_range(session: pd.DataFrame, cfg: ORBConfig) -> tuple[float, float] | None:
    if session.empty:
        return None
    open_ts = _session_open(session)
    end = open_ts + pd.Timedelta(minutes=cfg.range_minutes)
    win = session.loc[(session.index >= open_ts) & (session.index < end)]
    if win.empty:
        return None
    if win["high"].isna().any() or win["low"].isna().any():
        return None
    return float(win["high"].max()), float(win["low"].min())


def prior_session_atr(prior_bars: pd.DataFrame | None, cfg: ORBConfig) -> float | None:
    if prior_bars is None or prior_bars.empty:
        return None
    prev_close = prior_bars["close"].shift(1)
    hl = prior_bars["high"] - prior_bars["low"]
    hc = (prior_bars["high"] - prev_close).abs()
    lc = (prior_bars["low"] - prev_close).abs()
    tr = pd.concat([hl, hc, lc], axis=1).max(axis=1).dropna()
    if tr.empty:
        return None
    return float(tr.mean())


def stop_and_target(
    entry: float, direction: int, range_high: float, range_low: float,
    atr: float, cfg: ORBConfig,
) -> tuple[float, float, float]:
    if cfg.stop_mode == "boundary":
        stop = range_low if direction == 1 else range_high
    else:
        stop = entry - cfg.stop_atr_mult * atr if direction == 1 else entry + cfg.stop_atr_mult * atr
    risk = abs(entry - stop)
    target = entry + direction * cfg.rr * risk
    return float(stop), float(target), float(risk)


def find_orb_signal(
    session: pd.DataFrame, atr: float | None, cfg: ORBConfig
) -> ORBSignal | None:
    if session.empty or atr is None or atr <= 0:
        return None
    rng = opening_range(session, cfg)
    if rng is None:
        return None
    range_high, range_low = rng
    end = _session_open(session) + pd.Timedelta(minutes=cfg.range_minutes)
    after = session.loc[session.index >= end]
    if after.empty:
        return None

    min_break = cfg.min_break_atr * atr
    for ts, bar in after.iterrows():
        if pd.isna(bar["volume"]) or pd.isna(bar["close"]):
            continue
        seen_vol = session.loc[session.index <= ts, "volume"]
        if bar["volume"] < cfg.vol_mult * seen_vol.median():
            continue
        close = float(bar["close"])
        if close >= range_high + min_break:
            direction = 1
        elif close <= range_low - min_break:
            direction = -1
        else:
            continue
        stop, target, risk = stop_and_target(
            close, direction, range_high, range_low, atr, cfg
        )
        if risk <= 0:
            continue
        return ORBSignal(
            direction=direction, signal_bar=ts, range_high=range_high,
            range_low=range_low, atr=float(atr), entry=close, stop=stop, target=target,
        )
    return None
