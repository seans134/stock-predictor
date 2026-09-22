"""Pure detection functions for the TJR model: fractal swings and Fair Value Gaps.

These are the correctness-critical primitives; everything the backtest concludes rests on them
being right, so they are pure functions over plain sequences and unit-tested with hand-built cases.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class FVG:
    """A 3-candle Fair Value Gap.

    kind: "bull" or "bear".
    top/bottom: the price bounds of the gap (top >= bottom).
    index: the index of the middle (impulse) candle.
    """
    kind: str
    top: float
    bottom: float
    index: int


def swing_highs(highs: Sequence[float], wing: int = 2) -> list[int]:
    """Indices of fractal swing highs: a bar strictly higher than `wing` bars on each side."""
    n = len(highs)
    out: list[int] = []
    for i in range(wing, n - wing):
        h = highs[i]
        if all(h > highs[i - k] for k in range(1, wing + 1)) and \
           all(h > highs[i + k] for k in range(1, wing + 1)):
            out.append(i)
    return out


def swing_lows(lows: Sequence[float], wing: int = 2) -> list[int]:
    """Indices of fractal swing lows: a bar strictly lower than `wing` bars on each side."""
    n = len(lows)
    out: list[int] = []
    for i in range(wing, n - wing):
        lo = lows[i]
        if all(lo < lows[i - k] for k in range(1, wing + 1)) and \
           all(lo < lows[i + k] for k in range(1, wing + 1)):
            out.append(i)
    return out


def find_fvgs(highs: Sequence[float], lows: Sequence[float]) -> list[FVG]:
    """Find 3-candle Fair Value Gaps.

    Bullish FVG (middle candle i): lows[i+1] > highs[i-1] -> gap [highs[i-1], lows[i+1]].
    Bearish FVG (middle candle i): highs[i+1] < lows[i-1] -> gap [highs[i+1], lows[i-1]].
    """
    out: list[FVG] = []
    for i in range(1, len(highs) - 1):
        if lows[i + 1] > highs[i - 1]:
            out.append(FVG("bull", top=lows[i + 1], bottom=highs[i - 1], index=i))
        elif highs[i + 1] < lows[i - 1]:
            out.append(FVG("bear", top=lows[i - 1], bottom=highs[i + 1], index=i))
    return out


def last_swing_before(swings: Sequence[int], idx: int) -> int | None:
    """Most recent swing index strictly before position `idx`, else None."""
    prior = [s for s in swings if s < idx]
    return prior[-1] if prior else None
