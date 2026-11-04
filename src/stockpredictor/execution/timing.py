"""Conservative five-minute execution clock; all timestamps are bar starts.

Completed bars arrive five seconds after close. The first eligible scheduled
opening reference is therefore t+2, not t+1. This is a bar-data approximation,
not a promise of a market fill at the opening print.
"""
from __future__ import annotations

import pandas as pd

from stockpredictor.data.availability import BAR_INGEST_LATENCY

BAR_DURATION = pd.Timedelta(minutes=5)
HORIZON_BARS = 3
ENTRY_OFFSET = BAR_DURATION * 2
EXIT_OFFSET = ENTRY_OFFSET + BAR_DURATION * HORIZON_BARS
EXECUTION_VERSION = "causal-bars-v2"


def decision_time(bar_start):
    return bar_start + BAR_DURATION + BAR_INGEST_LATENCY


def entry_time(bar_start):
    return bar_start + ENTRY_OFFSET


def exit_time(bar_start):
    return bar_start + EXIT_OFFSET


def barrier_exit(bar, direction: int, stop: float, target: float | None):
    """Return (reference_price, reason, at_open), or None.

    Gap stops use the opening price. Resting target limits receive only their
    limit price, even after a favorable gap. Ambiguous intrabar paths resolve
    stop-first; capital stays reserved until bar end for intrabar fills.
    """
    opening, high, low = float(bar["open"]), float(bar["high"]), float(bar["low"])
    if direction * (opening - stop) <= 0:
        return opening, "stop", True
    if target is not None and direction * (opening - target) >= 0:
        return target, "target", True
    if (low <= stop if direction == 1 else high >= stop):
        return stop, "stop", False
    if target is not None and (high >= target if direction == 1 else low <= target):
        return target, "target", False
    return None
