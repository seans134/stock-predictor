"""Unit tests for the TJR correctness-critical detectors (fractals, FVGs)."""
from stockpredictor.tjr.indicators import (
    FVG,
    find_fvgs,
    last_swing_before,
    swing_highs,
    swing_lows,
)


def test_swing_high_simple_peak():
    assert swing_highs([1, 2, 3, 2, 1], wing=2) == [2]


def test_swing_low_simple_trough():
    assert swing_lows([5, 4, 3, 4, 5], wing=2) == [2]


def test_no_swing_when_flat():
    assert swing_highs([3, 3, 3, 3, 3], wing=2) == []
    assert swing_lows([3, 3, 3, 3, 3], wing=2) == []


def test_multiple_swing_highs():
    assert swing_highs([1, 3, 1, 1, 4, 1, 1], wing=1) == [1, 4]


def test_bullish_fvg_detected():
    fvgs = find_fvgs([5, 20, 25], [3, 8, 10])
    assert fvgs == [FVG("bull", top=10.0, bottom=5.0, index=1)]


def test_bearish_fvg_detected():
    fvgs = find_fvgs([12, 9, 6], [10, 3, 2])
    assert fvgs == [FVG("bear", top=10.0, bottom=6.0, index=1)]


def test_no_fvg_when_bars_overlap():
    assert find_fvgs([10, 11, 10], [8, 9, 8]) == []


def test_last_swing_before():
    assert last_swing_before([2, 5, 9], 7) == 5
    assert last_swing_before([2, 5, 9], 2) is None
    assert last_swing_before([], 4) is None
