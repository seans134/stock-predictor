# ORB Strategy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Opening-Range Breakout (ORB) trading strategy, integrated with Stage 1 as an optional watchlist filter, and back-test it under the existing walk-forward discipline.

**Architecture:** A new pure `orb/` package (signal logic + event-driven per-session simulator) emits trades in the existing portfolio schema. A backward-compatible extension to `execution/portfolio.py` replays ORB's variable-holding-period trades through the existing risk engine. A new `orb` CLI subcommand runs two arms — pure ORB and Stage 1-gated ORB — across the same folds as `cmd_backtest`, leaving the 42-session holdout untouched.

**Tech Stack:** Python 3.11+, pandas 2.x, numpy, LightGBM (Stage 1, reused), pytest.

## Global Constraints

- Python `>=3.11`; pandas `>=2.0`; numpy `>=1.26` (from `pyproject.toml`).
- All timestamps timezone-aware US/Eastern at the feature layer; `from stockpredictor.sessions import ET`.
- Bar columns are lower-case (`open, high, low, close, volume, trade_count, vwap`) as stored by `BarStore`.
- Direction constants come from `stockpredictor.execution.decisions`: `LONG = "long"`, `SHORT = "short"`.
- Cost is one round-trip figure in return units: `CostModel().round_trip_cost` (currently 11e-4). Never hard-code costs.
- Missing data never trades (return None / skip); values are never zero-filled.
- No look-ahead: ATR uses only prior sessions; the breakout scan and opening range use only bars up to the confirming bar; entry fills at the next bar's open.
- Tests are flat in `tests/`, named `test_*.py`; build bar frames inline with tz-aware ET timestamps; assert with `pytest.approx`.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## File Structure

- **Create** `src/stockpredictor/orb/__init__.py` — package marker.
- **Create** `src/stockpredictor/orb/strategy.py` — `ORBConfig`, `ORBSignal`, `opening_range`, `prior_session_atr`, `stop_and_target`, `find_orb_signal`. Pure, no I/O, no training.
- **Create** `src/stockpredictor/orb/backtest.py` — `ORBTrade`, `ORB_TRADE_COLUMNS`, `simulate_orb_session`, `run_orb`. Event-driven fills; emits portfolio-schema trades.
- **Modify** `src/stockpredictor/execution/portfolio.py` — `run_portfolio` honors optional `entry_ts`/`exit_ts`.
- **Modify** `src/stockpredictor/__main__.py` — add `cmd_orb` + `orb` subparser.
- **Create** `tests/test_orb_strategy.py`, `tests/test_orb_backtest.py`, `tests/test_portfolio_variable_exit.py`, `tests/test_orb_integration.py`.

---

### Task 1: ORB signal logic (`orb/strategy.py`)

**Files:**
- Create: `src/stockpredictor/orb/__init__.py`
- Create: `src/stockpredictor/orb/strategy.py`
- Test: `tests/test_orb_strategy.py`

**Interfaces:**
- Consumes: `LONG`, `SHORT` from `stockpredictor.execution.decisions`.
- Produces:
  - `ORBConfig(range_minutes=15, atr_lookback_sessions=1, stop_atr_mult=1.0, rr=1.0, min_break_atr=0.10, vol_mult=1.0, stop_mode="atr")` (frozen dataclass)
  - `ORBSignal(direction: int, signal_bar: pd.Timestamp, range_high: float, range_low: float, atr: float, entry: float, stop: float, target: float)` (frozen dataclass)
  - `opening_range(session: pd.DataFrame, cfg: ORBConfig) -> tuple[float, float] | None`
  - `prior_session_atr(prior_bars: pd.DataFrame | None, cfg: ORBConfig) -> float | None`
  - `stop_and_target(entry: float, direction: int, range_high: float, range_low: float, atr: float, cfg: ORBConfig) -> tuple[float, float, float]` returning `(stop, target, risk)`
  - `find_orb_signal(session: pd.DataFrame, atr: float | None, cfg: ORBConfig) -> ORBSignal | None`
  - `session` is one ticker-day of RTH 5-min bars, ET-`DatetimeIndex`, chronological, lower-case OHLCV columns.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_orb_strategy.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.orb.strategy import (
    ORBConfig,
    find_orb_signal,
    opening_range,
    prior_session_atr,
    stop_and_target,
)
from stockpredictor.sessions import ET

DAY = dt.date(2024, 3, 18)


def _bars(rows):
    """rows: list of (hh, mm, open, high, low, close, volume)."""
    idx, data = [], []
    for hh, mm, o, h, lo, c, v in rows:
        idx.append(pd.Timestamp(DAY.year, DAY.month, DAY.day, hh, mm, tz=ET))
        data.append({"open": o, "high": h, "low": lo, "close": c, "volume": v})
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


def _flat_session(open_ts_hh=9, open_ts_mm=30, n=12, price=100.0, vol=1000):
    rows = []
    for k in range(n):
        t = pd.Timestamp(DAY.year, DAY.month, DAY.day, 9, 30, tz=ET) + pd.Timedelta(minutes=5 * k)
        rows.append((t.hour, t.minute, price, price, price, price, vol))
    return _bars(rows)


def test_opening_range_first_15_minutes():
    # three 5-min bars 09:30/09:35/09:40 form the 15-min range; later bars ignored.
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 102, 98, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 105, 95, 100, 1000),  # outside the 15-min window
    ])
    assert opening_range(session, ORBConfig(range_minutes=15)) == (102.0, 98.0)


def test_opening_range_empty():
    assert opening_range(_bars([]), ORBConfig()) is None


def test_prior_session_atr_true_range():
    prior = _bars([
        (9, 30, 100, 102, 99, 101, 1000),   # TR = 3 (no prev close)
        (9, 35, 101, 103, 100, 102, 1000),  # TR = max(3, |103-101|, |100-101|) = 3
    ])
    atr = prior_session_atr(prior, ORBConfig())
    assert atr == pytest.approx(3.0)


def test_prior_session_atr_none_when_empty():
    assert prior_session_atr(None, ORBConfig()) is None
    assert prior_session_atr(_bars([]), ORBConfig()) is None


def test_long_breakout_confirmed():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),   # range hi=101, lo=99
        (9, 45, 100, 103, 100, 102.5, 5000),  # close 102.5 > 101 + 0.1*ATR, high volume
    ])
    sig = find_orb_signal(session, atr=2.0, cfg=ORBConfig(range_minutes=15, min_break_atr=0.1, vol_mult=1.0))
    assert sig is not None
    assert sig.direction == 1
    assert sig.signal_bar == pd.Timestamp(DAY.year, DAY.month, DAY.day, 9, 45, tz=ET)


def test_short_breakout_confirmed():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 100, 96, 96.5, 5000),   # close 96.5 < 99 - 0.1*ATR
    ])
    sig = find_orb_signal(session, atr=2.0, cfg=ORBConfig(range_minutes=15))
    assert sig is not None and sig.direction == -1


def test_break_too_small_rejected():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 101.1, 100, 101.05, 5000),  # only 0.05 over hi; needs 0.1*ATR=0.2
    ])
    assert find_orb_signal(session, atr=2.0, cfg=ORBConfig(min_break_atr=0.1)) is None


def test_low_volume_break_rejected():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 100),  # strong break, but volume far below median
    ])
    assert find_orb_signal(session, atr=2.0, cfg=ORBConfig(vol_mult=1.0)) is None


def test_no_break_returns_none():
    assert find_orb_signal(_flat_session(), atr=2.0, cfg=ORBConfig()) is None


def test_causality_truncating_future_bars():
    session = _bars([
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 5000),  # signal forms here
        (9, 50, 102, 110, 90, 95, 9000),      # wild future bar
    ])
    cfg = ORBConfig()
    full = find_orb_signal(session, atr=2.0, cfg=cfg)
    truncated = find_orb_signal(session.iloc[:4], atr=2.0, cfg=cfg)
    assert full is not None and truncated is not None
    assert full.signal_bar == truncated.signal_bar
    assert full.direction == truncated.direction


def test_stop_and_target_atr_long():
    stop, target, risk = stop_and_target(
        entry=102.0, direction=1, range_high=101, range_low=99, atr=2.0,
        cfg=ORBConfig(stop_mode="atr", stop_atr_mult=1.0, rr=2.0),
    )
    assert stop == pytest.approx(100.0)      # 102 - 1*2
    assert risk == pytest.approx(2.0)
    assert target == pytest.approx(106.0)    # 102 + 2*2


def test_stop_and_target_boundary_long():
    stop, target, risk = stop_and_target(
        entry=102.0, direction=1, range_high=101, range_low=99, atr=2.0,
        cfg=ORBConfig(stop_mode="boundary", rr=1.0),
    )
    assert stop == pytest.approx(99.0)       # opposite range side
    assert target == pytest.approx(105.0)    # 102 + 1*(102-99)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_orb_strategy.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockpredictor.orb'`.

- [ ] **Step 3: Create the package marker**

```python
# src/stockpredictor/orb/__init__.py
```
(empty file)

- [ ] **Step 4: Implement `orb/strategy.py`**

```python
# src/stockpredictor/orb/strategy.py
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

from stockpredictor.execution.decisions import LONG, SHORT  # noqa: F401  (re-exported meaning)

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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_orb_strategy.py -v`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add src/stockpredictor/orb/__init__.py src/stockpredictor/orb/strategy.py tests/test_orb_strategy.py
git commit -m "$(printf 'Add ORB signal logic: opening range, prior-session ATR, confirmed breakout\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: ORB event-driven backtest (`orb/backtest.py`)

**Files:**
- Create: `src/stockpredictor/orb/backtest.py`
- Test: `tests/test_orb_backtest.py`

**Interfaces:**
- Consumes: `ORBConfig`, `find_orb_signal`, `prior_session_atr`, `stop_and_target` (Task 1); `CostModel` from `stockpredictor.execution.costs`; `LONG`, `SHORT` from `stockpredictor.execution.decisions`; `ET` from `stockpredictor.sessions`.
- Produces:
  - `ORB_TRADE_COLUMNS = ["ticker", "date", "bar_start", "direction", "gross_ret", "net_ret", "q10", "entry_ts", "exit_ts"]`
  - `ORBTrade` (frozen dataclass) with fields: `ticker, date, signal_bar, entry_bar, exit_ts, direction, entry, stop, target, exit, exit_reason, gross_ret, net_ret, risk_frac_return`
  - `simulate_orb_session(session: pd.DataFrame, atr: float, cfg: ORBConfig, costs: CostModel, ticker: str, day) -> ORBTrade | None`
  - `run_orb(rth_by_ticker: dict[str, pd.DataFrame], cfg: ORBConfig, costs: CostModel, universe_filter: set[tuple[str, object]] | None = None) -> pd.DataFrame` (returns the `ORB_TRADE_COLUMNS` frame)

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_orb_backtest.py
import datetime as dt

import pandas as pd
import pytest

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.orb.backtest import ORB_TRADE_COLUMNS, run_orb, simulate_orb_session
from stockpredictor.orb.strategy import ORBConfig
from stockpredictor.sessions import ET

DAY = dt.date(2024, 3, 18)
PRIOR = dt.date(2024, 3, 15)


def _session(day, rows):
    idx, data = [], []
    for hh, mm, o, h, lo, c, v in rows:
        idx.append(pd.Timestamp(day.year, day.month, day.day, hh, mm, tz=ET))
        data.append({"open": o, "high": h, "low": lo, "close": c, "volume": v})
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


# A long breakout at 09:45, then a bar that reaches the target.
LONG_DAY_ROWS = [
    (9, 30, 100, 101, 99, 100, 1000),
    (9, 35, 100, 101, 99, 100, 1000),
    (9, 40, 100, 101, 99, 100, 1000),   # range hi=101, lo=99
    (9, 45, 100, 103, 100, 102.5, 5000),  # confirm long (close>101+0.2)
    (9, 50, 103, 108, 102, 107, 5000),    # entry at 09:50 open=103; target hits
    (9, 55, 107, 108, 106, 107, 5000),
]


def test_entry_at_next_bar_open_and_target():
    session = _session(DAY, LONG_DAY_ROWS)
    # ATR=2, stop_atr_mult=1 -> entry 103, stop 101, risk 2, rr=1 -> target 105.
    trade = simulate_orb_session(session, atr=2.0, cfg=ORBConfig(rr=1.0), costs=CostModel(),
                                 ticker="AAA", day=DAY)
    assert trade is not None
    assert trade.entry == pytest.approx(103.0)          # 09:50 open
    assert trade.entry_bar == pd.Timestamp(2024, 3, 18, 9, 50, tz=ET)
    assert trade.exit_reason == "target"
    assert trade.exit == pytest.approx(105.0)
    assert trade.gross_ret == pytest.approx((105.0 - 103.0) / 103.0)
    assert trade.net_ret == pytest.approx(trade.gross_ret - CostModel().round_trip_cost)
    assert trade.q10_hint() == pytest.approx(-trade.risk_frac_return)  # see note below


def test_stop_fills_first_when_bar_spans_both():
    rows = [
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 5000),   # confirm long
        (9, 50, 103, 106, 100, 104, 5000),     # entry 103; bar spans stop 101 AND target 105
    ]
    trade = simulate_orb_session(_session(DAY, rows), atr=2.0, cfg=ORBConfig(rr=1.0),
                                 costs=CostModel(), ticker="AAA", day=DAY)
    assert trade.exit_reason == "stop"
    assert trade.exit == pytest.approx(101.0)


def test_time_stop_at_eod():
    rows = [
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 103, 100, 102.5, 5000),   # confirm long, entry next bar
        (9, 50, 103, 103.5, 102, 103, 5000),   # neither stop(101) nor target(105) hit
        (9, 55, 103, 103.5, 102, 103.2, 5000),
    ]
    trade = simulate_orb_session(_session(DAY, rows), atr=2.0, cfg=ORBConfig(rr=1.0),
                                 costs=CostModel(), ticker="AAA", day=DAY)
    assert trade.exit_reason == "time"
    assert trade.exit == pytest.approx(103.2)              # last bar close
    assert trade.exit_ts == pd.Timestamp(2024, 3, 18, 9, 55, tz=ET)


def test_short_pnl_sign():
    rows = [
        (9, 30, 100, 101, 99, 100, 1000),
        (9, 35, 100, 101, 99, 100, 1000),
        (9, 40, 100, 101, 99, 100, 1000),
        (9, 45, 100, 100, 96, 96.5, 5000),    # confirm short, entry next bar
        (9, 50, 96, 96, 90, 92, 5000),        # entry 96; target = 96 - (96-98)=... see ATR
    ]
    # ATR=2, stop_atr_mult=1 -> short entry 96, stop 98, risk 2, rr=1 -> target 94.
    trade = simulate_orb_session(_session(DAY, rows), atr=2.0, cfg=ORBConfig(rr=1.0),
                                 costs=CostModel(), ticker="AAA", day=DAY)
    assert trade.direction == -1
    assert trade.exit_reason == "target"
    assert trade.exit == pytest.approx(94.0)
    assert trade.gross_ret == pytest.approx((-1) * (94.0 - 96.0) / 96.0)  # positive


def test_no_signal_returns_none():
    flat = _session(DAY, [(9, 30 + 5 * k // 1, 100, 100, 100, 100, 1000) for k in range(6)])
    # build a genuinely flat session explicitly:
    flat = _session(DAY, [
        (9, 30, 100, 100, 100, 100, 1000),
        (9, 35, 100, 100, 100, 100, 1000),
        (9, 40, 100, 100, 100, 100, 1000),
        (9, 45, 100, 100, 100, 100, 1000),
    ])
    assert simulate_orb_session(flat, atr=2.0, cfg=ORBConfig(), costs=CostModel(),
                                ticker="AAA", day=DAY) is None


def _prior_flat(day):
    return _session(day, [
        (9, 30, 100, 102, 99, 101, 1000),
        (9, 35, 101, 103, 100, 102, 1000),
    ])


def test_run_orb_schema_and_columns():
    bars = pd.concat([_prior_flat(PRIOR), _session(DAY, LONG_DAY_ROWS)])
    trades = run_orb({"AAA": bars}, cfg=ORBConfig(rr=1.0), costs=CostModel())
    assert list(trades.columns) == ORB_TRADE_COLUMNS
    assert len(trades) == 1
    row = trades.iloc[0]
    assert row["ticker"] == "AAA"
    assert row["direction"] == LONG
    assert row["q10"] < 0                       # -risk_frac_return
    assert row["entry_ts"] == pd.Timestamp(2024, 3, 18, 9, 50, tz=ET)


def test_run_orb_universe_filter_gates_names():
    bars = pd.concat([_prior_flat(PRIOR), _session(DAY, LONG_DAY_ROWS)])
    both = {"AAA": bars, "BBB": bars}
    pure = run_orb(both, cfg=ORBConfig(rr=1.0), costs=CostModel())
    gated = run_orb(both, cfg=ORBConfig(rr=1.0), costs=CostModel(),
                    universe_filter={("AAA", DAY)})
    assert set(pure["ticker"]) == {"AAA", "BBB"}
    assert set(gated["ticker"]) == {"AAA"}


def test_run_orb_skips_first_session_no_prior_atr():
    # Only one session available -> no prior ATR -> no trades.
    trades = run_orb({"AAA": _session(DAY, LONG_DAY_ROWS)}, cfg=ORBConfig(), costs=CostModel())
    assert trades.empty
```

> **Note on `q10_hint()`:** add a tiny convenience method on `ORBTrade` returning `-self.risk_frac_return` so the test reads clearly; the emitted frame column `q10` uses the same value.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_orb_backtest.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockpredictor.orb.backtest'`.

- [ ] **Step 3: Implement `orb/backtest.py`**

```python
# src/stockpredictor/orb/backtest.py
"""Event-driven ORB backtest: signal -> next-bar-open fill -> stop/target/EOD.

Emits trades in the portfolio schema so the existing risk engine can replay
them. Conservative fill: a bar spanning both stop and target resolves as a
stop. Round-trip cost is subtracted from every trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.orb.strategy import ORBConfig, find_orb_signal, prior_session_atr, stop_and_target
from stockpredictor.sessions import ET

ORB_TRADE_COLUMNS = [
    "ticker", "date", "bar_start", "direction",
    "gross_ret", "net_ret", "q10", "entry_ts", "exit_ts",
]


@dataclass(frozen=True)
class ORBTrade:
    ticker: str
    date: object
    signal_bar: pd.Timestamp
    entry_bar: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: int
    entry: float
    stop: float
    target: float
    exit: float
    exit_reason: str          # "target" | "stop" | "time"
    gross_ret: float
    net_ret: float
    risk_frac_return: float

    def q10_hint(self) -> float:
        return -self.risk_frac_return


def simulate_orb_session(
    session: pd.DataFrame, atr: float, cfg: ORBConfig, costs: CostModel,
    ticker: str, day,
) -> ORBTrade | None:
    sig = find_orb_signal(session, atr, cfg)
    if sig is None:
        return None
    after = session.loc[session.index > sig.signal_bar]
    if after.empty:
        return None
    entry_bar = after.index[0]
    entry = float(after.iloc[0]["open"])
    stop, target, risk = stop_and_target(
        entry, sig.direction, sig.range_high, sig.range_low, atr, cfg
    )
    if risk <= 0:
        return None

    forward = session.loc[session.index >= entry_bar]
    exit_px = exit_reason = exit_ts = None
    for ts, bar in forward.iterrows():
        lo, hi = float(bar["low"]), float(bar["high"])
        hit_stop = lo <= stop <= hi
        hit_tgt = lo <= target <= hi
        if hit_stop:                       # stop-first when both
            exit_px, exit_reason, exit_ts = stop, "stop", ts
            break
        if hit_tgt:
            exit_px, exit_reason, exit_ts = target, "target", ts
            break
    if exit_px is None:                    # EOD time-stop
        exit_px = float(forward.iloc[-1]["close"])
        exit_reason, exit_ts = "time", forward.index[-1]

    gross = sig.direction * (exit_px - entry) / entry
    net = gross - costs.round_trip_cost
    return ORBTrade(
        ticker=ticker, date=day, signal_bar=sig.signal_bar, entry_bar=entry_bar,
        exit_ts=exit_ts, direction=sig.direction, entry=entry, stop=stop,
        target=target, exit=exit_px, exit_reason=exit_reason,
        gross_ret=float(gross), net_ret=float(net), risk_frac_return=risk / entry,
    )


def run_orb(
    rth_by_ticker: dict[str, pd.DataFrame], cfg: ORBConfig, costs: CostModel,
    universe_filter: set[tuple[str, object]] | None = None,
) -> pd.DataFrame:
    rows = []
    for ticker, bars in rth_by_ticker.items():
        if bars is None or bars.empty:
            continue
        et = bars.copy()
        et.index = et.index.tz_convert(ET)
        et = et.sort_index()
        by_day = {d: g for d, g in et.groupby(et.index.date)}
        days = sorted(by_day)
        for i, day in enumerate(days):
            if universe_filter is not None and (ticker, day) not in universe_filter:
                continue
            if i < cfg.atr_lookback_sessions:
                continue
            prior = pd.concat(
                [by_day[days[j]] for j in range(i - cfg.atr_lookback_sessions, i)]
            )
            atr = prior_session_atr(prior, cfg)
            if atr is None:
                continue
            trade = simulate_orb_session(by_day[day], atr, cfg, costs, ticker, day)
            if trade is None:
                continue
            rows.append({
                "ticker": ticker,
                "date": day,
                "bar_start": trade.signal_bar,
                "direction": LONG if trade.direction == 1 else SHORT,
                "gross_ret": trade.gross_ret,
                "net_ret": trade.net_ret,
                "q10": -trade.risk_frac_return,
                "entry_ts": trade.entry_bar,
                "exit_ts": trade.exit_ts,
            })
    return pd.DataFrame(rows, columns=ORB_TRADE_COLUMNS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_orb_backtest.py -v`
Expected: PASS (all). If `test_short_pnl_sign` reveals an off-by-one in the short target, re-check `stop_and_target`: for a short, `target = entry + (-1)*rr*risk`, so entry 96, risk 2, rr 1 -> target 94. The bar low 90 <= 94 <= high 96, so target fills.

- [ ] **Step 5: Commit**

```bash
git add src/stockpredictor/orb/backtest.py tests/test_orb_backtest.py
git commit -m "$(printf 'Add ORB event-driven backtest with next-bar fills and EOD time-stop\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 3: Variable-holding-period support in the risk engine (`execution/portfolio.py`)

**Files:**
- Modify: `src/stockpredictor/execution/portfolio.py:88-90` (the per-row entry/exit timing inside `run_portfolio`)
- Test: `tests/test_portfolio_variable_exit.py`

**Interfaces:**
- Consumes: existing `RiskConfig`, `run_portfolio`, `size_position`; `HORIZON_BARS` from `stockpredictor.execution.simulator`.
- Produces: `run_portfolio` now reads optional `entry_ts`/`exit_ts` columns; behavior unchanged when they are absent.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_portfolio_variable_exit.py
import datetime as dt

import pandas as pd
import pytest

from stockpredictor.execution.portfolio import RiskConfig, run_portfolio

DAY = dt.date(2024, 3, 18)
ET = "America/New_York"


def _ts(hh, mm, day=DAY):
    return pd.Timestamp(day.year, day.month, day.day, hh, mm, tz=ET)


def _orb_trade(signal_hh, signal_mm, entry, exit_, net_ret, q10=-0.02, ticker="AAA", day=DAY):
    return {
        "ticker": ticker, "date": day, "direction": "long",
        "bar_start": _ts(signal_hh, signal_mm, day),
        "gross_ret": net_ret, "net_ret": net_ret, "q10": q10,
        "entry_ts": entry, "exit_ts": exit_,
    }


def test_variable_exit_realizes_pnl_at_exit_ts():
    # Position held 09:50 -> 10:40 (much longer than the 15-min default).
    trades = pd.DataFrame([
        _orb_trade(9, 45, _ts(9, 50), _ts(10, 40), 0.02),
    ])
    result = run_portfolio(trades, RiskConfig(max_positions=10))
    assert result["n_taken"] == 1
    # Curve point lands at the ORB exit time, not at a fixed 15-min horizon.
    assert result["equity_curve"].index[-1] == _ts(10, 40)


def test_variable_exit_concurrency_counts_full_span():
    # Two ORB trades overlap 09:50-10:40 and 10:00-10:30; cap 1 blocks the second.
    trades = pd.DataFrame([
        _orb_trade(9, 45, _ts(9, 50), _ts(10, 40), 0.01, ticker="AAA"),
        _orb_trade(9, 55, _ts(10, 0), _ts(10, 30), 0.01, ticker="BBB"),
    ])
    result = run_portfolio(trades, RiskConfig(max_positions=1))
    assert result["n_taken"] == 1
    assert result["n_skipped_concurrency"] == 1


def test_fallback_without_ts_columns_unchanged():
    # No entry_ts/exit_ts -> the existing fixed 15-min behavior (regression pin).
    trades = pd.DataFrame([{
        "ticker": "AAA", "date": DAY, "direction": "long",
        "bar_start": _ts(10, 0), "gross_ret": 0.01, "net_ret": 0.01, "q10": -0.01,
    }])
    result = run_portfolio(trades, RiskConfig(max_positions=10))
    assert result["n_taken"] == 1
    # Fixed horizon exit = bar_start + 5*(HORIZON_BARS+1) = 10:00 + 20min = 10:20.
    assert result["equity_curve"].index[-1] == _ts(10, 20)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_portfolio_variable_exit.py -v`
Expected: FAIL — the first two fail because exit is computed at a fixed 15-min horizon (curve lands at 10:10 / wrong concurrency); the fallback test passes.

- [ ] **Step 3: Edit `run_portfolio` to honor optional timestamps**

Replace the per-row timing lines (currently lines 88-90):

```python
    for row in ordered.itertuples():
        entry_ts = row.bar_start + pd.Timedelta(minutes=5)  # next bar open
        exit_ts = row.bar_start + pd.Timedelta(minutes=5 * (HORIZON_BARS + 1))
```

with:

```python
    has_explicit_ts = {"entry_ts", "exit_ts"}.issubset(trades.columns)
    for row in ordered.itertuples():
        if has_explicit_ts:
            entry_ts = row.entry_ts
            exit_ts = row.exit_ts
        else:
            entry_ts = row.bar_start + pd.Timedelta(minutes=5)  # next bar open
            exit_ts = row.bar_start + pd.Timedelta(minutes=5 * (HORIZON_BARS + 1))
```

Also update the docstring of `run_portfolio` to note: "If the frame carries
`entry_ts`/`exit_ts` columns (e.g. ORB's variable holding periods) they drive the
entry/exit timing; otherwise the fixed 15-minute Stage 2 horizon is assumed."

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_portfolio_variable_exit.py tests/test_portfolio.py -v`
Expected: PASS (new file) and PASS (existing `test_portfolio.py`, unchanged).

- [ ] **Step 5: Commit**

```bash
git add src/stockpredictor/execution/portfolio.py tests/test_portfolio_variable_exit.py
git commit -m "$(printf 'Support variable-holding-period trades in run_portfolio (optional entry/exit ts)\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 4: `orb` CLI subcommand — two arms across walk-forward folds (`__main__.py`)

**Files:**
- Modify: `src/stockpredictor/__main__.py` (add `cmd_orb` after `cmd_backtest`; register the `orb` subparser in `main`)
- Test: `tests/test_orb_integration.py`

**Interfaces:**
- Consumes: everything from Tasks 1-3, plus the existing Stage 1 pipeline pieces already imported by `cmd_backtest` (`build_stage1_features`, `Stage1Classifier`, `DirichletCalibrator`, `proba_frame`, `CLASS_ORDER`, `TimeOfDayStats`, `classify_sessions`, `qualify_windows`, `session_windows`, `LABEL_AS_OF_DELAY`), the walk-forward harness (`generate_folds`, `run_walkforward`, `WalkForwardConfig`), `RiskConfig`, `run_portfolio`, `summarize`.
- Produces: a `stockpredictor orb ...` command that writes one experiment JSON and `trades_<name>_pure.csv` / `trades_<name>_gated.csv`, and prints per-fold + across-fold tables for both arms plus each arm's portfolio metrics.

The integration test exercises the two-arm logic directly (no DB needed); the full CLI is verified by a manual smoke run at the end.

- [ ] **Step 1: Write the failing integration test**

```python
# tests/test_orb_integration.py
"""Two-arm ORB: the gated arm is a strict subset of the pure arm's ticker-days,
and both replay through the risk engine. Uses synthetic bars only (no DB)."""

import datetime as dt

import pandas as pd

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.portfolio import RiskConfig, run_portfolio
from stockpredictor.orb.backtest import run_orb
from stockpredictor.orb.strategy import ORBConfig
from stockpredictor.sessions import ET

PRIOR = dt.date(2024, 3, 15)
DAY = dt.date(2024, 3, 18)


def _session(day, rows):
    idx, data = [], []
    for hh, mm, o, h, lo, c, v in rows:
        idx.append(pd.Timestamp(day.year, day.month, day.day, hh, mm, tz=ET))
        data.append({"open": o, "high": h, "low": lo, "close": c, "volume": v})
    return pd.DataFrame(data, index=pd.DatetimeIndex(idx))


_PRIOR = [(9, 30, 100, 102, 99, 101, 1000), (9, 35, 101, 103, 100, 102, 1000)]
_BREAKOUT = [
    (9, 30, 100, 101, 99, 100, 1000),
    (9, 35, 100, 101, 99, 100, 1000),
    (9, 40, 100, 101, 99, 100, 1000),
    (9, 45, 100, 103, 100, 102.5, 5000),
    (9, 50, 103, 106, 102, 105, 5000),
]


def test_gated_is_subset_of_pure_and_portfolio_runs():
    bars = pd.concat([_session(PRIOR, _PRIOR), _session(DAY, _BREAKOUT)])
    universe = {"AAA": bars, "BBB": bars, "CCC": bars}
    cfg = ORBConfig(rr=1.0)

    pure = run_orb(universe, cfg, CostModel())
    watchlist = {("AAA", DAY), ("BBB", DAY)}  # Stage 1 flagged only two names
    gated = run_orb(universe, cfg, CostModel(), universe_filter=watchlist)

    pure_pairs = set(zip(pure["ticker"], pure["date"]))
    gated_pairs = set(zip(gated["ticker"], gated["date"]))
    assert gated_pairs <= pure_pairs
    assert gated_pairs == {("AAA", DAY), ("BBB", DAY)}

    # Both arms replay through the extended risk engine without error.
    for arm in (pure, gated):
        result = run_portfolio(arm, RiskConfig(max_positions=10))
        assert result["n_taken"] >= 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_orb_integration.py -v`
Expected: FAIL initially only if imports are missing; since Tasks 1-3 exist, this test should actually PASS already (it exercises `run_orb`/`run_portfolio`, which are done). Confirm it PASSES — it is the regression guard for the two-arm contract that `cmd_orb` depends on. If it fails, fix `run_orb` before touching the CLI.

- [ ] **Step 3: Add `cmd_orb` to `__main__.py`**

Insert this function after `cmd_backtest` (before `cmd_ablation`). It mirrors
`cmd_backtest`'s loading/Stage-1 blocks and adds the two ORB arms.

```python
def cmd_orb(args: argparse.Namespace) -> int:
    import pandas as pd

    from stockpredictor.data.asof import AsOfData
    from stockpredictor.data.store import BarStore
    from stockpredictor.execution.costs import CostModel
    from stockpredictor.execution.portfolio import RiskConfig, run_portfolio
    from stockpredictor.execution.simulator import summarize
    from stockpredictor.labeling.labeler import LabelConfig, classify_sessions, qualify_windows
    from stockpredictor.labeling.tod import TimeOfDayStats
    from stockpredictor.labeling.windows import LABEL_AS_OF_DELAY, session_windows
    from stockpredictor.orb.backtest import run_orb
    from stockpredictor.orb.strategy import ORBConfig
    from stockpredictor.sessions import ET, default_calendar
    from stockpredictor.stage1.calibration import DirichletCalibrator
    from stockpredictor.stage1.features import build_stage1_features
    from stockpredictor.stage1.model import (
        CLASS_ORDER, Stage1Classifier, Stage1ModelConfig, proba_frame,
    )
    from stockpredictor.validation.folds import WalkForwardConfig, generate_folds
    from stockpredictor.validation.harness import run_walkforward

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    wf_config = WalkForwardConfig(
        train_sessions=args.train, test_sessions=args.test, step_sessions=args.step,
        calibrate_sessions=args.calibrate, embargo_sessions=args.embargo,
        holdout_sessions=args.holdout,
    )
    calendar = default_calendar()
    sessions = calendar.sessions_between(start, end)
    session_set = set(sessions)
    folds = generate_folds(sessions, wf_config)
    if not folds:
        print("No folds fit in the requested range.", file=sys.stderr)
        return 1
    print(f"{len(folds)} folds over {len(sessions)} sessions ({args.holdout} held out)")

    store = BarStore(args.db)
    try:
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if args.tickers else store.tickers()
        )
        data = AsOfData(store)
        range_close = pd.Timestamp(calendar.session_close(sessions[-1]))
        as_of = range_close + LABEL_AS_OF_DELAY
        range_open = pd.Timestamp(calendar.session_open(sessions[0])) - pd.Timedelta("6h")
        print(f"Loading extended-hours bars for {len(tickers)} tickers...")
        bars_by_ticker = {
            t: data.get_bars(t, as_of=as_of, lookback=as_of - range_open) for t in tickers
        }
    finally:
        store.close()

    opens = {d: pd.Timestamp(calendar.session_open(d)) for d in sessions}
    closes = {d: pd.Timestamp(calendar.session_close(d)) for d in sessions}

    def rth_only(bars):
        if bars.empty:
            return bars
        day = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
        ts = pd.Series(bars.index.tz_convert(ET), index=bars.index)
        mask = day.isin(session_set) & (ts >= day.map(opens)) & (ts < day.map(closes))
        return bars.loc[mask.to_numpy()]

    rth_by_ticker = {t: rth_only(b) for t, b in bars_by_ticker.items()}

    print("Building Stage 1 pre-market features (for the gated arm)...")
    s1_features = build_stage1_features(bars_by_ticker, calendar)
    print("Building 15-min windows for Stage 1 labels...")
    window_frames = []
    for ticker, bars in rth_by_ticker.items():
        if bars.empty:
            continue
        et_dates = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
        for day, day_bars in bars.groupby(et_dates):
            window_frames.append(session_windows(day_bars, ticker, day, calendar))
    windows = pd.concat(window_frames, ignore_index=True)

    label_config = LabelConfig()
    s1_config = Stage1ModelConfig()
    costs = CostModel()
    orb_config = ORBConfig(
        range_minutes=args.range_minutes, stop_atr_mult=args.stop_atr_mult, rr=args.rr,
        min_break_atr=args.min_break_atr, vol_mult=args.vol_mult, stop_mode=args.stop_mode,
    )
    class_index = {label: i for i, label in enumerate(CLASS_ORDER)}
    p_cols = [f"p_{c}" for c in CLASS_ORDER]
    trades_pure_all, trades_gated_all = [], []

    def label_rows(day_set, stats):
        segment = windows[windows["date"].isin(day_set)]
        labels = classify_sessions(
            qualify_windows(stats.transform(segment), label_config), calendar, label_config
        )
        return s1_features.merge(labels[["ticker", "date", "label"]], on=["ticker", "date"])

    def run_fold(fold):
        train_days = set(fold.train_days)
        test_days = set(fold.test_days)

        # Stage 1 watchlist for the gated arm.
        stats = TimeOfDayStats().fit(windows[windows["date"].isin(train_days)])
        s1_model = Stage1Classifier(s1_config).fit(label_rows(train_days, stats))
        calib_rows = label_rows(set(fold.calibrate_days), stats)
        calibrator = DirichletCalibrator().fit(
            s1_model.predict_proba(calib_rows)[p_cols].to_numpy(),
            calib_rows["label"].map(class_index).to_numpy(),
        )
        test_snapshots = s1_features[s1_features["date"].isin(test_days)]
        scores = proba_frame(
            calibrator.transform(s1_model.predict_proba(test_snapshots)[p_cols].to_numpy()),
            test_snapshots.index,
        )
        snap = test_snapshots[["ticker", "date"]].copy()
        snap["score"] = scores["opportunity_score"]
        watchlist = set(
            zip(*snap[snap["score"] >= args.min_score][["ticker", "date"]].to_numpy().T)
        ) if not snap.empty else set()

        # ORB needs the prior session for ATR, so include the day before each test
        # day's bars per ticker. Slice each ticker to test days + one prior session.
        test_rth = {}
        for t, bars in rth_by_ticker.items():
            if bars.empty:
                continue
            et_dates = pd.Series(bars.index.tz_convert(ET).date, index=bars.index)
            keep_days = set(test_days)
            # add the immediately-preceding stored session per test day
            all_days = sorted(set(et_dates))
            idx_of = {d: i for i, d in enumerate(all_days)}
            for d in list(test_days):
                if d in idx_of and idx_of[d] > 0:
                    keep_days.add(all_days[idx_of[d] - 1])
            test_rth[t] = bars.loc[et_dates.isin(keep_days).to_numpy()]

        pure = run_orb(test_rth, orb_config, costs)
        gated = run_orb(test_rth, orb_config, costs, universe_filter=watchlist)
        # keep only trades whose DATE is a test day (prior session was ATR context only)
        pure = pure[pure["date"].isin(test_days)]
        gated = gated[gated["date"].isin(test_days)]
        trades_pure_all.append(pure)
        trades_gated_all.append(gated)

        m_pure = summarize(pure, len(pure), len(test_days))
        m_gated = summarize(gated, len(gated), len(test_days))
        print(f"  fold {fold.fold_id}: pure {m_pure['n_trades']} trades, "
              f"gated {m_gated['n_trades']} trades, watchlist {len(watchlist)}")
        return {
            "watchlist_size": len(watchlist),
            **{f"pure_{k}": v for k, v in m_pure.items()},
            **{f"gated_{k}": v for k, v in m_gated.items()},
        }

    record = run_walkforward(
        folds, run_fold, experiment_name=args.name,
        config_record={
            "walkforward": wf_config, "label": label_config, "stage1": s1_config,
            "costs": costs, "orb": orb_config, "min_score": args.min_score,
            "tickers": tickers, "range": [str(start), str(end)],
        },
    )

    def stitch_and_report(chunks, label):
        stitched = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        if not stitched.empty:
            path = Path("data") / f"trades_{args.name}_{label}.csv"
            stitched.sort_values("bar_start").to_csv(path, index=False)
            print(f"  {label} trades -> {path}")
        portfolio = run_portfolio(stitched, RiskConfig())
        print(f"  {label}: final ${portfolio['final_equity']:,.0f}  "
              f"return {portfolio['total_return']:+.2%}  maxDD {portfolio['max_drawdown']:.2%}  "
              f"taken {portfolio['n_taken']}")
        return portfolio

    print(f"\n{'fold':>4} {'test period':^23} {'wl':>4} {'pure n':>7} {'pure net':>9} "
          f"{'gated n':>8} {'gated net':>10}")
    for fr in record["folds"]:
        m = fr["metrics"]
        pn = f"{m['pure_avg_net_bps']:.1f}bp" if m["pure_n_trades"] else "  -"
        gn = f"{m['gated_avg_net_bps']:.1f}bp" if m["gated_n_trades"] else "  -"
        print(f"{fr['fold_id']:>4} {fr['test'][0]} .. {fr['test'][1]} "
              f"{m['watchlist_size']:>4} {m['pure_n_trades']:>7} {pn:>9} "
              f"{m['gated_n_trades']:>8} {gn:>10}")

    print("\nPortfolio (walk-forward test months stitched, risk engine v1):")
    stitch_and_report(trades_pure_all, "pure")
    stitch_and_report(trades_gated_all, "gated")
    print(f"\nExperiment record: {record['path']}")
    return 0
```

- [ ] **Step 4: Register the `orb` subparser**

In `main()`, after the `backtest` subparser block, add:

```python
    p_orb = sub.add_parser("orb", help="opening-range breakout backtest (pure vs Stage 1-gated)")
    p_orb.add_argument("--start", required=True)
    p_orb.add_argument("--end", required=True)
    p_orb.add_argument("--train", type=int, default=250)
    p_orb.add_argument("--test", type=int, default=21)
    p_orb.add_argument("--step", type=int, default=21)
    p_orb.add_argument("--calibrate", type=int, default=40)
    p_orb.add_argument("--embargo", type=int, default=1)
    p_orb.add_argument("--holdout", type=int, default=42)
    p_orb.add_argument("--range-minutes", type=int, default=15, choices=[15, 30])
    p_orb.add_argument("--rr", type=float, default=1.0)
    p_orb.add_argument("--stop-atr-mult", type=float, default=1.0)
    p_orb.add_argument("--min-break-atr", type=float, default=0.10)
    p_orb.add_argument("--vol-mult", type=float, default=1.0)
    p_orb.add_argument("--stop-mode", default="atr", choices=["atr", "boundary"])
    p_orb.add_argument("--min-score", type=float, default=0.5)
    p_orb.add_argument("--tickers", default=None)
    p_orb.add_argument("--name", default="orb-backtest")
    p_orb.add_argument("--db", default=str(DEFAULT_DB))
    p_orb.set_defaults(func=cmd_orb)
```

- [ ] **Step 5: Run the integration test and the full suite**

Run: `python -m pytest tests/test_orb_integration.py tests/ -q`
Expected: PASS (all). No previously-passing test regresses.

- [ ] **Step 6: Manual smoke run of the CLI on a short range**

Run (adjust dates to data you hold; pick ~1 fold's worth so it's quick):

```bash
python -m stockpredictor orb --start 2024-08-01 --end 2024-12-31 --train 60 --test 21 --step 21 --calibrate 20 --holdout 10 --tickers AAPL,MSFT,NVDA,SPY,QQQ --name orb-smoke
```

Expected: prints the per-fold pure/gated table, writes `data/trades_orb-smoke_pure.csv` and `data/trades_orb-smoke_gated.csv`, prints both portfolios, and an experiment JSON path under `data/experiments/`. Confirm the JSON exists and both CSVs have rows (or a clean "no trades" if the short window genuinely produced none).

- [ ] **Step 7: Commit**

```bash
git add src/stockpredictor/__main__.py tests/test_orb_integration.py
git commit -m "$(printf 'Add orb CLI: pure vs Stage 1-gated ORB across walk-forward folds\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Full-suite gate (after all tasks)

Run: `python -m pytest tests/ -q`
Expected: all tests pass. Then do the honest read: inspect `data/experiments/<stamp>_orb-backtest.json` and the two trades CSVs. The number that matters is **`*_avg_net_bps`** per arm and each arm's **portfolio total return / max drawdown**. Compare gated vs pure to answer whether Stage 1 adds value to ORB. Treat all figures as plumbing checks (modeled fills/costs, no quotes) and confirm the swept configs on the untouched 42-session holdout only once, after selection.

---

## Self-Review

**1. Spec coverage:**
- Opening range 15/30 → `ORBConfig.range_minutes` + `--range-minutes {15,30}` (Task 1, 4). ✓
- Stop-entry on confirmed break, next-bar-open fill → `find_orb_signal` + `simulate_orb_session` (Task 1, 2). ✓
- Confirmation = min-break-ATR + volume → `find_orb_signal` (Task 1), tests for both rejections. ✓
- ATR stop + R-multiple target + EOD time-stop → `stop_and_target` + `simulate_orb_session` (Task 1, 2). ✓
- ATR = prior-session true range → `prior_session_atr` (Task 1), leak test. ✓
- stop_mode atr default, boundary available → `ORBConfig.stop_mode` + tests (Task 1). ✓
- Two arms (pure/gated), toggle via `universe_filter` → `run_orb` (Task 2), `cmd_orb` builds watchlist (Task 4), integration test asserts subset (Task 4). ✓
- No hard cap on names/day; one trade per ticker-session → `run_orb` loop + `find_orb_signal` returns first signal (Task 2). ✓
- Sizing via existing `size_position` with `q10 = -risk_frac_return` → emitted `q10` column (Task 2). ✓
- `run_portfolio` variable exits, backward-compatible → Task 3 + fallback regression test. ✓
- Walk-forward reuse, 42-session holdout, experiment JSON + trades CSV → `cmd_orb` via `run_walkforward` (Task 4). ✓
- Leakage controls (as-of Stage 1 cutoff, calendar boundaries, causal scan) → reused `build_stage1_features`; strategy causality test; ATR prior-session-only. ✓
- Caveats carried into report → full-suite gate note. ✓

**2. Placeholder scan:** No TBD/TODO; every code step has runnable code; test bodies are complete. ✓

**3. Type consistency:** `ORBConfig`/`ORBSignal` fields and `stop_and_target`/`find_orb_signal`/`simulate_orb_session`/`run_orb` signatures match across Tasks 1-4; `ORB_TRADE_COLUMNS` matches the dict keys emitted by `run_orb` and the columns `run_portfolio` reads (`bar_start`, `net_ret`, `q10`, `entry_ts`, `exit_ts`). Direction stored as `LONG`/`SHORT` strings in the frame, `+1/-1` int inside `ORBTrade`/`ORBSignal` — consistent with the codebase's `simulator`/`portfolio` string convention. ✓
