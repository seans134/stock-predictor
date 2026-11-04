# Pairs / Statistical Arbitrage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a market-neutral daily pairs/statistical-arbitrage strategy — cointegration-selected pairs traded on a z-score reversion rule — back-tested under the existing walk-forward discipline.

**Architecture:** A new pure `pairs/` package: numpy cointegration stats (Task 1), per-fold pair selection (Task 2), a daily z-score reversion backtest emitting portfolio-schema trades (Task 3), and a daily-close panel loader + `pairs` CLI (Task 4). Trades reuse the existing `run_portfolio` (its Task-3 variable-exit support already handles multi-day holds); no risk-engine change needed.

**Tech Stack:** Python 3.11+, pandas 2.x, numpy, pytest. No new dependencies.

## Global Constraints

- Python `>=3.11`; pandas `>=2.0`; numpy `>=1.26`.
- Timestamps tz-aware US/Eastern at the feature layer; `from stockpredictor.sessions import ET`.
- Bar columns lower-case (`open, high, low, close, volume, ...`).
- Direction constants from `stockpredictor.execution.decisions`: `LONG = "long"`, `SHORT = "short"`.
- Cost is one round-trip figure in return units: `CostModel().round_trip_cost`. A pair trade pays **2 ×** that (two legs). Never hard-code costs.
- Missing data never trades (skip / return None / drop); values never zero-filled.
- No look-ahead: pair selection and `alpha/beta/mu/sigma` come from **train days only**; entry fills at the **next session's close** after the z-crossing; the daily panel is built through the as-of accessor at range end.
- No new pip dependencies (ADF is hand-implemented in numpy).
- Tests flat in `tests/`, named `test_*.py`; build frames inline with tz-aware ET where timestamps matter; assert with `pytest.approx`.
- Commit messages end with:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`

---

## File Structure

- **Create** `src/stockpredictor/pairs/__init__.py` — package marker.
- **Create** `src/stockpredictor/pairs/stats.py` — `ADF_CRIT_5PCT`, `ols_hedge_ratio`, `adf_tstat`. Pure numpy.
- **Create** `src/stockpredictor/pairs/selection.py` — `PairsConfig`, `PairModel`, `select_pairs`.
- **Create** `src/stockpredictor/pairs/backtest.py` — `PAIRS_TRADE_COLUMNS`, `PairTrade`, `spread_zscore`, `simulate_pair`, `run_pairs`.
- **Create** `src/stockpredictor/pairs/data.py` — `load_daily_closes`.
- **Modify** `src/stockpredictor/__main__.py` — add `cmd_pairs` + `pairs` subparser.
- **Create** `tests/test_pairs_stats.py`, `tests/test_pairs_selection.py`, `tests/test_pairs_backtest.py`, `tests/test_pairs_data.py`, `tests/test_pairs_integration.py`.

---

### Task 1: Cointegration stats (`pairs/stats.py`)

**Files:**
- Create: `src/stockpredictor/pairs/__init__.py`
- Create: `src/stockpredictor/pairs/stats.py`
- Test: `tests/test_pairs_stats.py`

**Interfaces:**
- Produces:
  - `ADF_CRIT_5PCT = -2.86`
  - `ols_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float]` returning `(alpha, beta)` for `log_a = alpha + beta*log_b`.
  - `adf_tstat(series: np.ndarray, maxlag: int = 1) -> float` — ADF t-statistic on the lagged-level coefficient; more negative ⇒ more stationary. Returns `0.0` when the series is too short to test.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pairs_stats.py
import numpy as np
import pytest

from stockpredictor.pairs.stats import ADF_CRIT_5PCT, adf_tstat, ols_hedge_ratio


def test_ols_hedge_ratio_recovers_known_beta():
    rng = np.random.default_rng(0)
    log_b = np.linspace(1.0, 3.0, 200) + rng.normal(0, 0.01, 200)
    log_a = 2.0 + 1.5 * log_b + rng.normal(0, 1e-6, 200)
    alpha, beta = ols_hedge_ratio(log_a, log_b)
    assert alpha == pytest.approx(2.0, abs=1e-2)
    assert beta == pytest.approx(1.5, abs=1e-2)


def test_adf_stationary_series_is_significantly_negative():
    # Stationary AR(1): x_t = 0.3 x_{t-1} + e_t  -> strongly mean-reverting.
    rng = np.random.default_rng(1)
    n = 400
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + rng.normal(0, 1)
    t_stat = adf_tstat(x)
    assert t_stat < ADF_CRIT_5PCT  # rejects the unit root


def test_adf_random_walk_is_not_significant():
    # Random walk: x_t = x_{t-1} + e_t -> unit root, should NOT be significant.
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(0, 1, 400))
    t_stat = adf_tstat(x)
    assert t_stat > ADF_CRIT_5PCT  # fails to reject the unit root


def test_adf_short_series_returns_zero():
    assert adf_tstat(np.array([1.0, 2.0, 3.0])) == 0.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pairs_stats.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockpredictor.pairs'`.

- [ ] **Step 3: Create the package marker**

```python
# src/stockpredictor/pairs/__init__.py
```
(empty file)

- [ ] **Step 4: Implement `pairs/stats.py`**

```python
# src/stockpredictor/pairs/stats.py
"""Cointegration statistics (pure numpy, no I/O).

Engle-Granger step 1 is an OLS hedge ratio; the residual spread's stationarity
is judged by an Augmented Dickey-Fuller t-statistic implemented here so the
project keeps no extra dependency. The ADF regression is

    d(y)_t = c + rho * y_{t-1} + sum_{i=1..maxlag} gamma_i * d(y)_{t-i} + e_t

and the returned statistic is the t-ratio of rho; a more negative value is
stronger evidence against a unit root (i.e. a more stationary, mean-reverting
series). A fixed large-sample 5% Dickey-Fuller critical value is used as the
gate (documented approximation; no small-sample p-value interpolation).
"""

from __future__ import annotations

import numpy as np

ADF_CRIT_5PCT = -2.86  # constant, no trend, large-sample DF 5% critical value


def ols_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float]:
    log_a = np.asarray(log_a, dtype=float)
    log_b = np.asarray(log_b, dtype=float)
    X = np.column_stack([np.ones_like(log_b), log_b])
    coef, *_ = np.linalg.lstsq(X, log_a, rcond=None)
    return float(coef[0]), float(coef[1])


def adf_tstat(series: np.ndarray, maxlag: int = 1) -> float:
    y = np.asarray(series, dtype=float)
    dy = np.diff(y)                       # dy[k] = y[k+1] - y[k]
    n = len(dy)
    if n - maxlag < 10:                   # too short to estimate reliably
        return 0.0
    resp = dy[maxlag:]                     # d(y)_t
    level = y[maxlag:n]                    # y_{t-1}
    const = np.ones_like(resp)
    lags = [dy[maxlag - i: n - i] for i in range(1, maxlag + 1)]
    X = np.column_stack([level, const] + lags)
    coef, *_ = np.linalg.lstsq(X, resp, rcond=None)
    resid = resp - X @ coef
    dof = len(resp) - X.shape[1]
    if dof <= 0:
        return 0.0
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0
    se_rho = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    if se_rho == 0.0:
        return 0.0
    return float(coef[0] / se_rho)
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pairs_stats.py -v`
Expected: PASS (all). If `test_adf_stationary_series_is_significantly_negative` is marginal, note the seed is fixed so it is deterministic; do not weaken the assertion — investigate the regression indexing instead.

- [ ] **Step 6: Commit**

```bash
git add src/stockpredictor/pairs/__init__.py src/stockpredictor/pairs/stats.py tests/test_pairs_stats.py
git commit -m "$(printf 'Add pairs cointegration stats: OLS hedge ratio and numpy ADF t-stat\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 2: Pair selection (`pairs/selection.py`)

**Files:**
- Create: `src/stockpredictor/pairs/selection.py`
- Test: `tests/test_pairs_selection.py`

**Interfaces:**
- Consumes: `ols_hedge_ratio`, `adf_tstat`, `ADF_CRIT_5PCT` (Task 1).
- Produces:
  - `PairsConfig(min_corr=0.80, max_pairs=20, adf_crit=ADF_CRIT_5PCT, z_entry=2.0, z_exit=0.5, z_stop=3.0, max_hold=20)` (frozen dataclass)
  - `PairModel(a: str, b: str, alpha: float, beta: float, mu: float, sigma: float, adf_t: float)` (frozen dataclass)
  - `select_pairs(train_closes: pd.DataFrame, cfg: PairsConfig) -> list[PairModel]` — `train_closes` indexed by date, columns = tickers, values = close.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pairs_selection.py
import numpy as np
import pandas as pd
import pytest

from stockpredictor.pairs.selection import PairsConfig, select_pairs


def _panel(series: dict) -> pd.DataFrame:
    n = len(next(iter(series.values())))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(series, index=idx)


def test_cointegrated_pair_selected_random_walk_rejected():
    rng = np.random.default_rng(0)
    n = 300
    b = np.cumsum(rng.normal(0, 0.01, n)) + 4.0          # random-walk log-price base
    noise = np.zeros(n)                                   # stationary AR(1) residual
    for t in range(1, n):
        noise[t] = 0.4 * noise[t - 1] + rng.normal(0, 0.01)
    a = 1.0 + 1.2 * b + noise                             # cointegrated with b
    # An independent random walk, correlated enough to pass the pre-filter only by chance:
    c = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    closes = _panel({"A": np.exp(a), "B": np.exp(b), "C": np.exp(c)})
    models = select_pairs(closes, PairsConfig(min_corr=0.0, max_pairs=10))
    pairs = {(m.a, m.b) for m in models}
    assert ("A", "B") in pairs
    assert ("A", "C") not in pairs and ("B", "C") not in pairs


def test_correlation_prefilter_excludes_uncorrelated():
    rng = np.random.default_rng(1)
    n = 300
    a = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    b = np.cumsum(rng.normal(0, 0.01, n)) + 4.0           # independent -> low corr
    closes = _panel({"A": np.exp(a), "B": np.exp(b)})
    models = select_pairs(closes, PairsConfig(min_corr=0.95, max_pairs=10))
    assert models == []


def test_max_pairs_caps_and_sorts_by_adf():
    # Build 3 cointegrated pairs of varying stationarity; expect the cap to apply.
    rng = np.random.default_rng(2)
    n = 300
    base = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    closes = {"BASE": np.exp(base)}
    for k, phi in enumerate([0.2, 0.5, 0.8]):
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = phi * noise[t - 1] + rng.normal(0, 0.01)
        closes[f"P{k}"] = np.exp(1.0 + base + noise)
    panel = _panel(closes)
    models = select_pairs(panel, PairsConfig(min_corr=0.0, max_pairs=2))
    assert len(models) == 2
    assert models[0].adf_t <= models[1].adf_t  # most stationary first


def test_models_use_train_only_and_skip_constant_series():
    rng = np.random.default_rng(3)
    n = 200
    b = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    closes = _panel({"A": np.exp(1.0 + b), "B": np.exp(b), "K": np.full(n, 5.0)})
    models = select_pairs(closes, PairsConfig(min_corr=0.0, max_pairs=10))
    # Constant column K has zero-variance spread with anything -> never selected.
    assert all("K" not in (m.a, m.b) for m in models)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pairs_selection.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockpredictor.pairs.selection'`.

- [ ] **Step 3: Implement `pairs/selection.py`**

```python
# src/stockpredictor/pairs/selection.py
"""Per-fold pair selection from formation (train) data only.

Sector metadata is unavailable, so "economically related" is approximated by a
formation-window return-correlation pre-filter; survivors are then gated by
Engle-Granger cointegration (OLS hedge ratio + ADF on the residual spread) and
ranked by ADF t-statistic. Everything here is computed on the train window only,
so nothing leaks from the trading window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpredictor.pairs.stats import ADF_CRIT_5PCT, adf_tstat, ols_hedge_ratio


@dataclass(frozen=True)
class PairsConfig:
    min_corr: float = 0.80
    max_pairs: int = 20
    adf_crit: float = ADF_CRIT_5PCT
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.0
    max_hold: int = 20  # trading sessions


@dataclass(frozen=True)
class PairModel:
    a: str
    b: str
    alpha: float
    beta: float
    mu: float
    sigma: float
    adf_t: float


def select_pairs(train_closes: pd.DataFrame, cfg: PairsConfig) -> list[PairModel]:
    # Keep only tickers with a complete formation series (never fill gaps).
    closes = train_closes.dropna(axis=1, how="any")
    if closes.shape[1] < 2:
        return []
    log = np.log(closes)
    rets = log.diff().dropna()
    if len(rets) < 10:
        return []
    corr = rets.corr()
    tickers = list(closes.columns)

    models: list[PairModel] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            c = corr.loc[a, b]
            if not np.isfinite(c) or c < cfg.min_corr:
                continue
            la = log[a].to_numpy()
            lb = log[b].to_numpy()
            alpha, beta = ols_hedge_ratio(la, lb)
            spread = la - alpha - beta * lb
            sigma = float(spread.std())
            if not np.isfinite(sigma) or sigma <= 0.0:
                continue
            adf_t = adf_tstat(spread)
            if adf_t >= cfg.adf_crit:
                continue
            models.append(
                PairModel(a=a, b=b, alpha=alpha, beta=beta,
                          mu=float(spread.mean()), sigma=sigma, adf_t=adf_t)
            )
    models.sort(key=lambda m: m.adf_t)  # most negative (most stationary) first
    return models[: cfg.max_pairs]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pairs_selection.py -v`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add src/stockpredictor/pairs/selection.py tests/test_pairs_selection.py
git commit -m "$(printf 'Add pairs selection: correlation pre-filter + Engle-Granger cointegration\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 3: Z-score reversion backtest (`pairs/backtest.py`)

**Files:**
- Create: `src/stockpredictor/pairs/backtest.py`
- Test: `tests/test_pairs_backtest.py`, `tests/test_pairs_integration.py`

**Interfaces:**
- Consumes: `PairModel`, `PairsConfig` (Task 2); `CostModel` (`stockpredictor.execution.costs`); `LONG`, `SHORT` (`stockpredictor.execution.decisions`).
- Produces:
  - `PAIRS_TRADE_COLUMNS = ["ticker","date","bar_start","direction","gross_ret","net_ret","q10","entry_ts","exit_ts"]`
  - `PairTrade` (frozen dataclass): `pair, entry_date, exit_date, direction, entry_z, exit_z, exit_reason, gross_ret, net_ret, risk_frac_return`
  - `spread_zscore(model, closes_a: pd.Series, closes_b: pd.Series) -> pd.Series`
  - `simulate_pair(model, test_closes: pd.DataFrame, cfg, costs, session_ts: dict) -> list[PairTrade]`
  - `run_pairs(models: list[PairModel], test_closes: pd.DataFrame, cfg, costs, session_ts: dict) -> pd.DataFrame`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pairs_backtest.py
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.pairs.backtest import (
    PAIRS_TRADE_COLUMNS,
    run_pairs,
    simulate_pair,
    spread_zscore,
)
from stockpredictor.pairs.selection import PairModel, PairsConfig
from stockpredictor.sessions import ET

# beta=1, alpha=0, so spread = log A - log B; mu=0, sigma=0.01 for easy z arithmetic.
MODEL = PairModel(a="A", b="B", alpha=0.0, beta=1.0, mu=0.0, sigma=0.01, adf_t=-5.0)


def _dates(n):
    return [dt.date(2024, 3, 1) + dt.timedelta(days=k) for k in range(n)]


def _panel(a_prices, b_prices):
    idx = _dates(len(a_prices))
    return pd.DataFrame({"A": a_prices, "B": b_prices}, index=idx)


def _session_ts(dates):
    return {d: pd.Timestamp(d.year, d.month, d.day, 16, 0, tz=ET) for d in dates}


def test_spread_zscore_matches_hand_computation():
    closes = _panel([np.e ** 0.02, np.e ** 0.00], [1.0, 1.0])  # log A = 0.02, 0.0; log B = 0
    z = spread_zscore(MODEL, closes["A"], closes["B"])
    assert z.iloc[0] == pytest.approx(0.02 / 0.01)  # (0.02-0)/0.01 = 2.0
    assert z.iloc[1] == pytest.approx(0.0)


def _z_to_prices(z_path):
    # Build A,B so that (logA - logB - mu)/sigma == z; keep B=1 (logB=0), logA = z*sigma.
    a = [float(np.exp(z * MODEL.sigma)) for z in z_path]
    b = [1.0] * len(z_path)
    return _panel(a, b)


def test_long_a_short_b_on_negative_z_then_reversion_exit():
    # z: -3 (signal), fill next day, then revert to 0 (exit). Pad tail so entry is allowed.
    closes = _z_to_prices([-3.0, -3.0, 0.0, 0.0, 0.0])
    cfg = PairsConfig(z_entry=2.0, z_exit=0.5, z_stop=4.0, max_hold=10)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1
    tr = trades[0]
    assert tr.direction == 1                                   # long A / short B
    assert tr.entry_date == closes.index[1]                    # filled the day AFTER the -3 signal
    assert tr.exit_reason == "reversion"
    # A rose from exp(-3*sigma) to exp(0)=1 while B flat -> long-A/short-B is profitable.
    assert tr.gross_ret > 0
    assert tr.net_ret == pytest.approx(tr.gross_ret - 2 * CostModel().round_trip_cost)


def test_short_a_long_b_on_positive_z():
    closes = _z_to_prices([3.0, 3.0, 0.0, 0.0, 0.0])
    cfg = PairsConfig(z_entry=2.0, z_exit=0.5, z_stop=4.0, max_hold=10)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1 and trades[0].direction == -1
    assert trades[0].gross_ret > 0  # A fell to par, short-A/long-B profits


def test_divergence_stop():
    # enter at -2.5, then diverge to -4.5 (|z| >= z_stop) before reverting.
    closes = _z_to_prices([-2.5, -2.5, -4.5, -4.5, -4.5])
    cfg = PairsConfig(z_entry=2.0, z_exit=0.5, z_stop=4.0, max_hold=10)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1 and trades[0].exit_reason == "stop"


def test_time_stop():
    # enter, never revert, never diverge past stop -> exit at max_hold.
    z = [-2.5, -2.5] + [-1.5] * 6
    closes = _z_to_prices(z)
    cfg = PairsConfig(z_entry=2.0, z_exit=0.2, z_stop=5.0, max_hold=3)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1 and trades[0].exit_reason == "time"


def test_window_end_force_flat():
    # enter near the end, no exit trigger -> forced flat on the last session.
    closes = _z_to_prices([0.0, -2.5, -2.5, -1.5])
    cfg = PairsConfig(z_entry=2.0, z_exit=0.1, z_stop=5.0, max_hold=50)
    trades = simulate_pair(MODEL, closes, cfg, CostModel(), _session_ts(closes.index))
    assert len(trades) == 1 and trades[0].exit_reason == "window_end"
    assert trades[0].exit_date == closes.index[-1]


def test_run_pairs_schema_and_direction_string():
    closes = _z_to_prices([-3.0, -3.0, 0.0, 0.0, 0.0])
    df = run_pairs([MODEL], closes, PairsConfig(z_entry=2.0, z_exit=0.5),
                   CostModel(), _session_ts(closes.index))
    assert list(df.columns) == PAIRS_TRADE_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["ticker"] == "A/B"
    assert df.iloc[0]["direction"] == LONG
    assert df.iloc[0]["q10"] < 0
    assert df.iloc[0]["exit_ts"] > df.iloc[0]["entry_ts"]


def test_run_pairs_skips_missing_columns():
    closes = _z_to_prices([-3.0, -3.0, 0.0])
    other = PairModel(a="X", b="Y", alpha=0.0, beta=1.0, mu=0.0, sigma=0.01, adf_t=-5.0)
    df = run_pairs([other], closes, PairsConfig(), CostModel(), _session_ts(closes.index))
    assert df.empty and list(df.columns) == PAIRS_TRADE_COLUMNS
```

```python
# tests/test_pairs_integration.py
"""A diverge-then-reconverge pair yields one profitable round-trip; selection
uses train days only (truncating trade-window data does not change the model)."""

import datetime as dt

import numpy as np
import pandas as pd

from stockpredictor.execution.costs import CostModel
from stockpredictor.pairs.backtest import run_pairs
from stockpredictor.pairs.selection import PairsConfig, select_pairs
from stockpredictor.sessions import ET


def _panel(series):
    n = len(next(iter(series.values())))
    idx = [dt.date(2024, 1, 1) + dt.timedelta(days=k) for k in range(n)]
    return pd.DataFrame(series, index=idx)


def test_select_then_trade_roundtrip():
    rng = np.random.default_rng(7)
    n_train = 250
    base = np.cumsum(rng.normal(0, 0.01, n_train)) + 4.0
    noise = np.zeros(n_train)
    for t in range(1, n_train):
        noise[t] = 0.4 * noise[t - 1] + rng.normal(0, 0.01)
    train = _panel({"A": np.exp(1.0 + base + noise), "B": np.exp(base)})
    models = select_pairs(train, PairsConfig(min_corr=0.0, max_pairs=5))
    assert any((m.a, m.b) == ("A", "B") for m in models)

    model = next(m for m in models if (m.a, m.b) == ("A", "B"))
    # Trade window: force the spread to diverge (below -z_entry) then reconverge to mu.
    sig, mu = model.sigma, model.mu
    z_path = [-3.0, -3.0, -1.0, 0.0, 0.0]
    a = [float(np.exp(mu + z * sig + model.alpha)) for z in z_path]  # logA-alpha-beta*logB=mu+z*sig, logB=0
    test = _panel({"A": a, "B": [1.0] * len(z_path)})
    session_ts = {d: pd.Timestamp(d.year, d.month, d.day, 16, 0, tz=ET) for d in test.index}
    trades = run_pairs([model], test, PairsConfig(z_entry=2.0, z_exit=0.5),
                       CostModel(), session_ts)
    assert len(trades) == 1
    assert trades.iloc[0]["gross_ret"] > 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_pairs_backtest.py tests/test_pairs_integration.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockpredictor.pairs.backtest'`.

- [ ] **Step 3: Implement `pairs/backtest.py`**

```python
# src/stockpredictor/pairs/backtest.py
"""Daily z-score reversion backtest for cointegrated pairs.

Decision at a session's close, fill at the NEXT session's close (no look-ahead).
Dollar-neutral legs: a pair trade's return is direction*(r_A - r_B) on one unit
of per-leg capital; both legs pay a round trip, so cost = 2 * round_trip_cost.
The hedge ratio drives the z-signal, not leg sizing. Emits portfolio-schema
trades so the existing risk engine can replay them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.pairs.selection import PairModel, PairsConfig

PAIRS_TRADE_COLUMNS = [
    "ticker", "date", "bar_start", "direction",
    "gross_ret", "net_ret", "q10", "entry_ts", "exit_ts",
]


@dataclass(frozen=True)
class PairTrade:
    pair: str
    entry_date: object
    exit_date: object
    direction: int              # +1 long-A/short-B, -1 short-A/long-B
    entry_z: float
    exit_z: float
    exit_reason: str            # "reversion" | "stop" | "time" | "window_end"
    gross_ret: float
    net_ret: float
    risk_frac_return: float


def spread_zscore(model: PairModel, closes_a: pd.Series, closes_b: pd.Series) -> pd.Series:
    spread = np.log(closes_a) - model.alpha - model.beta * np.log(closes_b)
    return (spread - model.mu) / model.sigma


def simulate_pair(
    model: PairModel, test_closes: pd.DataFrame, cfg: PairsConfig,
    costs: CostModel, session_ts: dict,
) -> list[PairTrade]:
    a, b = model.a, model.b
    if a not in test_closes.columns or b not in test_closes.columns:
        return []
    df = test_closes[[a, b]].dropna()
    if len(df) < 2:
        return []
    z = spread_zscore(model, df[a], df[b])
    dates = list(df.index)
    prices_a = df[a].to_numpy()
    prices_b = df[b].to_numpy()
    cost2 = 2.0 * costs.round_trip_cost
    risk_frac = max(cfg.z_stop - cfg.z_entry, 0.0) * model.sigma

    trades: list[PairTrade] = []
    pos = None              # dict: direction, entry_idx, entry_a, entry_b, entry_z
    pending = None          # (direction, decision_z) awaiting next-session fill
    last = len(dates) - 1

    def close_position(exit_idx: int, reason: str) -> None:
        nonlocal pos
        r_a = prices_a[exit_idx] / pos["entry_a"] - 1.0
        r_b = prices_b[exit_idx] / pos["entry_b"] - 1.0
        gross = pos["direction"] * (r_a - r_b)
        trades.append(PairTrade(
            pair=f"{a}/{b}", entry_date=dates[pos["entry_idx"]], exit_date=dates[exit_idx],
            direction=pos["direction"], entry_z=pos["entry_z"], exit_z=float(z.iloc[exit_idx]),
            exit_reason=reason, gross_ret=float(gross), net_ret=float(gross - cost2),
            risk_frac_return=float(risk_frac),
        ))
        pos = None

    for idx in range(len(dates)):
        zt = float(z.iloc[idx])
        # 1) fill a pending entry at THIS close (the session after the signal)
        if pending is not None and pos is None:
            direction, entry_z = pending
            pos = {"direction": direction, "entry_idx": idx,
                   "entry_a": prices_a[idx], "entry_b": prices_b[idx], "entry_z": entry_z}
            pending = None
            continue
        # 2) manage an open position
        if pos is not None:
            held = idx - pos["entry_idx"]
            if abs(zt) <= cfg.z_exit:
                close_position(idx, "reversion")
            elif abs(zt) >= cfg.z_stop:
                close_position(idx, "stop")
            elif held >= cfg.max_hold:
                close_position(idx, "time")
            elif idx == last:
                close_position(idx, "window_end")
            continue
        # 3) flat: look for an entry signal, but only if a fill day AND an exit day remain
        if idx < last - 1:
            if zt <= -cfg.z_entry:
                pending = (1, zt)
            elif zt >= cfg.z_entry:
                pending = (-1, zt)

    if pos is not None:  # safety: force flat on the last session
        close_position(last, "window_end")
    return trades


def run_pairs(
    models: list[PairModel], test_closes: pd.DataFrame, cfg: PairsConfig,
    costs: CostModel, session_ts: dict,
) -> pd.DataFrame:
    rows = []
    for model in models:
        for tr in simulate_pair(model, test_closes, cfg, costs, session_ts):
            rows.append({
                "ticker": tr.pair,
                "date": tr.entry_date,
                "bar_start": session_ts[tr.entry_date],
                "direction": LONG if tr.direction == 1 else SHORT,
                "gross_ret": tr.gross_ret,
                "net_ret": tr.net_ret,
                "q10": -tr.risk_frac_return,
                "entry_ts": session_ts[tr.entry_date],
                "exit_ts": session_ts[tr.exit_date],
            })
    return pd.DataFrame(rows, columns=PAIRS_TRADE_COLUMNS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_pairs_backtest.py tests/test_pairs_integration.py -v`
Expected: PASS (all). If `test_window_end_force_flat` or the entry-day math is off by one, re-read the loop: a pending signal is set on the decision day, filled on the *next* session's close (the `continue` after filling), and exits are only evaluated on sessions strictly after the fill.

- [ ] **Step 5: Commit**

```bash
git add src/stockpredictor/pairs/backtest.py tests/test_pairs_backtest.py tests/test_pairs_integration.py
git commit -m "$(printf 'Add pairs z-score reversion backtest with next-session fills and 2-leg costs\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

### Task 4: Daily-close loader + `pairs` CLI (`pairs/data.py`, `__main__.py`)

**Files:**
- Create: `src/stockpredictor/pairs/data.py`
- Modify: `src/stockpredictor/__main__.py` (add `cmd_pairs` after `cmd_orb`; register the `pairs` subparser)
- Test: `tests/test_pairs_data.py`

**Interfaces:**
- Consumes: `AsOfData`, `BarStore`, `SessionCalendar`/`default_calendar`, `ET`, `LABEL_AS_OF_DELAY`, plus Tasks 1-3 (`PairsConfig`, `select_pairs`, `run_pairs`), the walk-forward harness (`generate_folds`, `run_walkforward`, `WalkForwardConfig`), `CostModel`, `RiskConfig`, `run_portfolio`, `summarize`.
- Produces:
  - `load_daily_closes(data: AsOfData, calendar, tickers: list[str], start: date, end: date) -> pd.DataFrame` — index = session date, columns = tickers, values = RTH close; missing ticker-days NaN.
  - `stockpredictor pairs ...` CLI: one experiment JSON + `trades_<name>.csv`, per-fold + across-fold tables, portfolio metrics.

- [ ] **Step 1: Write the failing test for `load_daily_closes`**

```python
# tests/test_pairs_data.py
import datetime as dt

import pandas as pd

from stockpredictor.data.asof import AsOfData
from stockpredictor.data.store import BarStore
from stockpredictor.pairs.data import load_daily_closes
from stockpredictor.sessions import ET, default_calendar


def _insert_session(store, ticker, day, closes_by_hhmm):
    """Insert 5-min RTH bars for one session; available immediately."""
    rows = []
    for hhmm, close in closes_by_hhmm.items():
        h, m = map(int, hhmm.split(":"))
        ts = pd.Timestamp(day.year, day.month, day.day, h, m, tz=ET)
        rows.append((ts, close))
    idx = pd.DatetimeIndex([r[0] for r in rows]).tz_convert("UTC")
    df = pd.DataFrame(
        {
            "open": [r[1] for r in rows], "high": [r[1] for r in rows],
            "low": [r[1] for r in rows], "close": [r[1] for r in rows],
            "volume": [1000.0] * len(rows), "trade_count": [10] * len(rows),
            "vwap": [r[1] for r in rows],
            "available_ts": [ts + pd.Timedelta(minutes=6) for ts in idx],
        },
        index=idx,
    )
    store.insert_bars(ticker, df, source="test", timeframe="5min")


def test_load_daily_closes_last_rth_close_per_session(tmp_path):
    db = tmp_path / "bars.sqlite"
    store = BarStore(db)
    cal = default_calendar()
    # Two consecutive sessions for two tickers.
    d1, d2 = dt.date(2024, 3, 18), dt.date(2024, 3, 19)
    for day, last in [(d1, 101.0), (d2, 103.0)]:
        _insert_session(store, "AAA", day, {"09:35": 100.0, "15:55": last})
        _insert_session(store, "BBB", day, {"09:35": 50.0, "15:55": last / 2})
    store.close()

    store = BarStore(db)
    try:
        panel = load_daily_closes(AsOfData(store), cal, ["AAA", "BBB"], d1, d2)
    finally:
        store.close()
    assert list(panel.columns) == ["AAA", "BBB"]
    assert panel.loc[d1, "AAA"] == 101.0   # last RTH close of the session
    assert panel.loc[d2, "AAA"] == 103.0
    assert panel.loc[d2, "BBB"] == 103.0 / 2
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_pairs_data.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'stockpredictor.pairs.data'`.

- [ ] **Step 3: Implement `pairs/data.py`**

```python
# src/stockpredictor/pairs/data.py
"""Daily RTH-close panel from the 5-minute store, for pairs formation/trading.

One column per ticker, one row per ET session, value = the last regular-session
5-minute bar's close. Fetched through the as-of accessor at the range end
(finished sessions). Missing ticker-days stay NaN and are never zero-filled.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from stockpredictor.data.asof import AsOfData
from stockpredictor.labeling.windows import LABEL_AS_OF_DELAY
from stockpredictor.sessions import ET, SessionCalendar


def load_daily_closes(
    data: AsOfData, calendar: SessionCalendar, tickers: list[str],
    start: dt.date, end: dt.date,
) -> pd.DataFrame:
    sessions = calendar.sessions_between(start, end)
    if not sessions:
        return pd.DataFrame()
    session_set = set(sessions)
    range_open = pd.Timestamp(calendar.session_open(sessions[0]))
    range_close = pd.Timestamp(calendar.session_close(sessions[-1]))
    as_of = range_close + LABEL_AS_OF_DELAY
    opens = {d: pd.Timestamp(calendar.session_open(d)) for d in sessions}
    closes = {d: pd.Timestamp(calendar.session_close(d)) for d in sessions}

    series: dict[str, pd.Series] = {}
    for ticker in tickers:
        bars = data.get_bars(ticker, as_of=as_of, lookback=as_of - range_open)
        if bars.empty:
            continue
        et = bars.index.tz_convert(ET)
        day = pd.Series(et.date, index=bars.index)
        ts = pd.Series(et, index=bars.index)
        mask = day.isin(session_set) & (ts >= day.map(opens)) & (ts < day.map(closes))
        rth = bars.loc[mask.to_numpy()]
        if rth.empty:
            continue
        rth_day = pd.Series(rth.index.tz_convert(ET).date, index=rth.index)
        series[ticker] = rth.groupby(rth_day)["close"].last()
    if not series:
        return pd.DataFrame()
    panel = pd.DataFrame(series).sort_index()
    panel.index.name = "date"
    return panel
```

- [ ] **Step 4: Run the loader test**

Run: `python -m pytest tests/test_pairs_data.py -v`
Expected: PASS.

- [ ] **Step 5: Add `cmd_pairs` to `__main__.py`**

Insert after `cmd_orb` (before `cmd_ablation`):

```python
def cmd_pairs(args: argparse.Namespace) -> int:
    import pandas as pd

    from stockpredictor.data.asof import AsOfData
    from stockpredictor.data.store import BarStore
    from stockpredictor.execution.costs import CostModel
    from stockpredictor.execution.portfolio import RiskConfig, run_portfolio
    from stockpredictor.execution.simulator import summarize
    from stockpredictor.pairs.backtest import run_pairs
    from stockpredictor.pairs.data import load_daily_closes
    from stockpredictor.pairs.selection import PairsConfig, select_pairs
    from stockpredictor.sessions import default_calendar
    from stockpredictor.validation.folds import WalkForwardConfig, generate_folds
    from stockpredictor.validation.harness import run_walkforward

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    wf_config = WalkForwardConfig(
        train_sessions=args.train, test_sessions=args.test, step_sessions=args.step,
        embargo_sessions=args.embargo, holdout_sessions=args.holdout,
    )
    calendar = default_calendar()
    sessions = calendar.sessions_between(start, end)
    folds = generate_folds(sessions, wf_config)
    if not folds:
        print("No folds fit in the requested range.", file=sys.stderr)
        return 1
    print(f"{len(folds)} folds over {len(sessions)} sessions ({args.holdout} held out)")
    session_ts = {d: pd.Timestamp(calendar.session_close(d)) for d in sessions}

    store = BarStore(args.db)
    try:
        tickers = (
            [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if args.tickers else store.tickers()
        )
        print(f"Loading daily closes for {len(tickers)} tickers...")
        panel = load_daily_closes(AsOfData(store), calendar, tickers, start, end)
    finally:
        store.close()
    print(f"  panel: {panel.shape[0]} sessions x {panel.shape[1]} tickers")

    costs = CostModel()
    cfg = PairsConfig(
        min_corr=args.min_corr, max_pairs=args.max_pairs, z_entry=args.z_entry,
        z_exit=args.z_exit, z_stop=args.z_stop, max_hold=args.max_hold,
    )
    all_trades: list = []

    def run_fold(fold):
        train_days = list(fold.train_days)
        test_days = list(fold.test_days)
        train_panel = panel.loc[panel.index.isin(train_days)]
        test_panel = panel.loc[panel.index.isin(test_days)]
        models = select_pairs(train_panel, cfg)
        trades = run_pairs(models, test_panel, cfg, costs, session_ts)
        all_trades.append(trades)
        metrics = summarize(trades, len(trades), len(test_days))
        metrics["n_pairs"] = len(models)
        print(f"  fold {fold.fold_id}: {len(models)} pairs, {metrics['n_trades']} trades")
        return metrics

    record = run_walkforward(
        folds, run_fold, experiment_name=args.name,
        config_record={
            "walkforward": wf_config, "costs": costs, "pairs": cfg,
            "tickers": tickers, "range": [str(start), str(end)],
        },
    )

    print(f"\n{'fold':>4} {'test period':^23} {'pairs':>5} {'trades':>6} {'hit':>6} "
          f"{'gross':>7} {'net':>7} {'total':>8}")
    for fr in record["folds"]:
        m = fr["metrics"]
        hit = f"{m['hit_rate']:.1%}" if m["n_trades"] else "  -"
        gross = f"{m['avg_gross_bps']:.1f}bp" if m["n_trades"] else "  -"
        net = f"{m['avg_net_bps']:.1f}bp" if m["n_trades"] else "  -"
        print(f"{fr['fold_id']:>4} {fr['test'][0]} .. {fr['test'][1]} "
              f"{m['n_pairs']:>5} {m['n_trades']:>6} {hit:>6} {gross:>7} {net:>7} "
              f"{m['total_net_return']:>8.4f}")

    stitched = pd.concat([t for t in all_trades if not t.empty], ignore_index=True) \
        if any(not t.empty for t in all_trades) else pd.DataFrame()
    if not stitched.empty:
        path = Path("data") / f"trades_{args.name}.csv"
        stitched.sort_values("bar_start").to_csv(path, index=False)
        print(f"\nTrades -> {path}")
    portfolio = run_portfolio(stitched, RiskConfig())
    print("\nPortfolio (walk-forward test months stitched, risk engine v1):")
    print(f"  final ${portfolio['final_equity']:,.0f}  return {portfolio['total_return']:+.2%}  "
          f"maxDD {portfolio['max_drawdown']:.2%}  taken {portfolio['n_taken']}")
    print(f"\nExperiment record: {record['path']}")
    return 0
```

- [ ] **Step 6: Register the `pairs` subparser**

In `main()`, after the `orb` subparser block, add:

```python
    p_pairs = sub.add_parser("pairs", help="market-neutral pairs / statistical-arbitrage backtest")
    p_pairs.add_argument("--start", required=True)
    p_pairs.add_argument("--end", required=True)
    p_pairs.add_argument("--train", type=int, default=250)
    p_pairs.add_argument("--test", type=int, default=21)
    p_pairs.add_argument("--step", type=int, default=21)
    p_pairs.add_argument("--embargo", type=int, default=1)
    p_pairs.add_argument("--holdout", type=int, default=42)
    p_pairs.add_argument("--min-corr", type=float, default=0.80)
    p_pairs.add_argument("--max-pairs", type=int, default=20)
    p_pairs.add_argument("--z-entry", type=float, default=2.0)
    p_pairs.add_argument("--z-exit", type=float, default=0.5)
    p_pairs.add_argument("--z-stop", type=float, default=3.0)
    p_pairs.add_argument("--max-hold", type=int, default=20)
    p_pairs.add_argument("--tickers", default=None)
    p_pairs.add_argument("--name", default="pairs-backtest")
    p_pairs.add_argument("--db", default=str(DEFAULT_DB))
    p_pairs.set_defaults(func=cmd_pairs)
```

- [ ] **Step 7: Run the loader test and the full suite**

Run: `python -m pytest tests/test_pairs_data.py tests/ -q`
Expected: PASS (all), no regressions.

- [ ] **Step 8: Manual smoke run of the CLI**

Run (adjust dates to your data; a handful of related names so pairs can form):

```bash
python -m stockpredictor pairs --start 2024-07-08 --end 2026-09-21 --train 250 --test 21 --tickers XLK,QQQ,AAPL,MSFT,NVDA,AMD,AVGO,XLF,JPM,BAC,WFC,C --name pairs-smoke
```

Expected: prints the per-fold table (pairs selected, trades, gross/net bp), writes `data/trades_pairs-smoke.csv` (or a clean "no trades"), prints the portfolio line, and an experiment JSON under `data/experiments/`. Capture the console output and JSON path in your report. If it errors, debug it (likely a wiring bug). If it runs very long (>10 min) or the environment can't run it, report DONE_WITH_CONCERNS with the automated tests passing and the smoke run pending. Do not commit generated `data/` artifacts.

- [ ] **Step 9: Commit**

```bash
git add src/stockpredictor/pairs/data.py src/stockpredictor/__main__.py tests/test_pairs_data.py
git commit -m "$(printf 'Add pairs CLI and daily-close loader; walk-forward stat-arb backtest\n\nCo-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>')"
```

---

## Full-suite gate (after all tasks)

Run: `python -m pytest tests/ -q`
Expected: all pass. Then the honest read: inspect `data/experiments/<stamp>_pairs-backtest.json` and `trades_pairs-backtest.csv`. The number that matters is per-fold **`avg_net_bps`** and **`avg_gross_bps`** (gross is the signal to watch, since the 2-leg modeled cost is the dominant drag), plus the portfolio return/DD. A positive gross is the encouraging outcome; confirm any candidate config on the untouched 42-session holdout only once, after selection. All figures are plumbing checks (modeled fills/costs, no quotes).

---

## Self-Review

**1. Spec coverage:**
- Daily bars from 5-min store → `load_daily_closes` (Task 4). ✓
- Correlation pre-filter + Engle-Granger cointegration, train-only → `select_pairs` (Task 2) using `stats.py` (Task 1). ✓
- z-score reversion signal, train μ/σ, next-session fill → `spread_zscore` + `simulate_pair` (Task 3). ✓
- Entry/exit/stop/time/window-end → `simulate_pair` with tests for each. ✓
- Dollar-neutral 2-leg P&L, `net = gross − 2·round_trip` → `simulate_pair` + test. ✓
- Risk-engine reuse, `q10 = −risk_frac_return`, entry_ts/exit_ts → `run_pairs` emit (Task 3), `run_portfolio` unchanged. ✓
- Walk-forward, formation=train/trade=test, 42 holdout, experiment JSON + CSV → `cmd_pairs` (Task 4). ✓
- No Stage 1 → `cmd_pairs` has no Stage 1 block. ✓
- No new dependency → ADF hand-implemented in numpy (Task 1). ✓
- Leakage controls (train-only selection, next-session fill, as-of loader) → covered in Tasks 1-4 + tests. ✓

**2. Placeholder scan:** No TBD/TODO; every code step has runnable code; test bodies complete. ✓

**3. Type consistency:** `PairModel`/`PairsConfig` fields and `select_pairs`/`spread_zscore`/`simulate_pair`/`run_pairs`/`load_daily_closes` signatures match across Tasks 1-4. `PAIRS_TRADE_COLUMNS` matches the dict keys emitted by `run_pairs` and the columns `run_portfolio` reads (`bar_start`, `date`, `net_ret`, `q10`, `entry_ts`, `exit_ts`). Direction stored as `LONG`/`SHORT` strings in the frame, `+1/-1` int inside `PairTrade`/`simulate_pair` — consistent with the ORB/simulator convention. `session_ts` maps `date → tz-aware ET timestamp` in both `run_pairs` and `cmd_pairs`. ✓
