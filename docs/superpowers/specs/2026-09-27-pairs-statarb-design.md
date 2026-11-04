# Pairs / Statistical Arbitrage — Design Spec

**Date:** 2026-09-27
**Status:** Approved for planning
**Author:** brainstorming session (seans134 + Claude)

## 1. Goal

Add a market-neutral pairs / statistical-arbitrage strategy to StockPredictor and
back-test it on the existing bar store, under the existing walk-forward discipline
with the 42-session holdout untouched.

Rationale: every prior experiment (Stage 2 direction ≈ 50%, ORB gross ≈ 0, TJR not
significant) has hit the same wall — 5-minute bars do not carry a tradeable
*directional* signal. Pairs trading is **market-neutral**: it profits from the
*relative* mispricing of two cointegrated names reconverging, so it never needs the
directional prediction that has failed. This is the one approach whose mechanism
structurally sidesteps the project's binding constraint.

**Success criterion for the build (not for profitability):** a leak-free, tested,
fold-validated pairs backtest that selects cointegrated pairs on formation data only,
trades a z-score reversion rule out-of-sample, applies realistic two-leg costs, and
reports after-cost per-trade and portfolio metrics. Whether pairs has edge is an
empirical result read off the folds (and, only if a candidate config emerges, the
holdout).

## 2. Scope

### In scope
- New `pairs/` package: numpy cointegration stats, per-fold pair selection, daily
  z-score reversion backtest, daily-close panel loader.
- Daily bars derived from the 5-min store (per ET session), via the as-of accessor.
- Sector-scoping via a **formation-window correlation pre-filter** (no external
  metadata), then **Engle–Granger cointegration** (numpy OLS + numpy ADF, no new
  dependency).
- Market-neutral (dollar-neutral) two-leg P&L with two-leg round-trip costs.
- Reuse of `run_portfolio` (Task-3 variable-exit support) for sizing/concurrency/halt.
- A `pairs` CLI subcommand; walk-forward experiment JSON + trades CSV.
- Unit + integration tests, including explicit leakage/causality tests.

### Out of scope (YAGNI)
- No Stage 1 integration (single-name pre-market scanner does not apply to
  relative value). No gated/pure arms.
- No Johansen / multivariate cointegration; Engle–Granger two-asset only.
- No half-life filter or Kalman dynamic hedge ratio in v1 (static train-window β).
- No beta-neutral leg sizing (dollar-neutral only; β drives the signal, not leg $).
- No intraday execution; daily formation and daily trading.
- No new data sources, no quote/tick data.

## 3. Design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Horizon | **Daily** bars from the 5-min store | Canonical Gatev/cointegration approach; ~2yr ≈ 500 daily obs/pair; less exposed to the no-quotes problem than intraday. |
| Universe | All stored tickers (258 names + ETFs) | ETFs are natural anchors; correlation pre-filter finds the related subset. |
| Sector scoping | **Formation-window return-correlation pre-filter** (`corr ≥ min_corr`) | No sector metadata in repo; leak-free proxy for "economically related"; bounds the pair search. |
| Cointegration | **Engle–Granger**: numpy OLS hedge ratio + numpy ADF on the residual | No new dependency; testable in isolation. |
| Selection | corr pre-filter → ADF stationarity gate → rank by ADF t-stat → top `max_pairs` | Small, bounded traded set; limits multiple-comparisons exposure. |
| Signal | daily z-score of the log-price spread, z from **train** μ/σ | PIT: no test data in the normalization. |
| Entry / exit | enter at `|z| ≥ z_entry` (2.0), exit at `|z| ≤ z_exit` (0.5) | Standard reversion bands. |
| Stops | divergence stop `|z| ≥ z_stop` (3.0); time-stop `max_hold` days (20); force-flat at test-window end | Bounded holding; divergence protection. |
| Leg sizing | **dollar-neutral** (equal $ per leg); β used only for the spread/z signal | Robust, standard; avoids β-leverage surprises. |
| Costs | pair round trip = **2 × `CostModel.round_trip_cost`** (both legs) | Two legs, entry+exit; the dominant sensitivity. |
| Risk / portfolio | reuse `run_portfolio`; each pair trade is one row with `q10 = −spread-stop-risk`, `entry_ts`/`exit_ts` | One risk engine; reuses Task-3 variable-exit support. |
| Stage 1 | **not used** | Relative-value strategy; Stage 1 is single-name pre-market opportunity. |
| Validation | `generate_folds`; formation = train days, trading = test days; **42-session holdout untouched**; experiment JSON + `trades_<name>.csv` | Consistent with and comparable to prior experiments. |

## 4. Architecture

### 4.1 New package `src/stockpredictor/pairs/`

**`pairs/data.py`** — daily close panel.
```
def load_daily_closes(data: AsOfData, calendar: SessionCalendar,
                      tickers: list[str], start: date, end: date) -> pd.DataFrame
```
Returns a DataFrame indexed by session `date`, one column per ticker, values = the
session's **RTH close** (last regular-session 5-min bar's close), fetched through the
as-of accessor at range end and resampled per ET session. Missing ticker-days stay
NaN (never zero-filled).

**`pairs/stats.py`** — pure numpy cointegration math (no I/O).
```
def ols_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float]
    # returns (alpha, beta) for log_a = alpha + beta*log_b, via least squares

def adf_tstat(series: np.ndarray, maxlag: int = 1) -> float
    # Augmented Dickey-Fuller t-statistic on the coefficient of the lagged level
    # in  d(series)_t = c + rho*series_{t-1} + sum_i gamma_i * d(series)_{t-i} + e_t
    # More negative => stronger rejection of a unit root (more stationary).

ADF_CRIT_5PCT = -2.86  # constant, no trend, large-sample Dickey-Fuller 5% critical value
```

**`pairs/selection.py`** — per-fold pair selection from formation data.
```
@dataclass(frozen=True)
class PairModel:
    a: str; b: str
    alpha: float; beta: float
    mu: float; sigma: float      # train spread mean/std for z-scoring
    adf_t: float                 # residual ADF t-stat (selection strength)

@dataclass(frozen=True)
class PairsConfig:
    min_corr: float = 0.80
    max_pairs: int = 20
    adf_crit: float = ADF_CRIT_5PCT
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.0
    max_hold: int = 20           # trading sessions

def select_pairs(train_closes: pd.DataFrame, cfg: PairsConfig) -> list[PairModel]
```
`select_pairs`:
1. Restrict to tickers with a complete-enough train series (drop columns with too
   many NaNs); compute daily log returns.
2. Correlation pre-filter: keep candidate pairs (a,b), a<b, with return corr ≥ `min_corr`.
3. For each candidate: `alpha,beta = ols_hedge_ratio(log P_a, log P_b)` on train;
   residual spread `s = log P_a − alpha − beta·log P_b`; `adf_t = adf_tstat(s)`.
   Keep pairs with `adf_t < cfg.adf_crit` (stationary residual → cointegrated).
   Record `mu=s.mean()`, `sigma=s.std()` (train only). Skip if `sigma <= 0`.
4. Rank by `adf_t` ascending (most negative first); return the top `max_pairs`.

**`pairs/backtest.py`** — z-score reversion backtest.
```
PAIRS_TRADE_COLUMNS = ["ticker","date","bar_start","direction",
                       "gross_ret","net_ret","q10","entry_ts","exit_ts"]

@dataclass(frozen=True)
class PairTrade:
    pair: str; entry_date; exit_date; direction  # +1 long-A/short-B, -1 short-A/long-B
    entry_z: float; exit_z: float; exit_reason: str  # "reversion"|"stop"|"time"|"window_end"
    gross_ret: float; net_ret: float; risk_frac_return: float

def spread_zscore(model: PairModel, closes_a: pd.Series, closes_b: pd.Series) -> pd.Series
    # z_t = ((log a - alpha - beta*log b) - mu) / sigma, using the model's train mu/sigma

def simulate_pair(
    model: PairModel,
    test_closes: pd.DataFrame,              # index = test session date, columns include a, b
    cfg: PairsConfig,
    costs: CostModel,
    session_ts: dict,                       # session date -> tz-aware ET timestamp (session close)
) -> list[PairTrade]

def run_pairs(
    models: list[PairModel],
    test_closes: pd.DataFrame,              # test-day close panel (columns = tickers)
    cfg: PairsConfig,
    costs: CostModel,
    session_ts: dict,                       # session date -> tz-aware ET timestamp
) -> pd.DataFrame                           # PAIRS_TRADE_COLUMNS
```
`run_pairs` calls `simulate_pair` for every model (skipping a pair whose a/b column is
absent from `test_closes`) and returns the concatenated `PAIRS_TRADE_COLUMNS` frame.
`session_ts` maps each session date to that session's close timestamp
(`calendar.session_close(day)`), which becomes the row's `bar_start`/`entry_ts`/`exit_ts`.

### 4.2 Signal, P&L, and sizing (precise)

- z-score on test days from the model's **train** `alpha,beta,mu,sigma`.
- **Entry** (no open position for this pair):
  - `z ≤ −z_entry` → spread cheap → **long A / short B**, `direction=+1`.
  - `z ≥ +z_entry` → spread rich → **short A / long B**, `direction=−1`.
  - Entry fills at the **next session's close** after the crossing day (decision at
    close of day t, act at close of t+1 — no look-ahead, mirrors the Stage-2 timing
    rule adapted to daily).
- **Exit** (position open): first of — `|z| ≤ z_exit` ("reversion"); `|z| ≥ z_stop`
  ("stop"); held `max_hold` sessions ("time"); last test session ("window_end").
  Exit fills at that session's close.
- **P&L (dollar-neutral, per unit leg capital):** with leg returns
  `r_A = A_exit/A_entry − 1`, `r_B = B_exit/B_entry − 1`,
  `gross_ret = direction × (r_A − r_B)`; `net_ret = gross_ret − 2·costs.round_trip_cost`.
- **Sizing risk for the engine:** `risk_frac_return = max(z_stop − z_entry, 0)·sigma`
  (the spread move, in log-return units, from entry band to stop band), emitted as
  `q10 = −risk_frac_return` so `size_position` risks `risk_per_trade` at the divergence stop.
- One open position per pair at a time; a pair can re-enter after it closes.

### 4.3 Trade schema & CLI

`run_pairs` emits `PAIRS_TRADE_COLUMNS`, where `ticker` = the pair label `"A/B"`,
`bar_start` = the crossing (decision) session timestamp, `entry_ts` = the fill
session, `exit_ts` = the exit session (all tz-aware ET, e.g. session close).

**`cmd_pairs` in `__main__.py`**, mirroring `cmd_orb`'s walk-forward skeleton:
1. Load the daily close panel for all tickers over the range (once); build
   `session_ts = {day: calendar.session_close(day) for day in sessions}`.
2. Per fold: `models = select_pairs(panel.loc[train_days], cfg)`; then
   `run_pairs(models, panel.loc[test_days], cfg, costs, session_ts)`; collect trades.
3. Summarize with the existing `summarize`; stitch through `run_portfolio(RiskConfig())`.
4. Persist one experiment JSON via `run_walkforward` (recording `PairsConfig`, the
   per-fold selected-pair count, and metrics) and `trades_<name>.csv`.
5. Print per-fold and across-fold tables (pairs selected, trades, trades/day, hit
   rate, avg gross bp, avg net bp, total net) plus portfolio final equity / return / DD.

CLI args: `--start --end --train --test --step --embargo --holdout`
`--min-corr --max-pairs --z-entry --z-exit --z-stop --max-hold --tickers --name --db`.
Defaults match the other commands (train 250, test 21, step 21, embargo 1, holdout 42;
no calibrate segment — pairs needs no calibration).

## 5. Look-ahead / leakage controls (explicit, tested)

1. `alpha, beta, mu, sigma` and pair selection come from **train days only**; a test
   asserts they are unchanged when test-day data is truncated.
2. ADF and correlation use the formation (train) window only.
3. Entry fills at the **next** session's close after the crossing (never the crossing
   day's close).
4. The daily close panel is built through the as-of accessor at range end (finished
   sessions), consistent with the rest of the project.
5. Session set from the calendar; folds from `generate_folds`; holdout reserved.

## 6. Testing plan (TDD)

**`pairs/stats.py`**
- `ols_hedge_ratio` recovers a known β on synthetic `log_a = 2 + 1.5·log_b + tiny noise`.
- `adf_tstat` is strongly negative for a stationary AR(1) (φ≈0.3) series and near
  zero / non-significant for a cumulative random walk.

**`pairs/selection.py`**
- A synthetic cointegrated pair (`a = β·b + stationary noise`, both as price series)
  is selected; an independent random-walk pair is rejected by the ADF gate.
- Correlation pre-filter drops a low-correlation candidate before the ADF step.
- `max_pairs` caps the returned list; `mu/sigma/alpha/beta` come from train only.
- A near-constant series (`sigma ≤ 0`) is skipped, not divided by zero.

**`pairs/backtest.py`**
- `spread_zscore` matches a hand-computed z on a tiny series.
- `z ≤ −z_entry` opens long-A/short-B; `z ≥ +z_entry` opens short-A/long-B.
- entry fills at the next session's close (not the crossing day's).
- reversion exit at `|z| ≤ z_exit`; divergence exit at `|z| ≥ z_stop`; time-stop at
  `max_hold`; window-end force-flat on the last test session.
- `net_ret = gross_ret − 2·round_trip_cost`; dollar-neutral P&L sign correct for
  both directions (long-A/short-B profits when A outperforms B).
- `q10 = −risk_frac_return`.

**Integration (`run_pairs` on a synthetic 2-ticker panel)**
- A constructed diverge-then-reconverge series produces one round-trip trade with the
  expected direction and a positive gross return; selection uses only train days.

## 7. Honest caveats (carried into the report)

- **No historical quotes** → both legs' fills and the `CostModel` are modeled; the
  **two-leg cost (22 bp round trip with the placeholder)** is the dominant
  sensitivity, so the **gross** figure is the signal to watch, net likely needs real
  quotes.
- ~2 years, hand-picked universe → survivorship bias; pairs decay as relationships
  break (mergers, guidance shocks).
- The correlation pre-filter + `max_pairs` bound the search, but selecting pairs is
  still a search; the recorded experiment JSON preserves the config, and the
  42-session holdout is inspected only once, only if a candidate config emerges.
- numpy ADF uses a fixed large-sample 5% critical value (no small-sample p-value
  interpolation) — a documented approximation of the Engle–Granger gate.

## 8. File-level change summary

- **New:** `src/stockpredictor/pairs/__init__.py`, `pairs/data.py`, `pairs/stats.py`,
  `pairs/selection.py`, `pairs/backtest.py`.
- **New tests:** `tests/test_pairs_stats.py`, `tests/test_pairs_selection.py`,
  `tests/test_pairs_backtest.py`, `tests/test_pairs_integration.py`.
- **Edit:** `src/stockpredictor/__main__.py` (add `cmd_pairs` + `pairs` subparser).
- **No change** to `execution/portfolio.py` (Task-3 variable-exit support already
  covers pairs' variable holding periods).
