# ORB Strategy — Design Spec

**Date:** 2026-09-27
**Status:** Approved for planning
**Author:** brainstorming session (seans134 + Claude)

## 1. Goal

Add an Opening-Range Breakout (ORB) trading strategy to StockPredictor and
back-test it on the existing 5-minute bar store. The ORB trader takes an
intraday breakout of each session's opening range, sized by the existing risk
engine, and is evaluated under the existing walk-forward discipline with the
42-session holdout left untouched.

Stage 1's pre-market outputs feed the ORB trader as an optional **watchlist
filter**: the strategy runs in two comparison arms — *Stage 1-gated* (trade only
names Stage 1 flags pre-open) and *pure ORB* (trade the whole universe) — so we
can measure whether Stage 1 adds value to ORB, mirroring the existing
`cmd_ablation` pattern.

**Success criterion for the build (not for profitability):** a leak-free,
tested, fold-validated ORB backtest that runs both arms, persists an experiment
record + trades CSV, and reports after-cost per-trade and portfolio metrics
alongside the existing baselines. Whether ORB has positive edge is an empirical
result to be read off the holdout, not a build requirement.

## 2. Scope

### In scope
- New rule-based `orb/` package: opening-range + confirmed-breakout signal logic
  and an event-driven per-session trade simulator.
- ATR-based stop (default, swept) and R-multiple target with an end-of-day
  time-stop.
- Two range lengths (15-min, 30-min) and two R-multiples (1R, 2R) as swept config.
- Two universe arms (Stage 1-gated, pure) reported side by side.
- A backward-compatible extension to `execution/portfolio.py` so ORB's
  variable-holding-period trades replay correctly through the existing risk engine.
- A new `orb` CLI subcommand mirroring `cmd_backtest`.
- Unit + integration tests, including explicit look-ahead/causality tests.

### Out of scope (YAGNI)
- Opposite-range-boundary stop is **implemented as a config option but not swept**
  by default (kept as a robustness check).
- No new data sources, no quote/tick data, no news gating of ORB (news stays a
  separate concern).
- No live/paper trading; backtest only.
- No changes to Stage 1/Stage 2 model internals.

## 3. Design decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Opening range | Test **15-min and 30-min** (three / six 5-min bars) | Matches 5-min granularity; both are standard ORB windows. |
| Entry | Stop-entry on the first **bar close** beyond the range boundary that also passes confirmation; **fill at the next bar's open** | No look-ahead; consistent with Stage 2 timing (predict at bar close, act at next open). |
| "Strength" / confirmation | Break clears boundary by ≥ `min_break_atr × ATR` **and** breakout-bar volume ≥ `vol_mult ×` trailing-median volume | Breakout-quality gate is the sole strength signal (per user choice); Stage 1 score is a separate universe filter, not the strength gate. |
| Exit | **ATR stop** (`entry ∓ stop_atr_mult × ATR`) + **R-multiple target** (`entry ± rr × risk`, test 1R/2R) + **EOD time-stop** | Volatility-normalized risk across a heterogeneous 257-name universe; consistent R and sizing. |
| Stop placement (default) | **ATR-based**, swept | Vol-normalization worth one tuned parameter; disciplined by train-only selection + holdout. Opposite-boundary available but not swept. |
| Stage 1 role | Toggleable **watchlist filter**; report **gated vs pure** arms | Directly measures Stage 1's marginal value to ORB (ablation-style). |
| Trades per day | **No hard cap** — every eligible name whose breakout confirms trades; one trade per ticker-session (first confirmed breakout) | User's "trade up to any number of stocks per day if the indication is strong enough." |
| Sizing | Existing `size_position`, fed `q10 = −(|entry−stop|/entry)` | ORB's exact stop maps onto the risk engine's downside slot; `risk_per_trade` is then risked precisely at the stop. No new sizing code. |
| Risk / portfolio | Reuse `run_portfolio` with a backward-compatible extension for variable exit times | One risk engine, not two. |
| Validation | Reuse `generate_folds` / `run_walkforward`; **42-session holdout untouched**; persist experiment JSON + `trades_<name>.csv` | Consistent with and comparable to prior experiments. |

## 4. Architecture

### 4.1 New package `src/stockpredictor/orb/`

**`orb/strategy.py`** — pure, causal signal construction (no I/O, no training).

```
@dataclass(frozen=True)
class ORBConfig:
    range_minutes: int = 15              # opening-range length (15 or 30)
    atr_lookback_sessions: int = 1       # prior RTH sessions used for the ATR scale
    stop_atr_mult: float = 1.0           # stop distance in ATR units
    rr: float = 1.0                      # R-multiple target
    min_break_atr: float = 0.10          # break must clear boundary by this × ATR
    vol_mult: float = 1.0                # breakout-bar volume >= vol_mult × in-session median
    stop_mode: str = "atr"               # "atr" (default) or "boundary"

@dataclass(frozen=True)
class ORBSignal:
    direction: int                       # +1 long, -1 short
    entry: float                         # next-bar open (set by the backtest at fill)
    stop: float
    target: float
    signal_bar: pd.Timestamp             # the confirming bar's start (decision time)

def opening_range(session, cfg) -> tuple[float, float] | None
def prior_session_atr(prior_bars, cfg) -> float | None   # ATR from PRIOR session(s) only
def find_orb_signal(session, atr, cfg) -> ORBSignal | None
```

- `session` is one ticker-day of RTH 5-min bars, ET-indexed, chronological
  (OHLCV; columns lower-case as stored, adapter handles naming).
- `opening_range` = (high, low) over the first `range_minutes` from the session
  open (from the calendar, not inferred from present bars).
- **ATR definition (leak-free by construction):** `prior_session_atr` = mean
  5-minute true range over the previous `atr_lookback_sessions` completed RTH
  session(s). It uses **only bars from days strictly before the trading day**, so
  it is known before today's open and cannot leak. True range uses the standard
  max(high−low, |high−prev_close|, |low−prev_close|). Returns None when there is
  no prior-session data (that ticker-day is then skipped — missing data never
  trades). Passing `atr` in as a precomputed scalar keeps `find_orb_signal` pure.
- `find_orb_signal` scans bars **strictly after** the opening-range window in
  time order. The first bar whose **close** is beyond a boundary by
  ≥ `min_break_atr × atr` **and** whose volume ≥ `vol_mult ×` the median volume of
  the bars seen so far this session (causal, through the confirming bar) fires the
  signal. `stop`/`target` computed from `stop_mode`. `entry` is left as the
  confirming bar's close as a placeholder; the backtest overwrites it with the
  actual next-bar open at fill. Returns the first valid signal or None.

**`orb/backtest.py`** — event-driven fill/exit for one session, then a runner.

```
@dataclass
class ORBTrade:
    ticker: str; date; signal_bar; entry_bar; direction
    entry; stop; target; exit; exit_reason  # "target"|"stop"|"time"
    gross_ret; net_ret; risk_frac_return     # |entry-stop|/entry, for sizing

def simulate_orb_session(session, atr, cfg, costs, ticker, day) -> ORBTrade | None
def run_orb(rth_by_ticker, calendar, cfg, costs, universe_filter=None) -> pd.DataFrame
```

- `run_orb` receives per-ticker **continuous RTH bar frames** (the same
  `rth_by_ticker` `cmd_orb` already builds). For each ticker it walks sessions in
  order, computing each day's ATR from the prior `atr_lookback_sessions` session(s)
  via `prior_session_atr`, then calls `simulate_orb_session` on that day's bars.
  A day with no prior-session data (or outside the `universe_filter` watchlist, in
  the gated arm) is skipped.
- `simulate_orb_session`: find signal (given the precomputed `atr`) → fill at the
  **next bar's open** after the confirming bar → from the fill bar forward resolve
  stop/target/EOD:
  - If a bar's range spans **both** stop and target, **stop fills first**
    (conservative, matches `tjr/backtest.py`).
  - EOD time-stop = close of the last session bar.
  - `gross_ret = direction × (exit − entry)/entry`; `net_ret = gross_ret −
    costs.round_trip_cost`; `risk_frac_return = |entry − stop|/entry`.
  - Returns None if no signal, or the fill bar / a valid stop distance doesn't exist.
- `run_orb` iterates ticker-days (optionally filtered to a watchlist set of
  `(ticker, date)` pairs for the gated arm) and returns a trades DataFrame in the
  **portfolio-compatible schema** (see 4.3).

### 4.2 Trade schema (portfolio-compatible)

`run_orb` emits columns:
`ticker, date, bar_start, direction, gross_ret, net_ret, q10, entry_ts, exit_ts`
where:
- `bar_start` = signal (confirming) bar start,
- `direction` ∈ {"long","short"} (reuse `LONG`/`SHORT` constants),
- `q10 = −risk_frac_return` (so `size_position` risks `risk_per_trade` at the stop),
- `entry_ts` = fill bar start (next bar after signal),
- `exit_ts` = the resolved exit bar's start.

### 4.3 Extension to `execution/portfolio.py` (backward-compatible)

`run_portfolio` currently hardwires the exit at `bar_start + 5·(HORIZON_BARS+1)`
minutes and the entry at `bar_start + 5` min. Change:
- If the trades frame has `entry_ts` **and** `exit_ts` columns, use them for the
  entry and exit timing of each trade.
- Otherwise fall back to the current fixed 15-min computation (existing callers
  — Stage 2 `cmd_backtest` — are unchanged).
- Sizing (`size_position(equity, q10, config)`), concurrency cap, and daily-loss
  halt are unchanged. ORB's `q10 = −risk_frac_return` makes the existing sizing
  risk `risk_per_trade` at the stop.

A regression test pins the fallback path (no `entry_ts`/`exit_ts` → identical
behavior to today).

### 4.4 CLI: `cmd_orb` in `__main__.py`

Mirrors `cmd_backtest`. Per fold:
1. Load extended-hours bars (Stage 1 needs pre-market) and derive RTH bars
   (reuse the `rth_only` helper pattern already in `cmd_backtest`).
2. **Gated arm watchlist:** train Stage 1 on train days (windows → labels →
   classifier → Dirichlet calibrate on the calibrate segment → score test-day
   pre-market snapshots), exactly as `cmd_backtest` does. Watchlist =
   `{(ticker, date) : opportunity_score ≥ min_score}` (no top-K cap).
3. Run `run_orb` twice on the test days: once with `universe_filter=None`
   (**pure**), once with `universe_filter=watchlist` (**gated**).
4. Summarize each arm with the existing `summarize` (per-trade after-cost stats)
   and stitch each arm's trades through the extended `run_portfolio`.
5. Persist one experiment JSON via `run_walkforward` recording every swept config;
   write `trades_<name>_pure.csv` and `trades_<name>_gated.csv`.
6. Print a per-fold and across-fold table for both arms (trades, trades/day,
   hit rate, avg gross bp, avg net bp, total net) plus each arm's portfolio
   final equity / return / max drawdown, side by side.

CLI args (mirroring `cmd_backtest` where shared):
`--start --end --train --test --step --calibrate --embargo --holdout`
`--range-minutes {15,30}` `--rr` `--stop-atr-mult` `--min-break-atr` `--vol-mult`
`--stop-mode {atr,boundary}` `--min-score` `--tickers --name --db`.
Defaults chosen to match the other commands (train 250, test 21, step 21,
calibrate 40, embargo 1, holdout 42).

## 5. Look-ahead / leakage controls (explicit, tested)

1. **ATR** uses only prior-session bars; **opening range and the breakout scan**
   use only bars up to the confirming bar. A truncation test asserts that removing
   all bars after the signal bar does not change a signal that already formed.
2. **Entry** is the bar *after* the confirming bar's close — never the confirming
   bar itself.
3. **Conservative fill**: stop assumed first when a bar spans both stop and target.
4. **Stage 1 watchlist** is decided from pre-market features as-of the 9:00 ET
   cutoff (`build_stage1_features` already enforces this), so it is known before
   the opening range even completes.
5. **Session boundaries** come from the calendar, never inferred from which bars
   are present (matches Stage 2 practice).

## 6. Testing plan (TDD)

**`orb/strategy.py`**
- opening range = high/low of first N minutes; None when bars missing.
- long breakout above high; short below low; direction correct.
- confirmation rejects a break that clears by < `min_break_atr × atr`.
- confirmation rejects a breakout bar with volume < `vol_mult ×` running median.
- no break in the session → None.
- **causality**: truncating bars after the signal bar leaves the signal identical.
- `prior_session_atr` uses only prior-session bars; None when no prior session.
- `stop_mode="boundary"` places the stop at the opposite range side.

**`orb/backtest.py`**
- entry filled at next bar's open after the confirming bar.
- target hit → exit at target, reason "target"; stop hit → "stop".
- bar spanning both → stop fills first.
- no target/stop touched → EOD time-stop at last bar close, reason "time".
- `net_ret = gross_ret − round_trip_cost`; short P&L sign correct.
- `q10 = −|entry−stop|/entry`.

**`execution/portfolio.py`**
- with `entry_ts`/`exit_ts`: a trade held across many bars realizes P&L at
  `exit_ts`, and concurrency counts it open for its full span.
- **fallback**: a frame without those columns reproduces current behavior
  (regression test against a small fixed input).

**Integration (`cmd_orb`, small synthetic or tiny real slice)**
- gated arm trades ⊆ watchlist; pure arm may trade names outside it.
- both arms produce an experiment record and trades CSVs.
- end-to-end runs without error on a short date range.

## 7. Honest caveats (carried into the report)

- **No historical quotes** on the free tier → entry/exit fills and the
  `CostModel` (5 bps spread placeholder) are modeled, not observed. Every result
  is a plumbing check until a paid quote feed exists.
- **~2 years, hand-picked universe** → survivorship bias and few independent
  observations.
- **Swept configs** (2 ranges × 2 R = 4 per arm) are a small search; the recorded
  experiment JSON preserves the count so results are judged against it, and the
  42-session holdout is inspected only once after selection.
- ORB's academic support (Holmberg et al. 2013; Concretum 2024) is moderate and
  explicitly cost-sensitive — the after-cost net figure is the number that matters.

## 8. File-level change summary

- **New:** `src/stockpredictor/orb/__init__.py`, `orb/strategy.py`,
  `orb/backtest.py`.
- **New tests:** `tests/orb/test_strategy.py`, `tests/orb/test_backtest.py`,
  `tests/execution/test_portfolio_variable_exit.py`,
  `tests/orb/test_orb_integration.py`.
- **Edit:** `src/stockpredictor/execution/portfolio.py` (optional entry/exit ts).
- **Edit:** `src/stockpredictor/__main__.py` (add `cmd_orb` + `orb` subparser).
