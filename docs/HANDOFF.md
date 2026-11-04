# Handoff — outstanding fixes & improvements

> **Research paused November 4, 2026: no edge found on bar-only data.** ORB, Stage 2,
> H3 and H1 all showed no positive gross return. The items below stay deferred
> until research resumes. See the "Immediate work queue" in
> `docs/EDGE_EXECUTION_PLAN.md` for what would justify resuming.

_Last updated: 2026-11-04 (status, holdout, pairs and paid-quote notes below); the
rest dates from 2026-09-28. For research sequencing, `docs/EDGE_EXECUTION_PLAN.md`
takes precedence over this file. Covers the ORB (merged to `main`) and pairs/stat-arb
(`feat/pairs-statarb`) work. Nothing here blocks merging; these are the deferred
minors and known limitations captured during code review so a later session can
pick them up. Re-verify file:line against current code before acting — line
numbers drift._

## Status snapshot

- **ORB strategy** — built, reviewed, merged to `main`, pushed. Full walk-forward
  result: **no edge** (gross ≈ 0; ~−12 bp/trade net; Stage 1 gating did not help).
  See `docs/trading-strategies-reference.md` and memory `orb-backtest-result`.
- **Pairs / statistical arbitrage** — built, reviewed, on `feat/pairs-statarb`
  (not yet merged). Final review clean; two Important findings (exit look-ahead;
  `run_portfolio` multi-day safety) already fixed and re-reviewed. **Not yet run
  on the full universe with the corrected multi-day portfolio path** — the only
  smoke figure (12 tickers, −1.4%) predates the fix, so re-run before trusting a
  portfolio equity/DD number.
- **Edge research (October 2026)** — H3 failed its gross-return screen, H1 is
  blocked on ten unresolved bars, and H2 has no verified events. See
  `docs/EDGE_EXECUTION_PLAN.md` and `docs/research/`.
- Full test suite: **191 passing** (as of 2026-11-04).

## Deferred — do before trusting pairs results

Deferred until the H1 decision. Register this as a fourth research family that
counts toward the search. The command below uses the older CLI rather than the
research runner, so confirm how it handles coverage before relying on it. Read
gross figures only for now: multi-day positions are not marked to market, so
portfolio drawdown is realized only.

- [ ] **Run the full-universe pairs backtest** now that `run_portfolio` is
  multi-day-safe:
  `python -m stockpredictor pairs --start 2024-07-08 --end 2026-09-21 --name pairs-full`
  Read per-fold `avg_gross_bps` / `avg_net_bps` from
  `data/experiments/<stamp>_pairs-full.json` (gross is the signal; the 2-leg
  modeled cost ≈ 22 bp is the dominant drag).
- **Correction: there is no untouched holdout.** This file used to recommend
  confirming on a 42-session holdout. The October manifest
  (`data/edge_research/split_manifest.json`) records no audited pristine
  historical holdout, so real confirmation has to come from registered future
  data.

## Code polish (non-blocking, ~minutes each)

- [ ] `src/stockpredictor/pairs/stats.py` — add one-line docstrings to
  `ols_hedge_ratio` and `adf_tstat` (param orientation, what `maxlag` does);
  comment the `n - maxlag < 10` minimum-obs threshold; optionally assert
  `len(log_a) == len(log_b)` for a clearer error than numpy's.
- [ ] `src/stockpredictor/pairs/selection.py` — comment the `len(rets) < 10`
  guard; and `tests/test_pairs_selection.py` has an **unused `import pytest`**
  (line ~3) — drop it.
- [ ] `src/stockpredictor/pairs/backtest.py` — the `q10 = -risk_frac_return`
  sizing proxy is in spread-σ units (`risk_frac = (z_stop − z_entry)·sigma`),
  which only approximates the trade's dollar-neutral P&L risk when `β ≠ 1`. Add a
  one-line comment noting it's a sizing heuristic, not a calibrated downside.
- [ ] `src/stockpredictor/pairs/backtest.py:118-119` — the post-loop
  `if pos is not None: close_position(last, "window_end")` is a defensive no-op
  in most paths; keep it, but the comment should say it's defensive rather than
  implying it always fires.
- [ ] `src/stockpredictor/__main__.py:681` (`cmd_backtest`) — `pd.concat(all_trades,
  …)` still raises the pandas empty/all-NA `FutureWarning` when a fold produced no
  trades. `cmd_orb` and `cmd_pairs` already guard this by dropping empty frames
  first; apply the same guard here so test/CLI output stays pristine.

## Refactor worth doing (medium)

- [ ] **Extract a shared loader/label helper.** `rth_only` and the
  extended-hours-bar loading block are now duplicated across `cmd_backtest`
  (`__main__.py:525`), `cmd_orb` (`:759`), and `cmd_pairs` (`:1071`), and the
  Stage 1 train/calibrate/score block is duplicated in `cmd_backtest`/`cmd_orb`.
  The look-ahead-safety logic (as-of cutoff, RTH masking) now lives in three/two
  places that must be kept in sync by hand. Extract e.g. `_load_rth_by_ticker(...)`
  and `_stage1_watchlist(...)` helpers. Do this as its own change with the full
  suite as the guard — it touches merged, working code.

## Test-coverage gaps (add when convenient)

- [ ] `pairs/backtest.py` — no regression test for the entry-guard boundary
  (a signal at `idx == last-1` must be suppressed → zero trades) or the
  `len(df) < 2` short-panel early return.
- [ ] `tests/test_pairs_data.py` — only asserts the happy path; add a case with a
  ticker missing a session to pin the "missing ticker-day stays NaN, never
  zero-filled" constraint.
- [ ] `pairs` CLI `session_ts` is built over the full session list including the
  42-session holdout tail — harmless unused entries, but tighten to `usable` days
  if touched.

## Cross-cutting / strategic (the real ceiling)

- [ ] **No historical bid/ask quotes on the free Polygon tier.** Every backtest
  here (ORB, pairs, Stage 2) uses a **placeholder `CostModel`** (11 bp round-trip,
  no impact/borrow) and modeled fills. Results are plumbing checks, not tradeable
  P&L. Pairs pays 2× that per round trip. **A paid quote feed would be a
  sharper cost measure, not a source of edge**: with gross edge near zero on ORB and
  H3, quotes would only measure costs more accurately. Do not buy any until a
  strategy shows gross edge, and then price only what that strategy needs (E17 in
  the edge plan). Until then, judge strategies on **gross** edge.
- [ ] **Commit-trailer inconsistency** on `feat/pairs-statarb`: a couple of
  commits are stamped `Co-Authored-By: Claude Sonnet 5` (a subagent's own session
  attribution) instead of `Claude Opus 4.8`. Cosmetic; squash-merge would erase it.
- [ ] **`run_portfolio` daily-halt attribution** for multi-day exits is
  approximate (a position realized on its exit day counts toward that day's
  `day_pnl`, resolved in entry-processing order, not a strict event merge). Fine
  for a daily market-neutral book; revisit only if the daily-loss halt becomes
  load-bearing for a strategy.

## How to run things

- ORB: `python -m stockpredictor orb --start … --end … [--tickers …] --name …`
- Pairs: `python -m stockpredictor pairs --start … --end … [--tickers …] --name …`
- Tests: `python -m pytest -q`
- Experiment records land in `data/experiments/`, trade blotters in
  `data/trades_<name>*.csv` (both git-ignored).
