# Execution repair and first development experiments

**October 21 correction:** the subsequent [coverage audit](2026-10-21-data-quality-status.md) found five wholly missing SPY evaluation sessions. The original H3 report below is preserved, but its daily statistics and complete-window decision are incomplete and must not be treated as a validated evaluation.

Implementation covers the first three gates of the edge plan, with remaining prerequisites below. No strategy is approved for capital deployment.

Verification: full test suite passed, 168 tests. One joblib warning reports unavailable physical-core detection and falls back to logical cores.

## Implemented

- Shared five-minute execution clock: completed signal bar plus five seconds, entry at signal-bar start plus ten minutes. Stage 2 labels and simulated returns now share arithmetic returns and the same entry/exit clock.
- Opening gaps through stops resolve at the opening reference; ambiguous intrabar barriers use stop-first ordering. Intrabar exits release capital at bar end.
- Portfolio replay processes actual entry/exit events, rejects duplicate constituents, limits gross capital, and includes initial equity in realized drawdown. Short sizing uses the adverse upper-return quantile.
- Development-only research runner with input/code hashes, preserved failed runs, experiment registrations, and a global exposure manifest. Saved experiments expose the interval July 8, 2024 through July 9, 2026; no historical confirmation set is claimed.
- H1 opening-volume continuation and H3 SPY closing momentum, their controls, zero-trade calendar days, cost sensitivity, and descriptive block-bootstrap intervals.

## First runs

Registered evaluation window: September 5, 2025 through July 8, 2026. These are exploratory observations on exposed development data.

| Experiment | Result | Decision |
|---|---|---|
| H3-20261020T233844-3988d3 | 118 trades; mean gross -0.287 bp, mean net -11.287 bp under assumed 11 bp costs; account return -0.266% at 2% allocation per trade | Fail this development screen; no observed gross edge |
| H1-20261020T233844-aba28e | Missing timely history for registered ticker JNJ | Failed run preserved; no performance result |
| H1-20261020T233927-e760c7 | Pre-evaluation coverage screen excluded JNJ, MRK, ABBV, MU; evaluation then stopped on incomplete selected DE session, September 10, 2025 | Blocked by data; no performance result |

Read-only inspection confirmed DE's 12:35 America/New_York five-minute bar is absent entirely on that date, rather than merely excluded by the availability filter. The evaluator deliberately refuses to silently discard this selected outcome.

Full local artifacts live under `data/edge_research/`: `split_manifest.json`, `registry/<id>.json`, and `runs/<id>/`. Completed runs include reports, daily returns, and trade logs; failed runs include `failed.json`. Original historical result files were not rerun or rewritten. The updated synthetic stop audit is `2026-10-20-audit-current.json`.

## Next actions, in order

1. Audit regular-session completeness across the registered H1 universe and dates, producing a missing-bar ledger before any further performance evaluation. Distinguish genuine no-trade intervals/halts from feed omissions using source evidence.
2. Recover verifiable missing bars from the existing provider where possible. Preserve original data and availability provenance; do not fabricate candles or backdate newly observed availability. If reconstruction is impossible, document a prospective missing-data policy before a new experiment; do not remove losing or incomplete selected trades after observing outcomes.
3. Register a new H1 run against the repaired dataset and code hashes, with unchanged economic rules. Preserve both failed attempts. Review gross and net results once against the ordinary-volume control.
4. Only if evidence warrants it, investigate execution feasibility. H3's current negative-gross result does not justify parameter optimization or purchasing quote data to rescue it. H2 requires the separate event-quality audit in Gate 4.

## Remaining limits

Gates 1–3 are not fully passed. Static surviving-name membership, corporate actions, synthetic historical availability, missing bars, close-auction execution, quote costs, and intraday mark-to-market risk remain unresolved. Legacy feature paths have not all been audited for historical revisions. Bootstrap intervals are descriptive, not selection-adjusted confirmation; a prospective sample, power/review rules, and multiplicity policy still need registration. The runner cannot promote strategies to live trading.

## Reproduction

From the repository root in PowerShell:

```powershell
.venv/Scripts/python.exe -m pytest -q --basetemp=.pytest_cache/edge-verification-new
.venv/Scripts/python.exe -m stockpredictor.research prepare --start 2025-09-05 --end 2026-07-08 --family H1
.venv/Scripts/python.exe -m stockpredictor.research run <new-registration-id>
```

Prepare creates a new registration; it does not repair data. Run refuses changed source/data hashes and an existing output directory. Do not rerun H1 until the completeness issue is addressed. Source or data changes require a new registration rather than modification of an old experiment.
