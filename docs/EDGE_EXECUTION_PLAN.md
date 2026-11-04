**StockPredictor: finding and capturing an edge**

Created October 20, 2026. Status: execution repairs and development research runner implemented; gates remain partially open. H3 failed its first gross-return screen; H1 is blocked by missing session data. See [implementation evidence and next actions](research/2026-10-20-implementation-status.md). This is the active sequence for the next research phase. It supersedes the assumption that the generic 15-minute ML predictor should be the next component optimized; the broader architecture in PROJECT_PLAN.md remains background context.

Research basis: [market-edge memo](research/2026-10-20-market-edge-research.md) and [saved-result audit](research/2026-10-20-audit.json).

**Objective and operating rules**

Find one reproducible strategy with positive expected returns after feasible execution costs, then establish that the same policy can operate prospectively. Finding no edge is a valid result. Profitability is an acceptance condition to investigate, not a promised deliverable.

Each gate ends in one of four recorded decisions: pass, fail, inconclusive, or blocked by data. A failed candidate is archived with its results; an inconclusive candidate is not promoted. Dates below are intentionally unspecified: elapsed development time is not evidence of an edge.

Initial scope is regular-session US stocks and ETFs, with long-only execution for the first deployable candidate. Shorts, overnight positions, and leverage require separate data and validation. Historical short-side results may be reported as diagnostics but cannot qualify a long-only policy. No new model complexity until a simple baseline establishes a signal worth modeling.

**Gate 1 — make the backtest trustworthy**

First implementation batch, in dependency order:

- [ ] **E01: Fix gap-through stops.** Update `src/stockpredictor/orb/backtest.py`; add meaningful regressions in `tests/test_orb_backtest.py` for long and short gaps, gap-through targets, and bars spanning both barriers. For stop-market approximations, an opening gap beyond the stop exits at the opening reference plus modeled adverse execution, not at the stop. Document unresolved intrabar ordering and missing bars/halts.
- [ ] **E02: Establish one decision/order clock.** Carry feature availability, decision, order-submission, entry, and exit times separately. Begin with ORB, then reconcile Stage 2 targets and its simulator before comparing their results. A completed candle plus five-second processing delay cannot fill at the immediately following opening print. With five-minute data, fill at the first later eligible bar opening, explicitly labeled an approximation. Do not change only the simulator while leaving training labels on the old timing.
- [ ] **E03: Correct exit chronology.** Use exact bar-opening time only for a gap exit known at the opening. For unresolved intrabar exits, avoid releasing capital at bar start as though the exit were already known. Document a conservative event-time convention shared by portfolio replay.
- [ ] **E04: Validate tradable universe and portfolio accounting.** Enforce instrument type, session calendar, liquidity eligibility, duplicate-position handling, available capital, and corporate actions. Explicitly reject unsupported symbols such as FX. Record survivorship limitations where point-in-time membership is unavailable; that dataset remains exploratory. Defer pairs promotion until two-leg capital and cost normalization is audited.
- [ ] **E05: Run regression checks.** Use `.venv/Scripts/python.exe -m pytest -q tests/test_orb_backtest.py tests/test_execution.py tests/test_portfolio.py tests/test_portfolio_variable_exit.py tests/test_folds.py tests/test_sessions.py`, followed by the full suite after shared execution changes. Add timing regressions in the affected modules. Rerun old strategies only on already exposed development dates.

Deliverable: a reviewed simulator change, passing regression tests, and a before/after report explaining any changed fills and results. Pass only when no order predates required information, stop gaps resolve correctly, and cash/exposure chronology is consistent. This gate establishes measurement, not alpha.

**Gate 2 — protect the evidence before running experiments**

- [ ] **E06: Inventory historical exposure.** Collect train/calibration/test ranges from all saved experiments and record any known informal inspection. Preserve hashes of input files and code revision. Do not infer that a date is pristine merely because one run labeled it holdout.
- [ ] **E07: Create a global split manifest.** Store explicit development, validation, and final-confirmation dates shared across strategies. Unknown historical exposure is treated as development. If no historical block is demonstrably untouched, reserve future observations for final confirmation.
- [ ] **E08: Create an append-only experiment registry.** Each entry records hypothesis, parent/version, universe, timing, feature definitions, entry/exit rules, selection process, primary endpoint, costs, stress scenarios, split manifest, statistical procedure, and decision. Include failed experiments and manual variations.
- [ ] **E09: Freeze inference and review rules.** Use full daily portfolio returns including zero-trade days. Register day/week dependence handling, the family of primary tests, multiple-testing adjustment, sample/power assumptions, review dates, and maximum observation budget before looking at results. Do not repeatedly check significance and stop on the first positive reading.

Deliverables: split manifest and registry entries for the bounded experiments below. Pass when every run can identify its data lineage and whether its results are exploratory or confirmatory. Deflated Sharpe can supplement inference; it is not a replacement for clean splits and executable fills.

**Gate 3 — test simple mechanisms before ML**

Register at most these three initial strategy families. Each change creates a new version and counts toward the research search. Evaluate long-only variants as the initial executable policies; label comparisons with long/short literature accordingly.

| ID | Hypothesis and first implementation | Primary comparison | Decision |
|---|---|---|---|
| H1 | Unusually high opening volume identifies long continuation opportunities. First-five-minute volume divided by the previous 14 opening windows; top 20 eligible names with ratio above one; positive opening candle; close-confirmed upward breakout; causal later-bar entry; scheduled session-end exit. Freeze the entry window and emergency exit rule before running. No Stage 1 ML filter. | Cash and a matched ordinary-activity policy with the same execution and risk rules | Reject if gross returns cannot plausibly cover execution, or net evidence fails. |
| H2 | Verified upward guidance revisions improve a fixed opening-continuation policy. Use the memo's 09:45 decision / delayed 09:50 approximation, prior comparable guidance, and session-end endpoint. | The identical 09:45/09:50 price/volume policy without the event condition, plus matched no-observed-event diagnostics | Require both positive absolute net expectancy and positive incremental value; event and control must share timing and risk. |
| H3 | Early positive market return predicts positive returns near the close. SPY primary; signal from previous close through the first half hour; scheduled entry 30 minutes before the calendar close; scheduled close exit. | Cash and unconditional long exposure over the same closing window | Reject if timing, costs, or unconditional market exposure explain the result. QQQ is a correlated robustness check. |

H1 can begin after Gates 1–2. H2 additionally requires Gate 4. H3 is a small separate comparison, not a fallback parameter sweep. The table is a bounded research agenda, not permission to run drafts with unspecified entry windows or risk rules: complete each registry entry first.

- [ ] **E10: Implement H1 with transparent feature and trade logs.** Separate the bar approximation from any later quote-level replication. Report the effect of each eligibility filter.
- [ ] **E11: Implement H3 with correct holiday/early-close handling.** Obtain the signal before order submission and verify the closing execution convention.
- [ ] **E12: Produce common reports.** Daily equity and exposure, gross/net trade expectancy, number of events and active dates, turnover, tail losses, drawdown, concentration, and confidence intervals. Show zero/2/5/11/20 bp costs as sensitivity scenarios, clearly distinguished from measured feasible costs. Do not use the best-looking cost scenario as the headline.

Deliverable: reproducible reports and one explicit decision per family. A positive gross result can justify execution-data work; it cannot authorize capital deployment. If both cheap baselines fail, retain the findings and assess H2 only on its distinct economic hypothesis, not on an assumption that adding news must rescue H1.

**Gate 4 — establish whether event information adds value**

- [ ] **E13: Build an event-quality dataset.** Store source text/version, issuer, event identity, publication/first-seen/ingestion/processing times, fiscal period, metric, prior and new guidance, comparability, extraction confidence, and exclusion reason. Keep earnings and guidance as separate fields. Historical publication-plus-delay remains an assumption; do not silently relabel it measured availability.
- [ ] **E14: Audit a fixed sample before inspecting returns.** Manually verify issuer, event type, direction, fiscal period, novelty, and source availability. Predefine acceptable error rates; critical timestamp or issuer errors block promotion. If insufficient genuine guidance events remain, report inadequate sample coverage rather than broadening the definition after seeing P&L.
- [ ] **E15: Capture prospective availability.** Implement persistent logging for original articles and revisions, processing completion, and unique event clusters. First demonstrate a bounded collection run; sustained operation needs a dependable host and feed. Do not presume a chat remains running indefinitely.
- [ ] **E16: Run H2 and matched diagnostics.** Match using information available at decision time; group syndicated stories and related issuer events. Primary outcome is the registered session-end policy; secondary horizons are separate diagnostic tests. Record all excluded and untraded events.

Deliverable: audited event dataset and paired comparison report. Pass only if the event restriction improves net outcomes with adequate uncertainty control, not merely because it picks volatile stocks or a handful of winners. If historical timestamps are unverifiable, results remain provisional until prospective confirmation.

**Gate 5 — prove the signal can be captured**

- [ ] **E17: Price the actual execution requirement.** For a surviving candidate, determine the required quote/trade history, timestamps, coverage, corporate actions, and licensing. Assess sample coverage before any data purchase. Do not buy broad data merely to rerun a signal with negative gross expectancy.
- [ ] **E18: Replay realistic orders.** Incorporate bid/ask, order size, observed latency, impact assumptions, fees, cancellations, missed fills, and adverse selection. Model limit orders only with defensible fill rules. No guaranteed stop prices or fills on price touches.
- [ ] **E19: Stress the frozen policy.** Test the registered higher-cost and delay scenarios; review concentration, extreme-event contribution, capacity, and market/sector exposure. Diagnostic removal of a dominant event is not permission to delete losing trades from the primary result.
- [ ] **E20: Make a promotion decision.** Require selection-adjusted evidence of positive net expectancy, useful performance against the registered baseline, acceptable risk under predeclared limits, and adequate sample coverage. Document all uncertainty. Freeze the candidate before final confirmation.

Deliverable: execution feasibility report and a frozen candidate configuration. Fail if the apparent edge relies on costs or timing the intended system cannot achieve. A better quote model does not create a signal.

**Gate 6 — operate prospectively without capital exposure**

- [ ] **E21: Implement shadow signals.** Run the exact frozen data, selection, and decision path prospectively. Archive signals before outcomes. Track data outages, processing latency, revisions, and rejected candidates.
- [ ] **E22: Connect paper execution.** Reconcile order acknowledgments, cancellations, positions, and P&L. Compare paper fills with contemporaneous quotes; broker paper fills can be optimistic and are not proof of live execution.
- [ ] **E23: Add operational controls.** Reject stale data and duplicate orders; enforce configured name/sector/gross exposure and loss limits; reconcile broker versus internal positions; test disconnects, restarts, and the emergency stop. Specify whether a halt cancels orders, exits positions, or both.
- [ ] **E24: Review at registered checkpoints.** Compare realized net results, fills, activity, and uncertainty with historical expectations. A 60–90-session operational trial is a planning checkpoint, not a sufficient sample by definition. Changes create a new policy version and require new prospective evidence.

Deliverable: a reproducible prospective record, fill-quality report, and operational incident log. Pass only when both operational reliability and the registered evidence standard are met. If volume is too low, remain inconclusive or stop at the predeclared observation budget.

**Gate 7 — act on a confirmed edge with controlled deployment**

- [ ] **E25: Prepare a concrete deployment packet.** Frozen strategy version, supporting evidence, selected broker/account and supported order types, data/feed requirements, maximum capital allocation, per-position and total exposure, loss/drawdown limits, expected execution range, stop conditions, and recovery procedure. Account eligibility and current broker rules must be verified at that time.
- [ ] **E26: Obtain explicit authorization for real-capital execution.** This plan authorizes research planning; it does not place trades or activate a live broker. Prepare the working paper-tested setup before asking for live authorization. Capital and acceptable loss are user decisions, not values inferred from a backtest.
- [ ] **E27: Use a capped pilot.** Measure actual fills against the frozen model. Do not increase allocation to recover losses. Scale only at predeclared reviews after sufficient live evidence and acceptable capacity; any increase stays within the approved allocation or requires a new authorization.
- [ ] **E28: Maintain the edge ledger.** Monitor expectancy uncertainty, execution drag, event coverage, exposures, and operational incidents. Immediate operational failures trigger the defined halt. Statistical degradation is reviewed using the registered monitoring rule, not an improvised response to every loss.

Deliverable: a controlled, auditable deployment—or a documented decision to remain in paper mode. A strategy's purpose is to produce risk-adjusted net value under actual constraints; maintaining trading activity is not an objective.

**Immediate work queue**

Issuer-evidence foundation implemented: [store and bounded collection proof](research/2026-10-21-issuer-evidence-store.md). Two historical issuer sources archived across four observations; one draft event remains blocked by review/timing checks. Next is issuer-release discovery, structured guidance review, and a bounded prospective pilot—not H2 performance evaluation yet.

H2 audit started: [event-quality findings](research/2026-10-21-event-quality-audit.md). Broad keyword candidates fail source/timestamp verification; 8 of 12 sampled candidates were unsupported in their stored text. Next independent implementation is a versioned issuer-event evidence store and bounded prospective availability collection. No H2 return test is justified yet.

Recovery completed: [fresh reconstruction results](research/2026-10-21-reconstruction-results.md). All five July sessions were recovered without changing the original database. H3's complete-window reconstruction fails the gross-return screen; ten unresolved stock bars still block H1. The strict [missing-data policy](research/2026-10-21-reconstruction-policy.md) was frozen before the fresh runs.

October 21 update: [coverage audit and source checks](research/2026-10-21-data-quality-status.md) completed; research runs now fail on incomplete evaluation coverage. Five missing sessions also invalidate the earlier H3 report as a complete-window evaluation. Both families remain blocked pending reconstruction and a registered missing-data policy.

1. E04/E10: audit and repair missing H1 session data; DE September 10, 2025 is the first confirmed blocker.
2. E10/E12: register a new unchanged-policy H1 experiment only after data repair; preserve failed attempts and compare with the ordinary-volume control.
3. E04: finish corporate-action and point-in-time universe checks; retain exploratory status until resolved.
4. E07–E09: register prospective confirmation dates, sample requirements, multiplicity, and review rules before any confirmation claim.
5. E13–E14: assess event data for H2 independently of the failed H3 screen; do not begin a model sweep.

Do not start a new model sweep while these prerequisites are unresolved. Update this checklist with evidence links and decisions as work is completed; checking a box requires its deliverable, not merely code being written.
