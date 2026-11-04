# Recovery and fresh strategy evaluation

Recovered **18,330 of 18,340 audited missing bars**, using 53 bounded, rate-limited provider requests. All five missing July sessions were recovered across 46 stocks and SPY. Ten isolated bars remain absent from the current source responses. The original database SHA-256 remains unchanged.

Recovery evidence, request/receipt timestamps, and source hashes are sealed under `data/edge_research/reconstruction-20261021/`. The [frozen policy](2026-10-21-reconstruction-policy.md) allows only additive reconstruction at audited missing timestamps. Historical decision availability remains an explicit assumption, separate from actual retrieval time. No trades or sessions were removed to make coverage pass.

## SPY closing momentum: failed development screen

Fresh experiment: `H3-20261021T164608-2eec63`. Original evaluation dates, parameters, costs, and control were retained. The complete 210-session evaluation now passes the coverage gate.

| Measure | Signal policy | Unconditional closing-window control |
|---|---:|---:|
| Trades | 120 | 210 |
| Mean gross return per trade | -0.113 bp | -0.284 bp |
| Mean net return at assumed 11 bp cost | -11.113 bp | -11.284 bp |

At 2% account allocation per trade, the signal policy returned -0.2664% over the evaluation period. Even its zero-cost account return was negative (-0.0027%). Its improvement relative to the control does not establish a profitable absolute edge. The prescribed decision is **fail this development screen**; do not tune this policy or purchase execution data on the strength of this result.

This completed reconstruction supersedes the earlier incomplete-window H3 report for this historical screen. It remains exploratory: the sample was previously exposed, availability is hypothetical, costs are assumed, and close execution is a proxy. The result is not a statistical proof that all closing-momentum strategies lack an edge.

## Opening-volume continuation: unresolved coverage

Fresh experiment: `H1-20261021T164607-1203eb`. Ten unresolved bars span COST, DE, and FDX. Its economic rules and universe are unchanged. The run stopped before performance evaluation at COST September 12, 2025, with status `blocked_by_data`. Its failed-run artifact is preserved. No H1 expectancy can be inferred from the recovery result.

## Next work

For H1, obtain source-level evidence for the ten intervals, or design and register a different execution model that can represent genuinely missing observations without assuming favorable fills. The previous trade-level provider probe returned HTTP 403, so aggregate re-fetches alone cannot establish what happened inside those intervals. No account upgrade or data purchase was made.

The next independent research candidate is H2's event-quality audit in Gate 4. Its event timestamps, issuer matching, and comparable guidance revisions must be audited before strategy returns are inspected. It should be evaluated on its own hypothesis, not used to rescue the rejected H3 policy.

Verification: **178 tests passed**, with one existing joblib physical-core-detection warning. Meaningful regressions cover whole-session recovery, unchanged original data, actual observation timestamps, changed evidence, duplicate recovery, and incomplete-session rejection.

Compact machine-readable evidence: [results](2026-10-21-reconstruction-results.json). Complete local run artifacts: `data/edge_research/runs/<experiment-id>/`.
