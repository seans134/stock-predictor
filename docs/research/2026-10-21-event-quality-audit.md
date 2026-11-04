# H2 event quality audit

Implemented an outcome-blind audit of news published September 5, 2025 through July 8, 2026 (UTC publication dates, not exchange-session assignments). It reads the news database only, freezes screening/review rules before classification, records input/source hashes, and writes a deterministic sample and candidate ledger. It does not access backtest returns or promote keyword matches to strategy events.

## Results

- 55,655 articles inspected; 4,185 mention guidance, outlook, or forecasts.
- 1,911 guidance mentions receive the legacy classifier's earnings label because earnings has priority. The audit records earnings and guidance separately without changing legacy model inputs.
- 1,164 broad upward candidates; 539 have multiple tagged tickers and 315 contain analyst language. These categories overlap and are not verified issuers/events.
- All 55,655 articles have exactly 900 seconds between publication and stored availability. These are historical assumptions, not measured first-seen or processing times.
- Fixed sample: 12 candidates plus 8 noncandidates. Manual inspection of stored titles/descriptions found 8 of 12 candidates unsupported as upward issuer-guidance revisions. Four contain revision claims needing source and novelty verification. No explicit upward revision was identified in the eight negative-sample descriptions. This small stratified sample does not estimate population precision or recall.

False positives include dividend increases, unchanged guidance, downward forecasts accompanied by increased competition, analyst consensus forecasts, and growth language elsewhere in a story. Merely finding an upward verb and a guidance noun is inadequate. The sample and results are preserved; no regex was tuned to improve this sample's score.

## Primary-source example

DuPont's [February 10 release](https://www.investors.dupont.com/news-and-media/press-release-details/2026/DuPont-Reports-Fourth-Quarter-and-Full-Year-2025-Results/default.aspx) gives FY2026 adjusted EPS guidance of $2.25–$2.30. Its [May 5 release](https://www.investors.dupont.com/news-and-media/press-release-details/2026/DuPont-Reports-First-Quarter-2026-Results/default.aspx) gives $2.35–$2.40 and describes a guidance increase. This supports a same-period upward revision in the issuer's own releases. It does not yet establish exact release timing, original system availability, accounting comparability, absence of intervening guidance, or whether the stored commentary was the first event report. The example remains ineligible for H2.

The current news schema lacks primary URLs, full source/version archives, actual first-seen timestamps, processing completion, and structured prior/new comparable guidance. **Zero events pass the complete verification requirement. H2 returns were not evaluated.**

## H1 missing-interval follow-up

Attempted a bounded one-minute aggregate investigation, saving responses separately. COST September 12 and DE September 10/October 15 requests timed out. FDX October 22 succeeded, but contains zero one-minute bars in the unresolved five-minute interval. The batch was interrupted after repeated connection failures; the remaining four session checks were not completed. No absence was inferred from an error response, and no five-minute candles were fabricated. H1 remains blocked under its frozen coverage policy.

## Next implementation

Build a versioned issuer-event evidence store, preserving source URLs, fetched content hashes, observation time, publication claims, processing completion, fiscal period, metric, units, old/new values, and adjudicated event clusters. Seed it from issuer releases rather than treating article summaries as verified events. A prospective bounded collection run is needed to measure availability; historical first-seen cannot be recovered by assigning a new artificial delay. Keep guidance initiation, reaffirmation, revision, analyst estimates, and dividend changes separate.

Verification: targeted audit regressions and the full suite pass (182 tests). Existing physical-core-detection warning only. Structured audit and sample assessments: [evidence](2026-10-21-event-quality-audit.json). Full candidate ledger and frozen sample: `data/edge_research/event-audit-20261021/`.
