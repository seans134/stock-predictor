# Issuer-event evidence store

Implemented `stockpredictor.news.evidence` and the bounded `stockpredictor.research.collect_evidence` CLI. The original market/news databases and historical experiment artifacts are unchanged. New local evidence is stored in `data/issuer_evidence/evidence.sqlite` and content-addressed HTML files in `data/issuer_evidence/blobs/`.

## Stored evidence and controls

- Issuer sources bind a canonical URL to one issuer/ticker. A conflicting issuer mapping is rejected.
- Each request records its actual start, receipt, completion, HTTP status and outcome. Failed requests cannot become valid empty source documents.
- Each distinct response body creates a hashed source version. Identical re-fetches add observations but preserve the version's first-observed timestamp. Changed pages create new versions. Archived bytes are checked against their hashes when used.
- Event claims pin exact prior/new source versions, fiscal period, metric, units, guidance ranges, publication claim and proposed event cluster. The current numeric-range schema is a scaffold for revision review, not a general event extractor.
- Reviews append a reviewer, decision, rationale, recorded time and critical checks. Database triggers reject updates/deletes to evidence tables. These controls prevent accidental application-level overwrites; they are not a tamper-proof external archive.
- Evidence readiness requires an accepted review, available/processed evidence, exact timezone-aware publication timing, bounded observation latency, freshness and an unambiguous upward revision. A later rejection supersedes acceptance only from its recorded time. Readiness is separate from trading eligibility; no path here places orders or emits an approved strategy signal.

Required review checks cover issuer, direction, accounting/metric comparability, whether prior guidance is the latest comparable statement, publication time, novelty and event clustering. A draft cluster ID is not proof that syndicated stories have been adjudicated.

## Bounded collection proof

Two collection rounds, 15 seconds apart, fetched the two previously identified DuPont release pages. All four requests returned HTTP 200; the second round returned the same hashes. Result: two sources, two content versions, four observations. Archived HTML was inspected to confirm the prior $2.25–$2.30 and new $2.35–$2.40 adjusted-EPS ranges used in the existing source-audit example.

Seeded event: `69932ff6e6e7a817b735dc9cdba856be6390f483e90bd49c8cbbcce6861ad2ce`. It is a pending FY2026 adjusted-EPS revision claim. The gate reports `critical_review_incomplete` and `exact_publication_time_unverified`; `strategy_eligible` remains false.

This verifies collection and preservation of current observations of historical pages. It does **not** measure latency to a newly published event, reconstruct original first-seen times, complete accounting/novelty review, or establish an H2 edge. No continuous collector or scheduler was installed.

Verification: **191 tests passed**, with the existing joblib physical-core-detection warning. New regressions cover unchanged/changed versions, failed observations, tampered archives, append-only records, backfilled/date-only publication claims, as-of review decisions, stale events and conflicting issuer mappings. [Recorded store status](2026-10-21-issuer-evidence-results.json).

## Commands

```powershell
.venv/Scripts/python.exe -m stockpredictor.research.collect_evidence --store data/issuer_evidence collect --sources docs/research/issuer-sources.json --rounds 2 --interval 15
.venv/Scripts/python.exe -m stockpredictor.research.collect_evidence --store data/issuer_evidence seed --example data/edge_research/event-audit-20261021/primary_source_example.json
```

Collection accepts 1–20 explicitly configured HTTPS sources and 1–3 rounds. Responses are capped at 4 MiB; redirects, non-HTML responses and unsuccessful statuses are recorded as failures. No credentials are required for the configured public releases. Review decisions are available through the store API; the CLI deliberately seeds pending claims only.

## Next implementation

Add issuer-release discovery and structured, source-backed extraction/review of comparable guidance, then observe new releases in a bounded prospective pilot. Keep each entity's identity, fiscal period, metric and accounting basis explicit. Resolve duplicate event clusters before constructing the matched H2 sample. The evidence store is ready for that work; H2 performance evaluation still requires verified events, a registered universe and timing policy, and complete trading data.
