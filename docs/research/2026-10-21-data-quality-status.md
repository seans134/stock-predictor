# Coverage audit and source investigation

The next implementation batch completed the session-coverage audit and added a mandatory coverage gate before research performance evaluation. Full suite: 172 passed; one existing joblib core-detection warning. No trading database rows were changed.

## Findings

H1's 46 registered stocks were audited from July 22, 2025 through July 8, 2026, including warmup: 11,132 ticker-sessions, 238 incomplete, 17,950 absent bars, and no gaps attributable solely to late stored versions.

- All 46 stocks lack July 1, 2, 6, 7, and 8, 2026: 230 entirely absent ticker-sessions.
- Eight other sessions contain ten absent bars: COST September 12, 2025; DE September 10 and October 15, 2025; FDX October 22, 23, December 26, 30, 2025 and April 27, 2026.
- SPY also lacks those five July dates. The previously completed H3 report therefore has incomplete evaluation coverage. Its observed-trade average remains an observation, but the daily statistics and decision are not a valid complete-window evaluation. The original artifact is preserved; this notice supersedes its completion status for research decisions.

The provider returned all 78 regular-session AAPL bars for July 1. This demonstrates that at least one whole-session gap is recoverable from current provider history. It does not establish coverage for every missing ticker/date or prove historical availability.

The provider's fresh DE September 10 response still lacks the 12:35 Eastern bar. A trade-level request for that interval returned HTTP 403, so we could not determine whether the gap reflects trade eligibility or another source issue. This is a provider rejection, not an automatic approval-review rejection. No subscription upgrade was made.

[Massive's aggregate documentation](https://massive.com/knowledge-base/article/why-are-there-missing-aggregates-in-massives-data) explains that intervals without eligible trades can lack bars. An absent bar alone does not prove a halt, an ingestion error, or a tradable unchanged price.

## Implemented controls and artifacts

- `research.coverage` reads only timestamps, using a consistent SQLite snapshot, the exchange calendar, registered universe/date boundaries, input hashes, and immutable output directories. It emits all missing intervals and per-session/per-ticker summaries. It deliberately does not calculate returns or change eligibility based on future coverage.
- Research runs now reject incomplete evaluation coverage before calculating performance. This strict gate includes genuine no-print intervals until a source-supported handling policy is separately registered. Missing entire sessions can no longer masquerade as zero-trade days.
- `research.provider_probe` saves bounded aggregate or five-minute trade responses separately, with observation timestamps, no historical-availability claim, and no database mutation. Credentials and response URLs are not logged. HTTP failure is not interpreted as an empty valid trade response.
- Compact evidence: [audit summary](2026-10-21-coverage-audit.json). Full ledgers: `data/edge_research/audits/coverage-20261021/` and `spy-coverage-20261021/`. Source responses: `data/edge_research/provider_checks/`.

## Next decision

Keep H1 and the complete-window H3 evaluation blocked. Recover the missing July sessions into a separate reconstruction dataset with retrieval provenance. Independently obtain evidence for the ten isolated missing intervals, or pre-register a conservative missing-data execution policy and explicitly label the reconstructed study exploratory. Do not forward-fill prices, delete selected trades, shorten the evaluation window after seeing performance, or assign recovered data a fabricated contemporaneous availability time. The existing loader intentionally cannot treat newly observed historical data as contemporaneous evidence.

Before another performance run, the reconstruction policy must specify how historical bars are represented separately from first-seen timestamps, how uncertain stop paths are bounded, and what coverage blocks evaluation. Re-register both strategies if their inputs or source change; keep the previous results and failures intact. This audit provides no evidence of a profitable strategy.

## Commands

```powershell
.venv/Scripts/python.exe -m stockpredictor.research.coverage data/edge_research/registry/H1-20261020T233927-e760c7.json --output data/edge_research/audits/<new-directory>
.venv/Scripts/python.exe -m stockpredictor.research.provider_probe DE 2025-09-10 --output data/edge_research/provider_checks/<new-file>.json
.venv/Scripts/python.exe -m stockpredictor.research.provider_probe DE 2025-09-10 --trade-window 12:35 --output data/edge_research/provider_checks/<new-trade-file>.json
```

Provider calls use the existing account and can be subject to its access limits. The probe does not paginate: `has_more=true` means the saved response is incomplete and cannot establish a full-window absence.
