# Frozen historical reconstruction policy

Policy identifier: `reconstruction-strict-v1`. Frozen in the recovery plan before downloads and before any reconstructed performance results. This follows the October 21 coverage audit; it is not a claim that these historical observations were available in real time.

## Dataset and provenance

The original SQLite database and every previous experiment remain unchanged. Source responses are saved separately under `data/edge_research/reconstruction-20261021/responses/`, with request ranges, unadjusted-price setting, actual request/receipt times, and hashes in a sealed manifest. Only the 18,340 audited absent timestamps across 46 stocks and SPY may be added. Returned bars outside that list are preserved as source evidence but not used in the reconstruction.

The 53 bounded requests cover the five missing July sessions and the isolated missing intervals. A successful response is not assumed to contain all requested bars. Remaining absences are explicitly counted. Invalid OHLCV, duplicate recovered timestamps, changed evidence, and attempts to replace an original observation are rejected.

Original stored availability timestamps remain labeled as historical assumptions. Recovered rows have actual retrieval times, not backdated availability. The exploratory runner separately models bar completion plus five seconds as the decision-availability assumption. This separation makes a historical simulation possible without implying that the new evidence proves contemporaneous access.

## Missing-data execution rule

Every evaluation session must contain the complete exchange-calendar five-minute grid. An unresolved gap blocks the entire affected strategy experiment before return measurement. No candles are interpolated, prices carried forward, dates shortened, instruments removed after coverage inspection, or selected outcomes deleted.

An absent price interval cannot establish whether a stop triggered or what fill was executable. Adjacent candle prices provide no defensible finite worst-case fill bound for that interval. Therefore this policy does not invent a stop-price fill, a next-observed-price fill, or a convenient no-trade explanation. A different gap policy would be a separately registered experiment with new source evidence and explicit execution assumptions.

This strict rule may block genuine intervals with no eligible prints. That is an acknowledged limitation rather than evidence that the market or provider failed. The previous DE trade-level request was rejected with HTTP 403, leaving the source cause unresolved.

## Experiment rules

Keep the original September 5, 2025–July 8, 2026 evaluation interval, original strategy parameters, universes, costs, and controls. Register fresh runs after sealing and validating the recovery manifest, including the code and data hashes. Each family passes its own coverage gate: SPY may be evaluated if complete even if stock gaps block H1. Preserve blocked attempts and completed results; do not tune parameters based on which reconstruction looks profitable.

Results remain exploratory. Reconstructed availability, static surviving names, corporate actions, unobserved execution costs, and scheduled-close proxies continue to limit interpretation. No result from this reconstruction qualifies as prospective confirmation or authorizes trading capital.
