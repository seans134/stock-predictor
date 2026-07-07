# Stock Market Prediction Project

## Project Summary

This project uses a two-stage artificial intelligence and machine-learning system to analyze stock-market data and identify potential intraday trades.

Stage 1 is a fast, direction-neutral classifier that filters a broad stock universe down to a few promising candidates. Stage 2 performs the detailed bullish or bearish analysis and produces the probability, expected return, confidence, entry, exit, and risk information used by the trading strategy.

The system will be developed and evaluated using historical backtesting and real-time paper trading before any live trading is considered. Evaluation will account for transaction costs, spread, slippage, risk, and performance against suitable baseline strategies.

## Initial Timeframe Strategy

The initial Stage 2 trading strategy will use **5-minute inputs, a 15-minute prediction horizon, and 30-minute trend confirmation**.

- **5-minute inputs:** Generate short-term features and trading signals from completed five-minute candles.
- **15-minute prediction horizon:** Estimate the direction, probability, and expected return over the following 15 minutes.
- **30-minute trend confirmation:** Confirm that a proposed trade agrees with the broader short-term market trend.

The model should be allowed to return a hold signal when its probability or confidence is insufficient. Risk controls and realistic execution costs will be applied before converting predictions into trades.

## Stage 1: Pre-Market Opportunity Scanner

Stage 1 will be a fast classification and ranking model that scans a broad, point-in-time universe of eligible stocks before the market opens. Its only purpose is to answer: **Which stocks deserve detailed Stage 2 analysis today?**

Stage 1 will remain direction-neutral. It will not decide whether a stock is bullish or bearish, select entries or exits, size positions, or place trades. Those responsibilities belong to Stage 2 and the risk engine.

High volatility alone does not make a stock predictable or profitable. The scanner will look for significant potential movement while accounting for whether the stock is sufficiently liquid and inexpensive enough to trade.

### Universe Scope

The first version is committed to **US equities traded during the regular New York session**. This is a deliberate scoping decision, not a limitation to work around later.

- **Market:** US-listed common stocks and ETFs (NYSE and Nasdaq primary listings). Foreign, over-the-counter, and non-US-session instruments are out of scope for v1.
- **Session:** Only the regular New York session (09:30–16:00 ET) is traded. Pre-market data before the cutoff is used to build Stage 1 features, but trades and Stage 2 forecasts are confined to the regular session. Extended-hours and overnight trading are out of scope.
- **Timezone:** All timestamps are normalized to US Eastern Time with explicit daylight-saving handling. A single session calendar governs the pre-market cutoff, opening bell, half-days, and holidays.
- **News and filings:** Sourcing is restricted to US-relevant, English-language news wires, company releases, and SEC EDGAR filings, which keeps ticker resolution, event classification, and timestamp integrity tractable.

Focusing on one market and one session keeps the microstructure consistent, so time-of-day normalization, liquidity thresholds, and halt/circuit-breaker rules all refer to a single regime. This produces denser, more comparable examples of each opportunity class than a globally diverse universe would.

Sector remains a **feature and a diversification constraint**, not a universe filter—the universe spans all sectors within US equities, and Stage 1 and Stage 2 use sector-relative movement and concentration limits rather than restricting to one sector. Expansion to other markets, sessions, or asset classes is deferred until the US-session system is validated.

### Universe and Eligibility Filters

Before classification, stocks must pass deterministic eligibility checks:

- Point-in-time membership in the supported trading universe
- Minimum price, average daily volume, and recent dollar-volume requirements
- Acceptable recent and pre-market bid/ask spreads
- Sufficient current data quality and freshness
- No known trading halt or unsupported corporate action
- Broker availability and short availability when relevant to later stages

Historical universe membership, delisted stocks, splits, and other corporate actions must be retained to avoid survivorship bias.

### Potential Inputs

- Overnight price gap and pre-market movement
- Recent 5-minute, 15-minute, and 30-minute returns
- Recent and historical volatility
- Relative trading volume, dollar volume, and available liquidity
- Recent and pre-market bid/ask spread behavior
- Movement relative to the stock's sector and the broader market
- Earnings announcements and other scheduled corporate events
- News event type, sentiment, relevance, novelty, source quality, and surprise
- Options-implied volatility when reliable data is available
- Broader bullish, bearish, volatile, or sideways market conditions
- Missing-data and feature-quality indicators

Raw price levels and ticker identity should not dominate the model. Features should be normalized relative to each stock's history, sector, market, and time of day so the model can generalize across stocks.

All inputs must have been available before a strict pre-market cutoff time. News publication and ingestion timestamps must be preserved to prevent future information from leaking into training. Duplicate and updated news articles must be grouped so one syndicated event is not counted many times.

### Forecast Components

Stage 1 will learn the following direction-neutral components when estimating whether a stock deserves Stage 2 analysis:

- **Abnormal volatility:** Will movement exceed the stock's normal volatility for that time of day?
- **Relative volume:** Will trading activity be unusually high relative to the stock's normal time-of-day volume?
- **Path quality:** Will movement have useful displacement rather than only choppy back-and-forth noise?
- **Cost coverage:** Is likely movement large enough relative to the spread, estimated slippage, fees, and other execution costs?
- **Activity persistence:** Is market interest likely to continue beyond the opening reaction?
- **Event impact:** Is company or market news sufficiently significant, relevant, novel, and surprising to sustain attention?
- **Liquidity:** Is the stock likely to support timely entries and exits of the intended position size without excessive spread, slippage, or price impact?

Liquidity is a critical forecasting component and an eligibility requirement. It can be measured using:

- Bid/ask spread in dollars, percentage terms, and basis points
- Spread stability and frequency of unusually wide spreads
- Share volume, dollar volume, and time-of-day relative volume
- Number of trades and average trade size
- Quoted bid and ask depth when reliable order-book data is available
- Frequency of zero-volume or stale-price bars
- Historical slippage and estimated price impact for the intended order size
- Rolling illiquidity measures, such as absolute return relative to dollar volume

Current pre-market liquidity should not be assumed to continue after the opening bell. The model should compare pre-market conditions with each stock's historical transition into regular-session liquidity and forecast whether acceptable execution conditions are likely during the target window.

## News Processing Pipeline

News processing will initially be a structured feature-engineering pipeline that feeds Stage 1. It will not be trained as a separate trading or news-impact model in the first version.

```text
Raw articles and official filings
→ timestamp validation
→ duplicate and update clustering
→ company and ticker resolution
→ event classification
→ relevance, novelty, surprise, and sentiment extraction
→ existing market-reaction features
→ time-decayed event aggregation
→ Stage 1 classifier
```

### Timestamp Integrity

Each article or filing should preserve:

- **Published time:** Timestamp claimed by the original publisher
- **First-seen time:** First appearance in the selected data feed
- **Ingested time:** Time the system received the content
- **Processed time:** Time all features became available to the models
- **Updated time:** Timestamp for a later correction or revision

Historical tests must use a conservative availability timestamp, normally the processed time. Content discovered or corrected later must not be inserted into an earlier prediction as if it had already been available. All timestamps must use a consistent timezone and handle daylight-saving transitions.

### Duplicate and Update Clustering

Articles that describe the same underlying event should be grouped using normalized titles and URLs, affected entities, event type, publication proximity, semantic similarity, and shared facts. Syndicated copies count as one event supported by multiple sources, not as independent events. Updated articles must be versioned so that later facts do not overwrite the information originally available.

### Company and Ticker Resolution

The pipeline must identify:

- Primary and secondary affected companies
- Relevant tickers, sectors, industries, and broader market exposures
- Company-specific relevance and sentiment when multiple companies appear
- Confidence in every entity-to-ticker association

Low-confidence associations should be rejected or marked as missing rather than assigned automatically.

### Event Classification

Initial event categories should include:

- Earnings and revenue results
- Guidance changes
- Mergers and acquisitions
- Regulatory and legal actions
- Product announcements, recalls, and safety issues
- Clinical trials and regulatory approvals
- Management changes
- Financing, dilution, buybacks, and dividends
- Analyst ratings and price-target changes
- Contracts and partnerships
- Macroeconomic, geopolitical, and sector events

Event type should remain separate from sentiment because events with similar wording can produce different market behavior.

### Extracted News Features

For each clustered event, the pipeline should produce:

- Ticker relevance and extraction confidence
- Event type and scheduled-versus-unexpected status
- Absolute sentiment strength for Stage 1
- Signed entity-level sentiment retained for possible Stage 2 use
- Novelty relative to recent company and sector news
- Surprise relative to expectations, consensus, or previous guidance when available
- Source quality and number of independent sources
- Recency and time-decayed event importance
- Conflicting-report or conflicting-sentiment score
- News and event volume relative to the stock's normal coverage

Missing news means no observed qualifying news in the available sources; it must not automatically be interpreted as negative news.

### Existing Market Reaction

The pipeline should measure how the market has already responded before the prediction cutoff:

- Price movement since the event became available
- Movement relative to the stock's normal volatility
- Pre-market relative volume and dollar volume
- Spread and liquidity changes
- Time elapsed since publication and processing
- Sector and market movement during the same period

The system should not automatically reject an event because the price has already moved. These features allow Stage 1 to learn whether similar events tend to produce persistent activity without assuming continuation or reversal.

### Reliability and Auditability

- Preserve the raw source, extracted fields, feature values, and processing version used for every prediction.
- Record feed outages, missing coverage, delayed articles, and vendor changes.
- Reject stale or incomplete processing results at the prediction cutoff.
- Keep extracted facts traceable to their original text and avoid unsupported generated claims.
- Treat official filings, company releases, news wires, analyst reports, and social sources as distinct source types.
- Verify that historical data licensing permits model training, storage, and backtesting.

### Initial Implementation Scope

The first version should implement timestamps, ticker resolution, duplicate clustering, event type, novelty, entity-level sentiment magnitude, source diversity, extraction confidence, and pre-market reaction. More complex expectation extraction and document embeddings should be added only after this baseline is validated.

### Future Experiment: Dedicated News-Impact Model

A separately trained news-impact model is deliberately deferred. A future experiment may use a pretrained financial text encoder with a small supervised prediction head to estimate:

- Probability that an event materially affects the stock
- Incremental abnormal volatility beyond the price-only baseline
- Incremental abnormal volume beyond the time-of-day baseline
- Probability that attention persists beyond the opening reaction
- Entity-specific directional sentiment and extraction confidence

The news model should predict incremental effects rather than simply learning that historically volatile or heavily traded stocks remain active.

Before adoption, compare the following with identical walk-forward test periods:

1. Stage 1 without news features
2. Stage 1 with the initial structured news pipeline
3. Stage 1 with the proposed trained news-impact model

The dedicated model should be retained only if it improves probability calibration, top-candidate capture, Stage 2 results after costs, and consistency across multiple market regimes. It also requires point-in-time historical news, trustworthy first-seen timestamps, reliable entity mapping, duplicate clustering, sufficient examples across event types, and appropriate licensing.

### Qualifying Opportunity Definition

Each historical regular trading session will be divided into non-overlapping 15-minute windows. A window qualifies as a potential opportunity when all of the following are true:

- **Abnormal movement:** The window's direction-neutral movement is unusually large versus the stock's own history for that time-of-day bucket.
- **Cost coverage:** The movement is a sufficient multiple of estimated round-trip execution cost.
- **Liquidity:** Dollar volume and trade frequency rank high enough within the stock's time-of-day history.
- **Execution quality:** Bid/ask spread and estimated price impact stay under fixed caps.
- **Not an artifact:** The movement is supported by real trading, not a single print, stale quote, or flagged-erroneous bar.

Each criterion is an explicit formula rather than a qualitative judgement. Normalization is per stock and per time-of-day bucket, and every statistic (`median_tod`, `IQR_tod`, percentile references) is estimated on the training portion of the fold only, using availability-time data:

```text
Let r_w   = direction-neutral movement of the window (abs log displacement or realized vol)
Let cost  = round-trip cost in return units (spread + 2·slippage_est + fees)

Abnormal movement:   z_move = (r_w − median_tod) / (IQR_tod + ε)      qualifies if z_move ≥ k_move
Cost coverage:       cost_multiple = r_w / cost                       qualifies if cost_multiple ≥ m_cost
Liquidity:           dollar_vol_pctile ≥ p_vol  AND  trades_pctile ≥ p_trades   (time-of-day percentile ranks)
Execution quality:   spread_bps ≤ s_max  AND  est_impact_bps ≤ i_max
Not an artifact:     trade_count ≥ n_min  AND  active_subbar_fraction ≥ f_min  AND  bar not flagged stale/erroneous

A window qualifies only if all five criteria hold.
```

The constants `{k_move, m_cost, p_vol, p_trades, s_max, i_max, n_min, f_min}` are selected on the training portion of each walk-forward fold and held fixed during that fold's out-of-sample test, under the same fold-isolation rules as every other learned value. Because thresholds are normalized by stock and time of day, no single absolute movement or volume requirement is applied across all stocks.

The label must use fixed definitions and must not assume perfect knowledge of the future high, low, entry, or exit. A qualifying window means the stock deserved deeper analysis; it does not mean that Stage 2 would necessarily have found or completed a profitable trade.

### Classification Target

Stage 1 will use one direction-neutral multiclass classification target:

1. **No qualifying opportunity:** The session contains too few qualifying windows or fails liquidity and execution-quality requirements.
2. **Opening-only opportunity:** Qualifying activity is concentrated in approximately the first 60 to 90 minutes after the opening bell and does not persist sufficiently afterward.
3. **Sustained intraday opportunity:** Multiple qualifying windows occur across the session, including activity beyond the opening period.

The classifier will output a calibrated probability for each class. Exact window counts and thresholds will be selected using training data and then held fixed during each out-of-sample test.

### Candidate Ranking

The scanner will rank eligible stocks using its calibrated class probabilities and pass only sufficiently strong candidates to Stage 2 for more detailed direction, return, and risk analysis.

The probability of any opportunity is:

```text
P(any opportunity) = 1 - P(no qualifying opportunity)
```

Because sustained opportunities are more useful for monitoring throughout the day, the initial ranking score will weight them more heavily:

```text
Opportunity score =
    0.40 × P(opening-only opportunity)
    + 1.00 × P(sustained intraday opportunity)
```

The weights and minimum probability threshold are starting assumptions that must be tuned using walk-forward validation rather than chosen from final test results. A stock must also pass liquidity, spread, data-quality, and concentration requirements before selection.

The scanner will use a minimum opportunity threshold plus a configurable maximum candidate count instead of always returning a fixed number. Ties can be broken using sustained-opportunity probability, liquidity quality, and lower estimated execution cost.

Example output:

```text
NVDA: 8% none, 22% opening-only, 70% sustained
AMD:  14% none, 31% opening-only, 55% sustained
TSLA: 18% none, 47% opening-only, 35% sustained
```

The scanner may reject every stock when none meet the required probability, liquidity, data-quality, and diversification thresholds. Candidate concentration limits may prevent the final list from containing too many highly correlated stocks from one sector or news event.

### Intraday Re-Scan Trigger

Stage 1 runs once before the opening bell and produces a candidate list for the session. Because the scanner classifies the whole session from a pre-market snapshot, it cannot see opportunities that only emerge after the open—for example, a midday news shock, an unexpected volume surge, or a halt resumption in a stock that looked unremarkable pre-market. Freezing the candidate list for the entire day is a blind spot for a system whose main target is *sustained* intraday activity.

Continuously re-running the full Stage 1 classifier over the entire universe every few minutes is rejected for two reasons:

- **Cost:** Stage 1 features (spread stability, liquidity depth, sector-relative movement, news aggregation) are heavier than Stage 2's per-candidate features, and Stage 1 runs over the full eligible universe rather than a handful of survivors. Recomputing that on a timer for thousands of stocks is far more expensive than Stage 2's small continuous loop.
- **Distribution mismatch:** The pre-market classifier is trained on pre-market snapshot features predicting a whole-session label. Feeding it intraday inputs at midday is out-of-distribution and not necessarily valid.

Instead, a cheap deterministic monitor runs continuously over the full eligible universe and only triggers the Stage 1 classifier on the specific stocks that trip it.

#### Deterministic Monitor

The monitor uses fixed, non-learned rules that are inexpensive to evaluate on every completed bar. It only decides *which stocks deserve a fresh Stage 1 evaluation*—it does not classify, rank, or make trading decisions. Trigger signals may include:

- Relative volume or dollar-volume z-score above a stock-and-time-of-day threshold
- Realized-volatility or price-displacement z-score above a threshold
- Bid/ask spread compressing into a tradeable range after being too wide pre-market
- A new qualifying news or corporate-event flag arriving after the pre-market cutoff
- Trading-halt resumption
- Movement relative to sector and market exceeding a threshold

Thresholds are normalized by stock and time of day, consistent with the rest of Stage 1, so one absolute rule is not applied to every stock.

#### Triggered Re-Classification

When a stock trips the monitor, it should ultimately be re-evaluated by the Stage 1 classifier using only information available at that moment, updating the candidate list mid-session. But the pre-market classifier is trained on one pre-market snapshot per day and applying it unchanged at midday is out-of-distribution, so intraday re-classification is introduced in stages rather than all at once.

The committed direction is a **single time-of-day-aware Stage 1 model** rather than a separate intraday variant, because one model keeps pre-market and intraday predictions consistent, shares training data across all evaluation points, and leaves only one artifact per stage to calibrate, monitor, and retrain. A separate intraday-only variant is kept in reserve as a fallback, adopted only if a single model cannot hold pre-market quality while also covering intraday.

- **v1 — deterministic monitor in shadow mode.** The pre-market model owns the candidate list. The monitor runs but only detects and logs intraday trip events under full point-in-time discipline; it does not modify the live candidate list or feed trades. This validates the base system first and, in the process, collects the labeled intraday evaluation examples needed to train the next model.
- **v2 — activated re-classification.** The Stage 1 features are redefined so the same model can run at any time of day with time-of-day and elapsed-session context, and the model is trained on both the pre-market snapshot and the logged intraday evaluation points. Once it passes walk-forward and shadow-mode evaluation without degrading the critical pre-market prediction, the trigger-to-re-classification loop is turned on so monitor trips update the live candidate list.

Throughout both stages, intraday triggering preserves the same leakage discipline: only data available at the trigger time may be used, and news processed after the trigger cutoff must not leak backward.

#### Throttling and Stability

To keep cost bounded and the candidate list stable, the monitor applies:

- A per-stock cooldown so a stock is not re-triggered on every bar
- Deduplication so one ongoing event does not fire repeatedly
- A cap on triggered re-classifications per interval
- Hysteresis so stocks do not oscillate on and off the candidate list near a threshold

#### Validation

The monitor is a filter for the classifier, not a model itself, but its thresholds are still learned or selected values and must be tuned on the training portion of each walk-forward fold—never on final test results. Evaluation should measure how many genuinely qualifying intraday opportunities the monitor captures, how many spurious triggers it produces, and the added computational load, compared with a static pre-market-only candidate list.

## Stage 2: Intraday Return Distribution Model

Stage 2 will analyze only the candidates selected by Stage 1. Its purpose is to forecast the distribution of each candidate's return over the next 15 minutes using completed 5-minute candles and 30-minute trend context.

Stage 2 will predict returns rather than a raw future price. Returns are more comparable across stocks and allow the system to represent direction, expected movement, downside risk, and uncertainty. Forecasted price levels may still be derived from the predicted returns for display.

```text
Stage 1 candidates
→ completed 5-minute market and context features
→ Stage 2 return-distribution forecast
→ cost, liquidity, trend, and risk checks
→ BUY, SHORT, or HOLD decision
→ execution and position-risk engine
```

### Prediction Timing and Target

At each prediction time:

1. Wait for the current 5-minute candle to close.
2. Build features using only information available at that moment.
3. Generate the Stage 2 forecast.
4. Assume entry at the next realistically executable quote, trade, or bar reference.
5. Measure the forecast target over the following three 5-minute candles.

The primary target is the 15-minute forward log return:

```text
Forward return = log(exit reference price / entry reference price)
```

Training and backtesting must not assume that an order can be filled at the closing price of the candle used to create the prediction. Entry and exit references must be defined consistently and include realistic latency assumptions.

### Forecast Outputs

The initial model will estimate the conditional return distribution using:

- 10th-percentile 15-minute return
- Median 15-minute return
- 90th-percentile 15-minute return
- Probability that the return exceeds estimated long-trade costs
- Probability that the return falls below estimated short-trade costs
- Expected maximum favorable excursion during the horizon
- Expected maximum adverse excursion during the horizon
- Forecast confidence and interval width

Example:

```text
10th-percentile return:             -0.42%
Median return:                       0.31%
90th-percentile return:              0.96%
Probability above long-trade costs: 68%
Maximum favorable excursion:         0.74%
Maximum adverse excursion:          -0.28%
Confidence:                         Moderate
```

Forecasted prices can be derived without making raw price the training target:

```text
Forecasted price = reference price × exp(forecasted log return)
```

Maximum favorable and adverse excursion are risk forecasts, not claims that the strategy can execute at the exact future high or low.

### Initial Model Choice

The first implementation will use gradient-boosted quantile regression as the baseline Stage 2 method. It is suitable for structured market features, supports nonlinear relationships, is fast enough for five-minute updates, and can estimate multiple parts of the return distribution.

LightGBM or a comparable quantile-boosting implementation should be evaluated first. If a strict single-artifact Stage 2 model is later required, a joint non-crossing quantile neural network may be tested. Sequence models, recurrent networks, and transformers will be deferred until they demonstrate meaningful walk-forward improvement over the boosted-tree baseline.

### Stage 2 Features

Potential features include:

- Recent 5-minute returns, ranges, candle shapes, and gaps
- Aggregated 15-minute movement and 30-minute trend context
- Distance from VWAP and rolling averages
- Realized volatility, volatility changes, and volatility-scaled returns
- Relative share volume, dollar volume, and volume acceleration
- Bid/ask spread, liquidity, trade frequency, and available depth
- Order imbalance when reliable data is available
- Overnight gap and movement from the previous close
- Stock performance relative to its sector and the broader market
- SPY, QQQ, relevant sector ETF, and volatility-market context
- Time of day and time remaining in the regular session
- Stage 1 class probabilities and opportunity score
- Structured news features and signed entity-level sentiment
- Earnings, company-event, and macroeconomic-event flags

Features should use returns, ratios, time-of-day normalization, and volatility-scaled values instead of allowing raw price or ticker identity to dominate the model.

### Role of 30-Minute Trend Confirmation

The 30-minute trend will be used as a context feature and potential strategy gate rather than as the Stage 2 prediction target. Agreement between the 15-minute return forecast and 30-minute trend may strengthen a signal. Strong disagreement may reduce confidence, reduce position size, or produce a hold decision.

### Confidence and Uncertainty

Stage 2 confidence should consider:

- Width between the lower and upper return quantiles
- Historical coverage and calibration of prediction intervals
- Stability of predictions across recent retraining windows
- Distance between current features and the training distribution
- Liquidity, spread, missing data, and feed quality
- Agreement with broader market and 30-minute context

A narrow interval is useful only when similar historical intervals achieved their intended coverage. Rolling conformal calibration may later be evaluated to adjust the quantile intervals, but coverage must be monitored because market behavior is non-stationary.

### Converting Forecasts into Decisions

Stage 2 supplies forecasts; the strategy and risk engine make trading decisions. A long signal may require:

```text
Probability of net positive return ≥ required threshold
Median forecasted return > estimated trading costs and minimum edge
Lower return quantile remains inside the allowed risk limit
Liquidity and spread remain acceptable
30-minute context is not strongly bearish
```

A short signal will use symmetrical requirements. Otherwise, the correct result is HOLD. Candidate probability thresholds, minimum edge, maximum risk, and trend rules must be selected using walk-forward validation rather than final test results.

The execution and risk engine—not Stage 2—will choose order type, final entry price, position size, stop, take-profit behavior, maximum holding time, and emergency controls.

### Stage 2 Evaluation

Stage 2 forecasting metrics should include:

- Quantile or pinball loss
- Prediction-interval coverage and interval width
- Median-return error
- Direction accuracy after estimated costs
- Probability calibration
- Favorable- and adverse-excursion error
- Results by stock, sector, time of day, Stage 1 class, and market regime

Economic evaluation must include spread, slippage, fees, latency, rejected or unfilled orders, turnover, drawdown, and risk-adjusted return. Compare Stage 2 with zero-return, recent momentum, VWAP, linear regression, and other simple baselines.

## Model Segmentation

Both Stage 1 and Stage 2 use **one global model per stage as the v1 baseline**—every eligible US stock is pooled into a single training set, consistent with the feature design that normalizes each stock relative to its own history, sector, market, and time of day. Pooling maximizes training data, generalizes to newly listed or thinly traded stocks, and keeps one artifact per stage to calibrate, monitor, and retrain.

Per-stock models are explicitly ruled out: intraday examples per ticker are far too few to fit a model each without overfitting, new listings have no history, and thousands of artifacts would collide with the walk-forward and minimum-sample discipline.

Segmentation is permitted only as a **bounded, evidence-gated refinement** of the global baseline, never as the starting point.

### Bounded Split-Out Rule

After fitting the global model, its errors are examined by segment, and at most **N** underperforming segments are split out into their own specialized models. The cap N bounds the number of extra artifacts.

- **Segments, not stocks:** A split unit is a pre-defined group—liquidity tier, sector, or tier-by-sector bucket—never an individual ticker. Splitting individual tickers would recreate the rejected per-stock model.
- **The cap is a ceiling, not a quota:** Being among the top N underperformers only makes a segment *eligible* to split. It is actually split out only if both of the following hold, so fewer than N may be split:
  - The segment clears the minimum-sample requirement on its own.
  - Its specialized model beats the global model *on that segment* out-of-sample on walk-forward folds, after the multiple-comparisons correction.
- **Global is the catch-all:** Every segment not split out, plus new listings and any segment too thin to qualify, is served by the global model.

### Selection Discipline

The underperformance margin, the cap N, and the segment definitions are named quantitative constants selected on the training portion of each walk-forward fold and held fixed during that fold's out-of-sample test, under the same fold-isolation rules as every other learned value. Which segments are split is re-derived at retraining rather than frozen forever, but with hysteresis so the set does not thrash between folds. Stage 1 and Stage 2 make this choice independently. This keeps segmentation subject to the plan's rule that added complexity must demonstrate repeatable out-of-sample improvement before it is accepted.

## Execution and Risk Engine

Stage 2 supplies return-distribution forecasts; it does not place trades. The execution and risk engine converts those forecasts into actual orders and owns everything Stage 1 and Stage 2 deliberately exclude: the cost model, position sizing, portfolio-level exposure, loss limits, and emergency controls. It is the last gate before capital is committed, and its limits protect capital even when the models are wrong.

Consistent with the quantitative-representation principle, every cost, size, and limit below is an explicit numeric value or formula, and every named constant is selected on the training portion of each walk-forward fold and held fixed during that fold's out-of-sample test.

### Cost Model

A single cost model estimates round-trip execution cost in return units and is used identically in backtesting and live trading, through the same point-in-time pipeline, so cost estimates cannot differ between the two:

```text
round_trip_cost = spread_cost + market_impact + fees + borrow_cost(if short)
```

- **Spread cost:** Estimated from the point-in-time bid/ask at the entry and exit references, not an idealized mid.
- **Market impact / slippage:** Estimated as a function of order size relative to available liquidity—for example a square-root impact model scaled by order size divided by interval or average daily volume, with coefficients fit per liquidity tier and time-of-day bucket from historical fills or executions. Impact grows with size, so it is recomputed for the intended position size rather than assumed constant.
- **Fees:** Commissions, regulatory, and exchange fees.

Backtesting must not assume a fill at the closing price of the candle used to form the prediction; entry and exit references and their latency assumptions follow the Stage 2 timing rules. This cost model is the shared source of the cost thresholds referenced by the Stage 1 qualifying-opportunity definition and the Stage 2 decision rules, so all stages price execution the same way.

### Short-Side Costs

A short signal must clear costs that a long signal does not:

- **Availability:** The name must be locatable and shortable at the intended size; if not, no short is taken. This extends the Stage 1 short-availability eligibility check into execution.
- **Borrow cost:** The annualized borrow fee is converted to a holding-period cost over the expected time in the trade and added to `round_trip_cost` for shorts. Hard-to-borrow names carry a higher, more volatile rate.
- **Rate cap:** If the borrow rate exceeds a fixed cap, or availability is unreliable, the short is rejected regardless of forecast strength.

Stage 2's symmetric short requirement is only satisfied after borrow cost is included in the cost threshold, so expensive-to-borrow names need a proportionally larger forecasted edge.

### Position Sizing

Sizing is risk-based rather than fixed. Positions are scaled so that a forecast's downside—its lower return quantile or expected adverse excursion from Stage 2—maps to a fixed fraction of capital at risk per trade, subject to a per-name cap. Size is reduced when Stage 2 confidence is low, when the 15-minute forecast disagrees with the 30-minute trend, or when liquidity or spread is marginal.

### Portfolio-Level Controls

Limits apply across the whole book, not just per trade:

- Maximum number of concurrent positions
- Per-name, gross, and net exposure caps
- Sector and correlation concentration limits, extending the Stage 1 candidate-concentration rule so a single sector cannot dominate the live book
- Per-event exposure limits so one news event driving several correlated names is not treated as several independent bets

### Loss Limits and Kill-Switches

- **Per-position:** A maximum adverse move and a maximum holding time, consistent with the Stage 2 horizon, force an exit.
- **Daily:** A daily loss limit halts new entries for the session once breached.
- **Drawdown de-risking:** Sustained drawdown reduces size or halts trading, and coordinates with the retraining regime-break guard so a crisis reduces exposure rather than triggering reactive retraining.
- **Data-quality kill-switch:** Stale feeds, missing data, or feed-quality failures suspend new entries and can flatten open risk, consistent with rejecting stale processing at the prediction cutoff.

### Decision Authority

The execution and risk engine—not Stage 2—chooses order type, final entry price, position size, stop, take-profit behavior, maximum holding time, and emergency controls. Stage 2 states what it expects to happen; the risk engine decides how much, if any, capital is exposed to that expectation.

## Feature Pipeline and Point-in-Time Correctness

Stage 1 and Stage 2 must be fed by a single feature pipeline that produces identical features in backtest and live operation. Train/serve skew—where historical training features differ from the features computed in production—is one of the most common and hardest-to-detect ways these systems silently fail, and no amount of walk-forward validation catches it because it lives outside the validated code.

A shared code path is necessary but not sufficient. It removes implementation divergence (the same logic computed two different ways) but does not, by itself, prevent point-in-time input differences, look-ahead hidden inside the shared code, or timing and latency mismatches. The actual guarantee is **point-in-time correctness enforced by construction**, with the shared code path as its mechanism.

### One Pipeline Behind an As-Of Interface

Feature code never reads raw data directly. It requests data through an as-of accessor that answers "what was knowable at time T":

```text
get_bars(ticker, as_of=T)
get_quotes(ticker, as_of=T)
get_news(ticker, as_of=T)
```

- In **backtest**, the accessor replays only rows whose availability timestamp is less than or equal to T.
- In **live**, the accessor returns only what has actually arrived by T.

Because the feature functions are identical and both modes see data through the same visibility rule, backtest and live features are constructed to match rather than merely expected to.

### Bitemporal Storage

Every stored observation carries both an event time and an availability time, so "as of T" is a filter rather than a guess:

- **Event time:** when the bar, quote, trade, or event occurred.
- **Availability time:** when the system could first have used it (first-seen or processed time).

The plan already requires this for news timestamps; the same discipline extends to bars and quotes, which can be revised, arrive late, or be partial or stale. Historical features must use the availability time, not the event time, so late-arriving or revised data is never treated as if it had been known earlier.

### Timing and Latency Semantics

Features are always computed "as of time T, using only what had arrived by T." Live computation happens at a real wall-clock moment with ingestion delay; backtest must reproduce that same delay rather than assuming an idealized bar-close instant. Consistent with the Stage 2 timing rules, a prediction may not use the closing information of the candle it is predicting from unless that information was genuinely available at T.

### Skew Canary

A shared code path can be uniformly wrong, and identical backtest and live features do not prove either is correct. Two independent checks guard this:

- **Shadow-mode skew check:** Run the live pipeline in shadow mode and log its features. Recompute the same timestamps through the backtest pipeline over the same raw data and assert the two agree within tolerance. This catches divergence caused by revisions, late data, and stale quotes that shared code alone cannot detect.
- **Look-ahead test:** Feed the pipeline a truncated history and confirm that no already-computed feature changes when future rows are appended. A feature that shifts when the future is revealed is using look-ahead. This catches leakage that a shared code path would otherwise hide by making backtest and live agree on the same optimistic value.

### Versioning

Every prediction records the feature-pipeline version used to produce it, so historical predictions remain reproducible and a pipeline change cannot silently alter the meaning of a stored feature. This complements the reliability and auditability requirements of the news pipeline and applies to all features, not only news.

## Quantitative Representation

Every input and output in the system is a well-defined numeric value or numeric encoding. Every label, threshold, and confidence level has an explicit formula rather than a qualitative description. This makes labels reproducible, thresholds tunable per fold, and the whole pipeline testable, and it closes the ambiguity in phrases such as "unusually large" or "Confidence: Moderate."

Being fully quantitative means well-defined numeric *representations*, not coercing every value into a single continuous number.

### Inputs

- **Categoricals are encoded, not label-numbered.** Event type, sector, and similar categorical inputs are represented with one-hot, frequency, or target encoding fit on the training portion only. They are never assigned arbitrary integers, because that invents a false ordering (e.g., treating an M&A as "greater than" a rating change) that the model would wrongly exploit.
- **Missingness is explicit, never zero-filled.** When a feature is unavailable—no order-book depth, no qualifying news—the pipeline stores a missingness indicator alongside any imputed value. It never silently writes `0`, because `0` and "unknown" are different and conflating them corrupts training. This generalizes the existing rule that missing news must not be read as negative news.
- **Units and normalization are explicit.** Every quantity carries its units and normalization basis—log returns, ratios, basis points, volatility-scaled values, or time-of-day percentile ranks—so values are comparable across stocks and raw price level or ticker identity cannot dominate.

### Outputs

- Every forecast output is a number or a calibrated probability.
- Qualitative labels shown to a human (for example, Stage 2's "Confidence: Moderate") are display mappings derived from a stored numeric value such as interval width and historical coverage. The qualitative label is never the stored or evaluated quantity.

### Thresholds and Labels

- Every threshold and label is an explicit formula with named constants (for example, the Stage 1 qualifying-opportunity criteria and the intraday-monitor triggers).
- Named constants are selected on the training portion of each walk-forward fold and held fixed during that fold's out-of-sample test, consistent with the fold-isolation rules.

## Retraining and Drift Policy

Retraining uses a scheduled cadence as the backbone and triggers as a safety net. The scheduled cadence is primary because the entire validation protocol is walk-forward—train, calibrate, advance the window, refit—so a scheduled retrain is that same advance-and-refit running in production and is the only cadence the backtest actually validated. Purely trigger-driven retraining would make live behavior match nothing that was tested and would lose reproducibility.

Timer-only retraining is still insufficient: a fixed schedule can lag a genuine regime break and can waste a refit when nothing has changed, and refitting on noise can hurt. Triggers close that gap, but trigger-only retraining is dangerous because triggers act on live outcomes—reacting to a drawdown can chase variance, and retraining reactively during a crisis fits the model to a transient that is about to revert.

### Tiered Cadence

- **Scheduled full retrain:** A fixed-cadence walk-forward retrain on the newest training window is the backbone. It advances the same process used in validation.
- **Scheduled recalibration:** The probability-calibration layer, kept separate from the classifier, is recalibrated more frequently than the full retrain. Recalibration is cheap and addresses much of the drift that would otherwise appear as degradation.
- **Triggered off-cycle retrain or recalibration:** Drift or degradation forces an off-cycle refit or recalibration outside the normal schedule.

### Trigger Signals

A forced off-cycle retrain or recalibration may be triggered by:

- Feature drift, for example a population-stability index above threshold or growing distance from the training distribution
- Calibration decay, for example rising Brier score or log loss versus validation
- Stage 2 prediction-interval coverage breaching its intended level
- A sustained gap between live performance and validation expectations
- A spike in missing data or feed-quality problems

Trigger thresholds are themselves selected values and are tuned on training data under the same fold-isolation rules, not chosen from live results after the fact.

### Regime-Break Guard

A trigger that fires during an identified crisis or volatility spike de-risks first—widening thresholds, reducing size, or halting—and defers the refit, rather than retraining into the transient. The correct first response to a regime break is to reduce exposure, not to move the training distribution toward the spike.

### Triggers Force, Validation Promotes

A trigger decides only *when to attempt* a retrain; it never promotes a model. Every candidate model, scheduled or triggered, must pass shadow-mode evaluation before it goes live, and a previous known-good model is retained for rollback. Separating the "when to try" decision from the "whether to trust" decision keeps trigger-driven retraining from reintroducing the selection bias the walk-forward protocol exists to prevent. Each promoted model records its training window, feature-pipeline version, and calibration data for reproducibility.

## Validation Protocol

Validation will measure forecast quality, candidate-ranking quality, downstream Stage 2 value, and live reliability. Random row-level train/test splits are prohibited.

### Chronological Splits

- Keep every stock from the same trading date in the same fold so market conditions cannot leak between training and testing.
- Use rolling or expanding walk-forward splits that reproduce how the system would have been trained and deployed at that time.
- Keep separate training, hyperparameter-tuning, probability-calibration, and test periods.
- Add a gap or embargo wherever target windows overlap. Stage 2 must purge overlapping 15-minute labels.
- Reserve one final chronological holdout period that is not inspected until model and threshold selection is complete.

Example development fold:

```text
Train:      January 2021 – December 2023
Calibrate:  January 2024 – March 2024
Test:       April 2024
Then advance the windows and repeat.
```

Exact durations will depend on available point-in-time data and must provide enough examples of every opportunity class.

### Fold Isolation

Every learned or selected value must use only the appropriate historical training portion of each fold, including:

- Scaling, normalization, imputation, and feature selection
- Stock-relative and time-of-day thresholds
- Class weights and hyperparameters
- Candidate probability thresholds and ranking weights
- News aggregation rules selected from data
- Probability calibration

The probability calibrator must use predictions from data that was not used to fit the underlying classifier.

### Stage 1 Filter Metrics

Stage 1 will be evaluated as a filter and ranker, not as a trading model. Report:

- Multiclass log loss, Brier score, and reliability diagrams
- Precision and recall for no-opportunity, opening-only, and sustained classes
- Precision among the top 5, 10, and 20 candidates
- Percentage of the day's qualifying opportunities captured
- Ranking quality and results by opportunity-score decile
- Days with no selected candidates
- Candidate-list stability, turnover, and sector concentration
- Results across a range of candidate thresholds and maximum list sizes

Classification accuracy alone is insufficient because the classes may be imbalanced and the candidate ranking is more important than predicting the majority class.

### Baselines and Ablations

Compare Stage 1 with:

- Random eligible candidates
- Largest pre-market gaps
- Highest pre-market relative volume
- Highest recent volatility
- A simple hand-built opportunity score
- The same classifier without news features
- The same classifier without each major feature group

The learned scanner must demonstrate repeatable improvement over simple baselines before its additional complexity is accepted.

### Leakage Controls

Explicit checks must prevent:

- News, article revisions, filings, or model features processed after the prediction cutoff
- Future index membership and omission of delisted stocks
- Full-session statistics appearing in pre-market features
- Normalization, imputation, or feature selection using future dates
- Split or corporate-action information unavailable at prediction time
- Multiple stocks from the same date being divided between training and testing
- Candidate thresholds or ranking weights being selected using final test results

### Generalization and Regime Tests

Report performance separately for unseen stocks, market-cap groups, sectors, earnings and non-earnings days, news and no-news candidates, and opening-only versus sustained opportunities. Evaluate bullish, bearish, sideways, quiet, volatile, and crisis periods separately.

Confidence intervals should be calculated by resampling whole trading days or weeks rather than treating individual stock rows as independent observations. Results should remain useful across neighboring thresholds instead of depending on one unusually successful setting.

### Minimum Sample Requirements

Rare regimes are exactly the ones that matter most for risk and the ones most likely to be underrepresented: crisis and high-volatility periods, the sustained-opportunity class, hard-to-borrow shorts, and earnings days. A metric computed on a thin slice can look excellent purely by chance.

- Require a minimum number of independent examples—counted by whole trading days or weeks, not individual stock rows—before a stratified metric is trusted or a model is promoted on the strength of that slice.
- Report the underlying example count alongside every per-stock, per-sector, per-regime, and per-class metric, so thin slices are visible rather than hidden inside an average.
- When a slice falls below its minimum, treat it as low-confidence: do not tune thresholds to it, do not rely on it for promotion, and widen its reported uncertainty accordingly.
- Confidence intervals for these slices follow the same day- or week-level resampling used elsewhere, so a handful of correlated rows is not mistaken for many independent observations.

### Downstream System Value

Compare Stage 2 operating across the full eligible universe with Stage 2 operating only on Stage 1 candidates. Measure:

- Stage 2 signal quality and opportunity capture
- Return after spread, slippage, fees, and rejected or unfilled orders
- Drawdown, risk-adjusted return, and turnover
- Performance by market regime and candidate class
- Candidate-processing time and resource savings

Stage 1 may reduce the total number of opportunities while improving average candidate quality and computational efficiency.

### Experiment and Deployment Discipline

- Record every feature set, model, threshold, and ranking formula tested.
- Do not repeatedly optimize against the final holdout period.
- Use purged or combinatorial validation as an additional overfitting stress test when practical.
- Require historical walk-forward validation, historical replay, live shadow predictions, and paper trading before any controlled live deployment.
- Monitor live probability calibration, feature drift, missing data, candidate distributions, and degradation relative to validation results.

### Selection Bias and Multiple Comparisons

Every feature set, model, and threshold tried across walk-forward folds is a separate experiment, and the best-looking configuration is partly luck. Purged and combinatorial validation reduce leakage but do not correct for the number of configurations searched, so a strong result must be discounted by how many were tried.

- Record the number of configurations, features, models, and threshold settings evaluated, so the size of the search is known when results are judged.
- Apply an explicit multiple-testing correction to strategy-level results—for example a deflated Sharpe ratio, or a reality-check or superior-predictive-ability test—so a selected configuration must remain significant after accounting for the number of trials.
- Prefer configurations that perform well across a neighborhood of settings over a single unusually good point, consistent with requiring results to remain useful across neighboring thresholds.
- Inspect the final chronological holdout only once, and report the trial-adjusted significance rather than the raw best result.

### Stage 1 Design Decisions

The following ideas have been deliberately removed or deferred:

- Stage 1 will not predict bullish or bearish direction; Stage 2 owns direction.
- Stage 1 will not calculate entry, exit, stop, or position size; Stage 2 and the risk engine own execution.
- A fixed number of candidates will not be forced when no stocks qualify.
- Raw ticker identity, raw price level, and a single generic news-sentiment score will not be primary signals.
- Classification accuracy alone will not be treated as evidence that the scanner is useful.
- Future high-low range will not be presented as an achievable trading return.
- Research-oriented data sources will not be assumed suitable for production-quality pre-market trading.
