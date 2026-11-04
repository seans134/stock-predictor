**Market-edge research for StockPredictor — October 20, 2026**

My recommendation is to stop tuning the generic 15-minute direction predictor and investigate a narrowly defined, event-conditioned continuation strategy. Use an opening-volume strategy as its transparent baseline, and a liquid-ETF closing-momentum strategy as a separate, inexpensive comparison. None currently has a demonstrated executable edge in this project. The objective is to discover whether a mechanism survives measurement, costs, and genuinely new observations, including the possibility that all three fail.

This memo combines primary-source research, inspection of the current implementation, all 21 saved experiment JSON files, four saved trade blotters, and news metadata. The calculations are reproducible with [audit_saved_results.py](audit_saved_results.py); numerical output is in [2026-10-20-audit.json](2026-10-20-audit.json). No new strategy was optimized, no reserved holdout price outcomes were opened, and no live trading or data purchase was performed. The bootstrap calculations below revisit existing results and are exploratory, not new validation.

**What our evidence actually says**

These are trade-count-weighted average returns, in basis points per trade. One basis point is 0.01%. They are not portfolio returns, annualized performance, or returns on account equity. Experiments overlap and must not be added together.

| Saved experiment / variant | Trades | Gross bp/trade | Net bp/trade |
|---|---:|---:|---:|
| Wide-universe Stage 2, all stocks | 5,572 | −0.73 | −11.73 |
| Wide-universe Stage 2, Stage 1 candidates | 206 | −0.59 | −11.59 |
| News features, all stocks | 5,393 | −0.55 | −11.55 |
| Fresh-news gate, all stocks | 425 | −1.97 | −12.97 |
| Sentiment-aligned, all stocks | 119 | +2.37 | −8.63 |
| Sentiment-aligned, Stage 1 candidates | 9 | +29.73 | +18.73 |
| Event + sentiment-aligned candidates | 7 | +23.17 | +12.17 |
| Morning-only model | 86 | +7.33 | −3.67 |
| Cost-aware morning model | 92 | +2.10 | −8.90 |
| ORB, ungated | 48,333 | −1.05 | −12.05 |
| ORB, Stage 1 gated | 7,179 | −1.93 | −12.93 |
| Pairs smoke run | 53 | +9.14 | −12.86 |

The Stage 2 rows cover September 5, 2025–April 7, 2026; full ORB covers September 5, 2025–July 8, 2026. The pairs smoke run covers July 9, 2025–July 9, 2026, but the handoff warns that it predates a portfolio fix. Its saved trade arithmetic is descriptive only; its portfolio performance is not validated.

The seven- and nine-trade positive subsets are not sufficient evidence for promotion. They were found among many related experiments and may be driven by individual events. Conversely, these results do not prove that all price signals or all news strategies fail. They establish failure or uncertainty for these implementations, samples, and execution assumptions.

I resampled whole active trading dates in five-date circular blocks, 10,000 times with a fixed seed, preserving same-day cross-sectional dependence. Approximate 95% intervals for mean gross trade return were:

| Strategy | Active dates | Gross interval, bp/trade |
|---|---:|---:|
| ORB, ungated | 209 | −1.60 to −0.49 |
| ORB, gated | 202 | −3.31 to −0.65 |
| Morning-only | 36 | −5.70 to +21.01 |
| Cost-aware morning | 43 | −9.50 to +14.92 |

These intervals are conditional on already selected strategies, use active dates rather than a complete daily equity series, and do not correct for repeated research. They provide useful uncertainty diagnostics, not a statistical certification. They also inherit simulator errors. Still, the economic distinction is clear: ORB needs a better signal; the morning model needs much stronger evidence as well as plausible costs.

For the morning model, replacing the assumed 11 bp cost with 5 bp would make its historical average +2.33 bp. That is a sensitivity calculation, not evidence that 5 bp is achievable. For ORB, even zero cost leaves the observed average negative. A paid quote feed can resolve execution uncertainty; it cannot manufacture expected returns.

**What an edge would have to be**

A useful hypothesis answers four questions: who is trading, why their demand is predictable, why it persists after we can act, and why competition has not eliminated the return after costs. Potential mechanisms include slow interpretation of complicated disclosures, gradual institutional execution, constrained hedging flows, or compensation for providing liquidity. A chart pattern is a measurement of behavior, not itself an economic explanation.

Our plausible advantage is selective participation at small size: we can ignore most securities and sessions, investigate a narrow class of events, and avoid trades whose expected move is too small. Small size reduces capacity constraints but grants no automatic informational advantage. Our current bars and delayed news do not support a credible speed race against professional news and market-making systems.

Expected net return, conditional on information available before the order, is the central object:

`edge(x, action) = E[realized executable return | x, action] − expected all-in costs(x, action)`

Risk limits constrain whether we may take that trade; they do not turn a negative expectation positive. Classification accuracy is also insufficient. A strategy winning 55% of the time with +10 bp winners and −20 bp losers has expected gross return `0.55×10 − 0.45×20 = −3.5 bp`. The current q50 gate estimates a median, and the cost-aware classifier estimates the probability of beating costs. Neither alone estimates expected payoff, including loss severity. A few quantiles also do not uniquely specify a return distribution.

For a simple initial policy, estimate net mean payoff with uncertainty and explicit tail diagnostics; retain calibrated quantiles as risk information. Any later probability model needs conditional win/loss magnitudes too. Prefer a trade only when a conservative estimate of its expected net payoff exceeds zero by a predeclared margin. That conservative estimate should come from adequately populated event groups or validated models, not a fictitiously precise confidence interval for every individual stock.

**What the literature supports—and where it contradicts the thesis**

| Evidence | What transfers to our research | What does not transfer |
|---|---|---|
| Chan, news versus no-news price moves | Event context can distinguish continuation from reversal | Monthly-horizon, older evidence does not establish a 15-minute trading rule |
| Heston–Sinha, news sentiment | Predictive horizon and positive/negative asymmetry matter | A vendor sentiment score is not equivalent to a tradable surprise |
| Ng–Tuna–Verdi, management forecasts | Credibility and the content of guidance can affect underreaction | It does not demonstrate profitability for our universe or timing |
| Zarattini–Barbon–Aziz, stocks-in-play ORB | Opening activity relative to prior sessions is a specific hypothesis | Published backtest returns are not an execution guarantee |
| Gao et al.; Baltussen et al., intraday momentum | Closing flows offer a separate, timed mechanism to test | We cannot infer dealer positions from candles |
| Ng–Rusticus–Verdi, earnings drift and costs | The largest apparent anomaly may sit where capture is expensive | Gross abnormal returns need not be achievable net returns |
| Christensen–Timmermann–Veliyev, earnings jumps | Modern price adjustment can be extremely fast | This does not rule out every event, market segment, or longer horizon |

Chan finds different subsequent behavior for news and comparable no-news moves, with important small-stock and short-sale limitations. His formation and holding periods are much longer than ours. Use the paper to motivate conditioning, not to justify an intraday profit claim. [Author manuscript](https://users.nber.org/~confer/2001/bfs01/chan.pdf).

Heston and Sinha report short predictive persistence for daily news and longer persistence for weekly aggregates, with asymmetric reactions to positive and negative stories. This makes our fixed 15-minute target an assumption to test, rather than an implication of news research. [Federal Reserve paper](https://www.federalreserve.gov/econres/feds/news-versus-sentiment-predicting-stock-returns-from-news-stories.htm).

Research on management forecasts links credibility to how much information is incorporated immediately versus subsequently. The practical hypothesis is about facts, expectations, and credibility—not generic positive language. [Ng, Tuna, and Verdi](https://dspace.mit.edu/entities/publication/d6ee4b92-1865-4304-97bb-965e56cdf26e).

There is substantial contrary evidence to confront. Ng, Rusticus, and Verdi show that transaction costs materially reduce implementable earnings-drift profits and that high-cost firms exhibit larger abnormal returns. [Published paper](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1475-679X.2008.00290.x). A manuscript posted in January 2026 by Christensen, Timmermann, and Veliyev finds that small trading frictions eliminate statistical significance for its post-announcement strategy in 2016–2020. Its main analysis concerns very liquid firms and after-hours earnings reactions, so this is serious counterevidence to easy earnings trading, not a universal rejection of slower drift. [Paper and scope](https://arxiv.org/html/2601.08962v2).

Published effects also decay. McLean and Pontiff study 97 predictors and find substantial deterioration outside original samples and after publication. Their aggregate percentages are not a haircut we can mechanically apply to our strategy; they are a reason to demand contemporary replication. [Journal of Finance](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365).

**First research priority: an auditable opening-volume continuation baseline**

Our existing implementation is not a replication of the prominent stocks-in-play ORB study:

| Dimension | Current code | Published design |
|---|---|---|
| Opening window | 15 minutes | 5 minutes |
| Activity measure | Current bar versus running same-session median | Opening volume versus prior 14 opening windows; top 20 eligible names |
| Direction | Either confirmed boundary break | First candle direction |
| Volatility scale | Mean true range of prior-session intraday bars | Daily 14-session ATR |
| Entry | Confirming close, then next-bar open | Stop entry at opening boundary |
| Exit | Fixed 1:1 target or stop | Stop or session end |

The study explicitly includes commission costs; its headline results do not establish our spread, impact, borrow, and stop-fill costs. The differences above warrant a bounded replication attempt, not adoption of its reported Sharpe or leverage. [Original study](https://alexandria.unisg.ch/server/api/core/bitstreams/3c2989c4-688d-4d78-8a71-f02690990d51/content).

Our own inference: same-session median volume is a different question from whether a company is unusually active today. It can also confound the normal opening-to-midday volume pattern. And a fixed small target truncates the large winners a continuation hypothesis is trying to capture. These are reasons to design a different test, not proof that changing them improves returns.

Build the historical opening-volume ratio using only completed previous sessions and decide eligibility only after today's opening interval is received. Separate a faithful execution-data replication from a five-minute-bar approximation; label the latter explicitly. Do not simulate boundary stop entries with perfect fills from OHLC. First estimate whether candidate moves have gross continuation beyond plausible costs using a causal, conservative entry rule. If there is no gross signal, stop. If there is, quote-level verification becomes worth pursuing.

The first baseline should have no learned Stage 1 gate, no sentiment gate, and no large parameter search. Stage 1 has not yet shown downstream benefit and may suppress exactly the rare cases of interest. Add it only as a separately scored ablation. Keep small size and portfolio constraints independent of the paper's risk allocation.

**Main edge hypothesis: verified business news plus persistent unusual demand**

This is my preferred research direction for the existing project, with low confidence that it will ultimately be profitable. Restrict initially to earnings and guidance events because they offer relatively structured facts. Exclude generic stock commentary, repeated summaries, routine promotional releases, and ambiguous ticker attribution. Treat merger, financing, clinical, and legal events as different mechanisms rather than pooling them into one sentiment bucket.

The hypothesis is: after a substantive, novel update, some liquid stocks continue repricing while investors interpret the information or execute gradually. Opening activity, the move relative to market/sector, and price response after realistic latency may distinguish those cases from fully priced or transient reactions. The competing explanation is equally important: all useful information is already incorporated, and apparent continuation is selection, beta, or execution bias.

Represent an event with separate fields for earnings surprise, revenue surprise, guidance change, metric and fiscal period, source, first observed time, processing completion, novelty, and disagreement. Actual-versus-consensus requires the consensus vintage available before the release. Without that data, use clearly named substitutes such as guidance versus the company's previous guidance, or price-reaction proxies. Do not label them analyst surprises. Earnings and guidance must be allowed to disagree; our priority-ordered keyword classifier can currently classify an earnings-plus-guidance story only as earnings.

SEC submissions and filing APIs are potential sources for official disclosure histories, with accession-level records and correction handling. Filing date alone is not a usable intraday availability timestamp, and SEC filing acceptance may follow an earlier company release. [SEC access documentation](https://www.sec.gov/search-filings/edgar-search-assistance/accessing-edgar-data).

Use a short event study before fitting ML. At a predeclared decision time, compare continuation after qualifying events against price/volume-matched observations without observed qualifying events. Match using only contemporaneously known characteristics: prior volatility, dollar volume, market/sector move, gap magnitude, and opening activity. Lack of observed news is a coverage condition, not proof no news existed. Estimate the event interaction while retaining the price-only baseline; matching is a predictive control, not causal identification of investor underreaction.

Measure actual stock returns and market/sector-adjusted diagnostic returns separately. A residual return is not realizable P&L unless the hedge is traded and its costs are included. Fit exposures on earlier data only. Compare a small, registered horizon family: 30 minutes, 90 minutes, and session end. The primary intraday endpoint should be session end, because the proposed mechanism is persistent repricing rather than a fixed 15-minute fluctuation. The other horizons diagnose decay and count as additional tests. A multi-day branch needs explicit scope expansion, overnight gap handling, dividends, borrow, and a longer purge interval.

Begin with continuous descriptive relationships and simple fixed policies. Only after a repeatable effect appears should a small model estimate which events to skip. Keep event identity together across splits; duplicate stories, multiple affected tickers, and overlapping response windows otherwise create pseudo-replication. Using a contemporary language model to extract historical events also requires guarding against memorized later outcomes: restrict extraction to supplied contemporaneous text, prohibit outcome questions, version outputs, and validate prospectively. More sophisticated text processing is not itself an edge.

Reject this hypothesis if the improvement disappears after realistic delays, publisher deduplication, price/volume controls, or feasible costs. Reject it if the result is solely a few unshortable names, one issuer, one extraordinary week, or one source whose historical annotations cannot be timestamped.

For a concrete first policy to register, I propose the following research specification. These choices are starting assumptions, not fitted or proven optimal:

- Primary event set: company earnings/guidance releases since the previous regular-session close, demonstrably available and processed by 09:45 ET. Earnings/guidance facts must refer to the correct period; contradictory or unresolvable facts are excluded by fixed rules before outcomes are seen.
- Primary direction: long continuation after an unambiguous upward guidance revision relative to the company's previously published comparable guidance. This avoids relying on unavailable historical consensus or short locates. Negative guidance is a separately registered diagnostic until borrow availability can be modeled. The restriction may leave too few observations; that would be a data limitation, not a reason to broaden the definition after inspecting returns.
- Activity: rank the eligible liquid stock universe by first-five-minute volume divided by its previous 14 completed opening-window mean; retain the top 20 above one. Evaluate the event subset within that fixed candidate set, and report how many events the activity filter removes.
- Confirmation: by 09:45, the latest completed candle closes above the first-five-minute high and the opening candle was positive. Submit only after all features are available. For a deliberately delayed bar-only approximation, use the 09:50 opening reference; do not call this a faithful stop-entry ORB replication.
- Initial research endpoint: scheduled liquidation at session end, with an order policy arranged before the applicable closing cutoff. Apply the same predefined emergency risk rules to event and control portfolios. Start the signal study with fixed-horizon outcomes rather than optimizing intrabar stops/targets; subsequently measure whether required risk exits preserve the effect.
- Controls: the identical timing/activity/confirmation policy without the event restriction, plus matched no-observed-event observations. Use equal small research notionals and the same portfolio constraints so differences cannot be attributed to leverage or sizing.

This specification deliberately tests a narrower guidance-continuation hypothesis than either generic sentiment or the published ORB. It would require new event extraction and timestamp validation before its results were credible. If the data cannot support it, do not substitute a generic positive-news label and claim to have tested the same mechanism.

**A separate low-complexity comparison: liquid-ETF closing momentum**

Gao and coauthors document that the market's first half-hour return, measured from the previous close, predicts the last half hour in their historical ETF sample. [Research record](https://profiles.wustl.edu/en/publications/market-intraday-momentum/). Baltussen and coauthors study a broader futures sample and link end-of-day continuation to hedging demand. These are distinct specifications; choose one before testing. [Published author copy](https://academicweb.nd.edu/~zda/intramom.pdf).

For our project, use SPY as a primary instrument and QQQ as a correlated robustness check, not a second independent discovery. First test a fixed closing-window policy using the predeclared return signal, no inferred dealer-gamma filter, and actual entry/exit timing. Compare against unconditional exposure over the same window, cash, and exposure-matched benchmarks. Respect early closes and achievable closing-order cutoffs. If the effect cannot survive feasible spreads and delayed entry, abandon it. This is attractive as a cheap falsifiable experiment, not because the paper guarantees current ETF alpha.

**Measurement problems to resolve before trusting a positive result**

1. **Stop gaps are demonstrably mishandled.** In `orb/backtest.py`, stops are hit only when the stop lies inside the bar's high–low interval. My synthetic reproduction enters at 103 with stop 101; the next bar opens at 98 and has high 100. The simulator ignores the stop and later records a target exit at 105. Correct logic must treat gap-through stops directionally and model the attainable fill, including halts and liquidity. Stop-first resolution when both barriers appear inside a bar does not solve gap-through execution. Stops do not guarantee their trigger price. [SEC investor bulletin](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/investor-bulletins-15).

2. **Availability and fill timing disagree.** The project assumes five seconds of bar ingestion latency, yet uses the immediately following bar's open after a close-confirmed signal. The prior bar cannot be processed five seconds after that open and still trade at that already observed opening print. With only five-minute bars, use an explicitly conservative later eligible bar or obtain finer execution data. The bias magnitude and direction require measurement. Review signal timestamps, entry timestamps, and exits together; an intrabar stop cannot be assumed to occur exactly at the bar start.

3. **Historical news availability is assumed.** All 125,649 articles have `available_ts − published_ts = 900 seconds`. The schema supports filtering, but these are not measured first-seen timestamps. The stored vendor sentiment has no independently established historical processing time. Fifteen minutes may be conservative for some articles and too optimistic for revised/backfilled content. Start a forward availability log and keep original versions; label historical news work provisional. Stage 2 currently counts articles rather than unique events, so syndication can amplify its features.

4. **Universe eligibility needs enforcement.** Saved ORB configuration includes `C:EURUSD` despite the stated US-equities scope. That does not establish that it generated trades, but it shows why the universe needs an explicit instrument whitelist and point-in-time eligibility check. A hand-picked surviving equity list is not a historical tradable universe. Retain delistings and corporate-action histories before claiming broad market performance.

5. **Pairs requires separate accounting.** The implementation uses a beta-estimated log spread for signals but dollar-neutral legs for returns. Those are different exposures when beta is not one. Its return is normalized by one leg's capital; total gross capital is twice that amount for equal legs. Costs, leverage limits, and portfolio sizing must use the same denominator. Cointegration is not a forecast guarantee; market neutrality removes some common exposure while leaving convergence and structural-break risk. The small old smoke run should not drive strategy selection.

6. **Cost assumptions need evidence.** Current single-name cost is 11 bp, with fixed spread, no size-sensitive impact, and no short borrow. Do not choose cost parameters to make results profitable. Estimate them externally or from forward records by liquidity, time, size, volatility, and order type. Limit orders introduce missed fills and adverse selection, not free execution. Only use quotes to refine a strategy after an economically meaningful gross signal appears; precise rejection is valuable too.

These are documented audit findings; production strategy code was not changed during this research request.

**A finite experiment plan**

| Order | Deliverable | Decision it enables |
|---|---|---|
| 1 | Repair stop gaps, align availability and order timestamps, validate universe and return denominators | Can the simulator measure the proposed trade honestly? |
| 2 | Freeze an experiment registry and globally reserved dates; inventory prior holdout exposure | Which observations remain genuinely unselected? |
| 3 | Run the causal opening-volume baseline and separate ETF control | Does either mechanism produce gross returns worth pricing? |
| 4 | Audit earnings/guidance event quality, then run the matched event study | Does event information add value beyond market activity? |
| 5 | Quote/fill and delay stress tests for survivors; simple portfolio replay | Can any signal be captured with our execution and capital? |
| 6 | Frozen prospective paper/shadow policy | Do data timing, selection, and forecast calibration persist forward? |

At registration, specify universe, signal, feature timestamps, decision time, order policy, primary horizon, cost scenarios, risk limits, statistical endpoint, and abandonment criteria. Record every tried variation, including apparently harmless changes such as long-only, a different cutoff, or removing a losing stock. Existing 21 experiment files are not 21 independent hypotheses, nor a complete count of informal research decisions.

Retain chronological date-level splits. For overlapping multi-day labels, purge according to actual label end times; the current one-session embargo is not automatically sufficient. A backward-looking feature using legitimately known history does not itself require purging an entire lookback. Inner training/calibration chooses thresholds; outer periods assess the fixed selection process. Once researchers adapt to those outer results, they become development evidence too.

Audit holdout dates across all experiment families. Reserving the last 42 sessions separately in runs with different end dates does not create a single globally untouched holdout. Forty-two sessions are also often too short to evaluate rare events or regime robustness. Do not consume a holdout to rescue a weak candidate. If no pristine historical block remains, make the final confirmation prospective.

Evaluate a complete daily portfolio return series, including zero-trade days, exposure, cash, simultaneous positions, costs, and financing. Report gross and net expectancy, drawdown, turnover, tail loss, concentration, and capacity. Use dependence-aware uncertainty, with days/weeks and event clusters as resampling units. Forty-eight thousand stock trades across 209 active dates are not forty-eight thousand independent market regimes.

Use selection-aware inference as a supplement. The Deflated Sharpe Ratio addresses performance inflation from multiple testing and non-normal returns; its inputs and trial dependence must be estimated honestly. [Bailey and López de Prado](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf). Probability-of-backtest-overfitting methods provide an additional selection-risk diagnostic when comparable candidate return histories exist. They do not cure leakage, bad fills, or regime change. [Bailey and coauthors](https://www.davidhbailey.com/dhbpapers/backtest-prob.pdf).

My proposed promotion standard is demanding: positive net performance under documented feasible costs; positive incremental value over the appropriate simple baseline; dependence-aware uncertainty that supports positive expectancy after accounting for the registered search; persistence across time blocks; and no dependence on one ticker, date, or implausible fill assumption. Check a predeclared higher-cost scenario, adverse execution delay, and leave-one-event-group-out sensitivity. These are diagnostics, not opportunities to remove losing observations and relabel the result validation. If evidence is insufficient, the status is inconclusive rather than profitable.

For intuition, under independent observations a rough confidence calculation gives `N ≈ (1.96×σ/μ)^2`: a 5 bp mean against 100 bp standard deviation needs about 1,537 observations merely for a nominal 95% interval excluding zero. This is not a power calculation; 80% power would require roughly 3,136 under the same simplified assumptions, before dependence and multiple testing. A 60–90-session paper run can test operations but may still provide little statistical evidence for a rare-event strategy.

**Where I would spend the effort**

First make the measurement trustworthy, then test opening-volume continuation with a simple baseline, then ask whether verified earnings/guidance facts improve it. Keep the ETF closing-window comparison small and independent in design. Defer a larger ML model, additional indicators, broad parameter sweeps, and more elaborate position sizing until there is a stable conditional payoff to learn.

The existing two-stage architecture can still be useful, but its jobs should become explicit: Stage 1 finds credible event-and-liquidity candidates; Stage 2 estimates actionable payoff and whether to abstain; execution verifies capture; the portfolio layer limits correlated losses. Stage 1 only earns its place if it improves downstream performance versus transparent selection.

The opportunity worth investigating is a narrow interaction between information, trading demand, and time. The evidence so far does not establish that opportunity. A successful next phase either demonstrates one with reproducible forward evidence or rejects the hypothesis cheaply enough to redirect the project without years of overfitting.
