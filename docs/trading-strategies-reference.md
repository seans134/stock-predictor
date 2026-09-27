# Trading Strategies Reference

A survey of well-researched trading strategies — academic factors and practitioner
intraday setups — with an honest read on the *quality of the evidence* behind each,
its strengths and failure modes, and **how feasible it is to make work inside this
specific project** (StockPredictor: free-tier Polygon 5-min consolidated bars, ~2yr
history, Polygon news + SEC EDGAR, two-stage ML, cost-aware non-overlap simulator,
walk-forward validation).

Two things this document deliberately does: it separates *does the effect exist in
the literature* from *can we capture it here*, and it refuses to dress up
practitioner lore as peer-reviewed fact.

---

## How to read this doc

### Evidence grade — how well-supported is the effect itself?

| Grade | Meaning |
|-------|---------|
| 🟢 **Strong** | Replicated across decades, markets, and asset classes in top journals; survives out-of-sample. |
| 🟡 **Moderate** | Published and real, but fragile: decays post-publication, regime-dependent, or heavily cost-sensitive. |
| 🔴 **Weak/Anecdotal** | Mostly practitioner/educator lore or a single fragile study; little rigorous, independent support. |

### Feasibility grade — how likely can *we* make it work on this stack?

This is the rating you asked for. It blends four things: (1) do we have the data,
(2) does the horizon fit ~2 years of history, (3) how cost-sensitive is it given we
have **no historical bid/ask quotes** on the free tier, and (4) does the effect even
have an edge to capture. It is deliberately harsh — "buildable" is not "profitable."

| Grade | Meaning for this project |
|-------|--------------------------|
| 🟢 **High** | Data + horizon fit our stack; realistic to backtest honestly and plausibly find edge. |
| 🟡 **Medium** | Buildable but with a real blocker — short history, cost sensitivity, or a signal we can only proxy. |
| 🔴 **Low** | Needs data we don't have (quotes/fundamentals/estimates), a horizon our history can't validate, or the edge is absent/eaten by costs. |

> **Standing caveat for every backtest here.** Our free tier has no historical
> quotes, so fill prices and spreads are modeled, not observed; our universe is
> hand-picked (survivorship bias); and 2 years is a *short* window. Treat every
> result as a plumbing check until a paid quote feed exists. Our own logs already
> show **no directional edge from price/volume bars alone** — that finding shapes
> the feasibility grades below.

---

## Part A — Academic / systematic strategies

Longer horizon (days to months), deep evidence base, mostly cross-sectional.

### A1. Cross-sectional momentum
- **Thesis:** Stocks that outperformed over the past 3–12 months keep outperforming over the next 3–12 months. Buy winners, short losers. Behavioral (underreaction) + risk-premium explanations.
- **Evidence:** 🟢 Strong. Jegadeesh & Titman (1993) — ~1.5%/month on a long/short winner–loser spread over 3–12m holds; replicated globally and across asset classes for 30+ years.
- **Strengths:** Robust, well-documented, low-to-moderate turnover, works cross-sectionally on equities.
- **Weaknesses:** Rare but savage "momentum crashes" (sharp rebounds off market bottoms); partially reverses beyond 12m; some post-publication decay.
- **Feasibility here — 🟡 Medium.** Data is fine (daily bars from our 5-min store), turnover and cost sensitivity are manageable. Blocker: **~2yr history gives very few independent monthly observations**, and our hand-picked 257-name universe is survivorship-biased — enough to prototype the signal, not enough to *validate* an edge with walk-forward honesty.

### A2. Time-series (absolute) momentum / trend-following
- **Thesis:** An asset's own past 12-month return predicts its next-month return; go long positive-trend, short negative-trend. The core of managed-futures/CTA trend.
- **Evidence:** 🟢 Strong. Moskowitz, Ooi & Pedersen (2012, JFE) — consistent across 58 futures/forwards (equity indices, FX, commodities, bonds) over 25+ years; effect persists ~12m then partially reverses.
- **Strengths:** Diversifies across assets, historically strong crisis-period ("crisis alpha") behavior, simple rules.
- **Weaknesses:** Cleanest on a broad multi-asset futures universe; single-stock equity trend is noisier. Whipsaws in choppy, range-bound regimes.
- **Feasibility here — 🔴 Low.** The evidence lives in **futures/multi-asset**, which we don't trade, and a 12-month lookback against 2yr of single-name equity history leaves almost nothing to test out-of-sample. Interesting to know, not a near-term build.

### A3. Value
- **Thesis:** Cheap stocks (low price relative to book, earnings, or cash flow) outperform expensive ones over the long run.
- **Evidence:** 🟢 Strong historically. Basu (1977); Fama & French (1992, 1993) value factor (HML). Note: value has had a long, well-publicized drawdown post-~2007, so the *live* premium is contested.
- **Strengths:** Deep theoretical + empirical base, low turnover, intuitive.
- **Weaknesses:** Long horizon (annual+), can underperform for a decade, needs clean fundamental data.
- **Feasibility here — 🔴 Low.** Two hard blockers: value needs **fundamentals** (book value, earnings) — we'd have to build a whole pipeline off EDGAR filings — and it's an **annual-horizon** effect that 2 years of history simply cannot validate. Wrong tool for an intraday stack.

### A4. Low-volatility / betting-against-beta (BAB)
- **Thesis:** Low-beta / low-volatility stocks earn higher *risk-adjusted* returns than high-beta ones — the opposite of what CAPM predicts.
- **Evidence:** 🟢 Strong. Frazzini & Pedersen (2014, JFE): BAB Sharpe ~0.78 for US stocks 1926–2012, and the effect appears in international equities, bonds, and FX. Explained via leverage/funding constraints.
- **Strengths:** High risk-adjusted returns, defensive, works across asset classes.
- **Weaknesses:** Long-short construction leans on leverage and shorting low-beta names; long horizon; crowded.
- **Feasibility here — 🔴 Low.** Long-horizon, low-turnover, leverage-dependent — none of which fits an intraday, cash, 2-year-history setup. Educational only.

### A5. Short-term reversal
- **Thesis:** Last week's/last month's losers bounce and winners give back over the next week/month — overreaction + liquidity provision.
- **Evidence:** 🟡 Moderate. Jegadeesh (1990) ~2%/month on prior-month sort; Lehmann (1990) weekly reversal. Real and long-known — **but** the profits are concentrated in illiquid names and are widely shown to be a payment for *providing liquidity*.
- **Strengths:** Short horizon (fits us better than A1–A4), strong raw statistical signal.
- **Weaknesses:** Largely an artifact of the bid-ask bounce and liquidity shocks; profits shrink or vanish after realistic transaction costs.
- **Feasibility here — 🔴 Low (borderline 🟡).** Horizon fits, but this effect is *the* textbook case of a signal eaten by spreads. Capturing it means modeling liquidity provision precisely — and we have **no quote data** to do that. High risk of a mirage backtest.

### A6. Pairs trading / statistical arbitrage
- **Thesis:** Two historically co-moving securities that diverge tend to reconverge; short the rich one, long the cheap one, market-neutral.
- **Evidence:** 🟡 Moderate. Gatev, Goetzmann & Rouwenhorst (2006, RFS): up to ~11%/yr excess on distance-matched pairs, 1962–2002, net of conservative costs — **but** profits declined markedly in later subperiods as the trade crowded.
- **Strengths:** Market-neutral, tractable to build, statistically principled, flexible horizon (works on daily *or* intraday bars).
- **Weaknesses:** Edge has decayed with competition; vulnerable to structural breaks (a "pair" stops being a pair after a merger, guidance shock, etc.).
- **Feasibility here — 🟢 High (relative to the rest of Part A).** This is the academic strategy that fits us best: **no fundamentals, no quotes strictly required, works on our 5-min or daily bars, and our universe already includes sector ETFs** (natural cointegration anchors). Cost-sensitive and decayed, so expectations should be modest — but it's genuinely backtestable with walk-forward, and market-neutrality sidesteps the directional-signal problem our logs keep hitting.

### A7. Post-earnings-announcement drift (PEAD) — bridges to your news work
- **Thesis:** After a big earnings surprise, prices drift in the surprise's direction for weeks because investors underreact.
- **Evidence:** 🟢 Strong historically. Bernard & Thomas (1989): top-vs-bottom SUE decile spread positive in 41/48 quarters, stronger after bigger surprises. Well-replicated — **though** multiple studies find PEAD has weakened over time.
- **Strengths:** One of the most durable anomalies; horizon (days–weeks) fits an intraday-to-swing stack; directly adjacent to our news pipeline.
- **Weaknesses:** Canonical construction needs a *standardized unexpected earnings* (SUE) measure = actual vs. **analyst consensus estimate**, which we don't have on the free tier. Concentrated in smaller, less-liquid names.
- **Feasibility here — 🟡 Medium.** The closest academic fit to what we're already building. We can't compute true SUE without consensus estimates, but we *can* proxy the surprise from the **price/volume reaction to the earnings event** (gap + volume) and from **Polygon news entity sentiment**, then test drift. That's a real, novel-ish experiment on our stack — the honest caveat is that a reaction-based proxy is a different (weaker) signal than SUE.

---

## Part B — Practitioner / intraday setups

Shorter horizon (minutes to a day). Be candid: formal, peer-reviewed evidence here is
much thinner than Part A. Where a real study exists, it's cited; where it's mostly
educator lore, the grade says so.

### B1. Opening-range breakout (ORB)
- **Thesis:** Define a range in the first N minutes; a break beyond it tends to continue for the session.
- **Evidence:** 🟡 Moderate — and unusually good for a day-trading setup. Holmberg, Lönnbark & Lundström (2013, *Finance Research Letters*) found ORB returns significantly above zero; a 2024 US-equity study (Concretum) reported a high Sharpe with near-zero beta on "stocks in play." **Both stress transaction costs can erase the edge.**
- **Strengths:** Simple, objective, intraday, and **directly aligned with your existing morning-window work** (your blotter already shows pre-noon trades are where the non-negative returns live).
- **Weaknesses:** Highly sensitive to fills/slippage and to the range definition (N, breakout threshold); prone to false breakouts in quiet sessions.
- **Feasibility here — 🟢 High.** Best-aligned strategy in the whole document. **5-min bars are exactly the right granularity, we already have a morning-window gate, VWAP, and a cost-aware simulator.** The one honest gap: no quotes means fill realism is modeled — so results ride on the CostModel assumptions. Strong candidate for the next experiment.

### B2. VWAP reversion / anchoring
- **Thesis:** Intraday price tends to revert toward VWAP; fade stretched deviations, or use VWAP as a directional filter.
- **Evidence:** 🔴 Weak (as a standalone alpha). VWAP is rigorously established as an *execution benchmark*, not as a documented return-predicting signal; reversion-to-VWAP alpha is mostly practitioner claim.
- **Strengths:** We already store a `vwap` column per bar — zero new data needed. Intuitive intraday anchor, useful as a *feature/filter* even if weak alone.
- **Weaknesses:** Little independent evidence of standalone edge; "reversion" and "trend-from-VWAP" are contradictory framings that are easy to overfit.
- **Feasibility here — 🟡 Medium.** Trivial to build (data's already there) and a sensible **feature** to feed Stage 2, but low odds as a standalone edge. Best treated as an input, not a strategy.

### B3. Gap-and-go / gap fade
- **Thesis:** A stock gapping on overnight news either continues ("go") or fills the gap ("fade"), depending on context.
- **Evidence:** 🟡 Moderate, mostly via its overlap with **PEAD (A7)** and overnight-return literature — the "gap" is often the market's reaction to an information event. As a standalone retail pattern it's more lore than proof.
- **Strengths:** Easy to detect on our bars; **we already use a "gap baseline" in Stage 1 evaluation**, so the plumbing exists. Ties overnight news to intraday follow-through.
- **Weaknesses:** Direction (continue vs. fill) is regime- and news-dependent; without classifying *why* the gap happened it's a coin flip.
- **Feasibility here — 🟢 High to build, 🟡 for edge.** Very buildable on our stack and a natural marriage of the **news pipeline + intraday bars**. The edge likely comes from *conditioning the gap on news sentiment/event type*, which is exactly the direction our project is already pushing.

### B4. Liquidity sweeps / stop-hunts (ICT / "smart money concepts" / TJR)
- **Thesis:** Price briefly spikes past an obvious level to trigger clustered stop orders, then reverses — trade the reversal.
- **Evidence:** 🔴 Weak. This is the honest one. Order-flow research *does* confirm stops cluster at round numbers and prior highs/lows — that part is real. But independent backtests of the **full "smart-money reversal" claim show no significant edge**: e.g., a StatOasis study found +0.119% 5-day forward return on SPY sweeps (547 events, t≈0.94 — not significant), similar on DIA, weaker elsewhere; unfiltered SMC setups report 38–48% win rates with long losing streaks. The *footprint* exists; the *premium* does not reliably.
- **Strengths:** Precisely defined, intraday, and **you've already implemented it** (`tjr/` backtest reading `bars.sqlite`).
- **Weaknesses:** No rigorous evidence of a standalone edge; heavy exposure to overfitting and to discretionary "context" rules that don't generalize; the reversal premium doesn't survive significance testing in independent tests.
- **Feasibility here — 🟢 to build (done), 🔴 for edge.** You *can* run it — you have. But set expectations from the evidence: treat it as a **feature/event detector** (a sweep is a real microstructure event worth flagging to Stage 2) rather than a standalone money-maker, and hold it to the same significance bar as everything else. Don't let a discretionary "it works if you read context right" narrative smuggle past walk-forward.

### B5. Order-flow / order-book imbalance
- **Thesis:** Short-horizon price direction is predictable from bid/ask imbalance and the sign of aggressive trades.
- **Evidence:** 🟢 Strong *in the microstructure literature* — but it requires tick-level trades-and-quotes data.
- **Strengths:** Among the most robust *very* short-horizon predictors that exist.
- **Weaknesses:** Needs L1/L2 quote and tick data; alpha decays in milliseconds-to-minutes; infrastructure-heavy.
- **Feasibility here — 🔴 Low (effectively infeasible).** We have **no historical quotes or ticks** on the free tier, full stop. Listed for completeness and to mark it clearly out of reach until a paid feed exists.

### B6. News-spike fade / news-driven drift
- **Thesis:** After a news-driven spike, price either continues (drift) or over-extends and fades — conditioned on the news content/sentiment.
- **Evidence:** 🟡 Moderate. Overlaps PEAD (A7) and a real short-term-reversal-vs-news literature (news-driven moves reverse *less* than non-news moves — the news/no-news distinction is documented).
- **Strengths:** **This is the thesis your whole news pipeline is built to test.** You have Polygon news with entity sentiment + EDGAR events.
- **Weaknesses:** Point-in-time integrity is hard (news timestamps vs. availability); sentiment scores are noisy; event classification is the crux.
- **Feasibility here — 🟡 Medium, and the most *strategically* aligned.** It's the natural next experiment given the project's direction: use news to *condition* the intraday setups above (does an ORB / gap continue or fade given the news?), which addresses the exact gap your logs identify — bars alone carry no direction, so the edge, if any, has to come from the news layer.

---

## Part C — Cross-cutting lessons (the part that decides whether any of this works)

1. **Post-publication decay is real and large.** McLean & Pontiff (2016): returns to published predictors are ~26% lower out-of-sample and ~58% lower post-publication. Anything in Part A is *known* to whoever you're trading against — assume the live edge is a fraction of the paper edge.
2. **Transaction costs are the graveyard.** Nearly every source above (ORB, short-term reversal, pairs, ICT) caveats that the effect shrinks or dies after realistic costs. Our cost-aware simulator is the right instinct; the missing quote feed means our cost model is the single biggest source of error in every result.
3. **We keep confirming bars carry activity, not direction.** Stage 1 finds *where* things happen (high precision@k); Stage 2 can't predict *which way* from price/volume alone (~50% direction). The feasibility grades reward strategies that either (a) are market-neutral (pairs, A6) or (b) add a genuinely new information source (news → A7, B3, B6). They penalize anything that just re-slices price/volume.
4. **Walk-forward + multiple-comparisons discipline is non-negotiable.** With ~19 experiments already logged and no correction yet, the risk of a false positive is high. Every strategy above must clear the *untouched holdout* and a significance bar — the ICT case (t≈0.94) is the cautionary tale.
5. **Short history caps ambition.** ~2 years kills anything with a monthly+ rebalance (A1–A4). It favors intraday/short-horizon strategies (Part B) where each session is a fresh observation.

### Feasibility summary — where to spend effort

| # | Strategy | Evidence | Feasibility here | One-line verdict |
|---|----------|:--------:|:----------------:|------------------|
| B1 | Opening-range breakout | 🟡 | 🟢 | Best-aligned; build next. Data + morning-window + sim all ready. |
| A6 | Pairs / stat-arb | 🟡 | 🟢 | Best academic fit; market-neutral sidesteps our direction problem. |
| B3 | Gap-and-go (news-conditioned) | 🟡 | 🟢/🟡 | Marries news pipeline + bars; edge lives in the conditioning. |
| A7 | PEAD (reaction-proxy) | 🟢 | 🟡 | Durable anomaly; we can only proxy the surprise, but it's novel here. |
| B6 | News-driven drift/fade | 🟡 | 🟡 | The project's core thesis; use news to condition B1/B3. |
| B2 | VWAP reversion | 🔴 | 🟡 | Weak alone — use as a Stage 2 feature, not a strategy. |
| A5 | Short-term reversal | 🟡 | 🔴 | Eaten by spreads; we lack quotes to trade it honestly. |
| B4 | ICT liquidity sweeps (TJR) | 🔴 | 🔴 edge | Already built; treat as event detector, not alpha. Hold to significance bar. |
| A1 | Cross-sectional momentum | 🟢 | 🟡 | Solid effect, but 2yr history can't validate a monthly rebalance. |
| A2 | Time-series momentum | 🟢 | 🔴 | Evidence is in futures; wrong universe + horizon for us. |
| A3 | Value | 🟢 | 🔴 | Needs fundamentals + decades; wrong tool for intraday. |
| A4 | Low-vol / BAB | 🟢 | 🔴 | Long-horizon, leverage-based; doesn't fit the stack. |
| B5 | Order-flow imbalance | 🟢 | 🔴 | Needs tick/quote data we don't have. Revisit on a paid feed. |

**If I had to pick three to pursue next on this stack:** ORB (B1), pairs/stat-arb
(A6), and news-conditioned gap/drift (B3/B6) — because they fit the data we actually
have, the horizon our history can validate, and the one lesson our own logs keep
teaching: the edge won't come from re-slicing price bars, it comes from
market-neutrality or a new information source.

---

## References

*Verified via web search, September 2026. Links are to the paper of record or a stable copy.*

**Academic factors**
- Jegadeesh, N. & Titman, S. (1993). "Returns to Buying Winners and Selling Losers." *Journal of Finance* 48(1), 65–91. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.1993.tb04702.x)
- Moskowitz, T., Ooi, Y.H. & Pedersen, L.H. (2012). "Time Series Momentum." *Journal of Financial Economics*. [NYU/Stern PDF](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf) · [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Time-Series-Momentum)
- Basu, S. (1977) and Fama, E. & French, K. (1992, 1993) — the value factor (HML). (Foundational; widely reproduced.)
- Frazzini, A. & Pedersen, L.H. (2014). "Betting Against Beta." *Journal of Financial Economics*. [NBER PDF](https://www.nber.org/system/files/working_papers/w16601/w16601.pdf) · [Stern PDF](https://pages.stern.nyu.edu/~lpederse/papers/BettingAgainstBeta.pdf)
- Jegadeesh, N. (1990) & Lehmann, B. (1990) — short-term reversal. Summary: [Alpha Architect](https://alphaarchitect.com/quantitative-momentum-research-short-term-return-reversal/)
- Gatev, E., Goetzmann, W.N. & Rouwenhorst, K.G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies* 19(3), 797–827. [Oxford](https://academic.oup.com/rfs/article-abstract/19/3/797/1646694) · [PDF](http://stat.wharton.upenn.edu/~steele/Courses/434/434Context/PairsTrading/PairsTradingGGR.pdf)
- Bernard, V. & Thomas, J. (1989). "Post-Earnings-Announcement Drift." *Journal of Accounting Research*. Overview: [Wikipedia](https://en.wikipedia.org/wiki/Post%E2%80%93earnings-announcement_drift) · decline over time: [Columbia CEASA PDF](https://business.columbia.edu/sites/default/files-efs/imce-uploads/CEASA/Events%20Page/PEAD_Declined_over_time.pdf)
- McLean, R.D. & Pontiff, J. (2016). "Does Academic Research Destroy Stock Return Predictability?" *Journal of Finance* 71(1), 5–32. [Wiley](https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12365)

**Intraday / practitioner**
- Holmberg, U., Lönnbark, C. & Lundström, C. (2013). "Assessing the profitability of intraday opening range breakout strategies." *Finance Research Letters* 10(1), 27–33. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1544612312000438) · [PDF](http://www.econ.umu.se/ueslpnr/ues845.pdf)
- Concretum Group (2024). "A Profitable Day Trading Strategy for the U.S. Equity Market" (ORB on stocks-in-play). [Concretum](https://concretumgroup.com/a-profitable-day-trading-strategy-for-the-u-s-equity-market/)
- ICT / Smart Money Concepts liquidity sweeps — independent backtest showing no significant edge: [StatOasis, "I Backtested ICT/SMC — What Survives"](https://statoasis.com/overfit/research/ict-backtest-what-survives). Concept references: [LuxAlgo](https://www.luxalgo.com/library/concept/liquidity-sweep/)
