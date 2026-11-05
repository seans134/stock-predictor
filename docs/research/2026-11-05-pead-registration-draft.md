# Post-earnings drift: registration draft

**Status: draft only.** Nothing is registered, no strategy code exists, and no returns or event-day price reactions were computed. Research stays paused until the owner approves this. All cutoffs below are proposals, chosen without looking at results, and can still change.

**Blocked: SEC coverage is not measured.** See [SEC coverage](#sec-coverage).

## Hypothesis

After a large earnings reaction on unusually high volume, the stock keeps moving in the same direction for days to weeks. Evidence exists in the literature, though it has weakened over time and is strongest in smaller, costlier stocks than this universe. The reaction is a price-based stand-in for a true earnings surprise, because analyst estimates are unavailable.

## Events

- **Source:** original 8-K filings with Item 2.02, using SEC acceptance time. Amendments (8-K/A) are not events.
- **First full session (S):** the same day if accepted before 09:30 ET on a session day, otherwise the next session. The press release can come before the filing, so S can be late. The signal then measures a reaction already under way, but never uses information from after the entry decision.
- **Repeat filings:** a filing within 15 sessions of the previous event is not a new event. Item 2.02 can also cover preannouncements, so precision needs a manual sample.

## Signal

ATR(14) is the mean true range over the 14 sessions ending at S-1, so it excludes the reaction. All three conditions must hold.

| Measure | Definition | Long | Short (research only) |
|---|---|---:|---:|
| Gap | (open S - close S-1) / ATR | at least +0.5 | at most -0.5 |
| Reaction | (close S - close S-1) / ATR | at least +1.0 | at most -1.0 |
| Volume | volume S / mean volume of the prior 14 sessions | at least 2.0 | at least 2.0 |

Entry is at the open of S+1. These cutoffs are round judgment calls, not fitted values, and must not change after any signal count or return has been seen.

## Arms and decision rule

| Role | Arm | Exit |
|---|---|---|
| **Primary** | Long-minus-short spread | Fixed 10-session hold |
| Secondary | Long-minus-short spread | Threshold exits |
| Secondary | Long only | Fixed 10-session hold |
| Secondary | Short only (research only) | Fixed 10-session hold |

The spread is the combined long and short book at equal notional per trade.

- **Pass:** mean daily net return of the primary arm is above zero after costs, and also above zero after subtracting SPY over the same holding periods. A pass earns one validation run on locked dates from July 10, 2026 onward, not trading.
- **Fail:** either figure is at or below zero. Archive with no tuning.
- **Inconclusive:** fewer than 100 trades on either side. Do not treat as a pass, and do not change cutoffs to get more trades.
- **Secondary arms:** count only if the primary passes, using a Holm adjustment across the three tests with day-block bootstrap p-values.

**Shorts are a removable research arm.** The owner has not explicitly approved them. Without borrow data they cannot count as tradable. If they are dropped, the long-only 10-session arm becomes primary, measured against SPY and the control. Survivorship also leaves out companies that collapsed, which understates short profits and flatters longs.

## Exits

- **Fixed hold:** exit at the open 10 sessions after entry.
- **Threshold arm (the one alternative):** from the entry price, stop-loss at 2 x ATR(14) and profit target at 4 x ATR(14), mirrored for shorts. Check at each daily close starting on the entry day, and exit at the next open, at the actual open price. A gap past the threshold therefore takes the full gap, and the realized loss can exceed 2 x ATR. Maximum hold is the same 10 sessions.
- **Next earnings:** a new Item 2.02 acceptance during a hold triggers exit at the next open, in both arms. Future report dates are not in this data, so this reacts to a filing instead of exiting before it. A strict "exit before the next report" needs an earnings calendar. With a 10-session hold, an overlap should be rare, and the report will count them.
- **Diagnostics only:** gross returns at 5, 10 and 20 sessions. They cannot be used to choose a holding period.

## Benchmarks

- SPY from the same entry open to the same exit open.
- The same stocks outside earnings windows: the same signal on sessions at least 15 sessions from any Item 2.02 event. If this control performs as well, the effect is not specific to earnings.

## Costs

These are assumptions, not measured costs. Cost sensitivity at 0, 2, 5, 11 and 20 bp will be reported but cannot replace the headline.

| Side | Round trip | Extra |
|---|---:|---|
| Long | 11 bp | None |
| Short | 16 bp (11 bp plus 5 bp) | Borrow at 100 bp per year, about 4 bp over 10 sessions |

Actual borrow on stocks after bad news can cost far more, which is why shorts stay research only.

## Universe and dates

- **Universe:** the 46 H1 stocks, a static list of surviving names.
- **Proposed evaluation window:** August 2024 through July 8, 2026, with earlier bars as warmup. The later H1 window would hold only about four earnings seasons.
- **Sizing:** fixed 2% of equity per trade, 20 positions at most, with the H1 gross limit of 40%.

## What must be fixed or built before running

**Portfolio replay** (`execution/portfolio.py`, `research/reporting.py`) was built for intraday trades:

- **No daily marks:** P&L is realized only at exit. Drawdown and worst-day figures would miss losses on open positions, and `daily_portfolio` books a whole multi-day trade on its exit day.
- **Intervals:** because of that lumping, daily block-bootstrap intervals would be wrong. Daily mark-to-market should come first, and the block length should cover the holding period.
- **Daily loss halt:** it only sees realized P&L, so it cannot react to open losses.
- **Position limits:** earnings cluster in the same weeks, so the 20-position cap could bind. Ties break by ticker order, so skipped trades must be counted and their effect reported.
- **Same-ticker overlap:** a new signal in a stock already held is skipped, which should also be counted.

**Data and strategy work:**

- **Coverage audit:** the existing audit starts July 22, 2025, so earlier bars are unaudited. The ten registered gaps and the reconstruction also need a policy for this longer window, and multi-day trades can hold through a gap at any time.
- **Dividends:** bars are unadjusted, and returns ignore dividends. Over 10 sessions this slightly understates longs and overstates shorts. It needs a dividend feed or a stated bias.
- **Splits:** bars are unadjusted, so a split inside the ATR, volume or holding window makes prices incomparable. Proposed rule: no trade if a split falls between S-14 and the planned exit. NFLX's split on November 17, 2025 is the only one found for July 2025 onward. Earlier dates still need checking.
- **Strategy code:** daily aggregation from five-minute bars, short trades, costs by side, the threshold arm, the control, and the decision report.

## Sample size risk

Forty-six stocks over about eight quarters would give roughly 370 events if coverage is complete. An unknown share will clear the signal. If it is 20% to 40%, that is only about 75 to 150 trades across both sides, which could fall short of the 100-per-side minimum. This range is an assumption, not a measurement.

The owner could approve counting signals before any forward return exists, using event-day data only, or widen the universe. That needs deciding before any forward return is seen.

## SEC coverage

**Not measured.** The SEC rejected the request for its ticker directory with HTTP 403 when using a plain client name, with no contact details. SEC automated access requires a declared contact. No contact was sent, and no browser identity was faked. No evidence file was written.

`research/earnings_dates.py` is ready and tested on synthetic filings, not on real SEC responses. It requires `SEC_USER_AGENT` in the environment or `.env`, so the contact never enters the repo:

```powershell
.venv/Scripts/python.exe -m stockpredictor.research.earnings_dates data/edge_research/registry/H1-20261104T214237-53e350.json --start 2024-07-08 --end 2026-11-04 --output data/edge_research/provider_checks/sec-item202-h1.json
```

It will report events per stock, gaps over 110 days as possible missed reports, filings under 60 days apart as possible non-earnings items, and acceptance timing. About eight or nine originals per stock would be expected for this date range.

## Open decisions for the owner

1. **SEC contact:** which contact should go in `SEC_USER_AGENT`?
2. **Shorts:** keep as a research arm or drop?
3. **Threshold design:** this draft checks daily closes with a 4 x ATR profit target. An alternative is a loss-only threshold checked on five-minute bars, which exits nearer the threshold but can be stopped out by intraday noise.
4. **Sample size:** count signals in advance, widen the universe, or accept the risk of an inconclusive result?
