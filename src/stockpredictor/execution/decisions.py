"""Decision rules: Stage 2 forecasts in, BUY / SHORT / HOLD out.

Stage 2 states what it expects; these rules decide whether that
expectation clears costs, risk, and trend gates. Every threshold is a
named constant in DecisionConfig — starting values for the walk-forward
harness to select per fold, never tuned on test results.

A long requires (shorts are symmetric):
- forecast edge:   q50 >= round_trip_cost + min_edge
- bounded downside: q10 >= -max_risk
- trend gate:      30-min return not strongly opposed
- room to exit:    enough session left for the full 15-min horizon

Anything else is HOLD. Position sizing, portfolio caps, and kill-switches
belong to the risk engine and are deliberately not here yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpredictor.execution.costs import CostModel

HOLD = "hold"
LONG = "long"
SHORT = "short"


@dataclass(frozen=True)
class DecisionConfig:
    min_edge_bps: float = 5.0        # required edge beyond round-trip cost
    max_risk_bps: float = 100.0      # worst acceptable q10 (q90 for shorts)
    trend_oppose_bps: float = 30.0   # 30-min move against the trade that vetoes it
    min_minutes_to_close: float = 20.0  # entry+horizon must fit in the session
    # Fresh-news gate: require at least this many articles in the last 2h
    # before trading. 0 disables the gate. Rows with unknown news (NaN,
    # no news store) never pass an active gate — missing data never trades.
    min_news_articles_2h: float = 0.0
    # Sentiment alignment: longs require positive 2h signed sentiment,
    # shorts require negative. Zero (no signed news) or NaN passes neither
    # direction, so this gate also implies recent signed news must exist.
    align_news_sentiment: bool = False
    # Entry-time window: only trade signals within this many minutes of
    # the open (0 disables). Motivated by the observed intraday decay of
    # trade quality; cutoff chosen from exploratory blotter analysis, so
    # results carry that caveat until holdout confirmation.
    max_minutes_since_open: float = 0.0


def decide(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    costs: CostModel,
    config: DecisionConfig,
) -> pd.Series:
    """Vectorized decisions for feature rows and aligned q10/q50/q90.

    `frame` must carry r30 (trend context) and minutes_to_close; rows with
    missing trend context fail the gate (missing data never trades).
    """
    q10 = predictions["q10"].to_numpy()
    q50 = predictions["q50"].to_numpy()
    q90 = predictions["q90"].to_numpy()
    r30 = frame["r30"].to_numpy()
    minutes_left = frame["minutes_to_close"].to_numpy()

    cost = costs.round_trip_cost
    edge = config.min_edge_bps / 1e4
    max_risk = config.max_risk_bps / 1e4
    oppose = config.trend_oppose_bps / 1e4

    room = minutes_left >= config.min_minutes_to_close
    trend_known = ~np.isnan(r30)

    if config.min_news_articles_2h > 0:
        news_2h = frame["news_articles_2h"].to_numpy(dtype=float)
        news_ok = ~np.isnan(news_2h) & (news_2h >= config.min_news_articles_2h)
        room = room & news_ok

    if config.max_minutes_since_open > 0:
        since_open = frame["minutes_since_open"].to_numpy(dtype=float)
        room = room & ~np.isnan(since_open) & (since_open <= config.max_minutes_since_open)

    if config.align_news_sentiment:
        sent_2h = frame["news_sent_signed_2h"].to_numpy(dtype=float)
        long_sent_ok = ~np.isnan(sent_2h) & (sent_2h > 0)
        short_sent_ok = ~np.isnan(sent_2h) & (sent_2h < 0)
    else:
        long_sent_ok = np.ones(len(frame), dtype=bool)
        short_sent_ok = np.ones(len(frame), dtype=bool)

    long_ok = (
        (q50 >= cost + edge)
        & (q10 >= -max_risk)
        & trend_known
        & (r30 > -oppose)
        & room
        & long_sent_ok
    )
    short_ok = (
        (q50 <= -(cost + edge))
        & (q90 <= max_risk)
        & trend_known
        & (r30 < oppose)
        & room
        & short_sent_ok
    )

    decisions = np.full(len(frame), HOLD, dtype=object)
    decisions[long_ok] = LONG
    decisions[short_ok] = SHORT
    return pd.Series(decisions, index=frame.index, name="decision")
