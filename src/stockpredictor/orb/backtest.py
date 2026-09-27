"""Event-driven ORB backtest: signal -> next-bar-open fill -> stop/target/EOD.

Emits trades in the portfolio schema so the existing risk engine can replay
them. Conservative fill: a bar spanning both stop and target resolves as a
stop. Round-trip cost is subtracted from every trade.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import LONG, SHORT
from stockpredictor.orb.strategy import ORBConfig, find_orb_signal, prior_session_atr, stop_and_target
from stockpredictor.sessions import ET

ORB_TRADE_COLUMNS = [
    "ticker", "date", "bar_start", "direction",
    "gross_ret", "net_ret", "q10", "entry_ts", "exit_ts",
]


@dataclass(frozen=True)
class ORBTrade:
    ticker: str
    date: object
    signal_bar: pd.Timestamp
    entry_bar: pd.Timestamp
    exit_ts: pd.Timestamp
    direction: int
    entry: float
    stop: float
    target: float
    exit: float
    exit_reason: str          # "target" | "stop" | "time"
    gross_ret: float
    net_ret: float
    risk_frac_return: float

    def q10_hint(self) -> float:
        return -self.risk_frac_return


def simulate_orb_session(
    session: pd.DataFrame, atr: float, cfg: ORBConfig, costs: CostModel,
    ticker: str, day,
) -> ORBTrade | None:
    sig = find_orb_signal(session, atr, cfg)
    if sig is None:
        return None
    after = session.loc[session.index > sig.signal_bar]
    if after.empty:
        return None
    entry_bar = after.index[0]
    entry = float(after.iloc[0]["open"])
    stop, target, risk = stop_and_target(
        entry, sig.direction, sig.range_high, sig.range_low, atr, cfg
    )
    if risk <= 0:
        return None

    forward = session.loc[session.index >= entry_bar]
    exit_px = exit_reason = exit_ts = None
    for ts, bar in forward.iterrows():
        lo, hi = float(bar["low"]), float(bar["high"])
        hit_stop = lo <= stop <= hi
        hit_tgt = lo <= target <= hi
        if hit_stop:                       # stop-first when both
            exit_px, exit_reason, exit_ts = stop, "stop", ts
            break
        if hit_tgt:
            exit_px, exit_reason, exit_ts = target, "target", ts
            break
    if exit_px is None:                    # EOD time-stop
        exit_px = float(forward.iloc[-1]["close"])
        exit_reason, exit_ts = "time", forward.index[-1]

    gross = sig.direction * (exit_px - entry) / entry
    net = gross - costs.round_trip_cost
    return ORBTrade(
        ticker=ticker, date=day, signal_bar=sig.signal_bar, entry_bar=entry_bar,
        exit_ts=exit_ts, direction=sig.direction, entry=entry, stop=stop,
        target=target, exit=exit_px, exit_reason=exit_reason,
        gross_ret=float(gross), net_ret=float(net), risk_frac_return=risk / entry,
    )


def run_orb(
    rth_by_ticker: dict[str, pd.DataFrame], cfg: ORBConfig, costs: CostModel,
    universe_filter: set[tuple[str, object]] | None = None,
) -> pd.DataFrame:
    rows = []
    for ticker, bars in rth_by_ticker.items():
        if bars is None or bars.empty:
            continue
        et = bars.copy()
        et.index = et.index.tz_convert(ET)
        et = et.sort_index()
        by_day = {d: g for d, g in et.groupby(et.index.date)}
        days = sorted(by_day)
        for i, day in enumerate(days):
            if universe_filter is not None and (ticker, day) not in universe_filter:
                continue
            if i < cfg.atr_lookback_sessions:
                continue
            prior = pd.concat(
                [by_day[days[j]] for j in range(i - cfg.atr_lookback_sessions, i)]
            )
            atr = prior_session_atr(prior, cfg)
            if atr is None:
                continue
            trade = simulate_orb_session(by_day[day], atr, cfg, costs, ticker, day)
            if trade is None:
                continue
            rows.append({
                "ticker": ticker,
                "date": day,
                "bar_start": trade.signal_bar,
                "direction": LONG if trade.direction == 1 else SHORT,
                "gross_ret": trade.gross_ret,
                "net_ret": trade.net_ret,
                "q10": -trade.risk_frac_return,
                "entry_ts": trade.entry_bar,
                "exit_ts": trade.exit_ts,
            })
    return pd.DataFrame(rows, columns=ORB_TRADE_COLUMNS)
