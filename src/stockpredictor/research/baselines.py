"""Fixed long-only H1/H3 policies, with explicit bar-data execution limitations."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpredictor.execution.timing import BAR_DURATION, barrier_exit, decision_time, entry_time
from stockpredictor.sessions import ET

# Explicit research allowlist, not a point-in-time membership dataset.
# Static selection limits historical generalization; no automatic regex whitelist.
STOCKS = tuple("AAPL MSFT AMZN NVDA AMD META GOOGL TSLA JPM BAC GS MS V MA XOM CVX COP "
               "UNH JNJ PFE MRK ABBV WMT COST HD LOW CAT DE BA GE DIS NFLX CRM ORCL ADBE "
               "INTC QCOM AVGO MU AMAT LRCX TXN KO PEP MCD NKE SBUX TGT UPS FDX".split())
INSTRUMENT_TYPES = {**dict.fromkeys(STOCKS, "US_COMMON_STOCK"), "SPY": "US_ETF", "QQQ": "US_ETF"}
TRADE_FIELDS = ["ticker", "date", "bar_start", "decision_ts", "entry_ts", "exit_ts",
                "direction", "gross_ret", "net_ret", "q10", "entry_price", "exit_price",
                "exit_reason", "rvol"]


@dataclass(frozen=True)
class BaselineConfig:
    lookback: int = 14
    top_n: int = 20
    min_price: float = 5.0
    min_daily_volume: float = 1_000_000
    min_dollar_volume: float = 20_000_000
    stop_daily_atr: float = .10
    max_entry_minutes: int = 60  # 10:30 ET; fixed before first run
    cost_bps: float = 11.0

    def __post_init__(self):
        if self.lookback < 1 or self.top_n < 1 or self.max_entry_minutes < 15:
            raise ValueError("Invalid lookback, selection count, or entry window")
        if any(not np.isfinite(v) or v <= 0 for v in
               (self.min_price, self.min_daily_volume, self.min_dollar_volume, self.stop_daily_atr)):
            raise ValueError("Invalid eligibility/risk parameter")
        if not np.isfinite(self.cost_bps) or self.cost_bps < 0:
            raise ValueError("Costs must be finite and nonnegative")


def validate_bars(frame: pd.DataFrame) -> None:
    if frame.index.tz is None or not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Bars must be sorted, unique, and timezone aware")
    values = frame[["open", "high", "low", "close", "volume"]]
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Nonfinite OHLCV")
    if (values[["open","high","low","close"]] <= 0).any().any() or (values.volume < 0).any():
        raise ValueError("Invalid price/volume")
    if ((values.high < values[["open","close","low"]].max(axis=1)) |
        (values.low > values[["open","close","high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC")
    if "available_ts" in frame:
        if (frame.available_ts > decision_time(frame.index)).any():
            raise ValueError("Late/revised bar supplied as contemporaneous data")


def sessions(frame: pd.DataFrame, calendar):
    frame = frame.copy()
    frame.index = frame.index.tz_convert(ET)
    groups = {}
    for day, bars in frame.groupby(frame.index.date):
        if not calendar.is_session(day):
            continue
        opening, closing = pd.Timestamp(calendar.session_open(day)), pd.Timestamp(calendar.session_close(day))
        bars = bars.loc[(bars.index >= opening) & (bars.index < closing)]
        if len(bars):
            validate_bars(bars)
            groups[day] = bars
    return groups


def gap_index(rows) -> dict:
    """Registered gaps as {ticker: Eastern bar starts}; empty means strict coverage."""
    gaps = {}
    for row in rows:
        gaps.setdefault(row["ticker"], []).append(row["event_ts"])
    return {ticker: pd.DatetimeIndex(pd.to_datetime(sorted(values), unit="s", utc=True)).tz_convert(ET)
            for ticker, values in gaps.items()}


def complete(bars, calendar, day, allowed=None):
    """Exact grid match. A registered gap must really be absent, or this fails."""
    expected = pd.date_range(calendar.session_open(day),
                             pd.Timestamp(calendar.session_close(day)) - BAR_DURATION,
                             freq=BAR_DURATION)
    if allowed is not None and len(allowed):
        expected = expected[~expected.isin(allowed)]
    return bars.index.equals(expected)


def require_complete_evaluation(data, calendar, days, registered_gaps=None):
    """Fail before measuring returns; missing sessions are not zero-trade days.

    This deliberately strict research gate also blocks genuine no-print gaps
    until a separate, source-supported handling policy is registered. Only the
    exact bars in such a policy are excused; any other gap still blocks.
    """
    registered_gaps = registered_gaps or {}
    for ticker, frame in data.items():
        grouped = sessions(frame, calendar)
        for day in days:
            bars = grouped.get(day)
            if bars is None or not complete(bars, calendar, day, registered_gaps.get(ticker)):
                raise ValueError(f"Incomplete evaluation coverage: {ticker} {day}; audit data before performance")


def trade_row(ticker, day, signal, entry, exit_ts, entry_px, exit_px, risk, reason, cost, rvol=np.nan):
    gross = exit_px / entry_px - 1.0
    return dict(ticker=ticker, date=day, bar_start=signal, decision_ts=decision_time(signal),
                entry_ts=entry, exit_ts=exit_ts, direction="long", gross_ret=gross,
                net_ret=gross - cost / 1e4, q10=-risk / entry_px,
                entry_price=entry_px, exit_price=exit_px, exit_reason=reason, rvol=rvol)


def gap_affected(trades, registered_gaps) -> pd.Series:
    """True where a registered gap falls inside the trade's holding period."""
    affected = pd.Series(False, index=trades.index)
    for ticker, gaps in (registered_gaps or {}).items():
        for gap in gaps:
            affected |= (trades.ticker == ticker) & (trades.entry_ts <= gap) & (gap < trades.exit_ts)
    return affected


def opening_volume(bars_by_ticker, calendar, eval_days, cfg=BaselineConfig(), ordinary=False,
                   registered_gaps=None, split_dates=None):
    """H1 + registered ordinary-activity control; missing future bars fail run.

    Selection reads only history and the opening bar. If a selected trade has
    incomplete future data, raise rather than silently deleting its outcome.

    Optional registered_gaps excuse exact no-print bars. History for those
    sessions uses the observed bars, a missing bar cannot signal, and a held
    position is next evaluated at the following observed bar. Optional
    split_dates make a ticker ineligible on an ex-date and until its lookback
    holds only post-split sessions, because raw volume and range are not
    comparable across a split.
    """
    registered_gaps, split_dates = registered_gaps or {}, split_dates or {}
    grouped, stats = {}, {}
    for ticker, frame in bars_by_ticker.items():
        if INSTRUMENT_TYPES.get(ticker) != "US_COMMON_STOCK":
            raise ValueError(f"Unsupported H1 instrument: {ticker}")
        grouped[ticker] = sessions(frame, calendar)
        rows = []
        previous_close = None
        all_days = calendar.sessions_between(min(grouped[ticker]), max(grouped[ticker])) if grouped[ticker] else []
        for day in all_days:
            bars = grouped[ticker].get(day)
            if bars is None or not complete(bars, calendar, day, registered_gaps.get(ticker)):
                rows.append(dict(day=day, volume=np.nan, dollar=np.nan, opening_volume=np.nan, tr=np.nan))
                previous_close = None
                continue
            high, low = bars.high.max(), bars.low.min()
            tr = high-low if previous_close is None else max(high-low, abs(high-previous_close), abs(low-previous_close))
            rows.append(dict(day=day, volume=bars.volume.sum(),
                             dollar=(bars.close * bars.volume).sum(),
                             opening_volume=bars.volume.get(pd.Timestamp(calendar.session_open(day)), np.nan),
                             tr=tr))
            previous_close = bars.close.iloc[-1]
        if rows:
            panel = pd.DataFrame(rows).set_index("day")
            stats[ticker] = panel.shift(1).rolling(cfg.lookback, min_periods=cfg.lookback).mean()
            for ex_date in split_dates.get(ticker, ()):
                first_after = sum(session < ex_date for session in panel.index)
                stats[ticker].iloc[first_after:first_after + cfg.lookback] = np.nan
    trades, selections = [], []
    for day in eval_days:
        candidates = []
        opening = pd.Timestamp(calendar.session_open(day))
        closing = pd.Timestamp(calendar.session_close(day))
        for ticker, history in stats.items():
            bars = grouped[ticker].get(day)
            if bars is None or day not in history.index or opening not in bars.index:
                continue
            prior = history.loc[day]
            first = bars.loc[opening]
            if prior.isna().any() or prior.opening_volume <= 0 or prior.tr <= 0:
                continue
            if first.open < cfg.min_price or prior.volume < cfg.min_daily_volume or prior.dollar < cfg.min_dollar_volume:
                continue
            rvol = first.volume / prior.opening_volume
            active = rvol <= 1 if ordinary else rvol > 1
            if active:
                candidates.append((float(rvol), ticker, float(prior.tr)))
        candidates.sort(key=lambda c: (-c[0], c[1]))
        for rank, (rvol, ticker, atr) in enumerate(candidates[:cfg.top_n], 1):
            bars = grouped[ticker][day]
            first = bars.loc[opening]
            selections.append(dict(date=day, ticker=ticker, rvol=rvol, rank=rank,
                                   selected_at=decision_time(opening)))
            if first.close <= first.open:
                continue
            # Only scan the registered entry window; no dependence on final-session statistics.
            signals = bars.loc[(bars.index > opening) &
                               (entry_time(bars.index) <= opening + pd.Timedelta(minutes=cfg.max_entry_minutes)) &
                               (bars.close > first.high)]
            if signals.empty:
                continue
            signal = signals.index[0]
            entry = entry_time(signal)
            expected_prefix = pd.date_range(opening, entry, freq=BAR_DURATION)
            excused = registered_gaps.get(ticker)
            absent = expected_prefix[~expected_prefix.isin(bars.index)]
            if entry in absent or (len(absent) and (excused is None or not absent.isin(excused).all())):
                raise ValueError(f"Missing bars before scheduled H1 entry: {ticker} {day}")
            if not complete(bars, calendar, day, excused):
                raise ValueError(f"Incomplete selected session; cannot silently discard outcome: {ticker} {day}")
            px = float(bars.loc[entry, "open"])
            risk = cfg.stop_daily_atr * atr
            stop = px - risk
            if stop <= 0:
                continue
            exit_px, exit_ts, reason = float(bars.close.iloc[-1]), closing, "scheduled_close"
            for ts, bar in bars.loc[entry:].iterrows():
                fill = barrier_exit(bar, 1, stop, None)
                if fill:
                    exit_px, reason, at_open = fill
                    exit_ts = ts if at_open else ts + BAR_DURATION
                    break
            trades.append(trade_row(ticker, day, signal, entry, exit_ts, px, exit_px, risk, reason, cfg.cost_bps, rvol))
    return pd.DataFrame(trades, columns=TRADE_FIELDS), pd.DataFrame(selections)


def closing_momentum(frame, calendar, eval_days, cfg=BaselineConfig(), unconditional=False):
    """SPY long if previous close -> first half-hour return is positive.

    Closing-price execution is a scheduled-close proxy, not a verified auction
    fill. Entry is scheduled well after the morning signal becomes available.
    """
    grouped = sessions(frame, calendar)
    rows = []
    for day in eval_days:
        opening = pd.Timestamp(calendar.session_open(day))
        closing = pd.Timestamp(calendar.session_close(day))
        entry = closing - pd.Timedelta(minutes=30)
        signal = opening + pd.Timedelta(minutes=25)
        previous_days = calendar.sessions_between(day - pd.Timedelta(days=10), day - pd.Timedelta(days=1))
        if not previous_days or day not in grouped:
            continue
        previous = grouped.get(previous_days[-1])
        bars = grouped[day]
        if previous is None or not complete(previous, calendar, previous_days[-1]) or signal not in bars.index:
            continue
        if not pd.date_range(opening, signal, freq=BAR_DURATION).isin(bars.index).all():
            continue
        if not unconditional and bars.loc[signal, "close"] <= previous.close.iloc[-1]:
            continue
        if not complete(bars, calendar, day):
            raise ValueError(f"Incomplete selected SPY session: {day}")
        px, exit_px = float(bars.loc[entry, "open"]), float(bars.close.iloc[-1])
        rows.append(trade_row("SPY", day, signal, entry, closing, px, exit_px,
                              px * .01, "scheduled_close", cfg.cost_bps))
    return pd.DataFrame(rows, columns=TRADE_FIELDS)
