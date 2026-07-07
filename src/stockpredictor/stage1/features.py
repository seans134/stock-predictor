"""Stage 1 pre-market feature construction.

One row per (ticker, session): everything the pre-market scanner may know
by the 9:00 ET cutoff, and nothing it may not.

Cutoff discipline: a 5-minute bar starting at time t completes at t+5min
and becomes available at t+5min+latency (the shared availability rule).
A pre-market bar is usable only if that availability timestamp is at or
before the cutoff — so the 08:55 bar, which arrives just after 9:00, is
excluded by construction rather than by hope.

Trailing normalizations (volatility, volume medians) use shift() so they
end at the *previous* session; nothing from the feature's own session or
later can enter them. Missing pre-market data stays NaN with an explicit
bar count, never zero-filled.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from stockpredictor.data.availability import BAR_INGEST_LATENCY, TIMEFRAME_DURATIONS
from stockpredictor.sessions import DEFAULT_PREMARKET_CUTOFF, ET, SessionCalendar

EPS = 1e-12

STAGE1_FEATURES = [
    "gap",              # log(pre-market last price / previous RTH close)
    "gap_norm",         # gap / trailing 20-day daily vol
    "pm_ret",           # pre-market return, first to last usable bar
    "pm_range",         # log(pre-market high / low)
    "pm_dollar_vol",
    "pm_vol_ratio",     # pre-market dollar vol vs trailing 20-session median
    "n_pm_bars",        # explicit missingness / thinness indicator
    "vol5",             # trailing 5-day daily-return vol (through yesterday)
    "vol20",            # trailing 20-day
    "prev_day_ret",
    "prev_day_range",
    "prev_rth_vol_ratio",  # yesterday's RTH dollar vol vs its trailing median
    "spy_gap",
    "spy_pm_ret",
    "rel_gap",          # gap minus SPY gap
]

META_COLUMNS = ["ticker", "date"]
PM_SESSION_START = dt.time(4, 0)


def _pm_usable_mask(et_index: pd.DatetimeIndex, cutoff: dt.time) -> np.ndarray:
    """Bars whose availability (start + duration + latency) is <= cutoff
    and which start at/after the 04:00 ET pre-market open."""
    duration = TIMEFRAME_DURATIONS["5min"] + BAR_INGEST_LATENCY
    available = et_index + duration
    times = et_index.time
    avail_times = available.time
    same_day = available.date == et_index.date
    return (
        (times >= PM_SESSION_START)
        & (avail_times <= cutoff)
        & same_day
    )


def _per_ticker_daily(
    bars: pd.DataFrame,
    calendar: SessionCalendar,
    cutoff: dt.time,
) -> pd.DataFrame:
    """Daily feature frame for one ticker from extended-hours bars."""
    et = bars.copy()
    et.index = et.index.tz_convert(ET)
    day = pd.Series(et.index.date, index=et.index)
    sessions = set(calendar.sessions_between(et.index[0].date(), et.index[-1].date()))
    in_session_day = day.isin(sessions).to_numpy()
    et = et.loc[in_session_day]
    day = day.loc[in_session_day]

    opens = {d: pd.Timestamp(calendar.session_open(d)) for d in sessions}
    closes = {d: pd.Timestamp(calendar.session_close(d)) for d in sessions}
    ts = pd.Series(et.index, index=et.index)
    rth_mask = (ts >= day.map(opens)) & (ts < day.map(closes))

    rth = et.loc[rth_mask.to_numpy()]
    rth_day = pd.Series(rth.index.date, index=rth.index)
    price_ref = rth["vwap"].fillna(rth["close"])
    daily = pd.DataFrame(
        {
            "rth_close": rth.groupby(rth_day)["close"].last(),
            "rth_high": rth.groupby(rth_day)["high"].max(),
            "rth_low": rth.groupby(rth_day)["low"].min(),
            "rth_dollar": (price_ref * rth["volume"]).groupby(rth_day).sum(),
        }
    )
    daily.index.name = "date"
    daily = daily.sort_index()

    daily_ret = np.log(daily["rth_close"] / daily["rth_close"].shift(1))
    daily["prev_close"] = daily["rth_close"].shift(1)
    daily["prev_day_ret"] = daily_ret.shift(1)
    daily["prev_day_range"] = np.log(daily["rth_high"] / daily["rth_low"]).shift(1)
    daily["vol5"] = daily_ret.shift(1).rolling(5, min_periods=3).std()
    daily["vol20"] = daily_ret.shift(1).rolling(20, min_periods=10).std()
    daily["prev_rth_vol_ratio"] = daily["rth_dollar"].shift(1) / daily["rth_dollar"].shift(
        2
    ).rolling(20, min_periods=10).median().replace(0, np.nan)

    pm = et.loc[_pm_usable_mask(et.index, cutoff)]
    pm_day = pd.Series(pm.index.date, index=pm.index)
    pm_price_ref = pm["vwap"].fillna(pm["close"])
    pm_daily = pd.DataFrame(
        {
            "pm_first_open": pm.groupby(pm_day)["open"].first(),
            "pm_last_close": pm.groupby(pm_day)["close"].last(),
            "pm_high": pm.groupby(pm_day)["high"].max(),
            "pm_low": pm.groupby(pm_day)["low"].min(),
            "pm_dollar_vol": (pm_price_ref * pm["volume"]).groupby(pm_day).sum(),
            "n_pm_bars": pm.groupby(pm_day).size(),
        }
    )
    pm_daily.index.name = "date"
    daily = daily.join(pm_daily, how="left")
    daily["n_pm_bars"] = daily["n_pm_bars"].fillna(0).astype(int)

    daily["gap"] = np.log(daily["pm_last_close"] / daily["prev_close"])
    daily["gap_norm"] = daily["gap"] / (daily["vol20"] + EPS)
    daily["pm_ret"] = np.log(daily["pm_last_close"] / daily["pm_first_open"])
    daily["pm_range"] = np.log(daily["pm_high"] / daily["pm_low"])
    daily["pm_vol_ratio"] = daily["pm_dollar_vol"] / daily["pm_dollar_vol"].shift(1).rolling(
        20, min_periods=10
    ).median().replace(0, np.nan)
    return daily


def build_stage1_features(
    bars_by_ticker: dict[str, pd.DataFrame],
    calendar: SessionCalendar,
    cutoff: dt.time = DEFAULT_PREMARKET_CUTOFF,
    market_ticker: str = "SPY",
) -> pd.DataFrame:
    """Pre-market snapshot rows for every ticker-session.

    bars_by_ticker holds UTC-indexed extended-hours bars per ticker.
    SPY contributes market context to every row; if absent its features
    stay NaN (explicit missingness).
    """
    spy_context = None
    if market_ticker in bars_by_ticker and not bars_by_ticker[market_ticker].empty:
        spy = _per_ticker_daily(bars_by_ticker[market_ticker], calendar, cutoff)
        spy_context = spy[["gap", "pm_ret"]].rename(
            columns={"gap": "spy_gap", "pm_ret": "spy_pm_ret"}
        )

    frames = []
    for ticker, bars in bars_by_ticker.items():
        if bars.empty:
            continue
        daily = _per_ticker_daily(bars, calendar, cutoff)
        if spy_context is not None:
            daily = daily.join(spy_context, how="left")
        else:
            daily["spy_gap"] = np.nan
            daily["spy_pm_ret"] = np.nan
        daily["rel_gap"] = daily["gap"] - daily["spy_gap"]
        daily["ticker"] = ticker
        daily = daily.reset_index()
        frames.append(daily[META_COLUMNS + STAGE1_FEATURES])

    if not frames:
        return pd.DataFrame(columns=META_COLUMNS + STAGE1_FEATURES)
    return pd.concat(frames, ignore_index=True)
