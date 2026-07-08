"""Stage 2 feature and target construction.

One row per completed regular-session 5-minute bar. Features use only
information from that bar and earlier (backward-looking shifts within the
session, plus the previous session's close for the overnight gap). The
target looks strictly forward and follows the plan's timing rules:

    prediction time  = close of bar t (available at bar end + latency)
    entry reference  = OPEN of bar t+1  (never the close we predicted from)
    exit reference   = CLOSE of bar t+3 (15 minutes after entry bar start)
    target           = log(exit / entry)

Everything is a return, ratio, or time-of-day value — raw price levels and
ticker identity are not features. Missing values stay NaN (LightGBM treats
NaN as missing natively); they are never zero-filled.

Vectorized construction is backward-looking by construction (groupby +
shift), and tests assert the look-ahead property directly: truncating the
future must not change existing rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from stockpredictor.news.features import STAGE2_NEWS_FEATURES, NewsFeatureBuilder
from stockpredictor.sessions import ET, SessionCalendar

EPS = 1e-12

_PRICE_FEATURES = [
    "r5",                # last completed 5-min log return
    "r15",               # 15-min log return (3 bars)
    "r30",               # 30-min log return (6 bars)
    "rv30",              # realized vol: std of r5 over last 6 bars
    "rv60",              # realized vol over last 12 bars
    "range_frac",        # (high-low)/close of bar t
    "close_pos",         # where close sits in bar t's range [0,1]
    "vwap_dist",         # log(close / intraday cumulative vwap)
    "vol_ratio",         # volume vs rolling 12-bar median volume
    "vol_accel",         # volume vs previous bar volume
    "minutes_since_open",
    "minutes_to_close",
    "overnight_gap",     # log(session open / previous session close)
    "spy_r15",           # market context
    "spy_r30",
    "qqq_r15",
    "qqq_r30",
]

# News features are NaN when no news store is supplied (unknown, not zero).
FEATURE_COLUMNS = _PRICE_FEATURES + STAGE2_NEWS_FEATURES

META_COLUMNS = ["ticker", "date", "bar_start"]
TARGET_COLUMN = "fwd_ret_15m"

# Bars needed before the first usable feature row of a session.
WARMUP_BARS = 12


def _session_features(bars: pd.DataFrame, calendar: "SessionCalendar") -> pd.DataFrame:
    """Feature engineering within one ticker (index: ET bar start).
    Assumes regular-session bars only, in chronological order."""
    df = bars.copy()
    day = df.index.date
    grouped = df.groupby(day, group_keys=False)

    close = df["close"]
    df["r5"] = grouped["close"].apply(lambda c: np.log(c / c.shift(1)))
    df["r15"] = grouped["close"].apply(lambda c: np.log(c / c.shift(3)))
    df["r30"] = grouped["close"].apply(lambda c: np.log(c / c.shift(6)))
    df["rv30"] = df.groupby(day, group_keys=False)["r5"].apply(lambda r: r.rolling(6).std())
    df["rv60"] = df.groupby(day, group_keys=False)["r5"].apply(lambda r: r.rolling(12).std())

    df["range_frac"] = (df["high"] - df["low"]) / close
    df["close_pos"] = (close - df["low"]) / (df["high"] - df["low"] + EPS)

    # Cumulative intraday VWAP; cumsum restarts per day so nothing leaks
    # across sessions.
    price_ref = df["vwap"].fillna(close)
    df["_pv"] = price_ref * df["volume"]
    cum_pv = df.groupby(day)["_pv"].cumsum()
    cum_v = df.groupby(day)["volume"].cumsum()
    df["vwap_dist"] = np.log(close / (cum_pv / (cum_v + EPS)).replace(0, np.nan))
    df = df.drop(columns=["_pv"])

    df["vol_ratio"] = df["volume"] / grouped["volume"].apply(
        lambda v: v.rolling(12).median()
    ).replace(0, np.nan)
    prev_vol = grouped["volume"].apply(lambda v: v.shift(1))
    df["vol_accel"] = df["volume"] / prev_vol.replace(0, np.nan)

    # Session boundaries come from the calendar, never inferred from which
    # bars happen to be present — inferring the close from observed data is
    # look-ahead (caught by the truncation test) and inferring the open is
    # wrong whenever early bars are missing.
    minutes = pd.Series(
        (df.index - df.index.normalize()).total_seconds() / 60.0, index=df.index
    )
    unique_days = pd.unique(day)
    open_minutes = {}
    close_minutes = {}
    for d in unique_days:
        open_dt = calendar.session_open(d)
        close_dt = calendar.session_close(d)
        open_minutes[d] = open_dt.hour * 60 + open_dt.minute
        close_minutes[d] = close_dt.hour * 60 + close_dt.minute
    day_series = pd.Series(day, index=df.index)
    df["minutes_since_open"] = minutes - day_series.map(open_minutes)
    df["minutes_to_close"] = day_series.map(close_minutes) - minutes

    session_open_price = df.groupby(day)["open"].transform("first")
    last_close_per_day = df.groupby(day)["close"].last()
    prev_session_close = pd.Series(day, index=df.index).map(last_close_per_day.shift(1))
    df["overnight_gap"] = np.log(session_open_price / prev_session_close)
    return df


def _forward_target(df: pd.DataFrame) -> pd.Series:
    """15-min forward log return, entry at next bar open, exit at close
    three bars ahead. NaN when the horizon crosses the session end."""
    day = df.index.date
    grouped = df.groupby(day, group_keys=False)
    entry = grouped["open"].apply(lambda o: o.shift(-1))
    exit_ = grouped["close"].apply(lambda c: c.shift(-3))
    return np.log(exit_ / entry)


def build_features(
    bars_by_ticker: dict[str, pd.DataFrame],
    calendar: SessionCalendar,
    context_tickers: tuple[str, str] = ("SPY", "QQQ"),
    news: NewsFeatureBuilder | None = None,
) -> pd.DataFrame:
    """Build the Stage 2 training frame from per-ticker RTH bar frames
    (UTC-indexed, as-of filtered). Context tickers contribute market
    features to every row; absent context stays NaN (explicit missingness).
    """
    context: dict[str, pd.DataFrame] = {}
    for name in context_tickers:
        if name in bars_by_ticker:
            ctx = bars_by_ticker[name].copy()
            ctx.index = ctx.index.tz_convert(ET)
            feats = _session_features(ctx, calendar)
            context[name] = feats[["r15", "r30"]]

    frames = []
    for ticker, bars in bars_by_ticker.items():
        if bars.empty:
            continue
        df = bars.copy()
        df.index = df.index.tz_convert(ET)
        df = _session_features(df, calendar)
        df[TARGET_COLUMN] = _forward_target(df)

        for name, prefix in zip(context_tickers, ("spy", "qqq")):
            if name in context:
                joined = context[name].reindex(df.index)
                df[f"{prefix}_r15"] = joined["r15"]
                df[f"{prefix}_r30"] = joined["r30"]
            else:
                df[f"{prefix}_r15"] = np.nan
                df[f"{prefix}_r30"] = np.nan

        if news is not None:
            bar_ends = df.index + pd.Timedelta("5min")
            news_feats = news.stage2_features(ticker, bar_ends)
            news_feats.index = df.index
            df = df.join(news_feats)
        else:
            for col in STAGE2_NEWS_FEATURES:
                df[col] = np.nan

        df["ticker"] = ticker
        df["date"] = df.index.date
        df["bar_start"] = df.index
        frames.append(df.reset_index(drop=True)[META_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN]])

    if not frames:
        return pd.DataFrame(columns=META_COLUMNS + FEATURE_COLUMNS + [TARGET_COLUMN])
    return pd.concat(frames, ignore_index=True)
