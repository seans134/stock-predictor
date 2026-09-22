"""Data adapter: feed the TJR backtest from StockPredictor's bitemporal bar store.

Replaces the old yfinance downloader. Reads 5-minute bars out of ``data/bars.sqlite`` via
:class:`stockpredictor.data.store.BarStore`, normalizes them to US/Eastern, and exposes the same
``get_intraday`` / ``get_daily`` interface the rest of the TJR code already consumes.

Daily bars (for prior-day high/low) are derived by resampling the 5-minute series per ET calendar
day, so we need only the one 5-minute table the store already holds.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..data.store import BarStore

ET = "US/Eastern"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DB_PATH = _REPO_ROOT / "data" / "bars.sqlite"
_CACHE_DIR = _REPO_ROOT / "data" / "tjr" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# TJR code expects capitalized OHLCV column names; the store uses lowercase.
_RENAME = {"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}


def _cache_path(ticker: str, kind: str) -> Path:
    safe = ticker.replace(":", "_")
    return _CACHE_DIR / f"{safe}_{kind}.pkl"


def get_intraday(ticker: str, use_cache: bool = True, db_path: Path | None = None) -> pd.DataFrame:
    """All stored 5-minute bars for ``ticker``, indexed by tz-aware US/Eastern timestamp.

    Columns: Open, High, Low, Close, Volume. Uses the latest revision of each bar (as-of now).
    """
    path = _cache_path(ticker, "5min")
    if use_cache and path.exists():
        return pd.read_pickle(path)

    store = BarStore(db_path or _DB_PATH)
    try:
        as_of = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
        df = store.get_bars_asof(ticker, as_of=as_of, timeframe="5min")
    finally:
        store.close()
    if df.empty:
        raise RuntimeError(f"No 5min bars for {ticker!r} in {db_path or _DB_PATH}")

    df = df[["open", "high", "low", "close", "volume"]].rename(columns=_RENAME)
    df.index = df.index.tz_convert(ET)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df.to_pickle(path)
    return df


def get_daily(ticker: str, use_cache: bool = True, db_path: Path | None = None) -> pd.DataFrame:
    """Daily OHLC derived from the 5-minute series, indexed by tz-naive ET date.

    Used only for prior-day high/low (PDH/PDL); the backtest strips tz before comparing.
    """
    path = _cache_path(ticker, "1d")
    if use_cache and path.exists():
        return pd.read_pickle(path)

    intr = get_intraday(ticker, use_cache=use_cache, db_path=db_path)
    grp = intr.groupby(intr.index.normalize())
    daily = pd.DataFrame({
        "Open": grp["Open"].first(),
        "High": grp["High"].max(),
        "Low": grp["Low"].min(),
        "Close": grp["Close"].last(),
        "Volume": grp["Volume"].sum(),
    })
    daily.index = pd.DatetimeIndex(daily.index).tz_localize(None).normalize()
    daily = daily.sort_index()
    daily.to_pickle(path)
    return daily


if __name__ == "__main__":
    for tk in ["QQQ", "C:EURUSD"]:
        intr = get_intraday(tk, use_cache=False)
        daily = get_daily(tk, use_cache=False)
        print(f"{tk:10} intraday {intr.shape} {intr.index.min()} -> {intr.index.max()}; "
              f"daily {daily.shape}")
