"""Per-fold pair selection from formation (train) data only.

Sector metadata is unavailable, so "economically related" is approximated by a
formation-window return-correlation pre-filter; survivors are then gated by
Engle-Granger cointegration (OLS hedge ratio + ADF on the residual spread) and
ranked by ADF t-statistic. Everything here is computed on the train window only,
so nothing leaks from the trading window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stockpredictor.pairs.stats import ADF_CRIT_5PCT, adf_tstat, ols_hedge_ratio


@dataclass(frozen=True)
class PairsConfig:
    min_corr: float = 0.80
    max_pairs: int = 20
    adf_crit: float = ADF_CRIT_5PCT
    z_entry: float = 2.0
    z_exit: float = 0.5
    z_stop: float = 3.0
    max_hold: int = 20  # trading sessions


@dataclass(frozen=True)
class PairModel:
    a: str
    b: str
    alpha: float
    beta: float
    mu: float
    sigma: float
    adf_t: float


def select_pairs(train_closes: pd.DataFrame, cfg: PairsConfig) -> list[PairModel]:
    # Keep only tickers with a complete formation series (never fill gaps).
    closes = train_closes.dropna(axis=1, how="any")
    if closes.shape[1] < 2:
        return []
    log = np.log(closes)
    rets = log.diff().dropna()
    if len(rets) < 10:
        return []
    corr = rets.corr()
    tickers = list(closes.columns)

    models: list[PairModel] = []
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):
            a, b = tickers[i], tickers[j]
            c = corr.loc[a, b]
            if not np.isfinite(c) or c < cfg.min_corr:
                continue
            la = log[a].to_numpy()
            lb = log[b].to_numpy()
            alpha, beta = ols_hedge_ratio(la, lb)
            spread = la - alpha - beta * lb
            sigma = float(spread.std())
            if not np.isfinite(sigma) or sigma <= 0.0:
                continue
            adf_t = adf_tstat(spread)
            if adf_t >= cfg.adf_crit:
                continue
            models.append(
                PairModel(a=a, b=b, alpha=alpha, beta=beta,
                          mu=float(spread.mean()), sigma=sigma, adf_t=adf_t)
            )
    models.sort(key=lambda m: m.adf_t)  # most negative (most stationary) first
    return models[: cfg.max_pairs]
