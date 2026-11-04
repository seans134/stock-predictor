"""Cointegration statistics (pure numpy, no I/O).

Engle-Granger step 1 is an OLS hedge ratio; the residual spread's stationarity
is judged by an Augmented Dickey-Fuller t-statistic implemented here so the
project keeps no extra dependency. The ADF regression is

    d(y)_t = c + rho * y_{t-1} + sum_{i=1..maxlag} gamma_i * d(y)_{t-i} + e_t

and the returned statistic is the t-ratio of rho; a more negative value is
stronger evidence against a unit root (i.e. a more stationary, mean-reverting
series). A fixed large-sample 5% Dickey-Fuller critical value is used as the
gate (documented approximation; no small-sample p-value interpolation).
"""

from __future__ import annotations

import numpy as np

ADF_CRIT_5PCT = -2.86  # constant, no trend, large-sample DF 5% critical value


def ols_hedge_ratio(log_a: np.ndarray, log_b: np.ndarray) -> tuple[float, float]:
    log_a = np.asarray(log_a, dtype=float)
    log_b = np.asarray(log_b, dtype=float)
    X = np.column_stack([np.ones_like(log_b), log_b])
    coef, *_ = np.linalg.lstsq(X, log_a, rcond=None)
    return float(coef[0]), float(coef[1])


def adf_tstat(series: np.ndarray, maxlag: int = 1) -> float:
    y = np.asarray(series, dtype=float)
    dy = np.diff(y)                       # dy[k] = y[k+1] - y[k]
    n = len(dy)
    if n - maxlag < 10:                   # too short to estimate reliably
        return 0.0
    resp = dy[maxlag:]                     # d(y)_t
    level = y[maxlag:n]                    # y_{t-1}
    const = np.ones_like(resp)
    lags = [dy[maxlag - i: n - i] for i in range(1, maxlag + 1)]
    X = np.column_stack([level, const] + lags)
    coef, *_ = np.linalg.lstsq(X, resp, rcond=None)
    resid = resp - X @ coef
    dof = len(resp) - X.shape[1]
    if dof <= 0:
        return 0.0
    sigma2 = float(resid @ resid) / dof
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
    except np.linalg.LinAlgError:
        return 0.0
    se_rho = float(np.sqrt(sigma2 * xtx_inv[0, 0]))
    if se_rho == 0.0:
        return 0.0
    return float(coef[0] / se_rho)
