import numpy as np
import pytest

from stockpredictor.pairs.stats import ADF_CRIT_5PCT, adf_tstat, ols_hedge_ratio


def test_ols_hedge_ratio_recovers_known_beta():
    rng = np.random.default_rng(0)
    log_b = np.linspace(1.0, 3.0, 200) + rng.normal(0, 0.01, 200)
    log_a = 2.0 + 1.5 * log_b + rng.normal(0, 1e-6, 200)
    alpha, beta = ols_hedge_ratio(log_a, log_b)
    assert alpha == pytest.approx(2.0, abs=1e-2)
    assert beta == pytest.approx(1.5, abs=1e-2)


def test_adf_stationary_series_is_significantly_negative():
    # Stationary AR(1): x_t = 0.3 x_{t-1} + e_t  -> strongly mean-reverting.
    rng = np.random.default_rng(1)
    n = 400
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = 0.3 * x[t - 1] + rng.normal(0, 1)
    t_stat = adf_tstat(x)
    assert t_stat < ADF_CRIT_5PCT  # rejects the unit root


def test_adf_random_walk_is_not_significant():
    # Random walk: x_t = x_{t-1} + e_t -> unit root, should NOT be significant.
    rng = np.random.default_rng(2)
    x = np.cumsum(rng.normal(0, 1, 400))
    t_stat = adf_tstat(x)
    assert t_stat > ADF_CRIT_5PCT  # fails to reject the unit root


def test_adf_short_series_returns_zero():
    assert adf_tstat(np.array([1.0, 2.0, 3.0])) == 0.0
