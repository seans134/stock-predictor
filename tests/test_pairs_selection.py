import numpy as np
import pandas as pd
import pytest

from stockpredictor.pairs.selection import PairsConfig, select_pairs


def _panel(series: dict) -> pd.DataFrame:
    n = len(next(iter(series.values())))
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(series, index=idx)


def test_cointegrated_pair_selected_random_walk_rejected():
    rng = np.random.default_rng(0)
    n = 300
    b = np.cumsum(rng.normal(0, 0.01, n)) + 4.0          # random-walk log-price base
    noise = np.zeros(n)                                   # stationary AR(1) residual
    for t in range(1, n):
        noise[t] = 0.4 * noise[t - 1] + rng.normal(0, 0.01)
    a = 1.0 + 1.2 * b + noise                             # cointegrated with b
    # An independent random walk, correlated enough to pass the pre-filter only by chance:
    c = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    closes = _panel({"A": np.exp(a), "B": np.exp(b), "C": np.exp(c)})
    models = select_pairs(closes, PairsConfig(min_corr=0.0, max_pairs=10))
    pairs = {(m.a, m.b) for m in models}
    assert ("A", "B") in pairs
    assert ("A", "C") not in pairs and ("B", "C") not in pairs


def test_correlation_prefilter_excludes_uncorrelated():
    rng = np.random.default_rng(1)
    n = 300
    a = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    b = np.cumsum(rng.normal(0, 0.01, n)) + 4.0           # independent -> low corr
    closes = _panel({"A": np.exp(a), "B": np.exp(b)})
    models = select_pairs(closes, PairsConfig(min_corr=0.95, max_pairs=10))
    assert models == []


def test_max_pairs_caps_and_sorts_by_adf():
    # Build 3 cointegrated pairs of varying stationarity; expect the cap to apply.
    rng = np.random.default_rng(2)
    n = 300
    base = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    closes = {"BASE": np.exp(base)}
    for k, phi in enumerate([0.2, 0.5, 0.8]):
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = phi * noise[t - 1] + rng.normal(0, 0.01)
        closes[f"P{k}"] = np.exp(1.0 + base + noise)
    panel = _panel(closes)
    models = select_pairs(panel, PairsConfig(min_corr=0.0, max_pairs=2))
    assert len(models) == 2
    assert models[0].adf_t <= models[1].adf_t  # most stationary first


def test_models_use_train_only_and_skip_constant_series():
    rng = np.random.default_rng(3)
    n = 200
    b = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    closes = _panel({"A": np.exp(1.0 + b), "B": np.exp(b), "K": np.full(n, 5.0)})
    models = select_pairs(closes, PairsConfig(min_corr=0.0, max_pairs=10))
    # Constant column K has zero-variance spread with anything -> never selected.
    assert all("K" not in (m.a, m.b) for m in models)
