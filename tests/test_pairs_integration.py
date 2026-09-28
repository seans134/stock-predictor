"""A diverge-then-reconverge pair yields one profitable round-trip; selection
uses train days only (truncating trade-window data does not change the model)."""

import datetime as dt

import numpy as np
import pandas as pd

from stockpredictor.execution.costs import CostModel
from stockpredictor.pairs.backtest import run_pairs
from stockpredictor.pairs.selection import PairsConfig, select_pairs
from stockpredictor.sessions import ET


def _panel(series):
    n = len(next(iter(series.values())))
    idx = [dt.date(2024, 1, 1) + dt.timedelta(days=k) for k in range(n)]
    return pd.DataFrame(series, index=idx)


def test_select_then_trade_roundtrip():
    rng = np.random.default_rng(7)
    n_train = 250
    base = np.cumsum(rng.normal(0, 0.01, n_train)) + 4.0
    noise = np.zeros(n_train)
    for t in range(1, n_train):
        noise[t] = 0.4 * noise[t - 1] + rng.normal(0, 0.01)
    train = _panel({"A": np.exp(1.0 + base + noise), "B": np.exp(base)})
    models = select_pairs(train, PairsConfig(min_corr=0.0, max_pairs=5))
    assert any((m.a, m.b) == ("A", "B") for m in models)

    model = next(m for m in models if (m.a, m.b) == ("A", "B"))
    # Trade window: force the spread to diverge (below -z_entry) then reconverge to mu.
    sig, mu = model.sigma, model.mu
    z_path = [-3.0, -3.0, -1.0, 0.0, 0.0]
    a = [float(np.exp(mu + z * sig + model.alpha)) for z in z_path]  # logA-alpha-beta*logB=mu+z*sig, logB=0
    test = _panel({"A": a, "B": [1.0] * len(z_path)})
    session_ts = {d: pd.Timestamp(d.year, d.month, d.day, 16, 0, tz=ET) for d in test.index}
    trades = run_pairs([model], test, PairsConfig(z_entry=2.0, z_exit=0.5),
                       CostModel(), session_ts)
    assert len(trades) == 1
    assert trades.iloc[0]["gross_ret"] > 0
