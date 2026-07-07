import datetime as dt

import numpy as np
import pandas as pd
import pytest

from stockpredictor.execution.costs import CostModel
from stockpredictor.execution.decisions import HOLD, LONG, SHORT, DecisionConfig, decide
from stockpredictor.execution.simulator import simulate, summarize
from stockpredictor.sessions import ET

DAY = dt.date(2024, 3, 18)


def test_cost_model_matches_labeler_placeholder():
    costs = CostModel()
    # spread 5 + 2 * (5/2) slippage + 1 fee = 11 bps, same as LabelConfig.
    assert costs.round_trip_cost == pytest.approx(11e-4)

    from stockpredictor.labeling.labeler import LabelConfig

    assert costs.round_trip_cost == pytest.approx(LabelConfig().round_trip_cost)


def _decision_frame(**overrides):
    row = {
        "r30": 0.001,
        "minutes_to_close": 120.0,
    }
    row.update({k: v for k, v in overrides.items() if k in row})
    preds = {
        "q10": overrides.get("q10", -0.002),
        "q50": overrides.get("q50", 0.0030),  # 30bp, well above 11bp cost + 5bp edge
        "q90": overrides.get("q90", 0.008),
    }
    frame = pd.DataFrame([row])
    predictions = pd.DataFrame([preds])
    return frame, predictions


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({}, LONG),                                        # clears every gate
        ({"q50": 0.0010}, HOLD),                           # edge below cost+min_edge
        ({"q10": -0.02}, HOLD),                            # downside beyond risk cap
        ({"r30": -0.004}, HOLD),                           # trend strongly opposed
        ({"r30": np.nan}, HOLD),                           # missing trend never trades
        ({"minutes_to_close": 10.0}, HOLD),                # no room before the close
        ({"q50": -0.0030, "q10": -0.008, "q90": 0.002, "r30": -0.001}, SHORT),
        ({"q50": -0.0030, "q10": -0.008, "q90": 0.002, "r30": 0.004}, HOLD),  # short vs uptrend
    ],
)
def test_decision_gates(overrides, expected):
    frame, preds = _decision_frame(**overrides)
    result = decide(frame, preds, CostModel(), DecisionConfig())
    assert result.iloc[0] == expected


def _sim_rows(times_and_decisions, fwd=0.005):
    rows = []
    for hhmm, decision in times_and_decisions:
        hour, minute = map(int, hhmm.split(":"))
        rows.append(
            {
                "ticker": "AAA",
                "date": DAY,
                "bar_start": pd.Timestamp(2024, 3, 18, hour, minute, tz=ET),
                "decision": decision,
                "fwd_ret_15m": fwd,
            }
        )
    return pd.DataFrame(rows)


def test_simulator_blocks_overlapping_signals():
    frame = _sim_rows(
        [("10:00", LONG), ("10:05", LONG), ("10:10", LONG), ("10:15", LONG)]
    )
    trades = simulate(frame, CostModel())
    # 10:00 trade occupies through 10:15 exit; 10:05/10:10 skipped, 10:15 allowed.
    assert len(trades) == 2
    assert [ts.strftime("%H:%M") for ts in trades["bar_start"]] == ["10:00", "10:15"]


def test_simulator_pnl_arithmetic():
    frame = _sim_rows([("10:00", LONG), ("11:00", SHORT)], fwd=0.005)
    trades = simulate(frame, CostModel()).set_index("direction")
    cost = CostModel().round_trip_cost
    assert trades.loc[LONG, "gross_ret"] == pytest.approx(0.005)
    assert trades.loc[LONG, "net_ret"] == pytest.approx(0.005 - cost)
    assert trades.loc[SHORT, "gross_ret"] == pytest.approx(-0.005)
    assert trades.loc[SHORT, "net_ret"] == pytest.approx(-0.005 - cost)


def test_simulator_skips_undefined_forward_return():
    frame = _sim_rows([("15:50", LONG)], fwd=np.nan)
    trades = simulate(frame, CostModel())
    assert trades.empty


def test_summarize_counts():
    frame = _sim_rows([("10:00", LONG), ("11:00", HOLD), ("12:00", SHORT)], fwd=0.005)
    trades = simulate(frame, CostModel())
    summary = summarize(trades, n_decision_rows=3, n_days=1)
    assert summary["n_trades"] == 2
    assert summary["n_long"] == 1
    assert summary["n_short"] == 1
    assert summary["hold_fraction"] == pytest.approx(1 / 3)
    assert summary["hit_rate"] == pytest.approx(0.5)  # long wins, short loses

    empty = summarize(simulate(_sim_rows([]), CostModel()), 0, 1)
    assert empty["n_trades"] == 0 and empty["hold_fraction"] == 1.0
