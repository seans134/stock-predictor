import datetime as dt
from dataclasses import asdict
import json

import pandas as pd
import pytest

from stockpredictor.data.store import BarStore
from stockpredictor.data.synthetic import synthetic_bars
from stockpredictor.execution.portfolio import RiskConfig
from stockpredictor.research.__main__ import run
from stockpredictor.research.baselines import (BaselineConfig, gap_affected, gap_index, opening_volume,
                                               require_complete_evaluation)
from stockpredictor.research.corporate_actions import collect
from stockpredictor.research.governance import digest, register, write_new
from stockpredictor.research.reporting import screen_decision
from stockpredictor.sessions import SessionCalendar

DAY = dt.date(2024, 3, 18)
CFG = BaselineConfig(lookback=2)


@pytest.fixture(scope="module")
def calendar():
    return SessionCalendar(start="2024-01-01", end="2024-12-31")


def et(clock, day=DAY):
    return pd.Timestamp(f"{day} {clock}", tz="America/New_York")


def signal_bars(calendar, end=DAY):
    """Selected and entered on DAY at 09:45 at 100; the 0.2 stop is 99.8."""
    bars = synthetic_bars(calendar, dt.date(2024, 3, 11), end)
    bars[["open", "close", "vwap"]] = 100.0
    bars["high"], bars["low"], bars["volume"] = 101.0, 99.0, 100_000.0
    dates = bars.index.tz_convert("America/New_York").date
    bars.loc[dates == DAY, "low"] = 99.9
    bars.loc[et("09:30"), ["close", "volume"]] = [100.5, 300_000]
    bars.loc[et("09:35"), ["close", "high"]] = [102, 103]
    return bars


def gaps_for(*clocks, ticker="AAPL"):
    return gap_index([dict(ticker=ticker, event_ts=int(et(c).timestamp())) for c in clocks])


def test_only_the_exact_registered_gap_is_excused(calendar):
    bars = signal_bars(calendar).drop(et("12:35"))
    with pytest.raises(ValueError, match="Incomplete evaluation"):
        require_complete_evaluation({"AAPL": bars}, calendar, [DAY])
    require_complete_evaluation({"AAPL": bars}, calendar, [DAY], gaps_for("12:35"))
    with pytest.raises(ValueError, match="Incomplete evaluation"):
        require_complete_evaluation({"AAPL": bars.drop(et("13:00"))}, calendar, [DAY], gaps_for("12:35"))


def test_registered_gap_that_is_actually_present_blocks(calendar):
    with pytest.raises(ValueError, match="Incomplete evaluation"):
        require_complete_evaluation({"AAPL": signal_bars(calendar)}, calendar, [DAY], gaps_for("12:35"))


def test_unregistered_gap_still_fails_selected_trade(calendar):
    bars = signal_bars(calendar).drop(et("12:35"))
    with pytest.raises(ValueError, match="Incomplete selected"):
        opening_volume({"AAPL": bars}, calendar, [DAY], CFG, registered_gaps=gaps_for("13:00"))


def test_position_held_through_gap_exits_at_next_bar_open(calendar):
    bars = signal_bars(calendar)
    bars.loc[et("12:40"), ["open", "low", "close"]] = [99, 99, 99]
    trades, _ = opening_volume({"AAPL": bars.drop(et("12:35"))}, calendar, [DAY], CFG,
                               registered_gaps=gaps_for("12:35"))
    trade = trades.iloc[0]
    assert trade.exit_price == 99 and trade.exit_ts == et("12:40") and trade.exit_reason == "stop"
    assert gap_affected(trades, gaps_for("12:35")).all()


def test_gap_after_exit_does_not_mark_the_trade(calendar):
    bars = signal_bars(calendar)
    bars.loc[et("11:00"), ["open", "low", "close"]] = [99, 99, 99]
    trades, _ = opening_volume({"AAPL": bars.drop(et("12:35"))}, calendar, [DAY], CFG,
                               registered_gaps=gaps_for("12:35"))
    assert trades.iloc[0].exit_ts == et("11:00")
    assert not gap_affected(trades, gaps_for("12:35")).any()


def test_missing_bar_cannot_signal_but_a_later_bar_can(calendar):
    bars = signal_bars(calendar).drop(et("09:35"))
    bars.loc[et("09:40"), ["close", "high"]] = [102, 103]
    trades, _ = opening_volume({"AAPL": bars}, calendar, [DAY], CFG, registered_gaps=gaps_for("09:35"))
    assert trades.iloc[0].bar_start == et("09:40")
    assert trades.iloc[0].entry_ts == et("09:50")


def test_gap_session_keeps_the_stock_eligible_afterwards(calendar):
    prior = dt.date(2024, 3, 15)
    bars = signal_bars(calendar)
    gap = et("12:35", prior)
    gaps = gap_index([dict(ticker="AAPL", event_ts=int(gap.timestamp()))])
    trades, _ = opening_volume({"AAPL": bars.drop(gap)}, calendar, [DAY], CFG, registered_gaps=gaps)
    assert len(trades) == 1
    # Without the registered gap, the same session stays excluded from history.
    assert opening_volume({"AAPL": bars.drop(gap)}, calendar, [DAY], CFG)[0].empty


@pytest.mark.parametrize("ex_date,eligible", [
    (dt.date(2024, 3, 18), False),  # ex-date itself: all history is pre-split
    (dt.date(2024, 3, 15), False),  # one post-split session in a two-session lookback
    (dt.date(2024, 3, 14), True),   # the lookback is now wholly post-split
])
def test_split_blocks_until_lookback_is_post_split(calendar, ex_date, eligible):
    trades, _ = opening_volume({"AAPL": signal_bars(calendar)}, calendar, [DAY], CFG,
                               split_dates={"AAPL": [ex_date]})
    assert (len(trades) == 1) is eligible


@pytest.mark.parametrize("primary,control,stressed,expected", [
    (1.0, 0.5, 0.8, "pass_screen"),
    (0.0, -1.0, -1.0, "fail"),
    (1.0, 1.0, 1.0, "fail"),
    (1.0, 0.5, 0.5, "fail"),
    (1.0, -1.0, -0.1, "fail"),
])
def test_frozen_decision_rule(primary, control, stressed, expected):
    assert screen_decision(primary, control, stressed)[0] == expected


class Response:
    def __init__(self, status_code, payload=None):
        self.status_code, self._payload = status_code, payload or {}

    def json(self):
        return self._payload


def test_split_collection_records_failures_instead_of_no_splits():
    answers = {"AAA": Response(200, {"results": [
                   {"execution_date": "2025-11-17", "split_from": 1, "split_to": 10, "ticker": "AAA"}]}),
               "BBB": Response(403), "CCC": Response(200, {"results": [], "next_url": "x"}),
               "DDD": Response(200, {"results": []})}
    splits, failures = collect(list(answers), "2025-07-01", "2026-07-08", "k",
                               get=lambda url, params, headers, timeout: answers[params["ticker"]],
                               sleep=lambda _: None)
    assert splits == {"AAA": [{"execution_date": "2025-11-17", "split_from": 1, "split_to": 10}], "DDD": []}
    assert failures == {"BBB": "http_403", "CCC": "truncated_response"}


def test_runner_applies_gap_policy_stress_and_decision(tmp_path, calendar):
    data = tmp_path / "data"
    data.mkdir()
    bars = signal_bars(calendar, dt.date(2024, 3, 20))
    # Gains 2% if held to the close, so the failure comes from the stress rule alone.
    day = bars.index.tz_convert("America/New_York").date == DAY
    later = day & (bars.index >= et("12:40"))
    bars.loc[later, ["open", "close", "high"]] = [102, 102, 102.5]
    store = BarStore(data / "bars.sqlite")
    store.insert_bars("AAPL", bars.drop(et("12:35")), "synthetic")
    store.close()
    workspace = data / "edge_research"
    manifest = {"development_start": "2024-03-11", "development_end": "2024-03-20"}
    write_new(workspace / "split_manifest.json", manifest)
    write_new(data / "splits.json", {"splits": {}})
    policy = dict(version="h1-gap-v2", stress_net_return=-.05, split_ex_dates={},
                  registered_gaps=[dict(ticker="AAPL", event_ts=int(et("12:35").timestamp()))],
                  split_evidence=dict(path="data/splits.json", sha256=digest(data / "splits.json")))
    spec = dict(id="H1-test", family="H1", start="2024-03-18", end="2024-03-20", tickers=["AAPL"],
                parameters=asdict(CFG), risk=asdict(RiskConfig(max_name_fraction=.02, max_positions=20,
                                                               max_gross_fraction=.4, fixed_name_fraction=.02)),
                cost_scenarios_bps=[0, 11], data_sha256=digest(data / "bars.sqlite"),
                limitations=["synthetic"], gap_policy=policy)
    register(workspace / "registry/H1-test.json", spec, manifest, tmp_path)
    run(tmp_path, workspace, "H1-test")
    report = json.loads((workspace / "runs/H1-test/report.json").read_text())
    assert report["primary"]["daily_mean_bps"] > 0
    assert report["stress"]["gap_affected_signals"] == 1
    assert report["stress"]["daily_mean_bps"] < 0
    assert report["decision"] == "fail" and "stress" in report["decision_basis"]
    with pytest.raises(FileExistsError):
        run(tmp_path, workspace, "H1-test")
