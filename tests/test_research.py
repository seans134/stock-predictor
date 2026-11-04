import datetime as dt
import json
from dataclasses import asdict

import pandas as pd
import pytest

from stockpredictor.data.store import BarStore
from stockpredictor.data.synthetic import synthetic_bars
from stockpredictor.execution.portfolio import RiskConfig
from stockpredictor.research.__main__ import load_development, run
from stockpredictor.research.baselines import BaselineConfig, closing_momentum, opening_volume
from stockpredictor.research.governance import (digest, inventory, register, require_development,
                                               verify_registration, write_new)
from stockpredictor.research.reporting import compare
from stockpredictor.sessions import SessionCalendar


@pytest.fixture(scope="module")
def calendar():
    return SessionCalendar(start="2024-01-01", end="2024-12-31")


def test_inventory_locks_everything_after_exposed_dates(tmp_path):
    write_new(tmp_path / "old.json", {"folds": [{"train": ["2024-01-02", "2024-02-01"],
                                                "test": ["2024-02-05", "2024-02-09"]}]})
    manifest = inventory(tmp_path)
    require_development(manifest, "2024-01-02", "2024-02-09")
    for start, end in [("2024-01-01","2024-02-09"), ("2024-02-05","2024-02-12")]:
        with pytest.raises(ValueError, match="protected"):
            require_development(manifest, start, end)
    assert manifest["records"][0]["sha256"] == digest(tmp_path / "old.json")


def test_immutable_registration_detects_source_and_spec_changes(tmp_path):
    manifest = {"development_start": "2024-01-02", "development_end": "2024-03-01"}
    spec = {"start": "2024-02-01", "end": "2024-02-09"}
    record = register(tmp_path / "reg.json", spec, manifest, tmp_path)
    assert verify_registration(record, manifest, tmp_path) == spec
    with pytest.raises(FileExistsError):
        register(tmp_path / "reg.json", spec, manifest, tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src/code.py").write_text("changed")
    with pytest.raises(ValueError, match="Source"):
        verify_registration(record, manifest, tmp_path)
    record["spec"]["end"] = "2024-02-12"
    with pytest.raises(ValueError, match="modified"):
        verify_registration(record, manifest, tmp_path)


def test_loader_rejects_protected_dates_before_opening_database(tmp_path):
    manifest = {"development_start": "2024-01-02", "development_end": "2024-03-01"}
    with pytest.raises(ValueError, match="protected"):
        load_development(tmp_path / "does-not-exist.db", ["SPY"], "2024-03-04", "2024-03-05", manifest)
    with pytest.raises(ValueError, match="Unsupported"):
        load_development(tmp_path / "does-not-exist.db", ["C:EURUSD"], "2024-02-01", "2024-02-02", manifest)


def _constant_bars(calendar, start, end):
    frame = synthetic_bars(calendar, start, end)
    frame[["open","close","vwap"]] = 100.0
    frame["high"], frame["low"], frame["volume"] = 101.0, 99.0, 100_000.0
    return frame


def test_h1_uses_prior_opening_volume_and_delayed_entry(calendar):
    bars = _constant_bars(calendar, dt.date(2024,3,11), dt.date(2024,3,18))
    day = dt.date(2024,3,18)
    opening = pd.Timestamp(calendar.session_open(day))
    bars.loc[opening, ["close","volume"]] = [100.5, 300_000]
    bars.loc[opening+pd.Timedelta(minutes=5), ["close","high"]] = [102,103]
    trades, selected = opening_volume({"AAPL":bars}, calendar, [day], BaselineConfig(lookback=2))
    assert selected.iloc[0].rvol == 3
    assert trades.iloc[0].entry_ts == opening+pd.Timedelta(minutes=15)
    assert trades.iloc[0].decision_ts == opening+pd.Timedelta(minutes=10, seconds=5)
    # Changing future volume never changes the already made selection.
    changed = bars.copy()
    changed.loc[changed.index > opening, "volume"] *= 100
    _, second = opening_volume({"AAPL":changed}, calendar, [day], BaselineConfig(lookback=2))
    pd.testing.assert_frame_equal(selected, second)


def test_missing_future_outcome_is_not_dropped_silently(calendar):
    bars = _constant_bars(calendar, dt.date(2024,3,11), dt.date(2024,3,18))
    day = dt.date(2024,3,18)
    opening = pd.Timestamp(calendar.session_open(day))
    bars.loc[opening, ["close","volume"]] = [100.5,300_000]
    bars.loc[opening+pd.Timedelta(minutes=5), ["close","high"]] = [102,103]
    bars = bars.drop(pd.Timestamp(calendar.session_close(day))-pd.Timedelta(minutes=5))
    with pytest.raises(ValueError, match="Incomplete selected"):
        opening_volume({"AAPL":bars}, calendar, [day], BaselineConfig(lookback=2))


def test_closing_momentum_respects_early_close_and_long_only(calendar):
    day = dt.date(2024,11,29)
    bars = _constant_bars(calendar, dt.date(2024,11,27), day)
    signal = pd.Timestamp(calendar.session_open(day))+pd.Timedelta(minutes=25)
    bars.loc[signal, ["close","high"]] = [102,103]
    trades = closing_momentum(bars, calendar, [day])
    assert len(trades) == 1
    assert trades.iloc[0].entry_ts.hour == 12 and trades.iloc[0].entry_ts.minute == 30
    assert trades.iloc[0].exit_ts.hour == 13
    bars.loc[signal, ["close","low"]] = [98,97]
    assert closing_momentum(bars, calendar, [day]).empty
    assert len(closing_momentum(bars, calendar, [day], unconditional=True)) == 1


@pytest.mark.parametrize("reconstruct", [False, True])
def test_runner_end_to_end_logs_zero_days_and_refuses_rerun(tmp_path, calendar, reconstruct):
    data = tmp_path / "data"
    data.mkdir()
    bars = _constant_bars(calendar, dt.date(2024,3,11), dt.date(2024,3,20))
    # Only March 18 has a positive first-half-hour signal.
    bars.loc[pd.Timestamp("2024-03-18 09:55", tz="America/New_York"), ["close","high"]] = [102,103]
    store = BarStore(data / "bars.sqlite")
    missing = bars.index.tz_convert("America/New_York").date == dt.date(2024,3,19)
    store.insert_bars("SPY", bars[~missing] if reconstruct else bars, "synthetic")
    store.close()
    workspace = data / "edge_research"
    manifest = {"development_start":"2024-03-11", "development_end":"2024-03-20"}
    write_new(workspace / "split_manifest.json", manifest)
    spec = {"id":"H3-test", "family":"H3", "start":"2024-03-18","end":"2024-03-20",
            "tickers":["SPY"],"parameters":asdict(BaselineConfig()),
            "risk":asdict(RiskConfig(fixed_name_fraction=.02)), "cost_scenarios_bps":[0,11,20],
            "data_sha256":digest(data / "bars.sqlite"), "limitations":["synthetic"]}
    if reconstruct:
        from stockpredictor.research.recovery import POLICY, seal
        from stockpredictor.research.governance import fingerprint
        folder = data / "reconstruction"
        job = dict(ticker="SPY", start="2024-03-19", end="2024-03-19", file="SPY.json")
        plan = dict(policy=POLICY, database_sha256=spec["data_sha256"], jobs=[job],
                    missing=[dict(ticker="SPY", event_ts=int(ts.timestamp())) for ts in bars[missing].index])
        plan["sha256"] = fingerprint(plan)
        write_new(folder / "plan.json", plan)
        write_new(folder / "responses/SPY.json", dict(job=job, adjusted=False,
            received_at="2026-10-21T16:00:00Z", bars=[dict(t=int(ts.timestamp())*1000,
            o=row.open, h=row.high, l=row.low, c=row.close, v=row.volume) for ts, row in bars[missing].iterrows()]))
        seal(folder)
        spec["reconstruction"] = dict(directory="data/reconstruction",
            manifest_sha256=digest(folder / "manifest.json"), policy=POLICY)
    register(workspace / "registry/H3-test.json", spec, manifest, tmp_path)
    run(tmp_path, workspace, "H3-test")
    output = workspace / "runs/H3-test"
    daily = pd.read_csv(output / "daily.csv")
    report = json.loads((output / "report.json").read_text())
    assert len(daily) == 3 and (daily["return"] == 0).sum() == 2
    assert report["primary"]["n_taken"] == 1
    assert report["decision"] == "fail"
    assert report["primary"]["max_drawdown"] < 0
    assert digest(data / "bars.sqlite") == spec["data_sha256"]
    with pytest.raises(FileExistsError):
        run(tmp_path, workspace, "H3-test")
