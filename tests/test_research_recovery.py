import json

import pandas as pd
import pytest

from stockpredictor.research.governance import digest, fingerprint, write_new
from stockpredictor.research.recovery import POLICY, additions, overlay, seal


def evidence(tmp_path, duplicate=False):
    ts = int(pd.Timestamp("2026-07-01 13:30Z").timestamp())
    job = dict(ticker="SPY", start="2026-07-01", end="2026-07-01", file="SPY.json")
    plan = dict(policy=POLICY, database_sha256="original", jobs=[job],
                missing=[dict(ticker="SPY", event_ts=ts), dict(ticker="SPY", event_ts=ts+300)])
    plan["sha256"] = fingerprint(plan)
    write_new(tmp_path / "plan.json", plan)
    bar = dict(t=ts*1000, o=100., h=102., l=99., c=101., v=1000.)
    bars = [bar, bar] if duplicate else [bar, {**bar, "t": (ts+600)*1000}]
    write_new(tmp_path / "responses/SPY.json", dict(job=job, adjusted=False,
        received_at="2026-10-21T16:00:00+00:00", bars=bars))
    return ts


def test_recovery_only_adds_audited_timestamps_preserves_observation_time(tmp_path):
    ts = evidence(tmp_path)
    seal(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["recovered_bars"] == 1
    assert manifest["unresolved_bars"] == 1
    idx = pd.to_datetime([ts-300], unit="s", utc=True)
    original = pd.DataFrame(dict(open=[90.], high=[92.], low=[89.], close=[91.], volume=[10.],
                                 available_ts=idx+pd.Timedelta(seconds=305)), index=idx)
    result = overlay({"SPY": original}, tmp_path, digest(tmp_path / "manifest.json"), "original")["SPY"]
    assert len(result) == 2
    assert result.iloc[0].close == 91.
    assert "available_ts" not in result
    assert result.iloc[-1].recovery_observed_at == pd.Timestamp("2026-10-21T16:00:00Z")
    assert result.iloc[-1].modeled_available_ts == pd.Timestamp(ts+305, unit="s", tz="UTC")
    assert "available_ts" in original  # original object also remains unchanged


def test_recovery_rejects_source_tampering_and_existing_bar(tmp_path):
    ts = evidence(tmp_path)
    seal(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    frame = pd.DataFrame({"close": [1.]}, index=pd.to_datetime([ts], unit="s", utc=True))
    with pytest.raises(ValueError, match="replace"):
        overlay({"SPY": frame}, tmp_path, digest(tmp_path / "manifest.json"), "original")
    path = tmp_path / "responses/SPY.json"
    path.write_text(path.read_text()+" ")
    with pytest.raises(ValueError, match="evidence changed"):
        additions(tmp_path, manifest)


def test_recovery_rejects_duplicate_bars_and_cannot_reseal(tmp_path):
    evidence(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate"):
        seal(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_recovery_seal_is_immutable(tmp_path):
    evidence(tmp_path)
    seal(tmp_path)
    with pytest.raises(FileExistsError):
        seal(tmp_path)


def test_recovery_rejects_incorrect_database(tmp_path):
    evidence(tmp_path)
    seal(tmp_path)
    with pytest.raises(ValueError, match="database/policy"):
        overlay({}, tmp_path, digest(tmp_path / "manifest.json"), "changed")
