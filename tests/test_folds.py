import datetime as dt
import json

import pytest

from stockpredictor.validation.folds import (
    Fold,
    WalkForwardConfig,
    generate_folds,
    holdout_days,
)
from stockpredictor.validation.harness import run_walkforward

SESSIONS = [dt.date(2024, 1, 1) + dt.timedelta(days=i) for i in range(300)]


def _positions(days):
    return [SESSIONS.index(d) for d in days]


def test_rolling_fold_layout():
    config = WalkForwardConfig(
        train_sessions=100, test_sessions=20, step_sessions=20,
        calibrate_sessions=20, embargo_sessions=1,
    )
    folds = generate_folds(SESSIONS, config)
    # test_end = k*20 + 100+1+20+1+20 <= 300  ->  k <= 7.9  ->  8 folds
    assert len(folds) == 8

    f0 = folds[0]
    assert _positions(f0.train_days) == list(range(0, 100))
    assert _positions(f0.calibrate_days) == list(range(101, 121))
    assert _positions(f0.test_days) == list(range(122, 142))

    f1 = folds[1]
    assert _positions(f1.train_days) == list(range(20, 120))  # window slides


def test_expanding_train_grows_from_fixed_start():
    config = WalkForwardConfig(
        train_sessions=100, test_sessions=20, step_sessions=20,
        embargo_sessions=1, expanding=True,
    )
    folds = generate_folds(SESSIONS, config)
    assert folds[0].train_days[0] == folds[3].train_days[0] == SESSIONS[0]
    assert len(folds[0].train_days) == 100
    assert len(folds[3].train_days) == 160


def test_segments_are_disjoint_and_ordered():
    config = WalkForwardConfig(
        train_sessions=100, test_sessions=20, step_sessions=20,
        calibrate_sessions=20, embargo_sessions=2,
    )
    for fold in generate_folds(SESSIONS, config):
        train, calib, test = set(fold.train_days), set(fold.calibrate_days), set(fold.test_days)
        assert not (train & calib) and not (train & test) and not (calib & test)
        assert max(fold.train_days) < min(fold.calibrate_days) < min(fold.test_days)
        # Embargo: at least `embargo` sessions between adjacent segments.
        assert SESSIONS.index(min(fold.calibrate_days)) - SESSIONS.index(max(fold.train_days)) > 2
        assert SESSIONS.index(min(fold.test_days)) - SESSIONS.index(max(fold.calibrate_days)) > 2


def test_holdout_never_appears_in_folds():
    config = WalkForwardConfig(
        train_sessions=100, test_sessions=20, step_sessions=20,
        embargo_sessions=1, holdout_sessions=50,
    )
    folds = generate_folds(SESSIONS, config)
    reserved = set(holdout_days(SESSIONS, config))
    assert len(reserved) == 50
    assert max(reserved) == SESSIONS[-1]
    for fold in folds:
        for days in (fold.train_days, fold.calibrate_days, fold.test_days):
            assert not (set(days) & reserved)
    # Holdout shrinks the usable range and therefore the fold count.
    no_holdout = generate_folds(SESSIONS, WalkForwardConfig(
        train_sessions=100, test_sessions=20, step_sessions=20, embargo_sessions=1,
    ))
    assert len(folds) < len(no_holdout)


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        WalkForwardConfig(train_sessions=0, test_sessions=20, step_sessions=20)
    with pytest.raises(ValueError):
        WalkForwardConfig(train_sessions=10, test_sessions=20, step_sessions=20, embargo_sessions=-1)


def test_run_walkforward_writes_experiment_record(tmp_path):
    config = WalkForwardConfig(train_sessions=100, test_sessions=20, step_sessions=50)
    folds = generate_folds(SESSIONS, config)

    record = run_walkforward(
        folds,
        run_fold=lambda fold: {"n_train": len(fold.train_days)},
        experiment_name="unit-test",
        config_record={"walkforward": config},
        out_dir=tmp_path,
    )
    assert record["n_folds"] == len(folds)
    on_disk = json.loads((tmp_path / record["path"].split("\\")[-1]).read_text())
    assert on_disk["experiment"] == "unit-test"
    assert on_disk["config"]["walkforward"]["train_sessions"] == 100
    assert all(f["metrics"]["n_train"] == 100 for f in on_disk["folds"])
