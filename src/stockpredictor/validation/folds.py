"""Chronological walk-forward fold generation.

The unit of splitting is the whole trading day: every stock from the same
date lands in the same segment, so market conditions cannot leak between
training and testing. Random row-level splits are prohibited by the plan
and impossible to express here.

Segment layout per fold, in sessions:

    [ train ][ embargo ][ calibrate ][ embargo ][ test ]

- Rolling folds slide the train window forward by step_sessions per fold;
  expanding folds grow it from a fixed start.
- The embargo gap purges overlapping label windows and rolling-feature
  leakage across segment boundaries.
- holdout_sessions are cut off the end before any fold is generated: the
  final chronological holdout that must not be inspected until model and
  threshold selection is complete.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class WalkForwardConfig:
    train_sessions: int
    test_sessions: int
    step_sessions: int
    calibrate_sessions: int = 0
    embargo_sessions: int = 1
    expanding: bool = False
    holdout_sessions: int = 0

    def __post_init__(self):
        if min(self.train_sessions, self.test_sessions, self.step_sessions) <= 0:
            raise ValueError("train, test, and step session counts must be positive")
        if min(self.calibrate_sessions, self.embargo_sessions, self.holdout_sessions) < 0:
            raise ValueError("calibrate, embargo, and holdout counts must be non-negative")


@dataclass(frozen=True)
class Fold:
    fold_id: int
    train_days: tuple[dt.date, ...]
    calibrate_days: tuple[dt.date, ...]
    test_days: tuple[dt.date, ...]

    def summary(self) -> dict:
        return {
            "fold_id": self.fold_id,
            "train": [str(self.train_days[0]), str(self.train_days[-1])],
            "calibrate": (
                [str(self.calibrate_days[0]), str(self.calibrate_days[-1])]
                if self.calibrate_days
                else None
            ),
            "test": [str(self.test_days[0]), str(self.test_days[-1])],
        }


def generate_folds(sessions: Sequence[dt.date], config: WalkForwardConfig) -> list[Fold]:
    """Generate folds over an ordered list of trading days."""
    sessions = list(sessions)
    if sorted(sessions) != sessions:
        raise ValueError("sessions must be in chronological order")
    usable = sessions[: len(sessions) - config.holdout_sessions] if config.holdout_sessions else sessions

    folds: list[Fold] = []
    k = 0
    while True:
        train_start = 0 if config.expanding else k * config.step_sessions
        train_end = k * config.step_sessions + config.train_sessions
        cursor = train_end
        if config.calibrate_sessions:
            calib_start = cursor + config.embargo_sessions
            cursor = calib_start + config.calibrate_sessions
        else:
            calib_start = cursor  # empty
        test_start = cursor + config.embargo_sessions
        test_end = test_start + config.test_sessions
        if test_end > len(usable):
            break

        folds.append(
            Fold(
                fold_id=k,
                train_days=tuple(usable[train_start:train_end]),
                calibrate_days=tuple(
                    usable[calib_start : calib_start + config.calibrate_sessions]
                ),
                test_days=tuple(usable[test_start:test_end]),
            )
        )
        k += 1
    return folds


def holdout_days(sessions: Sequence[dt.date], config: WalkForwardConfig) -> tuple[dt.date, ...]:
    """The reserved final holdout. Do not touch until selection is complete."""
    if not config.holdout_sessions:
        return ()
    return tuple(sessions[len(sessions) - config.holdout_sessions :])
