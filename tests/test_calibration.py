import numpy as np
import pytest

from stockpredictor.stage1.calibration import DirichletCalibrator


def _overconfident_dataset(n: int, seed: int):
    """True probabilities, labels drawn from them, and an overconfident
    distortion (sharpened probabilities) as the 'raw classifier output'."""
    rng = np.random.default_rng(seed)
    true_p = rng.dirichlet(alpha=[4.0, 2.5, 2.0], size=n)
    labels = np.array([rng.choice(3, p=p) for p in true_p])
    sharpened = true_p ** 3
    sharpened /= sharpened.sum(axis=1, keepdims=True)
    return sharpened, labels


def _log_loss(proba: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(proba[np.arange(len(y)), y], 1e-12, 1))))


def test_calibration_improves_overconfident_probabilities():
    raw_fit, y_fit = _overconfident_dataset(2000, seed=1)
    raw_test, y_test = _overconfident_dataset(2000, seed=2)

    calibrator = DirichletCalibrator().fit(raw_fit, y_fit)
    calibrated = calibrator.transform(raw_test)

    assert calibrated.shape == raw_test.shape
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    # Held-out log loss must improve materially on overconfident input.
    assert _log_loss(calibrated, y_test) < _log_loss(raw_test, y_test) - 0.05


def test_calibration_handles_missing_class_in_fit():
    raw_fit, y_fit = _overconfident_dataset(500, seed=3)
    two_class = y_fit != 2  # calibrate segment happened to contain no 'sustained'
    calibrator = DirichletCalibrator().fit(raw_fit[two_class], y_fit[two_class])

    raw_test, _ = _overconfident_dataset(100, seed=4)
    calibrated = calibrator.transform(raw_test)
    assert calibrated.shape == (100, 3)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert (calibrated[:, 2] == 0).all()  # unseen class stays explicitly zero


def test_near_calibrated_input_is_not_wrecked():
    """Calibration on already-calibrated input should cost little."""
    rng = np.random.default_rng(5)
    true_p = rng.dirichlet(alpha=[4.0, 2.5, 2.0], size=3000)
    labels = np.array([rng.choice(3, p=p) for p in true_p])

    calibrator = DirichletCalibrator().fit(true_p[:1500], labels[:1500])
    calibrated = calibrator.transform(true_p[1500:])
    before = _log_loss(true_p[1500:], labels[1500:])
    after = _log_loss(calibrated, labels[1500:])
    assert after < before + 0.02
