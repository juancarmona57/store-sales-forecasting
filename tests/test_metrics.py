"""Tests for src/evaluation/metrics.py."""

import numpy as np
import pytest

from src.evaluation.metrics import rmsle


def test_rmsle_perfect_prediction():
    """RMSLE is 0 for perfect predictions."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    assert rmsle(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_rmsle_known_value():
    """RMSLE matches manual calculation."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.5, 2.5, 3.5])
    expected = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true)) ** 2))
    assert rmsle(y_true, y_pred) == pytest.approx(expected, rel=1e-6)


def test_rmsle_clips_negative_predictions():
    """Negative predictions are clipped to 0 before computing RMSLE."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([-1.0, 2.0, 3.0])
    result = rmsle(y_true, y_pred)
    assert result > 0


def test_rmsle_zero_values():
    """RMSLE handles zero values (log1p(0) = 0)."""
    y_true = np.array([0.0, 0.0, 1.0])
    y_pred = np.array([0.0, 0.0, 1.0])
    assert rmsle(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)
