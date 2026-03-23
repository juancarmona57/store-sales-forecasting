"""Tests for src/models/ensemble.py."""

import numpy as np
import pytest

from src.models.ensemble import weighted_average_ensemble, optimize_weights
from src.evaluation.metrics import rmsle


def test_weighted_average_equal_weights():
    """Equal weights produce simple mean."""
    preds = [
        np.array([1.0, 2.0, 3.0]),
        np.array([3.0, 4.0, 5.0]),
    ]
    weights = [0.5, 0.5]
    result = weighted_average_ensemble(preds, weights)
    expected = np.array([2.0, 3.0, 4.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_weighted_average_custom_weights():
    """Custom weights are applied correctly."""
    preds = [
        np.array([10.0, 20.0]),
        np.array([20.0, 40.0]),
    ]
    weights = [0.7, 0.3]
    result = weighted_average_ensemble(preds, weights)
    expected = np.array([13.0, 26.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_weighted_average_clips_negatives():
    """Ensemble predictions are clipped to >= 0."""
    preds = [np.array([-5.0, 1.0]), np.array([-3.0, 2.0])]
    weights = [0.5, 0.5]
    result = weighted_average_ensemble(preds, weights)
    assert (result >= 0).all()


def test_optimize_weights_returns_valid_weights():
    """Optimized weights sum to 1 and are all >= 0."""
    np.random.seed(42)
    y_true = np.abs(np.random.randn(100) * 50 + 100)
    preds = [
        y_true + np.random.randn(100) * 5,
        y_true + np.random.randn(100) * 10,
    ]
    weights = optimize_weights(preds, y_true)
    assert len(weights) == 2
    assert abs(sum(weights) - 1.0) < 1e-6
    assert all(w >= 0 for w in weights)
