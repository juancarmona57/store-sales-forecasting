"""Ensemble methods for combining multiple model predictions.

Supports weighted averaging with weight optimization via
scipy.optimize to minimize RMSLE on hold-out data.
"""

import logging
from typing import List

import numpy as np
from scipy.optimize import minimize

from src.evaluation.metrics import rmsle

logger = logging.getLogger(__name__)


def weighted_average_ensemble(
    predictions: List[np.ndarray],
    weights: List[float],
) -> np.ndarray:
    """Combine predictions using weighted average.

    Args:
        predictions: List of prediction arrays from different models.
        weights: List of weights (should sum to 1).

    Returns:
        Weighted average predictions, clipped to >= 0.
    """
    if len(predictions) != len(weights):
        raise ValueError(
            f"Number of predictions ({len(predictions)}) must match "
            f"number of weights ({len(weights)})"
        )

    result = np.zeros_like(predictions[0], dtype=np.float64)
    for pred, weight in zip(predictions, weights):
        result += weight * np.asarray(pred, dtype=np.float64)

    return np.clip(result, 0, None)


def optimize_weights(
    predictions: List[np.ndarray],
    y_true: np.ndarray,
) -> List[float]:
    """Find optimal ensemble weights by minimizing RMSLE.

    Uses scipy.optimize.minimize with constraints that weights
    sum to 1 and are all non-negative.

    Args:
        predictions: List of prediction arrays from different models.
        y_true: True target values.

    Returns:
        List of optimized weights.
    """
    n_models = len(predictions)

    def objective(weights):
        ensemble_pred = weighted_average_ensemble(predictions, weights.tolist())
        return rmsle(y_true, ensemble_pred)

    # Initial weights: equal
    x0 = np.ones(n_models) / n_models

    # Constraints: weights sum to 1
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    # Bounds: each weight between 0 and 1
    bounds = [(0.0, 1.0)] * n_models

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1000},
    )

    optimal_weights = result.x.tolist()
    logger.info(
        "Optimized ensemble weights: %s (RMSLE: %.6f)",
        [f"{w:.4f}" for w in optimal_weights],
        result.fun,
    )

    return optimal_weights
