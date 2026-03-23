"""Evaluation metrics for the Store Sales competition.

Primary metric: RMSLE (Root Mean Squared Logarithmic Error).
"""

import numpy as np
from numpy.typing import ArrayLike


def rmsle(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute Root Mean Squared Logarithmic Error.

    RMSLE = sqrt(mean((log(1 + y_pred) - log(1 + y_true))^2))

    Negative predictions are clipped to 0 before computation.

    Args:
        y_true: Array of true values (must be >= 0).
        y_pred: Array of predicted values (negatives clipped to 0).

    Returns:
        RMSLE score (lower is better).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_pred = np.clip(y_pred, 0, None)

    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return float(np.sqrt(np.mean(log_diff ** 2)))
