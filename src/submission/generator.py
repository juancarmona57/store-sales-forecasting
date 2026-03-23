"""Submission generation and post-processing.

Handles the final step of the pipeline: transforming raw model
predictions into a valid Kaggle submission file.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def postprocess_predictions(
    predictions: np.ndarray, threshold: float | None = None
) -> np.ndarray:
    """Post-process predictions for submission.

    - Clips negative values to 0
    - Rounds very small values to 0

    Args:
        predictions: Raw model predictions.
        threshold: Values below this are rounded to 0. Uses config default if None.

    Returns:
        Post-processed predictions.
    """
    from src.config import PREDICTION_CLIP_THRESHOLD

    if threshold is None:
        threshold = PREDICTION_CLIP_THRESHOLD
    preds = np.clip(predictions, 0, None)
    preds[preds < threshold] = 0.0
    return preds


def generate_submission(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    save_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate a Kaggle submission DataFrame.

    Args:
        test_df: Test DataFrame with 'id' column.
        predictions: Array of predictions (one per test row).
        save_path: If provided, saves submission CSV to this path.

    Returns:
        DataFrame with 'id' and 'sales' columns.
    """
    if len(test_df) != len(predictions):
        raise ValueError(
            f"Test rows ({len(test_df)}) != predictions ({len(predictions)})"
        )

    preds = postprocess_predictions(predictions)

    submission = pd.DataFrame({
        "id": test_df["id"].values,
        "sales": preds,
    })

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(save_path, index=False)
        logger.info("Submission saved to %s (%d rows)", save_path, len(submission))

    return submission
