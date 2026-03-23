"""Time series cross-validation utilities.

Provides expanding-window splits with configurable gap
to prevent data leakage in time series forecasting.
"""

from typing import Generator, Tuple

import numpy as np
import pandas as pd


class TimeSeriesSplitWithGap:
    """Expanding-window time series splitter with a gap between train and val.

    The gap prevents lag/rolling features from leaking future information
    into the training set.

    Args:
        n_splits: Number of cross-validation folds.
        gap_days: Number of days gap between train and validation sets.
        val_days: Number of days in each validation set.
    """

    def __init__(self, n_splits: int = 5, gap_days: int = 16, val_days: int = 16):
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.val_days = val_days

    def split(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/validation index splits.

        Args:
            df: DataFrame with a date column.
            date_col: Name of the date column.

        Yields:
            Tuples of (train_indices, val_indices) as numpy arrays.
        """
        dates = df[date_col].sort_values().unique()
        max_date = dates.max()

        for i in range(self.n_splits):
            val_end = max_date - pd.Timedelta(days=i * self.val_days)
            val_start = val_end - pd.Timedelta(days=self.val_days - 1)
            train_end = val_start - pd.Timedelta(days=self.gap_days)

            train_mask = df[date_col] <= train_end
            val_mask = (df[date_col] >= val_start) & (df[date_col] <= val_end)

            train_idx = df.index[train_mask].values
            val_idx = df.index[val_mask].values

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx

    def get_holdout_split(
        self, df: pd.DataFrame, date_col: str = "date"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get a single holdout split using the last val_days as validation.

        Args:
            df: DataFrame with a date column.
            date_col: Name of the date column.

        Returns:
            Tuple of (train_indices, val_indices).
        """
        max_date = df[date_col].max()
        val_start = max_date - pd.Timedelta(days=self.val_days - 1)
        train_end = val_start - pd.Timedelta(days=self.gap_days)

        train_mask = df[date_col] <= train_end
        val_mask = df[date_col] >= val_start

        return df.index[train_mask].values, df.index[val_mask].values
