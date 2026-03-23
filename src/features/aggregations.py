"""Statistical aggregation features.

Computes mean/median/std of sales per store, family, and store x family
over historical windows using expanding/rolling calculations.
"""

from typing import List

import pandas as pd
import numpy as np


def add_aggregation_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """Add statistical aggregation features per store x family.

    Uses shifted expanding/rolling windows to prevent data leakage.

    Args:
        df: DataFrame sorted by store_nbr, family, date.
        target_col: Column to aggregate.
        windows: Rolling window sizes. Defaults to [30, 90].

    Returns:
        DataFrame with aggregation features.
    """
    df = df.copy()
    windows = windows or [30, 90]
    group_cols = ["store_nbr", "family"]

    shifted = df.groupby(group_cols, observed=True)[target_col].shift(1)
    group_key = df[group_cols].apply(tuple, axis=1)

    for window in windows:
        grouped_rolling = shifted.groupby(group_key)
        df[f"{target_col}_store_family_mean_{window}"] = (
            grouped_rolling.rolling(window=window, min_periods=window).mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{target_col}_store_family_std_{window}"] = (
            grouped_rolling.rolling(window=window, min_periods=window).std()
            .reset_index(level=0, drop=True)
        )

    # Expanding mean (coefficient of variation proxy)
    expanding = (
        shifted.groupby(group_key)
        .expanding(min_periods=7)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[f"{target_col}_expanding_mean"] = expanding

    return df
