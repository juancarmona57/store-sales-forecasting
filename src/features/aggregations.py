"""Statistical aggregation features.

Computes mean/median/std of sales per store, family, and store x family
over historical windows using expanding/rolling calculations.
Uses SAFE_SHIFT to ensure availability at inference.
"""

from typing import List

import pandas as pd
import numpy as np

from src.config import SAFE_SHIFT


def add_aggregation_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    windows: List[int] | None = None,
) -> pd.DataFrame:
    """Add statistical aggregation features per store x family.

    Uses SAFE_SHIFT to prevent data leakage during 16-day forecast horizon.
    """
    df = df.copy()
    windows = windows or [30, 90]
    group_cols = ["store_nbr", "family"]

    shifted = df.groupby(group_cols, observed=True)[target_col].shift(SAFE_SHIFT)
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

    expanding = (
        shifted.groupby(group_key)
        .expanding(min_periods=7)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df[f"{target_col}_expanding_mean"] = expanding

    return df
