"""Lag and rolling window feature engineering.

Creates time-lagged values and rolling statistics (mean, std)
grouped by store x family to capture temporal patterns.
"""

from typing import List

import pandas as pd

from src.config import LAG_DAYS, ROLLING_WINDOWS


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    lags: List[int] | None = None,
    group_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Add lagged target features grouped by store and family.

    Args:
        df: DataFrame sorted by group_cols + date.
        target_col: Column to create lags for.
        lags: List of lag periods in days. Defaults to config.LAG_DAYS.
        group_cols: Columns to group by. Defaults to ["store_nbr", "family"].

    Returns:
        DataFrame with lag columns added (NaN where lag is unavailable).
    """
    df = df.copy()
    lags = lags or LAG_DAYS
    group_cols = group_cols or ["store_nbr", "family"]

    grouped = df.groupby(group_cols, observed=True)[target_col]

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = grouped.shift(lag)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    windows: List[int] | None = None,
    group_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std features grouped by store and family.

    Rolling windows are computed on shifted values (shift=1) to prevent
    including the current observation, which would cause data leakage.
    Rolling is applied within each group to prevent cross-group contamination.

    Args:
        df: DataFrame sorted by group_cols + date.
        target_col: Column to compute rolling stats for.
        windows: List of window sizes. Defaults to config.ROLLING_WINDOWS.
        group_cols: Columns to group by. Defaults to ["store_nbr", "family"].

    Returns:
        DataFrame with rolling mean/std columns added.
    """
    df = df.copy()
    windows = windows or ROLLING_WINDOWS
    group_cols = group_cols or ["store_nbr", "family"]

    for window in windows:
        # Shift by 1 within each group to exclude current row (prevent leakage)
        # Then apply rolling within each group to prevent cross-group contamination
        shifted = df.groupby(group_cols, observed=True)[target_col].shift(1)
        df[f"{target_col}_rolling_mean_{window}"] = (
            shifted.groupby(df[group_cols].apply(tuple, axis=1))
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{target_col}_rolling_std_{window}"] = (
            shifted.groupby(df[group_cols].apply(tuple, axis=1))
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

    return df
