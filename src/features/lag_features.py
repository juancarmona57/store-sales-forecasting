"""Lag and rolling window feature engineering.

Creates time-lagged values and rolling statistics (mean, std)
grouped by store x family to capture temporal patterns.
All lags and shifts use SAFE_SHIFT (>=16) to ensure features
are available during the 16-day test forecast horizon.
"""

from typing import List

import pandas as pd

from src.config import LAG_DAYS, ROLLING_WINDOWS, SAFE_SHIFT


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    lags: List[int] | None = None,
    group_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Add lagged target features grouped by store and family.

    All lags are >= SAFE_SHIFT (16) to ensure availability at inference.
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

    Uses SAFE_SHIFT (16) to prevent any leakage from the 16-day horizon.
    Rolling windows are computed on data shifted by SAFE_SHIFT days.
    """
    df = df.copy()
    windows = windows or ROLLING_WINDOWS
    group_cols = group_cols or ["store_nbr", "family"]

    for window in windows:
        shifted = df.groupby(group_cols, observed=True)[target_col].shift(SAFE_SHIFT)
        group_key = df[group_cols].apply(tuple, axis=1)
        df[f"{target_col}_rolling_mean_{window}"] = (
            shifted.groupby(group_key)
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        df[f"{target_col}_rolling_std_{window}"] = (
            shifted.groupby(group_key)
            .rolling(window=window, min_periods=1)
            .std()
            .reset_index(level=0, drop=True)
        )

    return df
