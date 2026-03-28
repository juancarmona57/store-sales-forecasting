"""Lag and rolling window feature engineering.

Creates time-lagged values and rolling statistics (mean, std)
grouped by store x family to capture temporal patterns.
"""

from typing import List

import numpy as np
import pandas as pd

from src.config import LAG_DAYS, ROLLING_WINDOWS, SAFE_SHIFT


def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    lags: List[int] | None = None,
    group_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Add lagged target features grouped by store and family."""
    df = df.copy()
    lags = lags or LAG_DAYS
    group_cols = group_cols or ["store_nbr", "family"]

    grouped = df.groupby(group_cols, observed=True)[target_col]

    for lag in lags:
        df[f"{target_col}_lag_{lag}"] = grouped.shift(lag)

    return df


def add_same_dow_lag_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    group_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Add same-day-of-week lag features.

    Instead of lag_16 (which could be any weekday), these lags capture
    the same weekday pattern: e.g., if today is Sunday, use last Sunday's sales.
    Multiples of 7 starting at >= 21 (safe for 16-day horizon):
      - lag_same_dow_3w = shift(21)  — same DOW, 3 weeks ago
      - lag_same_dow_4w = shift(28)
      - lag_same_dow_8w = shift(56)
      - lag_same_dow_52w = shift(364) — same DOW, ~1 year ago
    """
    df = df.copy()
    group_cols = group_cols or ["store_nbr", "family"]
    grouped = df.groupby(group_cols, observed=True)[target_col]

    # Same-DOW lags (multiples of 7, all >= 21 for safety)
    for weeks, shift_val in [(3, 21), (4, 28), (8, 56), (52, 364)]:
        df[f"{target_col}_lag_dow_{weeks}w"] = grouped.shift(shift_val)

    # Same-DOW rolling mean: average of last 4 same-weekday values (weeks 3,4,5,6)
    dow_lags = []
    for w in [3, 4, 5, 6]:
        col = f"_tmp_dow_{w}w"
        df[col] = grouped.shift(w * 7)
        dow_lags.append(col)

    df[f"{target_col}_dow_rolling_mean_4"] = df[dow_lags].mean(axis=1)
    df.drop(columns=dow_lags, inplace=True)

    return df


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str = "sales",
    windows: List[int] | None = None,
    group_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Add rolling mean and std features grouped by store and family.

    Uses shift(SAFE_SHIFT) to prevent leakage of current observation.
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
