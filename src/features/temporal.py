"""Temporal and calendar feature engineering.

Extracts time-based features from the date column including
day of week, month, cyclical encodings, and business day indicators.
"""

import numpy as np
import pandas as pd


def add_temporal_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """Add calendar and cyclical time features.

    Features added:
    - day, day_of_week, month, year, week_of_year, quarter, day_of_year
    - is_weekend, is_month_start, is_month_end
    - dow_sin, dow_cos (cyclical day of week)
    - month_sin, month_cos (cyclical month)

    Args:
        df: DataFrame with a datetime column.
        date_col: Name of the date column.

    Returns:
        New DataFrame with temporal features added.
    """
    df = df.copy()
    dt = df[date_col].dt

    # Basic calendar features
    df["day"] = dt.day
    df["day_of_week"] = dt.dayofweek
    df["month"] = dt.month
    df["year"] = dt.year
    df["week_of_year"] = dt.isocalendar().week.astype(int)
    df["quarter"] = dt.quarter
    df["day_of_year"] = dt.dayofyear

    # Boolean indicators
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = dt.is_month_start.astype(int)
    df["is_month_end"] = dt.is_month_end.astype(int)

    # Cyclical encoding (preserves circular nature of time)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    return df
