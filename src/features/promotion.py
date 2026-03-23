"""Promotion feature engineering.

Creates lag, rolling, and duration features for the onpromotion column.
"""

import pandas as pd
from typing import List


def add_promotion_features(
    df: pd.DataFrame,
    group_cols: List[str] | None = None,
) -> pd.DataFrame:
    """Add promotion-related features.

    Args:
        df: DataFrame with onpromotion column.
        group_cols: Columns to group by. Defaults to ["store_nbr", "family"].

    Returns:
        DataFrame with promotion features added.
    """
    df = df.copy()
    group_cols = group_cols or ["store_nbr", "family"]

    if "onpromotion" not in df.columns:
        return df

    grouped = df.groupby(group_cols, observed=True)["onpromotion"]

    # Lag features
    df["promo_lag_7"] = grouped.shift(7)
    df["promo_lag_14"] = grouped.shift(14)

    # Rolling promo frequency
    shifted = grouped.shift(1)
    df["promo_rolling_14"] = (
        shifted.groupby(df[group_cols].apply(tuple, axis=1))
        .rolling(14, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Consecutive promo days (duration)
    def _consecutive_promo(series):
        result = []
        count = 0
        for val in series:
            if val == 1:
                count += 1
            else:
                count = 0
            result.append(count)
        return result

    df["promo_duration"] = grouped.transform(
        lambda x: pd.Series(_consecutive_promo(x.values), index=x.index)
    )

    return df
