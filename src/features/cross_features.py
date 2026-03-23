"""Cross/interaction feature engineering.

Creates categorical interaction features by combining pairs of columns.
"""

import pandas as pd


def add_cross_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction features between key categorical columns.

    Creates: family x store_type, family x cluster, dow x family.

    Args:
        df: DataFrame with categorical and temporal columns.

    Returns:
        DataFrame with cross features added.
    """
    df = df.copy()

    if "family" in df.columns and "store_type" in df.columns:
        df["family_x_store_type"] = (
            df["family"].astype(str) + "_" + df["store_type"].astype(str)
        ).astype("category")

    if "family" in df.columns and "cluster" in df.columns:
        df["family_x_cluster"] = (
            df["family"].astype(str) + "_" + df["cluster"].astype(str)
        ).astype("category")

    if "day_of_week" in df.columns and "family" in df.columns:
        df["dow_x_family"] = (
            df["day_of_week"].astype(str) + "_" + df["family"].astype(str)
        ).astype("category")

    return df
