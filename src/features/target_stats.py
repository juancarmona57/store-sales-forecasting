"""Target encoding statistics.

Computes static historical statistics (mean, median) by various groupings
from training data. These are static lookup features that are always
available at inference time (unlike lag features).
"""

import pandas as pd
import numpy as np


def compute_target_stats(train_df: pd.DataFrame, target_col: str = "sales") -> dict:
    """Compute target statistics from training data.

    Returns a dict of DataFrames that can be merged onto any dataset.
    """
    stats = {}

    # Store x Family mean/median
    sf = train_df.groupby(["store_nbr", "family"])[target_col].agg(["mean", "median", "std"]).reset_index()
    sf.columns = ["store_nbr", "family", "sf_mean", "sf_median", "sf_std"]
    stats["store_family"] = sf

    # Store x Family x DayOfWeek mean
    train_df = train_df.copy()
    train_df["_dow"] = train_df["date"].dt.dayofweek
    sfd = train_df.groupby(["store_nbr", "family", "_dow"])[target_col].mean().reset_index()
    sfd.columns = ["store_nbr", "family", "_dow", "sf_dow_mean"]
    stats["store_family_dow"] = sfd

    # Family mean
    fm = train_df.groupby("family")[target_col].agg(["mean", "std"]).reset_index()
    fm.columns = ["family", "family_mean", "family_std"]
    stats["family"] = fm

    # Store mean
    sm = train_df.groupby("store_nbr")[target_col].agg(["mean", "std"]).reset_index()
    sm.columns = ["store_nbr", "store_mean", "store_std"]
    stats["store"] = sm

    return stats


def apply_target_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Merge pre-computed target statistics onto a DataFrame."""
    df = df.copy()
    df["_dow"] = df["date"].dt.dayofweek

    # Store x Family
    df = df.merge(stats["store_family"], on=["store_nbr", "family"], how="left")

    # Store x Family x DayOfWeek
    df = df.merge(stats["store_family_dow"], on=["store_nbr", "family", "_dow"], how="left")

    # Family
    df = df.merge(stats["family"], on="family", how="left")

    # Store
    df = df.merge(stats["store"], on="store_nbr", how="left")

    df.drop(columns=["_dow"], inplace=True)

    return df
