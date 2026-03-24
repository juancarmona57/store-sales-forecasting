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

    # Store x Family mean/median/std/quantiles
    sf = train_df.groupby(["store_nbr", "family"])[target_col].agg(
        ["mean", "median", "std", lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()
    sf.columns = ["store_nbr", "family", "sf_mean", "sf_median", "sf_std", "sf_q25", "sf_q75"]
    stats["store_family"] = sf

    # Store x Family zero ratio
    sf_zero = train_df.groupby(["store_nbr", "family"])[target_col].apply(
        lambda x: (x == 0).mean()
    ).reset_index()
    sf_zero.columns = ["store_nbr", "family", "sf_zero_ratio"]
    stats["store_family_zero"] = sf_zero

    # Store x Family x DayOfWeek mean
    train_df = train_df.copy()
    train_df["_dow"] = train_df["date"].dt.dayofweek
    sfd = train_df.groupby(["store_nbr", "family", "_dow"])[target_col].mean().reset_index()
    sfd.columns = ["store_nbr", "family", "_dow", "sf_dow_mean"]
    stats["store_family_dow"] = sfd

    # Family mean/std
    fm = train_df.groupby("family")[target_col].agg(["mean", "std"]).reset_index()
    fm.columns = ["family", "family_mean", "family_std"]
    stats["family"] = fm

    # Family x DayOfWeek mean
    fd = train_df.groupby(["family", "_dow"])[target_col].mean().reset_index()
    fd.columns = ["family", "_dow", "family_dow_mean"]
    stats["family_dow"] = fd

    # Store mean/std
    sm = train_df.groupby("store_nbr")[target_col].agg(["mean", "std"]).reset_index()
    sm.columns = ["store_nbr", "store_mean", "store_std"]
    stats["store"] = sm

    # Store x DayOfWeek mean
    sd = train_df.groupby(["store_nbr", "_dow"])[target_col].mean().reset_index()
    sd.columns = ["store_nbr", "_dow", "store_dow_mean"]
    stats["store_dow"] = sd

    return stats


def apply_target_stats(df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    """Merge pre-computed target statistics onto a DataFrame."""
    df = df.copy()
    df["_dow"] = df["date"].dt.dayofweek

    # Store x Family
    df = df.merge(stats["store_family"], on=["store_nbr", "family"], how="left")

    # Store x Family zero ratio
    df = df.merge(stats["store_family_zero"], on=["store_nbr", "family"], how="left")

    # Store x Family x DayOfWeek
    df = df.merge(stats["store_family_dow"], on=["store_nbr", "family", "_dow"], how="left")

    # Family
    df = df.merge(stats["family"], on="family", how="left")

    # Family x DayOfWeek
    df = df.merge(stats["family_dow"], on=["family", "_dow"], how="left")

    # Store
    df = df.merge(stats["store"], on="store_nbr", how="left")

    # Store x DayOfWeek
    df = df.merge(stats["store_dow"], on=["store_nbr", "_dow"], how="left")

    df.drop(columns=["_dow"], inplace=True)

    return df
