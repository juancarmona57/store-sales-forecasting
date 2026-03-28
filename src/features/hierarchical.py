"""Hierarchical aggregation features.

Captures cross-store and cross-family signals:
- Store-level total sales (how is the whole store doing?)
- Family-level national sales (how is this category nationally?)
- City-level totals (local economic conditions)
- Share features (what fraction of market does this store capture?)

All features use safe shift >= 16 to prevent leakage.
"""

import numpy as np
import pandas as pd

from src.config import SAFE_SHIFT


def add_hierarchical_features(df: pd.DataFrame, target_col: str = "sales") -> pd.DataFrame:
    """Add hierarchical aggregation features.

    These features let the model understand broader context:
    - "Grocery sales were up at ALL stores last week"
    - "This store usually captures 3% of national Beverages sales"
    """
    df = df.copy()

    if target_col not in df.columns:
        return df

    # Sort for consistent groupby operations
    df = df.sort_values(["date", "store_nbr", "family"]).reset_index(drop=True)

    # --- Store-level total sales (sum across all families per store per day) ---
    store_daily = df.groupby(["date", "store_nbr"])[target_col].transform("sum")
    df["_store_daily_total"] = store_daily

    # Safe-lagged store totals
    store_grouped = df.groupby("store_nbr")["_store_daily_total"]
    df["store_total_lag_16"] = store_grouped.shift(SAFE_SHIFT)
    df["store_total_lag_28"] = store_grouped.shift(28)

    # Store total rolling mean (safe shifted)
    shifted_store = store_grouped.shift(SAFE_SHIFT)
    df["store_total_rolling_7"] = (
        shifted_store.groupby(df["store_nbr"])
        .rolling(7, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    df.drop(columns=["_store_daily_total"], inplace=True)

    # --- Family-level national sales (sum across all stores per family per day) ---
    family_daily = df.groupby(["date", "family"])[target_col].transform("sum")
    df["_family_daily_total"] = family_daily

    family_grouped = df.groupby("family")["_family_daily_total"]
    df["family_national_lag_16"] = family_grouped.shift(SAFE_SHIFT)
    df["family_national_lag_28"] = family_grouped.shift(28)

    df.drop(columns=["_family_daily_total"], inplace=True)

    # --- City-level total sales ---
    if "city" in df.columns:
        city_daily = df.groupby(["date", "city"])[target_col].transform("sum")
        df["_city_daily_total"] = city_daily

        city_grouped = df.groupby("city")["_city_daily_total"]
        df["city_total_lag_16"] = city_grouped.shift(SAFE_SHIFT)

        df.drop(columns=["_city_daily_total"], inplace=True)

    # --- Cluster × Family lag (stores in same cluster, same family) ---
    if "cluster" in df.columns:
        cluster_fam = df.groupby(["date", "cluster", "family"])[target_col].transform("mean")
        df["_cluster_fam_avg"] = cluster_fam

        cf_grouped = df.groupby(["cluster", "family"])["_cluster_fam_avg"]
        df["cluster_family_lag_16"] = cf_grouped.shift(SAFE_SHIFT)

        df.drop(columns=["_cluster_fam_avg"], inplace=True)

    # --- Store type × Family lag ---
    if "store_type" in df.columns:
        type_fam = df.groupby(["date", "store_type", "family"])[target_col].transform("mean")
        df["_type_fam_avg"] = type_fam

        tf_grouped = df.groupby(["store_type", "family"])["_type_fam_avg"]
        df["store_type_family_lag_16"] = tf_grouped.shift(SAFE_SHIFT)

        df.drop(columns=["_type_fam_avg"], inplace=True)

    # Re-sort back to original order for consistency with rest of pipeline
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    return df
