"""Data preprocessing utilities.

Handles type casting, missing values, outlier clipping,
and data cleaning for all competition data files.
"""

import pandas as pd
import numpy as np


def preprocess_train(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the training DataFrame.

    - Clips negative sales to 0
    - Fills missing onpromotion with 0
    - Ensures correct dtypes
    - Sorts by store_nbr, family, date
    """
    df = df.copy()
    df["sales"] = df["sales"].clip(lower=0.0)
    df["onpromotion"] = df["onpromotion"].fillna(0).astype(int)

    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    return df


def preprocess_oil(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess oil prices. Forward-fills, back-fills, then interpolates."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill().interpolate(method="linear")
    return df


def preprocess_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess holidays and events. Adds is_transferred boolean."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    if "transferred" in df.columns:
        df["is_transferred"] = df["transferred"].astype(bool)
    else:
        df["is_transferred"] = False
    return df


def preprocess_transactions(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess transaction counts."""
    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    return df
