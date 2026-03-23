"""Data loading utilities for the Store Sales competition.

Loads all CSV files from the raw data directory, parses dates,
and provides merged DataFrames ready for feature engineering.
"""

from pathlib import Path
from typing import Tuple

import pandas as pd

from src.config import (
    RAW_DIR,
    DATE_COL,
)


def load_train(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load training data with parsed dates.

    Args:
        raw_dir: Path to directory containing raw CSV files.

    Returns:
        DataFrame with columns: id, date, store_nbr, family, sales, onpromotion.

    Raises:
        FileNotFoundError: If train.csv is not found in raw_dir.
    """
    path = raw_dir / "train.csv"
    if not path.exists():
        raise FileNotFoundError(f"train.csv not found in {raw_dir}")

    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df["sales"] = df["sales"].clip(lower=0.0)
    return df


def load_test(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load test data with parsed dates."""
    path = raw_dir / "test.csv"
    if not path.exists():
        raise FileNotFoundError(f"test.csv not found in {raw_dir}")
    return pd.read_csv(path, parse_dates=[DATE_COL])


def load_stores(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load store metadata."""
    path = raw_dir / "stores.csv"
    if not path.exists():
        raise FileNotFoundError(f"stores.csv not found in {raw_dir}")
    return pd.read_csv(path)


def load_oil(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load daily oil prices with parsed dates. Missing values are forward-filled then back-filled."""
    path = raw_dir / "oil.csv"
    if not path.exists():
        raise FileNotFoundError(f"oil.csv not found in {raw_dir}")
    df = pd.read_csv(path, parse_dates=[DATE_COL])
    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()
    return df


def load_holidays(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load holidays and events data."""
    path = raw_dir / "holidays_events.csv"
    if not path.exists():
        raise FileNotFoundError(f"holidays_events.csv not found in {raw_dir}")
    return pd.read_csv(path, parse_dates=[DATE_COL])


def load_transactions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load daily transaction counts per store."""
    path = raw_dir / "transactions.csv"
    if not path.exists():
        raise FileNotFoundError(f"transactions.csv not found in {raw_dir}")
    return pd.read_csv(path, parse_dates=[DATE_COL])


def load_raw_data(raw_dir: Path = RAW_DIR) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge all raw data files into train and test DataFrames."""
    train = load_train(raw_dir)
    test = load_test(raw_dir)
    stores = load_stores(raw_dir)
    oil = load_oil(raw_dir)
    transactions = load_transactions(raw_dir)

    stores = stores.rename(columns={"type": "store_type"})

    train = train.merge(stores, on="store_nbr", how="left")
    test = test.merge(stores, on="store_nbr", how="left")

    train = train.merge(oil, on=DATE_COL, how="left")
    test = test.merge(oil, on=DATE_COL, how="left")

    train = train.merge(transactions, on=[DATE_COL, "store_nbr"], how="left")

    return train, test
