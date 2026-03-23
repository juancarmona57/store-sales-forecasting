"""Tests for src/data/loader.py."""

import pandas as pd
import pytest

from src.data.loader import load_raw_data, load_train, load_test, load_stores


def test_load_train_returns_dataframe(tmp_data_dir):
    """load_train returns a DataFrame with expected columns."""
    df = load_train(tmp_data_dir / "raw")
    assert isinstance(df, pd.DataFrame)
    assert "date" in df.columns
    assert "sales" in df.columns
    assert "store_nbr" in df.columns
    assert "family" in df.columns


def test_load_train_date_is_datetime(tmp_data_dir):
    """date column is parsed as datetime."""
    df = load_train(tmp_data_dir / "raw")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_load_test_returns_dataframe(tmp_data_dir):
    """load_test returns a DataFrame without sales column."""
    df = load_test(tmp_data_dir / "raw")
    assert isinstance(df, pd.DataFrame)
    assert "sales" not in df.columns
    assert "id" in df.columns


def test_load_stores_returns_dataframe(tmp_data_dir):
    """load_stores returns store metadata."""
    df = load_stores(tmp_data_dir / "raw")
    assert "store_nbr" in df.columns
    assert "type" in df.columns
    assert "cluster" in df.columns


def test_load_raw_data_merges_all(tmp_data_dir):
    """load_raw_data returns merged train DataFrame with store and oil info."""
    train_merged, test_merged = load_raw_data(tmp_data_dir / "raw")
    assert "store_type" in train_merged.columns or "type" in train_merged.columns
    assert len(train_merged) > 0
    assert len(test_merged) > 0
