"""Integration tests for src/features/builder.py."""

import pandas as pd
import numpy as np
import pytest

from src.features.builder import build_features, get_feature_columns


def test_build_features_adds_temporal(sample_train_df, sample_stores_df, sample_oil_df, sample_holidays_df):
    """Builder adds temporal features."""
    merged = sample_train_df.merge(sample_stores_df.rename(columns={"type": "store_type"}), on="store_nbr", how="left")
    merged = merged.merge(sample_oil_df, on="date", how="left")
    result = build_features(merged, holidays_df=sample_holidays_df, is_train=True)
    assert "day_of_week" in result.columns
    assert "month" in result.columns


def test_build_features_adds_lags_for_train(sample_train_df, sample_stores_df, sample_oil_df, sample_holidays_df):
    """Builder adds lag features when is_train=True."""
    merged = sample_train_df.merge(sample_stores_df.rename(columns={"type": "store_type"}), on="store_nbr", how="left")
    merged = merged.merge(sample_oil_df, on="date", how="left")
    result = build_features(merged, holidays_df=sample_holidays_df, is_train=True)
    assert "sales_lag_16" in result.columns


def test_build_features_skips_lags_for_test(sample_test_df, sample_stores_df, sample_oil_df, sample_holidays_df):
    """Builder does not add lag features when is_train=False."""
    merged = sample_test_df.merge(sample_stores_df.rename(columns={"type": "store_type"}), on="store_nbr", how="left")
    merged = merged.merge(sample_oil_df, on="date", how="left")
    result = build_features(merged, holidays_df=sample_holidays_df, is_train=False)
    assert "sales_lag_16" not in result.columns


def test_get_feature_columns_excludes_target(sample_train_df, sample_stores_df, sample_oil_df, sample_holidays_df):
    """Feature columns exclude id, date, and sales."""
    merged = sample_train_df.merge(sample_stores_df.rename(columns={"type": "store_type"}), on="store_nbr", how="left")
    merged = merged.merge(sample_oil_df, on="date", how="left")
    result = build_features(merged, holidays_df=sample_holidays_df, is_train=True)
    feat_cols = get_feature_columns(result)
    assert "sales" not in feat_cols
    assert "id" not in feat_cols
    assert "date" not in feat_cols
    assert len(feat_cols) > 10  # Should have many features
