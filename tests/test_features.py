"""Tests for src/features/ modules."""

import pandas as pd
import numpy as np
import pytest

from src.features.temporal import add_temporal_features
from src.features.lag_features import add_lag_features, add_rolling_features


class TestTemporalFeatures:
    """Tests for temporal feature generation."""

    def test_adds_day_of_week(self, sample_train_df):
        """day_of_week column is added with values 0-6."""
        result = add_temporal_features(sample_train_df)
        assert "day_of_week" in result.columns
        assert result["day_of_week"].between(0, 6).all()

    def test_adds_month(self, sample_train_df):
        """month column is added with values 1-12."""
        result = add_temporal_features(sample_train_df)
        assert "month" in result.columns
        assert result["month"].between(1, 12).all()

    def test_adds_is_weekend(self, sample_train_df):
        """is_weekend is 1 for Saturday/Sunday, 0 otherwise."""
        result = add_temporal_features(sample_train_df)
        assert "is_weekend" in result.columns
        assert set(result["is_weekend"].unique()).issubset({0, 1})

    def test_adds_cyclical_features(self, sample_train_df):
        """sin/cos cyclical features are in range [-1, 1]."""
        result = add_temporal_features(sample_train_df)
        assert "dow_sin" in result.columns
        assert "dow_cos" in result.columns
        assert result["dow_sin"].between(-1, 1).all()
        assert result["dow_cos"].between(-1, 1).all()

    def test_does_not_modify_original(self, sample_train_df):
        """Original DataFrame is not modified."""
        original_cols = list(sample_train_df.columns)
        _ = add_temporal_features(sample_train_df)
        assert list(sample_train_df.columns) == original_cols

    def test_output_shape_rows_unchanged(self, sample_train_df):
        """Number of rows is unchanged."""
        result = add_temporal_features(sample_train_df)
        assert len(result) == len(sample_train_df)


class TestLagFeatures:
    """Tests for lag feature generation."""

    def test_adds_lag_columns(self, sample_train_df):
        """Lag columns are created for specified days."""
        df = sample_train_df.sort_values(["store_nbr", "family", "date"])
        result = add_lag_features(df, target_col="sales", lags=[7, 14])
        assert "sales_lag_7" in result.columns
        assert "sales_lag_14" in result.columns

    def test_lag_values_are_correct(self):
        """Lag 1 shifts values by 1 row within each group."""
        df = pd.DataFrame({
            "date": pd.date_range("2017-01-01", periods=5),
            "store_nbr": [1] * 5,
            "family": ["A"] * 5,
            "sales": [10.0, 20.0, 30.0, 40.0, 50.0],
        })
        result = add_lag_features(df, target_col="sales", lags=[1])
        assert result.loc[1, "sales_lag_1"] == 10.0
        assert pd.isna(result.loc[0, "sales_lag_1"])

    def test_rows_unchanged(self, sample_train_df):
        """Number of rows does not change (NaN introduced, not dropped)."""
        result = add_lag_features(sample_train_df, target_col="sales", lags=[7])
        assert len(result) == len(sample_train_df)


class TestRollingFeatures:
    """Tests for rolling window features."""

    def test_adds_rolling_mean(self, sample_train_df):
        """Rolling mean columns are created."""
        df = sample_train_df.sort_values(["store_nbr", "family", "date"])
        result = add_rolling_features(df, target_col="sales", windows=[7])
        assert "sales_rolling_mean_7" in result.columns

    def test_adds_rolling_std(self, sample_train_df):
        """Rolling std columns are created."""
        df = sample_train_df.sort_values(["store_nbr", "family", "date"])
        result = add_rolling_features(df, target_col="sales", windows=[7])
        assert "sales_rolling_std_7" in result.columns

    def test_rolling_no_future_leak(self):
        """Rolling window does not include the current row."""
        df = pd.DataFrame({
            "date": pd.date_range("2017-01-01", periods=10),
            "store_nbr": [1] * 10,
            "family": ["A"] * 10,
            "sales": [float(i) for i in range(10)],
        })
        result = add_rolling_features(df, target_col="sales", windows=[3])
        # Rolling mean at index 3 should be mean of [0,1,2] = 1.0 (shifted, not including current)
        assert result.loc[3, "sales_rolling_mean_3"] == pytest.approx(1.0, rel=1e-6)
