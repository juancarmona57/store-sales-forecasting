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
        """Rolling window uses SAFE_SHIFT (16) so no data leaks from forecast horizon."""
        # Need at least SAFE_SHIFT + window rows for non-NaN rolling values
        n = 30
        df = pd.DataFrame({
            "date": pd.date_range("2017-01-01", periods=n),
            "store_nbr": [1] * n,
            "family": ["A"] * n,
            "sales": [float(i) for i in range(n)],
        })
        result = add_rolling_features(df, target_col="sales", windows=[3])
        # With SAFE_SHIFT=16, index 18 rolling_mean_3 = mean of shifted values at 16,17,18
        # shifted by 16: values at idx 0,1,2 = mean(0,1,2) = 1.0
        assert result.loc[18, "sales_rolling_mean_3"] == pytest.approx(1.0, rel=1e-6)
        # First 16 rows should be NaN (no data available after shift)
        assert pd.isna(result.loc[0, "sales_rolling_mean_3"])


from src.features.promotion import add_promotion_features


class TestPromotionFeatures:

    def test_adds_promo_lag(self, sample_train_df):
        result = add_promotion_features(sample_train_df)
        assert "promo_lag_7" in result.columns

    def test_adds_promo_rolling(self, sample_train_df):
        result = add_promotion_features(sample_train_df)
        assert "promo_rolling_14" in result.columns

    def test_adds_promo_duration(self, sample_train_df):
        result = add_promotion_features(sample_train_df)
        assert "promo_duration" in result.columns
        assert (result["promo_duration"] >= 0).all()


from src.features.cross_features import add_cross_features


class TestCrossFeatures:

    def test_adds_family_store_type(self, sample_train_df):
        df = sample_train_df.copy()
        df["store_type"] = "A"
        result = add_cross_features(df)
        assert "family_x_store_type" in result.columns

    def test_adds_dow_family(self, sample_train_df):
        df = sample_train_df.copy()
        df["day_of_week"] = df["date"].dt.dayofweek
        result = add_cross_features(df)
        assert "dow_x_family" in result.columns


from src.features.aggregations import add_aggregation_features


class TestAggregationFeatures:

    def test_adds_store_family_mean(self, sample_train_df):
        df = sample_train_df.sort_values(["store_nbr", "family", "date"])
        result = add_aggregation_features(df)
        assert "sales_store_family_mean_30" in result.columns

    def test_no_future_leak(self, sample_train_df):
        """Aggregations use SAFE_SHIFT, so early rows per group have NaN."""
        df = sample_train_df.sort_values(["store_nbr", "family", "date"])
        result = add_aggregation_features(df)
        # First SAFE_SHIFT rows per group should have NaN (no history after shift)
        first_rows = result.groupby(["store_nbr", "family"]).head(1)
        assert first_rows["sales_store_family_mean_30"].isna().all()
        assert first_rows["sales_expanding_mean"].isna().all()
