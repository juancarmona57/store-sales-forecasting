"""Tests for src/features/external.py."""

import pandas as pd
import numpy as np
import pytest

from src.features.external import add_oil_features, add_holiday_features, add_transaction_features


class TestOilFeatures:
    """Tests for oil price feature engineering."""

    def test_adds_oil_lag(self, sample_train_df, sample_oil_df):
        """Oil lag features are created."""
        merged = sample_train_df.merge(sample_oil_df, on="date", how="left")
        result = add_oil_features(merged)
        assert "oil_lag_7" in result.columns

    def test_adds_oil_rolling_mean(self, sample_train_df, sample_oil_df):
        """Oil rolling mean is created."""
        merged = sample_train_df.merge(sample_oil_df, on="date", how="left")
        result = add_oil_features(merged)
        assert "oil_rolling_mean_28" in result.columns


class TestHolidayFeatures:
    """Tests for holiday feature engineering."""

    def test_adds_is_holiday(self, sample_train_df, sample_holidays_df):
        """is_holiday column is added."""
        result = add_holiday_features(sample_train_df, sample_holidays_df)
        assert "is_holiday" in result.columns
        assert set(result["is_holiday"].unique()).issubset({0, 1})

    def test_adds_days_to_next_holiday(self, sample_train_df, sample_holidays_df):
        """days_to_next_holiday column is added."""
        result = add_holiday_features(sample_train_df, sample_holidays_df)
        assert "days_to_next_holiday" in result.columns


class TestTransactionFeatures:
    """Tests for transaction feature engineering."""

    def test_adds_transaction_lag(self, sample_train_df, sample_transactions_df):
        """Transaction lag features are created."""
        merged = sample_train_df.merge(
            sample_transactions_df, on=["date", "store_nbr"], how="left"
        )
        result = add_transaction_features(merged)
        assert "transactions_lag_7" in result.columns
