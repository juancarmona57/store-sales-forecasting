"""Tests for src/features/ modules."""

import pandas as pd
import numpy as np
import pytest

from src.features.temporal import add_temporal_features


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
