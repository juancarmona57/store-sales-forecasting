"""Tests for src/evaluation/validation.py."""

import pandas as pd
import numpy as np
import pytest

from src.evaluation.validation import TimeSeriesSplitWithGap


class TestTimeSeriesSplitWithGap:

    @pytest.fixture
    def date_df(self):
        """DataFrame spanning 100 days."""
        return pd.DataFrame({
            "date": pd.date_range("2017-01-01", periods=100, freq="D"),
            "value": range(100),
        })

    def test_yields_correct_number_of_splits(self, date_df):
        splitter = TimeSeriesSplitWithGap(n_splits=3, gap_days=16, val_days=16)
        splits = list(splitter.split(date_df))
        assert len(splits) == 3

    def test_gap_between_train_and_val(self, date_df):
        """No overlap: train ends at least gap_days before val starts."""
        splitter = TimeSeriesSplitWithGap(n_splits=2, gap_days=16, val_days=16)
        for train_idx, val_idx in splitter.split(date_df):
            train_max = date_df.loc[train_idx, "date"].max()
            val_min = date_df.loc[val_idx, "date"].min()
            gap = (val_min - train_max).days
            assert gap >= 16

    def test_val_size_matches_val_days(self, date_df):
        splitter = TimeSeriesSplitWithGap(n_splits=2, gap_days=16, val_days=16)
        for _, val_idx in splitter.split(date_df):
            val_dates = date_df.loc[val_idx, "date"].nunique()
            assert val_dates <= 16

    def test_no_index_overlap(self, date_df):
        splitter = TimeSeriesSplitWithGap(n_splits=2, gap_days=16, val_days=16)
        for train_idx, val_idx in splitter.split(date_df):
            assert len(set(train_idx) & set(val_idx)) == 0

    def test_holdout_split(self, date_df):
        splitter = TimeSeriesSplitWithGap(gap_days=16, val_days=16)
        train_idx, val_idx = splitter.get_holdout_split(date_df)
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(set(train_idx) & set(val_idx)) == 0
