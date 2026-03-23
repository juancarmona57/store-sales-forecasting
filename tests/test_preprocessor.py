"""Tests for src/data/preprocessor.py."""

import pandas as pd
import numpy as np
import pytest

from src.data.preprocessor import preprocess_train, preprocess_oil, preprocess_holidays


def test_preprocess_train_clips_negative_sales(sample_train_df):
    """Negative sales are clipped to 0."""
    df = sample_train_df.copy()
    df.loc[0, "sales"] = -5.0
    result = preprocess_train(df)
    assert (result["sales"] >= 0).all()


def test_preprocess_train_fills_missing_onpromotion(sample_train_df):
    """Missing onpromotion values are filled with 0."""
    df = sample_train_df.copy()
    df.loc[0, "onpromotion"] = np.nan
    result = preprocess_train(df)
    assert result["onpromotion"].isna().sum() == 0


def test_preprocess_train_sorts_by_date(sample_train_df):
    """Result is sorted by date, store_nbr, family."""
    result = preprocess_train(sample_train_df)
    assert result["date"].is_monotonic_increasing or (
        result.groupby(["store_nbr", "family"])["date"].apply(
            lambda x: x.is_monotonic_increasing
        ).all()
    )


def test_preprocess_oil_fills_missing(sample_oil_df):
    """Missing oil prices are filled (no NaNs)."""
    result = preprocess_oil(sample_oil_df)
    assert result["dcoilwtico"].isna().sum() == 0


def test_preprocess_holidays_adds_is_transferred(sample_holidays_df):
    """Preprocessed holidays have is_transferred boolean column."""
    result = preprocess_holidays(sample_holidays_df)
    assert "is_transferred" in result.columns
