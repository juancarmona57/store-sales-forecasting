"""Shared test fixtures for the store-sales-forecasting test suite."""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Minimal train DataFrame for unit tests."""
    dates = pd.date_range("2017-01-01", periods=30, freq="D")
    rows = []
    for i, date in enumerate(dates):
        for store in [1, 2]:
            for family in ["GROCERY I", "BEVERAGES"]:
                rows.append({
                    "id": len(rows),
                    "date": date,
                    "store_nbr": store,
                    "family": family,
                    "sales": max(0.0, np.random.normal(100, 30)),
                    "onpromotion": np.random.randint(0, 2),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_test_df() -> pd.DataFrame:
    """Minimal test DataFrame for unit tests."""
    dates = pd.date_range("2017-01-31", periods=16, freq="D")
    rows = []
    for date in dates:
        for store in [1, 2]:
            for family in ["GROCERY I", "BEVERAGES"]:
                rows.append({
                    "id": len(rows),
                    "date": date,
                    "store_nbr": store,
                    "family": family,
                    "onpromotion": np.random.randint(0, 2),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def sample_stores_df() -> pd.DataFrame:
    """Minimal stores DataFrame."""
    return pd.DataFrame({
        "store_nbr": [1, 2],
        "city": ["Quito", "Guayaquil"],
        "state": ["Pichincha", "Guayas"],
        "type": ["A", "B"],
        "cluster": [1, 2],
    })


@pytest.fixture
def sample_oil_df() -> pd.DataFrame:
    """Minimal oil DataFrame."""
    dates = pd.date_range("2017-01-01", periods=46, freq="D")
    prices = np.random.uniform(40, 80, size=len(dates)).astype(float)
    prices[5] = np.nan
    return pd.DataFrame({"date": dates, "dcoilwtico": prices})


@pytest.fixture
def sample_holidays_df() -> pd.DataFrame:
    """Minimal holidays DataFrame."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01", "2017-01-02", "2017-01-15"]),
        "type": ["Holiday", "Holiday", "Transfer"],
        "locale": ["National", "Local", "National"],
        "locale_name": ["Ecuador", "Quito", "Ecuador"],
        "description": ["New Year", "Quito Festival", "Transfer New Year"],
        "transferred": [False, False, True],
    })


@pytest.fixture
def sample_transactions_df() -> pd.DataFrame:
    """Minimal transactions DataFrame."""
    dates = pd.date_range("2017-01-01", periods=30, freq="D")
    rows = []
    for date in dates:
        for store in [1, 2]:
            rows.append({
                "date": date,
                "store_nbr": store,
                "transactions": np.random.randint(500, 3000),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def tmp_data_dir(tmp_path: Path, sample_train_df, sample_test_df,
                 sample_stores_df, sample_oil_df, sample_holidays_df,
                 sample_transactions_df) -> Path:
    """Create a temporary data directory with all CSV files."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    sample_train_df.to_csv(raw_dir / "train.csv", index=False)
    sample_test_df.to_csv(raw_dir / "test.csv", index=False)
    sample_stores_df.to_csv(raw_dir / "stores.csv", index=False)
    sample_oil_df.to_csv(raw_dir / "oil.csv", index=False)
    sample_holidays_df.to_csv(raw_dir / "holidays_events.csv", index=False)
    sample_transactions_df.to_csv(raw_dir / "transactions.csv", index=False)

    submission_df = sample_test_df[["id"]].copy()
    submission_df["sales"] = 0.0
    submission_df.to_csv(raw_dir / "sample_submission.csv", index=False)

    return tmp_path
