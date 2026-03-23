"""External data feature engineering.

Processes oil prices, holidays/events, and transaction counts
into features suitable for time series modeling.
"""

import numpy as np
import pandas as pd


def add_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add oil price features to the DataFrame.

    Assumes df already has 'dcoilwtico' column from merge with oil data.

    Args:
        df: DataFrame with dcoilwtico column.

    Returns:
        DataFrame with oil features added.
    """
    df = df.copy()

    if "dcoilwtico" not in df.columns:
        return df

    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    df["oil_lag_7"] = df["dcoilwtico"].shift(7)
    df["oil_lag_14"] = df["dcoilwtico"].shift(14)
    df["oil_rolling_mean_28"] = df["dcoilwtico"].shift(1).rolling(28, min_periods=1).mean()
    df["oil_rolling_std_28"] = df["dcoilwtico"].shift(1).rolling(28, min_periods=1).std()
    df["oil_diff"] = df["dcoilwtico"].diff()

    return df


def add_holiday_features(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add holiday-related features to the DataFrame.

    Args:
        df: Main DataFrame with date column.
        holidays_df: Holidays DataFrame.

    Returns:
        DataFrame with holiday features added.
    """
    df = df.copy()

    national = holidays_df[
        (holidays_df["locale"] == "National") &
        (~holidays_df.get("transferred", pd.Series([False] * len(holidays_df))).astype(bool))
    ]["date"].unique()

    all_holiday_dates = holidays_df["date"].unique()

    df["is_holiday"] = df["date"].isin(all_holiday_dates).astype(int)
    df["is_national_holiday"] = df["date"].isin(national).astype(int)

    sorted_holidays = np.sort(all_holiday_dates)

    def _days_to_next(date):
        future = sorted_holidays[sorted_holidays > date]
        if len(future) == 0:
            return 30
        return (future[0] - date).days

    def _days_since_last(date):
        past = sorted_holidays[sorted_holidays <= date]
        if len(past) == 0:
            return 30
        return (date - past[-1]).days

    unique_dates = df["date"].unique()
    days_to = {d: _days_to_next(d) for d in unique_dates}
    days_since = {d: _days_since_last(d) for d in unique_dates}

    df["days_to_next_holiday"] = df["date"].map(days_to)
    df["days_since_last_holiday"] = df["date"].map(days_since)

    return df


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transaction count features grouped by store.

    Uses grouped rolling to prevent cross-store contamination.

    Args:
        df: DataFrame with transactions column (from merge).

    Returns:
        DataFrame with transaction features added.
    """
    df = df.copy()

    if "transactions" not in df.columns:
        return df

    grouped = df.groupby("store_nbr", observed=True)["transactions"]

    df["transactions_lag_7"] = grouped.shift(7)

    shifted = grouped.shift(1)
    df["transactions_rolling_mean_7"] = (
        shifted.groupby(df["store_nbr"])
        .rolling(7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["transactions_rolling_mean_14"] = (
        shifted.groupby(df["store_nbr"])
        .rolling(14, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )

    return df
