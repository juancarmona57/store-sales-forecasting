"""External data feature engineering.

Processes oil prices, holidays/events, and transaction counts
into features suitable for time series modeling.

V22 improvements:
- Regional/local holiday matching to store location (city/state)
- Bridge day detection, transferred holidays, pre/post-holiday effects
- Oil regime features (low/medium/high based on rolling percentile)
- Transaction enrichment (vs mean ratio, momentum, DOW deviation)
"""

import numpy as np
import pandas as pd


def add_oil_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add oil price features including regime and momentum indicators.

    Ecuador is oil-dependent: oil price changes affect consumer spending.
    """
    df = df.copy()

    if "dcoilwtico" not in df.columns:
        return df

    df["dcoilwtico"] = df["dcoilwtico"].ffill().bfill()

    # Basic oil features (proven in v18+)
    df["oil_lag_7"] = df["dcoilwtico"].shift(7)
    df["oil_lag_14"] = df["dcoilwtico"].shift(14)
    df["oil_rolling_mean_28"] = df["dcoilwtico"].shift(1).rolling(28, min_periods=1).mean()
    df["oil_rolling_std_28"] = df["dcoilwtico"].shift(1).rolling(28, min_periods=1).std()
    df["oil_diff"] = df["dcoilwtico"].diff()

    # NEW: Oil regime features (Ecuador economy = oil dependent)
    oil_90d = df["dcoilwtico"].shift(1).rolling(90, min_periods=30)
    oil_pct = df["dcoilwtico"].shift(1).rolling(90, min_periods=30).rank(pct=True)
    oil_regime = pd.cut(
        oil_pct, bins=[-0.01, 0.33, 0.66, 1.01], labels=[0, 1, 2]
    )
    df["oil_regime"] = oil_regime.astype(str).astype(float).fillna(1.0)

    # Oil momentum: % change over 30 days
    oil_30d_ago = df["dcoilwtico"].shift(30)
    df["oil_momentum_30d"] = (
        (df["dcoilwtico"] - oil_30d_ago) / (oil_30d_ago + 1)
    ).fillna(0)

    # Oil above 60-day moving average (expansion signal)
    oil_60d_ma = df["dcoilwtico"].shift(1).rolling(60, min_periods=20).mean()
    df["oil_above_60d_ma"] = (df["dcoilwtico"] > oil_60d_ma).astype(int)

    return df


def add_holiday_features(
    df: pd.DataFrame,
    holidays_df: pd.DataFrame,
) -> pd.DataFrame:
    """Add holiday features with regional/local matching to store location.

    Critical improvement: regional holidays only affect stores in that state,
    local holidays only affect stores in that city. Ecuador has transferred
    holidays, bridge days, and work days (makeup days).
    """
    df = df.copy()

    # --- Global holiday features (backward compatible) ---
    national = holidays_df[
        (holidays_df["locale"] == "National") &
        (~holidays_df.get("transferred", pd.Series([False] * len(holidays_df))).astype(bool))
    ]["date"].unique()

    all_holiday_dates = holidays_df[
        holidays_df["type"].isin(["Holiday", "Transfer", "Bridge", "Additional"])
    ]["date"].unique()

    df["is_holiday"] = df["date"].isin(all_holiday_dates).astype(int)
    df["is_national_holiday"] = df["date"].isin(national).astype(int)

    # Days to/since next/last holiday (global)
    sorted_holidays = np.sort(all_holiday_dates)

    def _days_to_next(date):
        future = sorted_holidays[sorted_holidays > date]
        return (future[0] - date).days if len(future) > 0 else 30

    def _days_since_last(date):
        past = sorted_holidays[sorted_holidays <= date]
        return (date - past[-1]).days if len(past) > 0 else 30

    unique_dates = df["date"].unique()
    days_to = {d: _days_to_next(d) for d in unique_dates}
    days_since = {d: _days_since_last(d) for d in unique_dates}
    df["days_to_next_holiday"] = df["date"].map(days_to)
    df["days_since_last_holiday"] = df["date"].map(days_since)

    # Holiday type ordinal
    type_map = {"National": 3, "Regional": 2, "Local": 1}
    holiday_types = holidays_df.drop_duplicates("date").set_index("date")["locale"].map(type_map)
    df["holiday_type"] = df["date"].map(holiday_types).fillna(0).astype(int)

    # Days to Christmas and New Year
    df["days_to_christmas"] = df["date"].apply(
        lambda d: (pd.Timestamp(d.year, 12, 25) - d).days
        if (pd.Timestamp(d.year, 12, 25) - d).days >= 0
        else (pd.Timestamp(d.year + 1, 12, 25) - d).days
    ).clip(0, 365)

    df["days_to_new_year"] = df["date"].apply(
        lambda d: (pd.Timestamp(d.year + 1, 1, 1) - d).days
        if d.month == 12 else (pd.Timestamp(d.year, 1, 1) - d).days
    ).clip(0, 365).abs()

    # --- NEW: Regional/Local holiday matching ---
    # Build per-date per-location holiday lookup
    regional_holidays = holidays_df[holidays_df["locale"] == "Regional"][
        ["date", "locale_name"]
    ].drop_duplicates()
    local_holidays = holidays_df[holidays_df["locale"] == "Local"][
        ["date", "locale_name"]
    ].drop_duplicates()

    # Regional: match locale_name to store state
    if "state" in df.columns:
        regional_set = set(zip(regional_holidays["date"], regional_holidays["locale_name"]))
        df["is_regional_holiday"] = df.apply(
            lambda r: int((r["date"], r.get("state", "")) in regional_set), axis=1
        )
    else:
        df["is_regional_holiday"] = 0

    # Local: match locale_name to store city
    if "city" in df.columns:
        local_set = set(zip(local_holidays["date"], local_holidays["locale_name"]))
        df["is_local_holiday"] = df.apply(
            lambda r: int((r["date"], r.get("city", "")) in local_set), axis=1
        )
    else:
        df["is_local_holiday"] = 0

    # Effective holiday: national OR matching regional/local
    df["is_effective_holiday"] = (
        df["is_national_holiday"] | df["is_regional_holiday"] | df["is_local_holiday"]
    ).astype(int)

    # --- NEW: Bridge days ---
    bridge_dates = holidays_df[holidays_df["type"] == "Bridge"]["date"].unique()
    df["is_bridge_day"] = df["date"].isin(bridge_dates).astype(int)

    # --- NEW: Work days (negative holidays — makeup days, people work) ---
    work_dates = holidays_df[holidays_df["type"] == "Work Day"]["date"].unique()
    df["is_work_day"] = df["date"].isin(work_dates).astype(int)

    # --- NEW: Pre/Post holiday effects ---
    # Day before holiday: shopping rush
    holiday_set = set(all_holiday_dates)
    one_day = pd.Timedelta(days=1)
    two_days = pd.Timedelta(days=2)

    pre_holiday = {d - one_day for d in all_holiday_dates}
    post_holiday_1 = {d + one_day for d in all_holiday_dates}
    post_holiday_2 = {d + two_days for d in all_holiday_dates}

    df["is_pre_holiday"] = df["date"].isin(pre_holiday).astype(int)
    df["is_post_holiday_1d"] = df["date"].isin(post_holiday_1).astype(int)
    df["is_post_holiday_2d"] = df["date"].isin(post_holiday_2).astype(int)

    return df


def add_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transaction count features with enrichment.

    New: ratio vs store mean, momentum, DOW deviation.
    """
    df = df.copy()

    if "transactions" not in df.columns:
        return df

    # Impute NaN transactions
    if df["transactions"].isna().any():
        store_means = (
            df.dropna(subset=["transactions"])
            .groupby("store_nbr")["transactions"]
            .apply(lambda x: x.tail(28).mean())
        )
        df["transactions"] = df.apply(
            lambda row: store_means.get(row["store_nbr"], 0)
            if pd.isna(row["transactions"]) else row["transactions"],
            axis=1,
        )

    from src.config import SAFE_SHIFT

    grouped = df.groupby("store_nbr", observed=True)["transactions"]

    # Safe-shifted lags
    df["transactions_lag_16"] = grouped.shift(SAFE_SHIFT)
    df["transactions_lag_21"] = grouped.shift(21)
    df["transactions_lag_28"] = grouped.shift(28)

    # Safe-shifted rolling means
    shifted = grouped.shift(SAFE_SHIFT)
    for w in [7, 14, 28]:
        df[f"transactions_rolling_mean_{w}"] = (
            shifted.groupby(df["store_nbr"])
            .rolling(w, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )

    # NEW: Transaction enrichment features
    # Ratio vs store historical mean (is store busier than usual?)
    store_tx_mean = (
        df.dropna(subset=["transactions"])
        .groupby("store_nbr")["transactions"]
        .mean()
    )
    df["tx_vs_store_mean"] = df.apply(
        lambda r: r["transactions_lag_16"] / (store_tx_mean.get(r["store_nbr"], 1) + 1)
        if pd.notna(r.get("transactions_lag_16")) else np.nan,
        axis=1,
    )

    # Transaction momentum: 7d vs 28d rolling ratio
    df["tx_rolling_trend"] = (
        df["transactions_rolling_mean_7"] / (df["transactions_rolling_mean_28"] + 1)
    )

    return df
