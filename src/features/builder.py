"""Feature builder orchestrator.

Coordinates all feature engineering modules into a single
pipeline that transforms raw merged data into a feature matrix.
"""

from typing import Optional

import pandas as pd

from src.data.loader import load_holidays
from src.data.preprocessor import preprocess_holidays
from src.features.temporal import add_temporal_features
from src.features.lag_features import add_lag_features, add_rolling_features, add_same_dow_lag_features
from src.features.external import add_oil_features, add_holiday_features, add_transaction_features
from src.features.promotion import add_promotion_features
from src.features.cross_features import add_cross_features
from src.features.aggregations import add_aggregation_features
from src.features.hierarchical import add_hierarchical_features
from src.features.target_stats import apply_target_stats
from src.config import RAW_DIR, TARGET_COL, CATEGORICAL_COLS


# Module-level storage for category mappings (set during training, reused at inference)
_category_mappings: dict = {}


def build_features(
    df: pd.DataFrame,
    holidays_df: Optional[pd.DataFrame] = None,
    is_train: bool = True,
    target_stats: Optional[dict] = None,
) -> pd.DataFrame:
    """Build complete feature matrix from merged raw data.

    Args:
        df: Merged DataFrame (with store, oil, transaction columns).
        holidays_df: Holidays DataFrame. If None, loads from config path.
        is_train: If True, saves category mappings for later use.
        target_stats: Pre-computed target stats dict. If provided, applied as features.

    Returns:
        Feature-enriched DataFrame.
    """
    global _category_mappings

    # 1. Temporal features
    df = add_temporal_features(df)

    # 2. Lag and rolling features
    if TARGET_COL in df.columns:
        df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = add_same_dow_lag_features(df)

    # 3. Oil features
    df = add_oil_features(df)

    # 4. Holiday features
    if holidays_df is None:
        try:
            holidays_df = preprocess_holidays(load_holidays())
        except FileNotFoundError:
            holidays_df = None

    if holidays_df is not None:
        df = add_holiday_features(df, holidays_df)

    # 5. Transaction features
    if "transactions" in df.columns:
        df = add_transaction_features(df)

    # 6. Promotion features
    df = add_promotion_features(df)

    # 7. Cross features
    df = add_cross_features(df)

    # 8. Aggregation features
    if TARGET_COL in df.columns:
        df = add_aggregation_features(df)

    # 8b. Hierarchical features (store totals, family nationals, cluster avgs)
    if TARGET_COL in df.columns:
        df = add_hierarchical_features(df)

    # 9. Target statistics
    if target_stats is not None:
        df = apply_target_stats(df, target_stats)

    # 10. Encode categoricals with CONSISTENT mappings
    all_cat_cols = list(CATEGORICAL_COLS)
    # Also include cross-feature categoricals
    for col in df.columns:
        if col.endswith("_x_store_type") or col.endswith("_x_cluster") or col.startswith("dow_x_"):
            if col not in all_cat_cols:
                all_cat_cols.append(col)

    if is_train:
        # Save category mappings from training data
        for col in all_cat_cols:
            if col in df.columns:
                cat = df[col].astype("category")
                _category_mappings[col] = cat.cat.categories
                df[col] = cat
    else:
        # Reuse saved category mappings for consistent encoding
        for col in all_cat_cols:
            if col in df.columns:
                if col in _category_mappings:
                    cat_type = pd.CategoricalDtype(categories=_category_mappings[col])
                    df[col] = df[col].astype(cat_type)
                else:
                    df[col] = df[col].astype("category")

    # 11. Wage Payment Cycle Features (Ecuador: public sector pays 15th and month-end)
    day_of_month = df["date"].dt.day
    is_month_end = df["date"].dt.is_month_end

    # Payday dates: 15th and last day of month
    df["is_payday"] = ((day_of_month == 15) | is_month_end).astype(int)

    # Days since last payday (0 = payday, 1 = day after, etc.)
    # Spike is 0-2 days AFTER payday, not on payday itself
    df["days_since_payday"] = day_of_month.apply(
        lambda d: d - 1 if d <= 15 else d - 15  # days since 1st or 15th
    ).clip(0, 16)

    # Post-payday window: the 0-2 days after payday (demand spike)
    df["is_post_payday_window"] = (
        (day_of_month.isin([15, 16, 17])) |  # quincena window
        (day_of_month.isin([1, 2, 3])) |  # post month-end window
        (is_month_end)  # month-end itself
    ).astype(int)

    # Quincena (mid-month pay): 15th-17th
    df["is_quincena"] = day_of_month.isin([15, 16, 17]).astype(int)

    # Fin de mes: last 3 days + first 2 of next month
    df["is_fin_de_mes"] = (
        (day_of_month >= 28) | (day_of_month <= 2)
    ).astype(int)

    # Payday decay: exponential decay from nearest payday
    def _payday_decay(d):
        dist_to_15 = abs(d - 15) if d <= 17 else (d - 15)
        dist_to_end = min(abs(d - 30), d) if d <= 3 else (30 - d) if d >= 28 else 15
        dist = min(dist_to_15, dist_to_end)
        return max(0, 1 - 0.25 * dist)
    df["payday_decay"] = day_of_month.apply(_payday_decay)

    # 12. Earthquake flag
    df["is_earthquake_period"] = (
        (df["date"] >= "2016-04-16") & (df["date"] <= "2016-05-15")
    ).astype(int)

    # 13. Sales velocity features (differences between lags)
    if f"{TARGET_COL}_lag_16" in df.columns and f"{TARGET_COL}_lag_28" in df.columns:
        df["sales_velocity_16_28"] = df[f"{TARGET_COL}_lag_16"] - df[f"{TARGET_COL}_lag_28"]
    if f"{TARGET_COL}_lag_16" in df.columns and f"{TARGET_COL}_lag_42" in df.columns:
        df["sales_velocity_16_42"] = df[f"{TARGET_COL}_lag_16"] - df[f"{TARGET_COL}_lag_42"]
    if f"{TARGET_COL}_lag_28" in df.columns and f"{TARGET_COL}_lag_364" in df.columns:
        df["sales_yoy_change"] = df[f"{TARGET_COL}_lag_28"] - df[f"{TARGET_COL}_lag_364"]

    # 14. Promotion x target stat interactions
    if "onpromotion" in df.columns and "sf_mean" in df.columns:
        df["promo_x_sf_mean"] = df["onpromotion"] * df["sf_mean"]
    if "onpromotion" in df.columns and "sf_dow_mean" in df.columns:
        df["promo_x_sf_dow_mean"] = df["onpromotion"] * df["sf_dow_mean"]

    # 15. Ratio features
    if f"{TARGET_COL}_lag_16" in df.columns and "sf_mean" in df.columns:
        df["lag16_to_mean_ratio"] = df[f"{TARGET_COL}_lag_16"] / (df["sf_mean"] + 1)

    # 16. Same-DOW lag ratios (trend detection via same weekday)
    if f"{TARGET_COL}_lag_dow_3w" in df.columns and f"{TARGET_COL}_lag_dow_4w" in df.columns:
        df["dow_trend_3w_4w"] = (
            df[f"{TARGET_COL}_lag_dow_3w"] / (df[f"{TARGET_COL}_lag_dow_4w"] + 1)
        )
    if f"{TARGET_COL}_lag_dow_4w" in df.columns and f"{TARGET_COL}_lag_dow_52w" in df.columns:
        df["dow_yoy_ratio"] = (
            df[f"{TARGET_COL}_lag_dow_4w"] / (df[f"{TARGET_COL}_lag_dow_52w"] + 1)
        )

    # 17. Recent trend: rolling 14d mean vs rolling 90d mean
    if "sf_recent_mean" in df.columns and "sf_mean" in df.columns:
        df["recent_vs_overall_ratio"] = df["sf_recent_mean"] / (df["sf_mean"] + 1)

    # 18. Promo lift: onpromotion × recent DOW sales (expected promo effect)
    if "onpromotion" in df.columns and f"{TARGET_COL}_dow_rolling_mean_4" in df.columns:
        df["promo_x_dow_avg"] = df["onpromotion"] * df[f"{TARGET_COL}_dow_rolling_mean_4"]

    # 19. Back-to-school season (Ecuador Sierra: school starts September)
    df["is_back_to_school"] = (
        (df["date"].dt.month == 8) & (df["date"].dt.day >= 15)
    ).astype(int)

    # 20. Oil × store_type interaction (wealthy areas less affected by oil drops)
    if "oil_regime" in df.columns and "store_type" in df.columns:
        # Encode store_type as ordinal for interaction (A=5 premium, E=1 small)
        st_raw = df["store_type"].astype(str)
        store_type_ord = st_raw.map({"A": 5, "B": 4, "C": 3, "D": 2, "E": 1}).fillna(3).astype(float)
        df["oil_x_store_type"] = df["oil_regime"].astype(float) * store_type_ord

    # 21. Effective holiday × payday interaction (double spending trigger)
    if "is_effective_holiday" in df.columns:
        df["holiday_x_payday"] = df["is_effective_holiday"] * df["is_post_payday_window"]

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature column names (excluding target, id, date).

    Transactions are now included since NaN values are imputed for test rows.
    """
    exclude = {"id", "date", TARGET_COL}
    return [c for c in df.columns if c not in exclude]
