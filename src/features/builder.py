"""Feature builder orchestrator.

Coordinates all feature engineering modules into a single
pipeline that transforms raw merged data into a feature matrix.
"""

from typing import Optional

import pandas as pd

from src.data.loader import load_holidays
from src.data.preprocessor import preprocess_holidays
from src.features.temporal import add_temporal_features
from src.features.lag_features import add_lag_features, add_rolling_features
from src.features.external import add_oil_features, add_holiday_features, add_transaction_features
from src.features.promotion import add_promotion_features
from src.features.cross_features import add_cross_features
from src.features.aggregations import add_aggregation_features
from src.features.target_stats import apply_target_stats
from src.config import RAW_DIR, TARGET_COL, CATEGORICAL_COLS


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
        is_train: If True, generates lag/rolling/agg features from target.
        target_stats: Pre-computed target stats dict. If provided, applied as features.

    Returns:
        Feature-enriched DataFrame.
    """
    # 1. Temporal features
    df = add_temporal_features(df)

    # 2. Lag and rolling features (safe lags >= 16, shift >= 16)
    if TARGET_COL in df.columns:
        df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
        df = add_lag_features(df)
        df = add_rolling_features(df)

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

    # 8. Aggregation features (uses safe shift)
    if TARGET_COL in df.columns:
        df = add_aggregation_features(df)

    # 9. Target statistics (static lookup features - always available)
    if target_stats is not None:
        df = apply_target_stats(df, target_stats)

    # 10. Encode categoricals
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # 11. Payday feature
    df["is_payday"] = ((df["date"].dt.day == 15) | (df["date"].dt.is_month_end)).astype(int)

    # 12. Earthquake flag
    df["is_earthquake_period"] = (
        (df["date"] >= "2016-04-16") & (df["date"] <= "2016-05-15")
    ).astype(int)

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get list of feature column names (excluding target, id, date)."""
    exclude = {"id", "date", TARGET_COL}
    return [c for c in df.columns if c not in exclude]
