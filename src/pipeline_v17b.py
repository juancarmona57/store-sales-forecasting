"""V17b Pipeline: Family-Grouped Models + Business Logic Segmentation.

Key insight: 33 product families have vastly different characteristics.
- HIGH volume (13 families, 39% of data): mean > 100, 12% zeros → full-power LGBM
- MEDIUM volume (6 families, 18%): 10 < mean <= 100, 21% zeros → moderate LGBM
- LOW volume (14 families, 42%): mean <= 10, 53% zeros → simple model, static means

Training separate models per segment captures family-specific patterns
without wasting model capacity on noise.

Also includes:
- Earthquake period exclusion
- Tweedie 1.1 (closer to Poisson, proven by winning solutions)
- Transaction forecasting for test period
"""

import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, SUBMISSIONS_DIR, TARGET_COL, SEED
from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess_train
from src.features.builder import build_features, get_feature_columns
from src.features.target_stats import compute_target_stats
from src.evaluation.metrics import rmsle
from src.models.lgbm_model import LGBMModel
from src.submission.generator import generate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EARTHQUAKE_START = pd.Timestamp("2016-04-16")
EARTHQUAKE_END = pd.Timestamp("2016-05-15")

# Family segments based on historical mean sales
HIGH_VOLUME = [
    "GROCERY I", "BEVERAGES", "PRODUCE", "CLEANING", "DAIRY",
    "BREAD/BAKERY", "POULTRY", "MEATS", "PERSONAL CARE", "DELI",
    "HOME CARE", "EGGS", "FROZEN FOODS",
]
MEDIUM_VOLUME = [
    "PREPARED FOODS", "LIQUOR,WINE,BEER", "SEAFOOD",
    "GROCERY II", "HOME AND KITCHEN I", "HOME AND KITCHEN II",
]
LOW_VOLUME = [
    "CELEBRATION", "LINGERIE", "LADIESWEAR", "PLAYERS AND ELECTRONICS",
    "AUTOMOTIVE", "LAWN AND GARDEN", "PET SUPPLIES", "BEAUTY",
    "SCHOOL AND OFFICE SUPPLIES", "MAGAZINES", "HARDWARE",
    "HOME APPLIANCES", "BABY CARE", "BOOKS",
]


def forecast_transactions(train_df, test_df):
    """Forecast daily transactions per store for test period."""
    logger.info("Forecasting transactions for test period...")
    transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=["date"])

    tx = transactions.copy()
    tx["dow"] = tx.date.dt.dayofweek
    tx["month"] = tx.date.dt.month
    tx["day"] = tx.date.dt.day
    tx["week"] = tx.date.dt.isocalendar().week.astype(int)
    tx["is_weekend"] = (tx.dow >= 5).astype(int)

    tx = tx.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    for lag in [16, 21, 28, 35]:
        tx[f"tx_lag_{lag}"] = tx.groupby("store_nbr")["transactions"].shift(lag)

    shifted = tx.groupby("store_nbr")["transactions"].shift(16)
    group_key = tx["store_nbr"]
    for w in [7, 14, 28]:
        tx[f"tx_rolling_mean_{w}"] = (
            shifted.groupby(group_key).rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

    tx = tx.dropna(subset=["transactions"]).reset_index(drop=True)
    cutoff = tx.date.max() - pd.Timedelta(days=32)
    train_tx = tx[tx.date <= cutoff]
    val_tx = tx[(tx.date > cutoff) & (tx.date <= tx.date.max())]

    feat_cols = [c for c in tx.columns if c not in ["date", "transactions"]]

    model = lgb.LGBMRegressor(
        objective="tweedie", tweedie_variance_power=1.5,
        learning_rate=0.05, num_leaves=64, n_estimators=500,
        random_state=SEED, verbose=-1,
    )
    model.fit(
        train_tx[feat_cols], train_tx["transactions"],
        eval_set=[(val_tx[feat_cols], val_tx["transactions"])],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)],
    )

    test_dates = test_df["date"].unique()
    stores = test_df["store_nbr"].unique()
    last_known = tx.groupby("store_nbr").tail(50)

    pred_rows = []
    for store in stores:
        store_tx = last_known[last_known.store_nbr == store].copy()
        for date in sorted(test_dates):
            row = {
                "store_nbr": store, "date": date,
                "dow": pd.Timestamp(date).dayofweek,
                "month": pd.Timestamp(date).month,
                "day": pd.Timestamp(date).day,
                "week": pd.Timestamp(date).isocalendar().week,
                "is_weekend": int(pd.Timestamp(date).dayofweek >= 5),
            }
            for lag in [16, 21, 28, 35]:
                ref_date = date - pd.Timedelta(days=lag)
                match = store_tx[store_tx.date == ref_date]
                row[f"tx_lag_{lag}"] = match["transactions"].values[0] if len(match) > 0 else np.nan
            recent = store_tx.sort_values("date").tail(28)
            for w in [7, 14, 28]:
                vals = recent["transactions"].tail(w)
                row[f"tx_rolling_mean_{w}"] = vals.mean() if len(vals) > 0 else np.nan
            pred_rows.append(row)

    pred_df = pd.DataFrame(pred_rows)
    for c in feat_cols:
        if c not in pred_df.columns:
            pred_df[c] = 0
    pred_df[feat_cols] = pred_df[feat_cols].fillna(0)
    pred_df["transactions_pred"] = np.clip(model.predict(pred_df[feat_cols]), 0, None)

    tx_map = pred_df.set_index(["store_nbr", "date"])["transactions_pred"].to_dict()
    logger.info("Transaction forecast: mean=%.1f, median=%.1f",
                pred_df.transactions_pred.mean(), pred_df.transactions_pred.median())
    return tx_map


def train_segment_model(segment_name, train_data, val_data, feature_cols, params):
    """Train a model for a specific family segment."""
    X_train = train_data[feature_cols]
    y_train = train_data[TARGET_COL].values
    X_val = val_data[feature_cols]
    y_val = val_data[TARGET_COL].values

    logger.info("  %s: train=%d, val=%d", segment_name, len(X_train), len(X_val))

    model = LGBMModel(params=params)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    score = rmsle(y_val, preds)
    logger.info("  %s RMSLE: %.6f", segment_name, score)

    return model, preds, score


def main():
    logger.info("=== V17b PIPELINE: Family-Grouped Models ===")

    # Load data
    train_df, test_df = load_raw_data()
    train_df = preprocess_train(train_df)

    # Forecast transactions
    tx_map = forecast_transactions(train_df, test_df)
    test_df["transactions"] = test_df.apply(
        lambda r: tx_map.get((r["store_nbr"], r["date"]), 0), axis=1
    )

    # Exclude earthquake period
    earthquake_mask = (
        (train_df["date"] >= EARTHQUAKE_START) & (train_df["date"] <= EARTHQUAKE_END)
    )
    train_df = train_df[~earthquake_mask].reset_index(drop=True)
    logger.info("Excluded %d earthquake-period rows", earthquake_mask.sum())

    # Compute target stats and build features
    target_stats = compute_target_stats(train_df)

    test_df["sales"] = np.nan
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    logger.info("Building features...")
    combined_featured = build_features(combined, is_train=True, target_stats=target_stats)

    train_max_date = train_df["date"].max()
    train_featured = combined_featured[combined_featured["date"] <= train_max_date].copy()
    test_featured = combined_featured[combined_featured["date"] > train_max_date].copy()

    train_featured = train_featured.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    test_featured = test_featured.reset_index(drop=True)

    # Filter to last 2 years
    cutoff_date = train_max_date - pd.Timedelta(days=730)
    train_featured = train_featured[train_featured["date"] >= cutoff_date].reset_index(drop=True)

    feature_cols = get_feature_columns(train_featured)
    logger.info("Features: %d train, %d test, %d cols",
                len(train_featured), len(test_featured), len(feature_cols))

    numeric_feat = [c for c in feature_cols if train_featured[c].dtype.name != "category"]
    train_featured[numeric_feat] = train_featured[numeric_feat].fillna(0)
    test_featured[numeric_feat] = test_featured[numeric_feat].fillna(0)

    # Validation split
    val_start = train_max_date - pd.Timedelta(days=15)
    gap_end = val_start - pd.Timedelta(days=16)

    train_mask = train_featured["date"] <= gap_end
    val_mask = train_featured["date"] >= val_start

    train_split = train_featured[train_mask].reset_index(drop=True)
    val_split = train_featured[val_mask].reset_index(drop=True)

    logger.info("Train=%d, Val=%d", len(train_split), len(val_split))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================
    # MODEL PARAMS PER SEGMENT
    # =============================================

    # HIGH volume: aggressive model, more capacity
    high_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "learning_rate": 0.01,
        "num_leaves": 256,
        "min_child_samples": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "n_estimators": 5000,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "max_bin": 511,
        "verbose": -1,
    }

    # MEDIUM volume: moderate model
    medium_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.3,
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 128,
        "min_child_samples": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "n_estimators": 3000,
        "reg_alpha": 0.2,
        "reg_lambda": 0.2,
        "max_bin": 255,
        "verbose": -1,
    }

    # LOW volume: simple model, high regularization, fewer leaves
    # These families are mostly zeros — model should learn to predict near-zero
    low_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 32,
        "min_child_samples": 100,
        "feature_fraction": 0.6,
        "bagging_fraction": 0.6,
        "bagging_freq": 1,
        "n_estimators": 2000,
        "reg_alpha": 0.5,
        "reg_lambda": 0.5,
        "max_bin": 127,
        "verbose": -1,
    }

    segments = {
        "HIGH": (HIGH_VOLUME, high_params),
        "MEDIUM": (MEDIUM_VOLUME, medium_params),
        "LOW": (LOW_VOLUME, low_params),
    }

    # =============================================
    # TRAIN SEGMENT MODELS
    # =============================================
    segment_models = {}
    all_val_preds = np.zeros(len(val_split))
    segment_scores = {}

    for seg_name, (families, params) in segments.items():
        logger.info("Training %s volume model...", seg_name)
        t0 = time.time()

        train_seg = train_split[train_split["family"].isin(families)]
        val_seg = val_split[val_split["family"].isin(families)]

        if len(train_seg) == 0 or len(val_seg) == 0:
            logger.warning("  Skipping %s — no data", seg_name)
            continue

        model, preds, score = train_segment_model(
            seg_name, train_seg, val_seg, feature_cols, params
        )
        segment_models[seg_name] = model
        segment_scores[seg_name] = score

        # Place predictions back into full val array
        val_seg_idx = val_split.index[val_split["family"].isin(families)]
        all_val_preds[val_seg_idx] = preds

        logger.info("  %s done in %.1fs", seg_name, time.time() - t0)

    # Overall segmented score
    y_val_all = val_split[TARGET_COL].values
    segmented_score = rmsle(y_val_all, all_val_preds)

    # =============================================
    # BASELINE: Single global model for comparison
    # =============================================
    logger.info("Training global baseline model...")
    t0 = time.time()
    global_model = LGBMModel(params=high_params)
    global_model.fit(
        train_split[feature_cols], train_split[TARGET_COL].values,
        val_split[feature_cols], val_split[TARGET_COL].values,
    )
    global_preds = global_model.predict(val_split[feature_cols])
    global_score = rmsle(y_val_all, global_preds)
    logger.info("Global baseline RMSLE: %.6f (%.1fs)", global_score, time.time() - t0)

    # =============================================
    # BLEND: Segmented + Global
    # =============================================
    best_blend_score = float("inf")
    best_alpha = 0.0
    for alpha in np.arange(0.0, 1.05, 0.05):
        blend = alpha * all_val_preds + (1 - alpha) * global_preds
        s = rmsle(y_val_all, blend)
        if s < best_blend_score:
            best_blend_score = s
            best_alpha = alpha

    blend_preds = best_alpha * all_val_preds + (1 - best_alpha) * global_preds

    logger.info("=" * 60)
    for seg_name, score in segment_scores.items():
        logger.info("  %s segment:    %.6f", seg_name, score)
    logger.info("  Segmented:       %.6f", segmented_score)
    logger.info("  Global TW 1.1:   %.6f", global_score)
    logger.info("  Best Blend:      %.6f (alpha=%.2f seg / %.2f global)",
                best_blend_score, best_alpha, 1 - best_alpha)
    logger.info("=" * 60)

    # =============================================
    # Generate test predictions
    # =============================================
    logger.info("Generating test predictions...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = test_featured[feature_cols].copy()
    X_test[numeric_feat] = X_test[numeric_feat].fillna(0)

    # Segmented predictions
    seg_test_preds = np.zeros(len(test_featured))
    for seg_name, (families, _) in segments.items():
        if seg_name not in segment_models:
            continue
        test_seg_mask = test_featured["family"].isin(families)
        seg_preds = segment_models[seg_name].predict(X_test[test_seg_mask.values])
        seg_test_preds[test_seg_mask.values] = seg_preds

    # Global predictions
    global_test_preds = global_model.predict(X_test)

    # Apply best blend
    final_pred = best_alpha * seg_test_preds + (1 - best_alpha) * global_test_preds
    final_pred = np.clip(final_pred, 0, None)

    logger.info("Predictions: mean=%.1f, median=%.1f, zero_pct=%.1f%%",
                final_pred.mean(), np.median(final_pred), (final_pred == 0).mean() * 100)

    save_path = SUBMISSIONS_DIR / "submission_v17b.csv"
    generate_submission(test_featured, final_pred, save_path=save_path)
    logger.info("Saved: %s", save_path)

    # Also save pure segmented submission
    seg_pred = np.clip(seg_test_preds, 0, None)
    save_path2 = SUBMISSIONS_DIR / "submission_v17b_segmented.csv"
    generate_submission(test_featured, seg_pred, save_path=save_path2)
    logger.info("Saved segmented: %s", save_path2)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
