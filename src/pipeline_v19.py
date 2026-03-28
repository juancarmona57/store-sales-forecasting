"""V19 Pipeline: Segment-Optimized Models.

Focus on improving MEDIUM and LOW segments:
- HIGH (13 families, val=0.19): Keep LGBM Tweedie 1.1 — already excellent
- MEDIUM (6 families, val=0.46): Dedicated LGBM with tuned params
- LOW (14 families, val=0.52): Use statistical means instead of GBM
  For LOW-volume families, GBM learns noise. Simple store×family×dow
  means from recent data are more robust predictions.

Also: ensemble segmented + global for best blend.
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


def compute_low_volume_predictions(train_df, target_df):
    """For LOW volume families, use statistical means instead of GBM.

    Compute store×family×dow recent means (last 90 days) as predictions.
    Falls back to store×family mean, then family×dow mean, then family mean.
    """
    recent = train_df[train_df["date"] >= train_df["date"].max() - pd.Timedelta(days=90)].copy()
    recent["_dow"] = recent["date"].dt.dayofweek

    # Level 1: store × family × dow mean (most specific)
    sfd = recent.groupby(["store_nbr", "family", "_dow"])[TARGET_COL].mean().reset_index()
    sfd.columns = ["store_nbr", "family", "_dow", "pred_sfd"]

    # Level 2: store × family mean
    sf = recent.groupby(["store_nbr", "family"])[TARGET_COL].mean().reset_index()
    sf.columns = ["store_nbr", "family", "pred_sf"]

    # Level 3: family × dow mean
    fd = recent.groupby(["family", "_dow"])[TARGET_COL].mean().reset_index()
    fd.columns = ["family", "_dow", "pred_fd"]

    # Level 4: family mean
    fm = recent.groupby("family")[TARGET_COL].mean().reset_index()
    fm.columns = ["family", "pred_fm"]

    # Merge all onto target
    result = target_df[["store_nbr", "family", "date"]].copy()
    result["_dow"] = result["date"].dt.dayofweek

    result = result.merge(sfd, on=["store_nbr", "family", "_dow"], how="left")
    result = result.merge(sf, on=["store_nbr", "family"], how="left")
    result = result.merge(fd, on=["family", "_dow"], how="left")
    result = result.merge(fm, on="family", how="left")

    # Hierarchical fallback
    result["pred"] = result["pred_sfd"]
    result["pred"] = result["pred"].fillna(result["pred_sf"])
    result["pred"] = result["pred"].fillna(result["pred_fd"])
    result["pred"] = result["pred"].fillna(result["pred_fm"])
    result["pred"] = result["pred"].fillna(0)

    return result["pred"].values


def main():
    logger.info("=== V19 PIPELINE: Segment-Optimized Models ===")

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

    # Build features
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
    y_val = val_split[TARGET_COL].values

    logger.info("Train=%d, Val=%d", len(train_split), len(val_split))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================
    # SEGMENT 1: HIGH VOLUME — Dedicated LGBM TW 1.1
    # =============================================
    logger.info("=== HIGH VOLUME MODEL ===")
    t0 = time.time()
    high_train = train_split[train_split["family"].isin(HIGH_VOLUME)]
    high_val = val_split[val_split["family"].isin(HIGH_VOLUME)]

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
    high_model = LGBMModel(params=high_params)
    high_model.fit(
        high_train[feature_cols], high_train[TARGET_COL].values,
        high_val[feature_cols], high_val[TARGET_COL].values,
    )
    high_preds = high_model.predict(high_val[feature_cols])
    high_score = rmsle(high_val[TARGET_COL].values, high_preds)
    logger.info("HIGH RMSLE: %.6f (%.1fs)", high_score, time.time() - t0)

    # =============================================
    # SEGMENT 2: MEDIUM VOLUME — Dedicated LGBM TW 1.3
    # =============================================
    logger.info("=== MEDIUM VOLUME MODEL ===")
    t0 = time.time()
    med_train = train_split[train_split["family"].isin(MEDIUM_VOLUME)]
    med_val = val_split[val_split["family"].isin(MEDIUM_VOLUME)]

    med_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.3,
        "metric": "rmse",
        "learning_rate": 0.015,
        "num_leaves": 128,
        "min_child_samples": 40,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "n_estimators": 4000,
        "reg_alpha": 0.15,
        "reg_lambda": 0.15,
        "max_bin": 511,
        "verbose": -1,
    }
    med_model = LGBMModel(params=med_params)
    med_model.fit(
        med_train[feature_cols], med_train[TARGET_COL].values,
        med_val[feature_cols], med_val[TARGET_COL].values,
    )
    med_preds = med_model.predict(med_val[feature_cols])
    med_score = rmsle(med_val[TARGET_COL].values, med_preds)
    logger.info("MEDIUM RMSLE: %.6f (%.1fs)", med_score, time.time() - t0)

    # =============================================
    # SEGMENT 3: LOW VOLUME — Statistical means (no GBM)
    # =============================================
    logger.info("=== LOW VOLUME PREDICTIONS (statistical means) ===")
    t0 = time.time()
    low_val = val_split[val_split["family"].isin(LOW_VOLUME)]

    # Use the raw train data for computing means (more straightforward)
    low_preds_stat = compute_low_volume_predictions(train_df, low_val)
    low_score_stat = rmsle(low_val[TARGET_COL].values, low_preds_stat)
    logger.info("LOW (stat means) RMSLE: %.6f (%.1fs)", low_score_stat, time.time() - t0)

    # Also train a simple GBM for comparison
    logger.info("Also training LOW GBM for comparison...")
    t0 = time.time()
    low_train = train_split[train_split["family"].isin(LOW_VOLUME)]
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
    low_model = LGBMModel(params=low_params)
    low_model.fit(
        low_train[feature_cols], low_train[TARGET_COL].values,
        low_val[feature_cols], low_val[TARGET_COL].values,
    )
    low_preds_gbm = low_model.predict(low_val[feature_cols])
    low_score_gbm = rmsle(low_val[TARGET_COL].values, low_preds_gbm)
    logger.info("LOW (GBM) RMSLE: %.6f (%.1fs)", low_score_gbm, time.time() - t0)

    # Blend stat + GBM for LOW
    best_low_blend = float("inf")
    best_low_alpha = 0.0
    for alpha in np.arange(0.0, 1.05, 0.05):
        blend = alpha * low_preds_stat + (1 - alpha) * low_preds_gbm
        s = rmsle(low_val[TARGET_COL].values, blend)
        if s < best_low_blend:
            best_low_blend = s
            best_low_alpha = alpha
    logger.info("LOW best blend: %.6f (alpha=%.2f stat / %.2f gbm)",
                best_low_blend, best_low_alpha, 1 - best_low_alpha)

    # =============================================
    # COMBINE SEGMENTS
    # =============================================
    # Assemble full validation predictions
    combined_seg_preds = np.zeros(len(val_split))
    high_idx = val_split.index[val_split["family"].isin(HIGH_VOLUME)]
    med_idx = val_split.index[val_split["family"].isin(MEDIUM_VOLUME)]
    low_idx = val_split.index[val_split["family"].isin(LOW_VOLUME)]

    combined_seg_preds[high_idx] = high_preds
    combined_seg_preds[med_idx] = med_preds

    # Use best low prediction method
    if best_low_blend < low_score_stat and best_low_blend < low_score_gbm:
        low_best = best_low_alpha * low_preds_stat + (1 - best_low_alpha) * low_preds_gbm
        low_method = f"blend(alpha={best_low_alpha:.2f})"
    elif low_score_stat < low_score_gbm:
        low_best = low_preds_stat
        low_method = "stat_means"
    else:
        low_best = low_preds_gbm
        low_method = "gbm"
    combined_seg_preds[low_idx] = low_best

    seg_score = rmsle(y_val, combined_seg_preds)
    logger.info("Combined segmented RMSLE: %.6f (LOW method: %s)", seg_score, low_method)

    # =============================================
    # GLOBAL MODEL for comparison + blending
    # =============================================
    logger.info("=== GLOBAL MODEL (TW 1.2) ===")
    t0 = time.time()
    global_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.2,
        "metric": "rmse",
        "learning_rate": 0.01,
        "num_leaves": 384,
        "min_child_samples": 40,
        "feature_fraction": 0.65,
        "bagging_fraction": 0.75,
        "bagging_freq": 1,
        "n_estimators": 5000,
        "reg_alpha": 0.15,
        "reg_lambda": 0.15,
        "max_bin": 511,
        "verbose": -1,
    }
    global_model = LGBMModel(params=global_params)
    global_model.fit(
        train_split[feature_cols], train_split[TARGET_COL].values,
        val_split[feature_cols], y_val,
    )
    global_preds = global_model.predict(val_split[feature_cols])
    global_score = rmsle(y_val, global_preds)
    logger.info("Global TW 1.2 RMSLE: %.6f (%.1fs)", global_score, time.time() - t0)

    # =============================================
    # BLEND SEGMENTED + GLOBAL
    # =============================================
    logger.info("Optimizing segmented + global blend...")

    # Per-segment blending: different alpha for HIGH vs MEDIUM+LOW
    best_overall = float("inf")
    best_alpha_high = 0.0
    best_alpha_rest = 0.0

    for ah in np.arange(0.0, 1.05, 0.05):
        for ar in np.arange(0.0, 1.05, 0.05):
            blend = np.zeros(len(val_split))
            blend[high_idx] = ah * high_preds + (1 - ah) * global_preds[high_idx]
            blend[med_idx] = ar * med_preds + (1 - ar) * global_preds[med_idx]
            blend[low_idx] = ar * low_best + (1 - ar) * global_preds[low_idx]
            s = rmsle(y_val, blend)
            if s < best_overall:
                best_overall = s
                best_alpha_high = ah
                best_alpha_rest = ar

    logger.info("=" * 60)
    logger.info("  HIGH segment:      %.6f", high_score)
    logger.info("  MEDIUM segment:    %.6f", med_score)
    logger.info("  LOW stat means:    %.6f", low_score_stat)
    logger.info("  LOW GBM:           %.6f", low_score_gbm)
    logger.info("  LOW best blend:    %.6f", best_low_blend)
    logger.info("  Combined Segmented:%.6f", seg_score)
    logger.info("  Global TW 1.2:     %.6f", global_score)
    logger.info("  Best per-seg blend:%.6f (HIGH=%.2f, REST=%.2f)",
                best_overall, best_alpha_high, best_alpha_rest)
    logger.info("=" * 60)

    # =============================================
    # GENERATE TEST PREDICTIONS
    # =============================================
    logger.info("Generating test predictions...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = test_featured[feature_cols].copy()
    X_test[numeric_feat] = X_test[numeric_feat].fillna(0)

    # Per-segment test predictions
    is_high_test = test_featured["family"].isin(HIGH_VOLUME).values
    is_med_test = test_featured["family"].isin(MEDIUM_VOLUME).values
    is_low_test = test_featured["family"].isin(LOW_VOLUME).values

    high_test_preds = high_model.predict(X_test[is_high_test])
    med_test_preds = med_model.predict(X_test[is_med_test])

    # LOW: compute stat means for test data
    low_test_preds_stat = compute_low_volume_predictions(
        train_df, test_featured[is_low_test]
    )
    if low_method.startswith("blend"):
        low_test_preds_gbm = low_model.predict(X_test[is_low_test])
        low_test_preds = best_low_alpha * low_test_preds_stat + (1 - best_low_alpha) * low_test_preds_gbm
    elif low_method == "stat_means":
        low_test_preds = low_test_preds_stat
    else:
        low_test_preds = low_model.predict(X_test[is_low_test])

    global_test_preds = global_model.predict(X_test)

    # Apply per-segment blend weights
    final_pred = np.zeros(len(test_featured))
    high_test_idx = np.where(is_high_test)[0]
    med_test_idx = np.where(is_med_test)[0]
    low_test_idx = np.where(is_low_test)[0]

    final_pred[high_test_idx] = best_alpha_high * high_test_preds + (1 - best_alpha_high) * global_test_preds[high_test_idx]
    final_pred[med_test_idx] = best_alpha_rest * med_test_preds + (1 - best_alpha_rest) * global_test_preds[med_test_idx]
    final_pred[low_test_idx] = best_alpha_rest * low_test_preds + (1 - best_alpha_rest) * global_test_preds[low_test_idx]

    final_pred = np.clip(final_pred, 0, None)

    logger.info("Predictions: mean=%.1f, median=%.1f, zero_pct=%.1f%%",
                final_pred.mean(), np.median(final_pred), (final_pred == 0).mean() * 100)

    save_path = SUBMISSIONS_DIR / "submission_v19.csv"
    generate_submission(test_featured, final_pred, save_path=save_path)
    logger.info("Saved: %s", save_path)

    # Also save pure segmented (no global blend)
    seg_test = np.zeros(len(test_featured))
    seg_test[high_test_idx] = high_test_preds
    seg_test[med_test_idx] = med_test_preds
    seg_test[low_test_idx] = low_test_preds
    seg_test = np.clip(seg_test, 0, None)
    save_path2 = SUBMISSIONS_DIR / "submission_v19_pure_seg.csv"
    generate_submission(test_featured, seg_test, save_path=save_path2)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
