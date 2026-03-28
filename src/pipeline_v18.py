"""V18 Pipeline: Hybrid Segmented + Multi-Tweedie Ensemble.

Proven findings applied:
1. HIGH volume families (13 fams) → dedicated TW 1.1 model (val RMSLE=0.19)
2. ALL families → global TW 1.1 model (val RMSLE=0.40)
3. ALL families → global TW 1.5 model (diversity)
4. For HIGH: use segmented model; for MEDIUM+LOW: use global blend
5. Earthquake exclusion, transaction forecasting
6. Optimize ensemble weights per segment

This is a "best of both worlds" approach.
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


def main():
    logger.info("=== V18 PIPELINE: Hybrid Segmented + Multi-Tweedie ===")

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

    X_train = train_split[feature_cols]
    y_train = train_split[TARGET_COL].values
    X_val = val_split[feature_cols]
    y_val = val_split[TARGET_COL].values

    logger.info("Train=%d, Val=%d", len(X_train), len(X_val))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================
    # MODEL 1: HIGH volume dedicated model (TW 1.1)
    # =============================================
    logger.info("Model 1: HIGH volume dedicated (TW 1.1)...")
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
    logger.info("HIGH model RMSLE: %.6f (%.1fs)", high_score, time.time() - t0)

    # =============================================
    # MODEL 2: Global model TW 1.1
    # =============================================
    logger.info("Model 2: Global TW 1.1...")
    t0 = time.time()
    global_params_11 = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "learning_rate": 0.01,
        "num_leaves": 256,
        "min_child_samples": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "n_estimators": 5000,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "max_bin": 511,
        "verbose": -1,
    }
    global_11 = LGBMModel(params=global_params_11)
    global_11.fit(X_train, y_train, X_val, y_val)
    g11_preds = global_11.predict(X_val)
    g11_score = rmsle(y_val, g11_preds)
    logger.info("Global TW 1.1 RMSLE: %.6f (%.1fs)", g11_score, time.time() - t0)

    # =============================================
    # MODEL 3: Global model TW 1.5
    # =============================================
    logger.info("Model 3: Global TW 1.5...")
    t0 = time.time()
    global_15 = LGBMModel()  # default params = TW 1.5
    global_15.fit(X_train, y_train, X_val, y_val)
    g15_preds = global_15.predict(X_val)
    g15_score = rmsle(y_val, g15_preds)
    logger.info("Global TW 1.5 RMSLE: %.6f (%.1fs)", g15_score, time.time() - t0)

    # =============================================
    # MODEL 4: Global model TW 1.2 (different num_leaves)
    # =============================================
    logger.info("Model 4: Global TW 1.2...")
    t0 = time.time()
    global_params_12 = {
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
    global_12 = LGBMModel(params=global_params_12)
    global_12.fit(X_train, y_train, X_val, y_val)
    g12_preds = global_12.predict(X_val)
    g12_score = rmsle(y_val, g12_preds)
    logger.info("Global TW 1.2 RMSLE: %.6f (%.1fs)", g12_score, time.time() - t0)

    # =============================================
    # HYBRID: HIGH segmented + Global for rest
    # =============================================
    logger.info("Building hybrid predictions...")
    is_high = val_split["family"].isin(HIGH_VOLUME).values

    # For HIGH families: blend segmented with global
    # For NON-HIGH families: use global only
    # Try different blends for HIGH families
    best_overall = float("inf")
    best_config = None

    for high_seg_w in np.arange(0.0, 1.05, 0.1):
        for g11_w in np.arange(0.0, 1.05, 0.1):
            for g15_w in np.arange(0.0, 1.05, 0.1):
                g12_w = 1.0 - g11_w - g15_w
                if g12_w < -0.01 or g12_w > 1.01:
                    continue
                g12_w = max(0, g12_w)

                # Global blend
                global_blend = g11_w * g11_preds + g15_w * g15_preds + g12_w * g12_preds

                # Hybrid for HIGH
                hybrid_preds = global_blend.copy()
                high_idx = np.where(is_high)[0]
                hybrid_preds[high_idx] = (
                    high_seg_w * high_preds
                    + (1 - high_seg_w) * global_blend[high_idx]
                )

                s = rmsle(y_val, hybrid_preds)
                if s < best_overall:
                    best_overall = s
                    best_config = {
                        "high_seg_w": high_seg_w,
                        "g11_w": g11_w, "g15_w": g15_w, "g12_w": g12_w,
                    }

    logger.info("=" * 60)
    logger.info("  HIGH dedicated:    %.6f", high_score)
    logger.info("  Global TW 1.1:     %.6f", g11_score)
    logger.info("  Global TW 1.5:     %.6f", g15_score)
    logger.info("  Global TW 1.2:     %.6f", g12_score)
    logger.info("  Best Hybrid:       %.6f", best_overall)
    logger.info("  Config: HIGH_seg=%.1f, G11=%.1f, G15=%.1f, G12=%.1f",
                best_config["high_seg_w"], best_config["g11_w"],
                best_config["g15_w"], best_config["g12_w"])
    logger.info("=" * 60)

    # Simple 3-model ensemble for comparison
    for w1 in np.arange(0.0, 0.6, 0.1):
        for w2 in np.arange(0.0, 0.6, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            blend = w1 * g11_preds + w2 * g15_preds + w3 * g12_preds
            s = rmsle(y_val, blend)
            if s < best_overall * 1.001:  # Within 0.1% of best
                logger.info("  Also good global blend: %.6f (%.1f/%.1f/%.1f)",
                            s, w1, w2, w3)

    # =============================================
    # Generate test predictions
    # =============================================
    logger.info("Generating test predictions...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = test_featured[feature_cols].copy()
    X_test[numeric_feat] = X_test[numeric_feat].fillna(0)

    is_high_test = test_featured["family"].isin(HIGH_VOLUME).values

    # Global predictions
    g11_test = global_11.predict(X_test)
    g15_test = global_15.predict(X_test)
    g12_test = global_12.predict(X_test)
    global_blend_test = (
        best_config["g11_w"] * g11_test
        + best_config["g15_w"] * g15_test
        + best_config["g12_w"] * g12_test
    )

    # HIGH segmented predictions
    high_test_preds = high_model.predict(X_test[is_high_test])

    # Hybrid final
    final_pred = global_blend_test.copy()
    high_test_idx = np.where(is_high_test)[0]
    final_pred[high_test_idx] = (
        best_config["high_seg_w"] * high_test_preds
        + (1 - best_config["high_seg_w"]) * global_blend_test[high_test_idx]
    )
    final_pred = np.clip(final_pred, 0, None)

    logger.info("Predictions: mean=%.1f, median=%.1f, zero_pct=%.1f%%",
                final_pred.mean(), np.median(final_pred), (final_pred == 0).mean() * 100)

    save_path = SUBMISSIONS_DIR / "submission_v18.csv"
    generate_submission(test_featured, final_pred, save_path=save_path)
    logger.info("Saved: %s", save_path)

    # Also save pure global blends for A/B testing
    for name, preds in [
        ("g11_g15_g12_blend", global_blend_test),
        ("g11_only", g11_test),
    ]:
        p = SUBMISSIONS_DIR / f"submission_v18_{name}.csv"
        generate_submission(test_featured, np.clip(preds, 0, None), save_path=p)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
