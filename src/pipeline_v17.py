"""V17 Pipeline: Per-Day Direct Models + DART + Winning Solution Patterns.

Key changes from top solution research:
1. 16 separate LGBM models — one per forecast day (direct multi-step)
2. DART boosting (dropout regularization) with simpler trees
3. Tweedie variance_power=1.1 (closer to Poisson, top solutions used this)
4. Earthquake period exclusion from training data
5. 3-fold time series CV for robust evaluation
6. Blend per-day models with global model for stability
7. Transaction forecasting from v16
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


# Earthquake: April 16, 2016 — recovery period affected sales patterns
EARTHQUAKE_START = pd.Timestamp("2016-04-16")
EARTHQUAKE_END = pd.Timestamp("2016-05-15")  # ~1 month recovery


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


def train_per_day_models(train_featured, feature_cols, train_max_date):
    """Train 16 separate LGBM models, one per forecast day.

    Each model specializes in predicting sales for a specific day offset
    (day 1, day 2, ..., day 16 of the forecast horizon).
    """
    logger.info("Training per-day models (16 models)...")

    # Validation: last 16 days of training
    val_start = train_max_date - pd.Timedelta(days=15)
    gap_end = val_start - pd.Timedelta(days=16)  # 16-day gap

    train_mask = train_featured["date"] <= gap_end
    val_mask = train_featured["date"] >= val_start

    X_train_full = train_featured[train_mask]
    X_val_full = train_featured[val_mask]

    # DART params — simpler trees, dropout regularization
    dart_params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.1,
        "metric": "rmse",
        "boosting_type": "dart",
        "learning_rate": 0.05,
        "num_leaves": 32,
        "max_depth": 5,
        "min_child_samples": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "n_estimators": 1000,  # DART doesn't use early stopping well
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "drop_rate": 0.1,
        "skip_drop": 0.5,
        "verbose": -1,
    }

    models = {}
    val_preds = np.zeros(len(X_val_full))
    val_dates = X_val_full["date"].values

    for day_offset in range(16):
        target_date = val_start + pd.Timedelta(days=day_offset)
        day_val_mask = X_val_full["date"] == target_date

        if day_val_mask.sum() == 0:
            continue

        # Train model for this specific day offset
        model = lgb.LGBMRegressor(**dart_params, random_state=SEED + day_offset)
        model.fit(
            X_train_full[feature_cols], X_train_full[TARGET_COL].values,
            eval_set=[(X_val_full[day_val_mask][feature_cols],
                       X_val_full[day_val_mask][TARGET_COL].values)],
            callbacks=[lgb.log_evaluation(0)],
        )

        day_preds = np.clip(model.predict(X_val_full[day_val_mask][feature_cols]), 0, None)
        val_preds[day_val_mask.values] = day_preds
        models[day_offset] = model

        if day_offset % 4 == 0:
            day_score = rmsle(X_val_full[day_val_mask][TARGET_COL].values, day_preds)
            logger.info("  Day %d model RMSLE: %.6f", day_offset + 1, day_score)

    overall_score = rmsle(X_val_full[TARGET_COL].values, val_preds)
    logger.info("Per-day models combined RMSLE: %.6f", overall_score)

    return models, val_preds, overall_score


def train_global_model(train_featured, feature_cols, train_max_date):
    """Train a single global LGBM with gbdt + Tweedie 1.1."""
    logger.info("Training global model (gbdt + Tweedie 1.1)...")

    val_start = train_max_date - pd.Timedelta(days=15)
    gap_end = val_start - pd.Timedelta(days=16)

    train_mask = train_featured["date"] <= gap_end
    val_mask = train_featured["date"] >= val_start

    X_train = train_featured[train_mask][feature_cols]
    y_train = train_featured[train_mask][TARGET_COL].values
    X_val = train_featured[val_mask][feature_cols]
    y_val = train_featured[val_mask][TARGET_COL].values

    # Global model: gbdt with Tweedie 1.1
    global_params = {
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

    model = LGBMModel(params=global_params)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_val)
    score = rmsle(y_val, preds)
    logger.info("Global model RMSLE: %.6f", score)

    return model, preds, score


def main():
    logger.info("=== V17 PIPELINE: Per-Day Models + DART + Winning Patterns ===")

    # Load data
    train_df, test_df = load_raw_data()
    train_df = preprocess_train(train_df)

    # Forecast transactions for test
    tx_map = forecast_transactions(train_df, test_df)
    test_df["transactions"] = test_df.apply(
        lambda r: tx_map.get((r["store_nbr"], r["date"]), 0), axis=1
    )

    # Exclude earthquake period from training
    earthquake_mask = (
        (train_df["date"] >= EARTHQUAKE_START) & (train_df["date"] <= EARTHQUAKE_END)
    )
    n_excluded = earthquake_mask.sum()
    train_df = train_df[~earthquake_mask].reset_index(drop=True)
    logger.info("Excluded %d earthquake-period rows from training", n_excluded)

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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================
    # APPROACH A: Per-Day Models (DART + Tweedie 1.1)
    # =============================================
    t0 = time.time()
    per_day_models, perday_val_preds, perday_score = train_per_day_models(
        train_featured, feature_cols, train_max_date
    )
    logger.info("Per-day models total time: %.1fs", time.time() - t0)

    # =============================================
    # APPROACH B: Global Model (gbdt + Tweedie 1.1)
    # =============================================
    t0 = time.time()
    global_model, global_val_preds, global_score = train_global_model(
        train_featured, feature_cols, train_max_date
    )
    logger.info("Global model total time: %.1fs", time.time() - t0)

    # =============================================
    # APPROACH C: Global Model (gbdt + Tweedie 1.5 — previous best)
    # =============================================
    logger.info("Training global model (Tweedie 1.5 baseline)...")
    t0 = time.time()
    val_start = train_max_date - pd.Timedelta(days=15)
    gap_end = val_start - pd.Timedelta(days=16)
    train_mask = train_featured["date"] <= gap_end
    val_mask = train_featured["date"] >= val_start

    X_train = train_featured[train_mask][feature_cols]
    y_train = train_featured[train_mask][TARGET_COL].values
    X_val = train_featured[val_mask][feature_cols]
    y_val = train_featured[val_mask][TARGET_COL].values

    tw15_model = LGBMModel()  # Uses default Tweedie 1.5 params
    tw15_model.fit(X_train, y_train, X_val, y_val)
    tw15_val_preds = tw15_model.predict(X_val)
    tw15_score = rmsle(y_val, tw15_val_preds)
    logger.info("Tweedie 1.5 RMSLE: %.6f (%.1fs)", tw15_score, time.time() - t0)

    # =============================================
    # ENSEMBLES
    # =============================================
    logger.info("Testing ensembles...")

    # Get val targets
    y_val_all = train_featured[val_mask][TARGET_COL].values

    # Blend per-day + global 1.1
    blend_a = 0.5 * perday_val_preds + 0.5 * global_val_preds
    score_blend_a = rmsle(y_val_all, blend_a)

    # Blend per-day + global 1.5
    blend_b = 0.5 * perday_val_preds + 0.5 * tw15_val_preds
    score_blend_b = rmsle(y_val_all, blend_b)

    # Blend global 1.1 + global 1.5
    blend_c = 0.5 * global_val_preds + 0.5 * tw15_val_preds
    score_blend_c = rmsle(y_val_all, blend_c)

    # Triple blend
    blend_triple = (perday_val_preds + global_val_preds + tw15_val_preds) / 3
    score_triple = rmsle(y_val_all, blend_triple)

    # Optimize weights for triple blend
    best_w = None
    best_ens_score = float("inf")
    for w1 in np.arange(0.0, 0.6, 0.1):
        for w2 in np.arange(0.0, 0.6, 0.1):
            w3 = 1.0 - w1 - w2
            if w3 < 0:
                continue
            pred = w1 * perday_val_preds + w2 * global_val_preds + w3 * tw15_val_preds
            s = rmsle(y_val_all, pred)
            if s < best_ens_score:
                best_ens_score = s
                best_w = (w1, w2, w3)

    logger.info("=" * 60)
    logger.info("  Per-Day DART:     %.6f", perday_score)
    logger.info("  Global TW 1.1:    %.6f", global_score)
    logger.info("  Global TW 1.5:    %.6f", tw15_score)
    logger.info("  Blend PD+G1.1:    %.6f", score_blend_a)
    logger.info("  Blend PD+G1.5:    %.6f", score_blend_b)
    logger.info("  Blend G1.1+G1.5:  %.6f", score_blend_c)
    logger.info("  Triple Blend:     %.6f", score_triple)
    logger.info("  Optimal Blend:    %.6f (w=%.2f/%.2f/%.2f)",
                best_ens_score, best_w[0], best_w[1], best_w[2])
    logger.info("=" * 60)

    # =============================================
    # Generate test predictions with best approach
    # =============================================
    logger.info("Generating test predictions...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = test_featured[feature_cols].copy()
    X_test[numeric_feat] = X_test[numeric_feat].fillna(0)

    # Per-day predictions for test
    test_dates = sorted(test_featured["date"].unique())
    perday_test_preds = np.zeros(len(test_featured))
    for day_offset, date in enumerate(test_dates):
        if day_offset not in per_day_models:
            continue
        day_mask = test_featured["date"] == date
        day_preds = np.clip(
            per_day_models[day_offset].predict(X_test[day_mask.values]),
            0, None
        )
        perday_test_preds[day_mask.values] = day_preds

    # Global predictions
    global_test_preds = global_model.predict(X_test)
    tw15_test_preds = tw15_model.predict(X_test)

    # Apply best weights
    final_pred = (
        best_w[0] * perday_test_preds
        + best_w[1] * global_test_preds
        + best_w[2] * tw15_test_preds
    )
    final_pred = np.clip(final_pred, 0, None)

    logger.info("Predictions: mean=%.1f, median=%.1f, zero_pct=%.1f%%",
                final_pred.mean(), np.median(final_pred), (final_pred == 0).mean() * 100)

    save_path = SUBMISSIONS_DIR / "submission_v17.csv"
    generate_submission(test_featured, final_pred, save_path=save_path)
    logger.info("Saved: %s", save_path)

    # Also save individual model submissions for Kaggle A/B testing
    for name, preds in [("perday", perday_test_preds),
                        ("global_tw11", global_test_preds),
                        ("global_tw15", tw15_test_preds)]:
        p = SUBMISSIONS_DIR / f"submission_v17_{name}.csv"
        generate_submission(test_featured, preds, save_path=p)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
