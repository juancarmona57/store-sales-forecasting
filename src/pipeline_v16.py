"""V16 Pipeline: Two-Stage Hurdle Model + Transaction Forecasting.

Stage 1: Binary classifier predicts P(sales > 0) per store×family×date
Stage 2: Regression predicts sales amount ONLY for predicted non-zero rows
Final prediction = P(non-zero) * predicted_amount

This addresses the #1 limitation: 31.3% zero-inflated data.
Also forecasts transactions for test period instead of static imputation.
"""

import json
import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

from src.config import MODELS_DIR, SUBMISSIONS_DIR, TARGET_COL, SEED
from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess_train
from src.features.builder import build_features, get_feature_columns
from src.features.target_stats import compute_target_stats
from src.evaluation.metrics import rmsle
from src.evaluation.validation import TimeSeriesSplitWithGap
from src.models.lgbm_model import LGBMModel
from src.submission.generator import generate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def forecast_transactions(train_df, test_df):
    """Forecast daily transactions per store for test period.

    Instead of static mean imputation, train a small LGBM model
    to predict transactions based on temporal + store features.
    """
    logger.info("Forecasting transactions for test period...")
    transactions = pd.read_csv("data/raw/transactions.csv", parse_dates=["date"])

    # Build features for transaction forecasting
    tx = transactions.copy()
    tx["dow"] = tx.date.dt.dayofweek
    tx["month"] = tx.date.dt.month
    tx["day"] = tx.date.dt.day
    tx["week"] = tx.date.dt.isocalendar().week.astype(int)
    tx["is_weekend"] = (tx.dow >= 5).astype(int)

    # Lag features for transactions (safe: >= 16)
    tx = tx.sort_values(["store_nbr", "date"]).reset_index(drop=True)
    for lag in [16, 21, 28, 35]:
        tx[f"tx_lag_{lag}"] = tx.groupby("store_nbr")["transactions"].shift(lag)

    # Rolling features
    shifted = tx.groupby("store_nbr")["transactions"].shift(16)
    group_key = tx["store_nbr"]
    for w in [7, 14, 28]:
        tx[f"tx_rolling_mean_{w}"] = (
            shifted.groupby(group_key).rolling(w, min_periods=1).mean()
            .reset_index(level=0, drop=True)
        )

    # Train/val split
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

    # Predict for test dates
    test_dates = test_df["date"].unique()
    stores = test_df["store_nbr"].unique()

    # Build test features using last known transactions
    last_known = tx.groupby("store_nbr").tail(50)  # Keep last 50 days per store

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
            # Compute lags from store history
            for lag in [16, 21, 28, 35]:
                ref_date = date - pd.Timedelta(days=lag)
                match = store_tx[store_tx.date == ref_date]
                row[f"tx_lag_{lag}"] = match["transactions"].values[0] if len(match) > 0 else np.nan

            # Rolling features from shifted data
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
    logger.info("=== V16 PIPELINE: Two-Stage Hurdle Model ===")

    # Load data
    train_df, test_df = load_raw_data()
    train_df = preprocess_train(train_df)

    # Forecast transactions for test
    tx_map = forecast_transactions(train_df, test_df)

    # Apply forecasted transactions to test
    test_df["transactions"] = test_df.apply(
        lambda r: tx_map.get((r["store_nbr"], r["date"]), 0), axis=1
    )

    # Compute target stats
    target_stats = compute_target_stats(train_df)

    # Build features
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
    logger.info("Features: %d train, %d test, %d cols", len(train_featured), len(test_featured), len(feature_cols))

    numeric_feat = [c for c in feature_cols if train_featured[c].dtype.name != "category"]
    train_featured[numeric_feat] = train_featured[numeric_feat].fillna(0)
    test_featured[numeric_feat] = test_featured[numeric_feat].fillna(0)

    # Create binary target
    train_featured["is_nonzero"] = (train_featured[TARGET_COL] > 0).astype(int)

    # Split
    splitter = TimeSeriesSplitWithGap()
    train_idx, val_idx = splitter.get_holdout_split(train_featured)

    X_train = train_featured.iloc[train_idx][feature_cols]
    y_train = train_featured.iloc[train_idx][TARGET_COL].values
    y_train_bin = train_featured.iloc[train_idx]["is_nonzero"].values

    X_val = train_featured.iloc[val_idx][feature_cols]
    y_val = train_featured.iloc[val_idx][TARGET_COL].values
    y_val_bin = train_featured.iloc[val_idx]["is_nonzero"].values

    logger.info("Train=%d, Val=%d", len(X_train), len(X_val))
    logger.info("Train non-zero rate: %.1f%%, Val non-zero rate: %.1f%%",
                y_train_bin.mean() * 100, y_val_bin.mean() * 100)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================
    # STAGE 1: Binary classifier (zero vs non-zero)
    # =============================================
    logger.info("STAGE 1: Training zero classifier...")
    t0 = time.time()
    clf = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=0.02,
        num_leaves=256,
        min_child_samples=50,
        feature_fraction=0.7,
        bagging_fraction=0.7,
        bagging_freq=1,
        n_estimators=3000,
        random_state=SEED,
        verbose=-1,
        is_unbalance=True,  # Handle imbalanced classes
    )
    clf.fit(
        X_train, y_train_bin,
        eval_set=[(X_val, y_val_bin)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)],
    )

    # Evaluate classifier
    prob_val = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val_bin, prob_val)

    # Find optimal threshold
    best_thresh, best_f1 = 0.5, 0
    for t in np.arange(0.3, 0.8, 0.05):
        pred_bin = (prob_val >= t).astype(int)
        f1 = f1_score(y_val_bin, pred_bin)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    logger.info("Stage 1 AUC: %.4f, Best F1: %.4f (threshold=%.2f) (%.1fs)",
                auc, best_f1, best_thresh, time.time() - t0)

    # =============================================
    # STAGE 2: Regression on non-zero rows only
    # =============================================
    logger.info("STAGE 2: Training regression on non-zero rows...")
    t0 = time.time()

    # Train only on non-zero sales
    nonzero_mask_train = y_train > 0
    X_train_nz = X_train[nonzero_mask_train]
    y_train_nz = y_train[nonzero_mask_train]

    nonzero_mask_val = y_val > 0
    X_val_nz = X_val[nonzero_mask_val]
    y_val_nz = y_val[nonzero_mask_val]

    logger.info("Non-zero train: %d, val: %d", len(X_train_nz), len(X_val_nz))

    reg = LGBMModel(params={
        "objective": "tweedie",
        "tweedie_variance_power": 1.2,
        "metric": "rmse",
        "learning_rate": 0.01,
        "num_leaves": 512,
        "min_child_samples": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "n_estimators": 5000,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "max_bin": 511,
        "verbose": -1,
    })
    reg.fit(X_train_nz, y_train_nz, X_val_nz, y_val_nz)
    reg_score = rmsle(y_val_nz, reg.predict(X_val_nz))
    logger.info("Stage 2 RMSLE (non-zero only): %.6f (%.1fs)", reg_score, time.time() - t0)

    # =============================================
    # COMBINED: Hurdle model prediction
    # =============================================
    logger.info("Combining stages...")

    # Validation: hurdle prediction
    prob_nonzero = clf.predict_proba(X_val)[:, 1]
    reg_pred = reg.predict(X_val)

    # Method A: Hard threshold
    pred_hard = np.where(prob_nonzero >= best_thresh, reg_pred, 0)
    score_hard = rmsle(y_val, pred_hard)

    # Method B: Soft (probability * amount)
    pred_soft = prob_nonzero * reg_pred
    score_soft = rmsle(y_val, pred_soft)

    # Method C: Pure regression (baseline comparison)
    reg_all = LGBMModel(params={
        "objective": "tweedie",
        "tweedie_variance_power": 1.2,
        "metric": "rmse",
        "learning_rate": 0.01,
        "num_leaves": 512,
        "min_child_samples": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "n_estimators": 5000,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "max_bin": 511,
        "verbose": -1,
    })
    reg_all.fit(X_train, y_train, X_val, y_val)
    pred_baseline = reg_all.predict(X_val)
    score_baseline = rmsle(y_val, pred_baseline)

    # Method D: Blend hurdle with baseline
    pred_blend = 0.5 * pred_hard + 0.5 * pred_baseline
    score_blend = rmsle(y_val, pred_blend)

    logger.info("=" * 60)
    logger.info("  Hurdle Hard:     %.6f", score_hard)
    logger.info("  Hurdle Soft:     %.6f", score_soft)
    logger.info("  Pure Regression: %.6f", score_baseline)
    logger.info("  Blend (50/50):   %.6f", score_blend)
    logger.info("=" * 60)

    # Pick best method
    methods = {
        "hard": (score_hard, pred_hard),
        "soft": (score_soft, pred_soft),
        "baseline": (score_baseline, pred_baseline),
        "blend": (score_blend, pred_blend),
    }
    best_name = min(methods, key=lambda k: methods[k][0])
    best_score = methods[best_name][0]
    logger.info("Best method: %s (%.6f)", best_name, best_score)

    # =============================================
    # Generate test predictions
    # =============================================
    logger.info("Generating test predictions...")
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    X_test = test_featured[feature_cols].copy()
    X_test[numeric_feat] = X_test[numeric_feat].fillna(0)

    if best_name == "hard":
        prob_test = clf.predict_proba(X_test)[:, 1]
        reg_test = reg.predict(X_test)
        final_pred = np.where(prob_test >= best_thresh, reg_test, 0)
    elif best_name == "soft":
        prob_test = clf.predict_proba(X_test)[:, 1]
        reg_test = reg.predict(X_test)
        final_pred = prob_test * reg_test
    elif best_name == "baseline":
        final_pred = reg_all.predict(X_test)
    else:  # blend
        prob_test = clf.predict_proba(X_test)[:, 1]
        reg_test = reg.predict(X_test)
        hard_test = np.where(prob_test >= best_thresh, reg_test, 0)
        baseline_test = reg_all.predict(X_test)
        final_pred = 0.5 * hard_test + 0.5 * baseline_test

    final_pred = np.clip(final_pred, 0, None)
    logger.info("Predictions: mean=%.1f, median=%.1f, zero_pct=%.1f%%",
                final_pred.mean(), np.median(final_pred), (final_pred == 0).mean() * 100)

    save_path = SUBMISSIONS_DIR / "submission_v16.csv"
    generate_submission(test_featured, final_pred, save_path=save_path)
    logger.info("Saved: %s", save_path)
    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
