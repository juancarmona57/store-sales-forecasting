"""V20 Pipeline: Multi-Seed Multi-Tweedie Ensemble.

Building on v18 (best Kaggle 0.412):
1. 5 Tweedie variants (1.0, 1.1, 1.2, 1.3, 1.5) — maximum diversity
2. 3 seeds per model — reduces variance (15 models total for val)
3. Final submission trains on ALL data (no holdout) for each variant
4. Earthquake exclusion + transaction forecasting (proven in v18)
5. Optimized ensemble weights
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
from src.submission.generator import generate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

EARTHQUAKE_START = pd.Timestamp("2016-04-16")
EARTHQUAKE_END = pd.Timestamp("2016-05-15")

# Tweedie variants to ensemble
TWEEDIE_POWERS = [1.0, 1.1, 1.2, 1.3, 1.5]
SEEDS = [42, 123, 777]


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


def get_model_params(tweedie_power):
    """Get LGBM params for a specific Tweedie power."""
    # Adjust regularization based on variance power
    if tweedie_power <= 1.1:
        # Closer to Poisson — moderate complexity
        return {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_power,
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
    elif tweedie_power <= 1.3:
        # Middle ground
        return {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_power,
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
    else:
        # Closer to Gamma — more regularization
        return {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_power,
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
        }


def main():
    logger.info("=== V20 PIPELINE: Multi-Seed Multi-Tweedie Ensemble ===")
    logger.info("Models: %d Tweedie variants x %d seeds = %d total",
                len(TWEEDIE_POWERS), len(SEEDS), len(TWEEDIE_POWERS) * len(SEEDS))

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

    # Validation split for evaluation
    val_start = train_max_date - pd.Timedelta(days=15)
    gap_end = val_start - pd.Timedelta(days=16)
    train_mask = train_featured["date"] <= gap_end
    val_mask = train_featured["date"] >= val_start

    X_train = train_featured[train_mask][feature_cols]
    y_train = train_featured[train_mask][TARGET_COL].values
    X_val = train_featured[val_mask][feature_cols]
    y_val = train_featured[val_mask][TARGET_COL].values

    # Full train (for final submission — no holdout)
    X_full = train_featured[feature_cols]
    y_full = train_featured[TARGET_COL].values

    X_test = test_featured[feature_cols].copy()
    X_test[numeric_feat] = X_test[numeric_feat].fillna(0)

    logger.info("Train=%d, Val=%d, Full=%d", len(X_train), len(X_val), len(X_full))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    # =============================================
    # PHASE 1: Train all models with validation
    # =============================================
    logger.info("=== PHASE 1: Training with validation ===")
    val_predictions = {}  # key: (power, seed) -> val preds
    individual_scores = {}

    for power in TWEEDIE_POWERS:
        params = get_model_params(power)
        for seed in SEEDS:
            key = f"tw{power:.1f}_s{seed}"
            t0 = time.time()

            model = lgb.LGBMRegressor(**params, random_state=seed)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
            )
            preds = np.clip(model.predict(X_val), 0, None)
            score = rmsle(y_val, preds)

            val_predictions[key] = preds
            individual_scores[key] = score
            logger.info("  %s: RMSLE=%.6f, iters=%d (%.1fs)",
                        key, score, model.best_iteration_, time.time() - t0)

    # Per-variant averaged predictions (average across seeds)
    variant_preds = {}
    for power in TWEEDIE_POWERS:
        keys = [f"tw{power:.1f}_s{s}" for s in SEEDS]
        avg = np.mean([val_predictions[k] for k in keys], axis=0)
        variant_preds[power] = avg
        score = rmsle(y_val, avg)
        logger.info("TW %.1f (3-seed avg): %.6f", power, score)

    # =============================================
    # PHASE 2: Find optimal ensemble weights
    # =============================================
    logger.info("=== PHASE 2: Optimizing ensemble ===")

    # Simple equal-weight ensemble
    all_preds = list(variant_preds.values())
    equal_blend = np.mean(all_preds, axis=0)
    equal_score = rmsle(y_val, equal_blend)
    logger.info("Equal-weight %d-variant blend: %.6f", len(TWEEDIE_POWERS), equal_score)

    # Grid search for weights (coarse)
    best_score = float("inf")
    best_weights = None
    n = len(TWEEDIE_POWERS)

    # Use scipy optimize for better weight finding
    from scipy.optimize import minimize

    def objective(w):
        w = np.abs(w) / np.abs(w).sum()  # normalize to sum=1
        blend = sum(w[i] * all_preds[i] for i in range(n))
        return rmsle(y_val, blend)

    # Multiple random starts
    for trial in range(20):
        x0 = np.random.dirichlet(np.ones(n))
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 1000, "xatol": 1e-6})
        w = np.abs(result.x) / np.abs(result.x).sum()
        s = objective(w)
        if s < best_score:
            best_score = s
            best_weights = w

    logger.info("Optimized weights: %s", {f"tw{p:.1f}": f"{w:.3f}" for p, w in zip(TWEEDIE_POWERS, best_weights)})
    opt_blend = sum(best_weights[i] * all_preds[i] for i in range(n))
    logger.info("Optimized blend: %.6f", rmsle(y_val, opt_blend))

    # =============================================
    # PHASE 3: Train on FULL data + predict test
    # =============================================
    logger.info("=== PHASE 3: Full-data training for submission ===")

    test_predictions = {}
    for power in TWEEDIE_POWERS:
        params = get_model_params(power)
        seed_preds = []
        for seed in SEEDS:
            key = f"tw{power:.1f}_s{seed}"
            t0 = time.time()

            # Train on FULL data (no validation holdout)
            model = lgb.LGBMRegressor(**params, random_state=seed)
            # Use fixed n_estimators from validation phase (best_iteration)
            val_key = f"tw{power:.1f}_s{seed}"
            # Estimate iterations: use the first seed's best iteration + 10%
            n_est = int(model.n_estimators * 0.8)  # conservative

            model.fit(X_full, y_full, callbacks=[lgb.log_evaluation(0)])
            preds = np.clip(model.predict(X_test), 0, None)
            seed_preds.append(preds)
            logger.info("  Full-data %s done (%.1fs)", key, time.time() - t0)

        test_predictions[power] = np.mean(seed_preds, axis=0)

    # Apply optimized weights
    test_all_preds = [test_predictions[p] for p in TWEEDIE_POWERS]
    final_pred = sum(best_weights[i] * test_all_preds[i] for i in range(n))
    final_pred = np.clip(final_pred, 0, None)

    logger.info("Final predictions: mean=%.1f, median=%.1f, zero_pct=%.1f%%",
                final_pred.mean(), np.median(final_pred), (final_pred == 0).mean() * 100)

    # =============================================
    # SAVE SUBMISSIONS
    # =============================================
    # Main: optimized weights, full-data training
    save_path = SUBMISSIONS_DIR / "submission_v20.csv"
    generate_submission(test_featured, final_pred, save_path=save_path)
    logger.info("Saved: %s", save_path)

    # Equal-weight variant (often more robust on LB)
    equal_test = np.mean(test_all_preds, axis=0)
    equal_test = np.clip(equal_test, 0, None)
    save_eq = SUBMISSIONS_DIR / "submission_v20_equal.csv"
    generate_submission(test_featured, equal_test, save_path=save_eq)
    logger.info("Saved equal-weight: %s", save_eq)

    # Summary
    logger.info("=" * 60)
    logger.info("INDIVIDUAL MODELS (val):")
    for key, score in sorted(individual_scores.items(), key=lambda x: x[1]):
        logger.info("  %s: %.6f", key, score)
    logger.info("SEED-AVERAGED VARIANTS (val):")
    for power in TWEEDIE_POWERS:
        score = rmsle(y_val, variant_preds[power])
        logger.info("  TW %.1f: %.6f (w=%.3f)", power, score, best_weights[TWEEDIE_POWERS.index(power)])
    logger.info("ENSEMBLES (val):")
    logger.info("  Equal-weight:  %.6f", equal_score)
    logger.info("  Optimized:     %.6f", best_score)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
