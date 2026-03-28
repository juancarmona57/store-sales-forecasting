"""V22 Pipeline: Business Logic + Hierarchical Features + Reduced Tree Complexity.

Building on v21 (Kaggle 0.40106):
1. Phase 1: Ecuador business logic (regional holidays, wage cycles, oil regime)
2. Phase 2: Hierarchical features (store/family/city/cluster aggregations)
3. CRITICAL: Reduced num_leaves (256-512 → 64-128) based on top solution research
4. Same multi-seed multi-Tweedie architecture (5 powers × 3 seeds)
5. All proven techniques: earthquake exclusion, tx forecast, structural zero filter
"""

import gc
import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.config import MODELS_DIR, SUBMISSIONS_DIR, TARGET_COL, SEED
from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess_train, filter_structural_zeros
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
    """Get LGBM params — REDUCED num_leaves based on top solution research.

    Key insight: v21 used 256-512 leaves, but top solutions use 16-64.
    Simpler trees generalize better for 16-day forecast horizon.
    """
    if tweedie_power <= 1.1:
        return {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_power,
            "metric": "rmse",
            "learning_rate": 0.01,
            "num_leaves": 64,        # Was 256 → REDUCED
            "max_depth": 7,          # NEW: explicit depth limit
            "min_child_samples": 100, # Was 50 → INCREASED
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
        return {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_power,
            "metric": "rmse",
            "learning_rate": 0.01,
            "num_leaves": 96,        # Was 384 → REDUCED
            "max_depth": 7,
            "min_child_samples": 80,  # Was 40 → INCREASED
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
        return {
            "objective": "tweedie",
            "tweedie_variance_power": tweedie_power,
            "metric": "rmse",
            "learning_rate": 0.01,
            "num_leaves": 128,       # Was 512 → REDUCED
            "max_depth": 8,
            "min_child_samples": 60,  # Was 30 → INCREASED
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
    logger.info("=== V22 PIPELINE: Business Logic + Hierarchical + Reduced Complexity ===")
    logger.info("NEW: regional holidays, wage cycles, oil regime, hierarchical features")
    logger.info("CRITICAL: num_leaves reduced (256-512 → 64-128) per top solution research")

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

    # Filter structural zeros
    train_df, n_removed = filter_structural_zeros(train_df)
    logger.info("Removed %d structural zero rows", n_removed)

    # Build features (now includes Phase 1 + Phase 2 improvements)
    target_stats = compute_target_stats(train_df)
    test_df["sales"] = np.nan
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    logger.info("Building features (v22: +regional holidays, +wage cycles, +hierarchical)...")
    combined_featured = build_features(combined, is_train=True, target_stats=target_stats)

    train_max_date = train_df["date"].max()
    train_featured = combined_featured[combined_featured["date"] <= train_max_date].copy()
    test_featured = combined_featured[combined_featured["date"] > train_max_date].copy()

    train_featured = train_featured.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    test_featured = test_featured.reset_index(drop=True)

    # Use last 2 years
    cutoff_date = train_max_date - pd.Timedelta(days=730)
    train_featured = train_featured[train_featured["date"] >= cutoff_date].reset_index(drop=True)

    feature_cols = get_feature_columns(train_featured)
    logger.info("Features: %d train, %d test, %d cols",
                len(train_featured), len(test_featured), len(feature_cols))

    # Log new features
    new_feat_keywords = [
        "regional", "local_holiday", "effective_holiday", "bridge", "work_day",
        "pre_holiday", "post_holiday", "quincena", "fin_de_mes", "payday_decay",
        "post_payday", "days_since_payday", "oil_regime", "oil_momentum", "oil_above",
        "oil_x_store", "back_to_school", "holiday_x_payday",
        "store_total", "family_national", "city_total", "cluster_family_lag",
        "store_type_family_lag", "tx_vs_store", "tx_rolling_trend",
    ]
    new_feats = [c for c in feature_cols if any(k in c for k in new_feat_keywords)]
    logger.info("New Phase 1+2 features (%d): %s", len(new_feats), new_feats)

    numeric_feat = [c for c in feature_cols if train_featured[c].dtype.name != "category"]
    train_featured[numeric_feat] = train_featured[numeric_feat].fillna(0)
    test_featured[numeric_feat] = test_featured[numeric_feat].fillna(0)

    # Validation split
    val_start = train_max_date - pd.Timedelta(days=15)
    gap_end = val_start - pd.Timedelta(days=16)
    train_mask = train_featured["date"] <= gap_end
    val_mask = train_featured["date"] >= val_start

    X_train = train_featured[train_mask][feature_cols]
    y_train = train_featured[train_mask][TARGET_COL].values
    X_val = train_featured[val_mask][feature_cols]
    y_val = train_featured[val_mask][TARGET_COL].values

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
    logger.info("=== PHASE 1: Training %d models (REDUCED num_leaves) ===",
                len(TWEEDIE_POWERS) * len(SEEDS))

    val_predictions = {}
    individual_scores = {}
    best_iterations = {}

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
            best_iterations[key] = model.best_iteration_
            logger.info("  %s: RMSLE=%.6f, iters=%d (%.1fs)",
                        key, score, model.best_iteration_, time.time() - t0)

            del model
            gc.collect()

    # Per-variant averages
    variant_preds = {}
    for power in TWEEDIE_POWERS:
        keys = [f"tw{power:.1f}_s{s}" for s in SEEDS]
        avg = np.mean([val_predictions[k] for k in keys], axis=0)
        variant_preds[power] = avg
        score = rmsle(y_val, avg)
        logger.info("TW %.1f (3-seed avg): %.6f", power, score)

    del val_predictions
    gc.collect()

    # =============================================
    # PHASE 2: Optimize ensemble weights
    # =============================================
    logger.info("=== PHASE 2: Optimizing ensemble weights ===")

    all_preds = [variant_preds[p] for p in TWEEDIE_POWERS]
    n = len(TWEEDIE_POWERS)

    equal_blend = np.mean(all_preds, axis=0)
    equal_score = rmsle(y_val, equal_blend)
    logger.info("Equal-weight blend: %.6f", equal_score)

    from scipy.optimize import minimize

    def objective(w):
        w = np.abs(w) / np.abs(w).sum()
        blend = sum(w[i] * all_preds[i] for i in range(n))
        return rmsle(y_val, blend)

    best_score = float("inf")
    best_weights = None
    for trial in range(30):
        x0 = np.random.dirichlet(np.ones(n))
        result = minimize(objective, x0, method="Nelder-Mead",
                          options={"maxiter": 2000, "xatol": 1e-7})
        w = np.abs(result.x) / np.abs(result.x).sum()
        s = objective(w)
        if s < best_score:
            best_score = s
            best_weights = w

    logger.info("Optimized weights: %s",
                {f"tw{p:.1f}": f"{w:.3f}" for p, w in zip(TWEEDIE_POWERS, best_weights)})
    logger.info("Optimized blend val RMSLE: %.6f", best_score)

    del variant_preds, all_preds, equal_blend
    del X_train, y_train, X_val, y_val
    gc.collect()

    # =============================================
    # PHASE 3: Full-data training
    # =============================================
    logger.info("=== PHASE 3: Full-data training ===")

    test_predictions = {}
    for power in TWEEDIE_POWERS:
        params = get_model_params(power)
        seed_preds = []
        for seed in SEEDS:
            key = f"tw{power:.1f}_s{seed}"
            t0 = time.time()

            n_iters = int(best_iterations[key] * 1.1)
            params_full = {**params, "n_estimators": n_iters}

            model = lgb.LGBMRegressor(**params_full, random_state=seed)
            model.fit(X_full, y_full, callbacks=[lgb.log_evaluation(0)])
            preds = np.clip(model.predict(X_test), 0, None)
            seed_preds.append(preds)
            logger.info("  Full-data %s: n_iters=%d (%.1fs)", key, n_iters, time.time() - t0)

            del model
            gc.collect()

        test_predictions[power] = np.mean(seed_preds, axis=0)
        del seed_preds
        gc.collect()

    # Apply weights
    test_all_preds = [test_predictions[p] for p in TWEEDIE_POWERS]
    final_pred_opt = np.clip(
        sum(best_weights[i] * test_all_preds[i] for i in range(n)), 0, None
    )
    final_pred_equal = np.clip(np.mean(test_all_preds, axis=0), 0, None)

    logger.info("Optimized: mean=%.1f, median=%.1f, zero=%.1f%%",
                final_pred_opt.mean(), np.median(final_pred_opt),
                (final_pred_opt == 0).mean() * 100)

    # Save
    save_opt = SUBMISSIONS_DIR / "submission_v22.csv"
    generate_submission(test_featured, final_pred_opt, save_path=save_opt)
    logger.info("Saved: %s", save_opt)

    save_eq = SUBMISSIONS_DIR / "submission_v22_equal.csv"
    generate_submission(test_featured, final_pred_equal, save_path=save_eq)
    logger.info("Saved equal: %s", save_eq)

    # Summary
    logger.info("=" * 60)
    logger.info("V22 RESULTS")
    logger.info("=" * 60)
    for key, score in sorted(individual_scores.items(), key=lambda x: x[1]):
        logger.info("  %s: %.6f (iters=%d)", key, score, best_iterations[key])
    logger.info("Equal: %.6f | Optimized: %.6f", equal_score, best_score)
    logger.info("Weights: %s",
                {f"tw{p:.1f}": f"{w:.1%}" for p, w in zip(TWEEDIE_POWERS, best_weights)})
    logger.info("New features: %d | Total features: %d", len(new_feats), len(feature_cols))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
