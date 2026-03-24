"""Full training and submission pipeline.

Orchestrates data loading, feature engineering, model training,
ensemble creation, and submission generation.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR, SUBMISSIONS_DIR, TARGET_COL,
    ENSEMBLE_WEIGHTS, SEED,
)
from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess_train
from src.features.builder import build_features, get_feature_columns
from src.features.target_stats import compute_target_stats
from src.evaluation.metrics import rmsle
from src.evaluation.validation import TimeSeriesSplitWithGap
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.models.catboost_model import CatBoostModel
from src.models.ensemble import weighted_average_ensemble, optimize_weights
from src.submission.generator import generate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load raw data, preprocess, and build features."""
    logger.info("Loading raw data...")
    train_df, test_df = load_raw_data()
    train_df = preprocess_train(train_df)

    # Compute target statistics BEFORE building features (from clean train data)
    logger.info("Computing target statistics...")
    target_stats = compute_target_stats(train_df)

    # For proper lag computation, concatenate train + test
    # Test rows have no sales - they'll get lag features from train history
    test_df["sales"] = np.nan
    if "transactions" not in test_df.columns:
        test_df["transactions"] = np.nan

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    logger.info("Building features on combined data (%d rows)...", len(combined))
    combined_featured = build_features(combined, is_train=True, target_stats=target_stats)

    # Split back into train and test
    train_mask = combined_featured["date"] <= train_df["date"].max()
    test_mask = combined_featured["date"] > train_df["date"].max()

    train_featured = combined_featured[train_mask].copy()
    test_featured = combined_featured[test_mask].copy()

    # Drop rows with NaN in target (from early rows lacking lag history)
    train_featured = train_featured.dropna(subset=[TARGET_COL])

    feature_cols = get_feature_columns(train_featured)
    logger.info("Features: %d train rows x %d cols", len(train_featured), len(feature_cols))

    # Check NaN counts in features
    nan_counts = train_featured[feature_cols].isna().sum()
    if nan_counts.any():
        logger.info("NaN in features (filling with 0):")
        for col in nan_counts[nan_counts > 0].index[:10]:
            logger.info("  %s: %d NaN", col, nan_counts[col])
    train_featured[feature_cols] = train_featured[feature_cols].fillna(0)
    test_featured[feature_cols] = test_featured[feature_cols].fillna(0)

    return train_featured, test_featured, feature_cols, target_stats


def train_models(train_df, feature_cols):
    """Train all three models with time-series validation."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    splitter = TimeSeriesSplitWithGap()
    train_idx, val_idx = splitter.get_holdout_split(train_df)

    X_train = train_df.iloc[train_idx][feature_cols]
    y_train = train_df.iloc[train_idx][TARGET_COL].values
    X_val = train_df.iloc[val_idx][feature_cols]
    y_val = train_df.iloc[val_idx][TARGET_COL].values

    logger.info("Train=%d  Val=%d", len(X_train), len(X_val))

    models = {}
    scores = {}
    val_predictions = []

    # LightGBM
    logger.info("Training LightGBM...")
    t0 = time.time()
    lgbm = LGBMModel()
    lgbm.fit(X_train, y_train, X_val, y_val)
    lgbm_preds = lgbm.predict(X_val)
    lgbm_score = rmsle(y_val, lgbm_preds)
    lgbm.save(MODELS_DIR / "lgbm_model.txt")
    models["lgbm"] = lgbm
    scores["lgbm"] = lgbm_score
    val_predictions.append(lgbm_preds)
    logger.info("LightGBM RMSLE: %.6f (%.1fs)", lgbm_score, time.time() - t0)

    # XGBoost
    logger.info("Training XGBoost...")
    t0 = time.time()
    xgb_model = XGBModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_preds = xgb_model.predict(X_val)
    xgb_score = rmsle(y_val, xgb_preds)
    xgb_model.save(MODELS_DIR / "xgb_model.json")
    models["xgb"] = xgb_model
    scores["xgb"] = xgb_score
    val_predictions.append(xgb_preds)
    logger.info("XGBoost RMSLE: %.6f (%.1fs)", xgb_score, time.time() - t0)

    # CatBoost
    logger.info("Training CatBoost...")
    t0 = time.time()
    cb_model = CatBoostModel()
    cb_model.fit(X_train, y_train, X_val, y_val)
    cb_preds = cb_model.predict(X_val)
    cb_score = rmsle(y_val, cb_preds)
    cb_model.save(MODELS_DIR / "catboost_model.cbm")
    models["catboost"] = cb_model
    scores["catboost"] = cb_score
    val_predictions.append(cb_preds)
    logger.info("CatBoost RMSLE: %.6f (%.1fs)", cb_score, time.time() - t0)

    # Optimize ensemble weights
    logger.info("Optimizing ensemble weights...")
    opt_weights = optimize_weights(val_predictions, y_val)
    ensemble_preds = weighted_average_ensemble(val_predictions, opt_weights)
    ensemble_score = rmsle(y_val, ensemble_preds)
    scores["ensemble"] = ensemble_score

    # Save weights
    weights_path = MODELS_DIR / "weights.json"
    with open(weights_path, "w") as f:
        json.dump({"weights": opt_weights, "scores": scores}, f, indent=2)

    logger.info("=" * 60)
    logger.info("MODEL COMPARISON (Hold-out RMSLE)")
    logger.info("=" * 60)
    for name, score in scores.items():
        marker = " <-- BEST" if score == min(scores.values()) else ""
        logger.info("  %-12s: %.6f%s", name, score, marker)
    logger.info("Weights: lgbm=%.3f, xgb=%.3f, catboost=%.3f",
                opt_weights[0], opt_weights[1], opt_weights[2])
    logger.info("=" * 60)

    return models, opt_weights


def generate_test_submission(models, weights, test_featured, feature_cols):
    """Generate submission from trained models and pre-computed test features."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    for col in feature_cols:
        if col not in test_featured.columns:
            test_featured[col] = 0

    X_test = test_featured[feature_cols]
    logger.info("Test features: %d rows x %d cols", len(X_test), len(feature_cols))

    # Check for NaN
    nan_test = X_test.isna().sum()
    if nan_test.any():
        logger.info("NaN in test features (filling with 0):")
        for col in nan_test[nan_test > 0].index[:10]:
            logger.info("  %s: %d", col, nan_test[col])
        X_test = X_test.fillna(0)

    # Generate predictions
    predictions = []
    for name in ["lgbm", "xgb", "catboost"]:
        preds = models[name].predict(X_test)
        predictions.append(preds)
        logger.info("  %s: mean=%.2f, median=%.2f, std=%.2f",
                     name, preds.mean(), np.median(preds), preds.std())

    # Ensemble
    ensemble_preds = weighted_average_ensemble(predictions, weights)
    logger.info("Ensemble: mean=%.2f, median=%.2f", ensemble_preds.mean(), np.median(ensemble_preds))

    # Generate submission - need to match test IDs
    save_path = SUBMISSIONS_DIR / "submission_v8.csv"
    submission = generate_submission(test_featured, ensemble_preds, save_path=save_path)
    logger.info("Submission saved: %s (%d rows)", save_path, len(submission))

    return submission


def main():
    parser = argparse.ArgumentParser(description="Store Sales Forecasting Pipeline")
    parser.add_argument("--train", action="store_true", help="Train models only")
    parser.add_argument("--submit", action="store_true", help="Generate submission only")
    args = parser.parse_args()

    if not args.train and not args.submit:
        args.train = True
        args.submit = True

    train_featured, test_featured, feature_cols, target_stats = load_and_prepare_data()

    if args.train:
        models, weights = train_models(train_featured, feature_cols)

        if args.submit:
            generate_test_submission(models, weights, test_featured, feature_cols)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
