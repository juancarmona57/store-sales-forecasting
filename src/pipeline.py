"""Full training and submission pipeline.

Orchestrates data loading, feature engineering, model training,
ensemble creation, and submission generation.

Usage:
    python -m src.pipeline             # Full pipeline
    python -m src.pipeline --train     # Train models only
    python -m src.pipeline --submit    # Generate submission only
"""

import argparse
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
from src.evaluation.metrics import rmsle
from src.evaluation.validation import TimeSeriesSplitWithGap
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.models.catboost_model import CatBoostModel
from src.models.ensemble import weighted_average_ensemble, optimize_weights
from src.submission.generator import generate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load raw data, preprocess, and build features."""
    logger.info("Loading raw data...")
    train_df, _test_df = load_raw_data()
    train_df = preprocess_train(train_df)

    logger.info("Building features for training data...")
    train_featured = build_features(train_df, is_train=True)

    # Drop rows with NaN in target
    train_featured = train_featured.dropna(subset=[TARGET_COL])

    feature_cols = get_feature_columns(train_featured)
    logger.info("Feature matrix: %d rows x %d features", len(train_featured), len(feature_cols))

    return train_featured, feature_cols


def train_models(train_df, feature_cols):
    """Train all three models with cross-validation."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    splitter = TimeSeriesSplitWithGap()
    train_idx, val_idx = splitter.get_holdout_split(train_df)

    X_train = train_df.iloc[train_idx][feature_cols]
    y_train = train_df.iloc[train_idx][TARGET_COL].values
    X_val = train_df.iloc[val_idx][feature_cols]
    y_val = train_df.iloc[val_idx][TARGET_COL].values

    logger.info("Train: %d rows, Val: %d rows", len(X_train), len(X_val))

    models = {}
    scores = {}

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
    logger.info("CatBoost RMSLE: %.6f (%.1fs)", cb_score, time.time() - t0)

    # Optimize ensemble weights
    logger.info("Optimizing ensemble weights...")
    val_predictions = [lgbm_preds, xgb_preds, cb_preds]
    opt_weights = optimize_weights(val_predictions, y_val)

    ensemble_preds = weighted_average_ensemble(val_predictions, opt_weights)
    ensemble_score = rmsle(y_val, ensemble_preds)
    scores["ensemble"] = ensemble_score

    logger.info("=" * 50)
    logger.info("MODEL COMPARISON (Hold-out RMSLE)")
    logger.info("=" * 50)
    for name, score in scores.items():
        marker = " <-- BEST" if score == min(scores.values()) else ""
        logger.info("  %-12s: %.6f%s", name, score, marker)
    logger.info("Ensemble weights: lgbm=%.3f, xgb=%.3f, catboost=%.3f",
                opt_weights[0], opt_weights[1], opt_weights[2])
    logger.info("=" * 50)

    return models, opt_weights, feature_cols


def generate_test_submission(models, weights, feature_cols):
    """Generate submission from trained models."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading test data...")
    _train_raw, test_merged = load_raw_data()

    logger.info("Building features for test data...")
    test_featured = build_features(test_merged, is_train=False)

    # Ensure feature columns exist (fill missing with 0)
    for col in feature_cols:
        if col not in test_featured.columns:
            test_featured[col] = 0

    X_test = test_featured[feature_cols]

    # Generate predictions from each model
    logger.info("Generating predictions...")
    predictions = []
    model_names = ["lgbm", "xgb", "catboost"]
    for name in model_names:
        preds = models[name].predict(X_test)
        predictions.append(preds)
        logger.info("  %s: mean=%.2f, std=%.2f", name, preds.mean(), preds.std())

    # Ensemble
    ensemble_preds = weighted_average_ensemble(predictions, weights)
    logger.info("Ensemble: mean=%.2f, std=%.2f", ensemble_preds.mean(), ensemble_preds.std())

    # Generate submission
    save_path = SUBMISSIONS_DIR / "submission.csv"
    submission = generate_submission(test_featured, ensemble_preds, save_path=save_path)
    logger.info("Submission saved: %s (%d rows)", save_path, len(submission))

    return submission


def main():
    parser = argparse.ArgumentParser(description="Store Sales Forecasting Pipeline")
    parser.add_argument("--train", action="store_true", help="Train models only")
    parser.add_argument("--submit", action="store_true", help="Generate submission only")
    args = parser.parse_args()

    # Default: run everything
    if not args.train and not args.submit:
        args.train = True
        args.submit = True

    if args.train:
        train_df, feature_cols = load_and_prepare_data()
        models, weights, feature_cols = train_models(train_df, feature_cols)

        if args.submit:
            generate_test_submission(models, weights, feature_cols)
    elif args.submit:
        logger.error("--submit without --train not yet supported (needs model loading)")


if __name__ == "__main__":
    main()
