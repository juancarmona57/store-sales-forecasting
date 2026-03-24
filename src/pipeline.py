"""Full training and submission pipeline.

Uses safe lags (>=16) that are always available during the 16-day test horizon.
No iterative prediction needed — avoids error accumulation.
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import MODELS_DIR, SUBMISSIONS_DIR, TARGET_COL, SEED
from src.data.loader import load_raw_data
from src.data.preprocessor import preprocess_train
from src.features.builder import build_features, get_feature_columns
from src.features.target_stats import compute_target_stats
from src.evaluation.metrics import rmsle
from src.evaluation.validation import TimeSeriesSplitWithGap
from src.models.lgbm_model import LGBMModel
from src.models.xgb_model import XGBModel
from src.models.ensemble import weighted_average_ensemble, optimize_weights
from src.submission.generator import generate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_and_prepare_data():
    """Load, preprocess, build features on combined train+test."""
    logger.info("Loading raw data...")
    train_df, test_df = load_raw_data()
    train_df = preprocess_train(train_df)

    logger.info("Computing target statistics...")
    target_stats = compute_target_stats(train_df)

    # Prepare test rows
    test_df["sales"] = np.nan
    if "transactions" not in test_df.columns:
        test_df["transactions"] = np.nan

    # Combine train + test for proper lag computation
    # With safe lags (>=16), test rows get lags from training data directly
    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = combined.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    logger.info("Building features on combined data (%d rows)...", len(combined))
    combined_featured = build_features(combined, is_train=True, target_stats=target_stats)

    # Split back
    train_max_date = train_df["date"].max()
    train_featured = combined_featured[combined_featured["date"] <= train_max_date].copy()
    test_featured = combined_featured[combined_featured["date"] > train_max_date].copy()

    train_featured = train_featured.dropna(subset=[TARGET_COL]).reset_index(drop=True)
    test_featured = test_featured.reset_index(drop=True)

    feature_cols = get_feature_columns(train_featured)
    logger.info("Features: %d train rows, %d test rows, %d cols",
                len(train_featured), len(test_featured), len(feature_cols))

    # Fill NaN in numeric features
    numeric_feat = [c for c in feature_cols if train_featured[c].dtype.name != "category"]
    train_featured[numeric_feat] = train_featured[numeric_feat].fillna(0)
    test_featured[numeric_feat] = test_featured[numeric_feat].fillna(0)

    return train_featured, test_featured, feature_cols, target_stats


def train_models(train_df, feature_cols):
    """Train LightGBM and XGBoost."""
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

    # Optimize ensemble
    logger.info("Optimizing ensemble weights...")
    opt_weights = optimize_weights(val_predictions, y_val)
    ensemble_preds = weighted_average_ensemble(val_predictions, opt_weights)
    ensemble_score = rmsle(y_val, ensemble_preds)
    scores["ensemble"] = ensemble_score

    weights_path = MODELS_DIR / "weights.json"
    with open(weights_path, "w") as f:
        json.dump({"weights": opt_weights, "scores": scores}, f, indent=2)

    logger.info("=" * 60)
    for name, score in scores.items():
        marker = " <-- BEST" if score == min(scores.values()) else ""
        logger.info("  %-12s: %.6f%s", name, score, marker)
    logger.info("=" * 60)

    return models, opt_weights


def generate_test_submission(models, weights, test_featured, feature_cols):
    """Generate submission from pre-computed test features."""
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)

    for col in feature_cols:
        if col not in test_featured.columns:
            test_featured[col] = 0

    X_test = test_featured[feature_cols]
    numeric_feat = [c for c in feature_cols if X_test[c].dtype.name != "category"]
    X_test[numeric_feat] = X_test[numeric_feat].fillna(0)

    logger.info("Generating predictions (%d rows)...", len(X_test))
    predictions = []
    for name in models:
        preds = models[name].predict(X_test)
        predictions.append(preds)
        logger.info("  %s: mean=%.1f, median=%.1f", name, preds.mean(), np.median(preds))

    ensemble_preds = weighted_average_ensemble(predictions, weights)
    logger.info("Ensemble: mean=%.1f, median=%.1f", ensemble_preds.mean(), np.median(ensemble_preds))

    save_path = SUBMISSIONS_DIR / "submission_v10.csv"
    submission = generate_submission(test_featured, ensemble_preds, save_path=save_path)
    logger.info("Submission saved: %s (%d rows)", save_path, len(submission))
    return submission


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    if not args.train and not args.submit:
        args.train = True
        args.submit = True

    train_featured, test_featured, feature_cols, _ = load_and_prepare_data()

    if args.train:
        models, weights = train_models(train_featured, feature_cols)
        if args.submit:
            generate_test_submission(models, weights, test_featured, feature_cols)

    logger.info("Pipeline complete!")


if __name__ == "__main__":
    main()
