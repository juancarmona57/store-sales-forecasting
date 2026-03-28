"""V15 Pipeline: 2-LGBM ensemble (Tweedie 1.5 + Tweedie 1.2).

Both fast, both proven. Different variance powers capture
different aspects of the data distribution.
"""

import json
import logging
import time

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
from src.models.ensemble import weighted_average_ensemble, optimize_weights
from src.submission.generator import generate_submission

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== V15 PIPELINE: 2-LGBM Ensemble ===")

    # Load and prepare
    logger.info("Loading raw data...")
    train_df, test_df = load_raw_data()
    train_df = preprocess_train(train_df)
    target_stats = compute_target_stats(train_df)

    test_df["sales"] = np.nan
    if "transactions" not in test_df.columns:
        test_df["transactions"] = np.nan

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
    logger.info("Features: %d train, %d test, %d cols", len(train_featured), len(test_featured), len(feature_cols))

    numeric_feat = [c for c in feature_cols if train_featured[c].dtype.name != "category"]
    train_featured[numeric_feat] = train_featured[numeric_feat].fillna(0)
    test_featured[numeric_feat] = test_featured[numeric_feat].fillna(0)

    # Split
    splitter = TimeSeriesSplitWithGap()
    train_idx, val_idx = splitter.get_holdout_split(train_featured)
    X_train = train_featured.iloc[train_idx][feature_cols]
    y_train = train_featured.iloc[train_idx][TARGET_COL].values
    X_val = train_featured.iloc[val_idx][feature_cols]
    y_val = train_featured.iloc[val_idx][TARGET_COL].values
    logger.info("Train=%d  Val=%d", len(X_train), len(X_val))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Model 1: Tweedie 1.5 (standard)
    logger.info("Training LGBM Tweedie 1.5...")
    t0 = time.time()
    m1 = LGBMModel()
    m1.fit(X_train, y_train, X_val, y_val)
    p1_val = m1.predict(X_val)
    s1 = rmsle(y_val, p1_val)
    m1.save(MODELS_DIR / "lgbm_tw15.txt")
    logger.info("Tweedie 1.5 RMSLE: %.6f (%.1fs)", s1, time.time() - t0)

    # Model 2: Tweedie 1.2 (closer to Poisson, different bias)
    logger.info("Training LGBM Tweedie 1.2...")
    t0 = time.time()
    params2 = {
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
    m2 = LGBMModel(params=params2)
    m2.fit(X_train, y_train, X_val, y_val)
    p2_val = m2.predict(X_val)
    s2 = rmsle(y_val, p2_val)
    m2.save(MODELS_DIR / "lgbm_tw12.txt")
    logger.info("Tweedie 1.2 RMSLE: %.6f (%.1fs)", s2, time.time() - t0)

    # Optimize ensemble
    logger.info("Optimizing weights...")
    opt_w = optimize_weights([p1_val, p2_val], y_val)
    ens_val = weighted_average_ensemble([p1_val, p2_val], opt_w)
    ens_score = rmsle(y_val, ens_val)
    logger.info("Ensemble RMSLE: %.6f (weights: %.3f / %.3f)", ens_score, opt_w[0], opt_w[1])

    # Predictions
    SUBMISSIONS_DIR.mkdir(parents=True, exist_ok=True)
    X_test = test_featured[feature_cols].copy()
    X_test[numeric_feat] = X_test[numeric_feat].fillna(0)

    p1_test = m1.predict(X_test)
    p2_test = m2.predict(X_test)
    ens_test = weighted_average_ensemble([p1_test, p2_test], opt_w)
    logger.info("Predictions: mean=%.1f, median=%.1f", ens_test.mean(), np.median(ens_test))

    save_path = SUBMISSIONS_DIR / "submission_v15.csv"
    generate_submission(test_featured, ens_test, save_path=save_path)
    logger.info("Saved: %s", save_path)

    logger.info("=" * 60)
    logger.info("  Tweedie 1.5: %.6f", s1)
    logger.info("  Tweedie 1.2: %.6f", s2)
    logger.info("  Ensemble:    %.6f", ens_score)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
