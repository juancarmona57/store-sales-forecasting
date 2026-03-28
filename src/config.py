"""Project configuration: paths, constants, and hyperparameter defaults."""

from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"

# --- Data Files ---
TRAIN_FILE = RAW_DIR / "train.csv"
TEST_FILE = RAW_DIR / "test.csv"
STORES_FILE = RAW_DIR / "stores.csv"
OIL_FILE = RAW_DIR / "oil.csv"
HOLIDAYS_FILE = RAW_DIR / "holidays_events.csv"
TRANSACTIONS_FILE = RAW_DIR / "transactions.csv"
SAMPLE_SUBMISSION_FILE = RAW_DIR / "sample_submission.csv"

# --- Feature Engineering ---
# Safe lags only: all >= 16 to guarantee availability for 16-day test horizon
# No iterative prediction needed — avoids error accumulation
LAG_DAYS = [16, 17, 21, 28, 35, 42, 56, 91, 182, 364]
ROLLING_WINDOWS = [7, 14, 28]
SAFE_SHIFT = 16  # shift >= 16 ensures rolling features use only known data
CATEGORICAL_COLS = ["family", "store_type", "cluster", "city", "state"]
TARGET_COL = "sales"
DATE_COL = "date"

# --- Validation ---
N_SPLITS = 5
VALIDATION_GAP_DAYS = 16
HOLDOUT_DAYS = 16

# --- Model Defaults ---
LGBM_PARAMS = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
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

XGB_PARAMS = {
    "objective": "reg:squaredlogerror",
    "learning_rate": 0.03,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_estimators": 3000,
    "verbosity": 0,
}

CATBOOST_PARAMS = {
    "loss_function": "RMSE",
    "learning_rate": 0.05,
    "depth": 6,
    "iterations": 1500,
    "verbose": 200,
    "l2_leaf_reg": 3,
    "border_count": 128,
}

# --- Ensemble ---
ENSEMBLE_WEIGHTS = {"lgbm": 0.5, "xgb": 0.3, "catboost": 0.2}

# --- Post-Processing ---
PREDICTION_CLIP_THRESHOLD = 0.01  # Values below this are rounded to 0

# --- Random State ---
SEED = 42
