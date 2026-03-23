"""Optuna-based hyperparameter optimization.

Provides model-specific search spaces and an optimization loop
that uses TimeSeriesSplit cross-validation for evaluation.
"""

import logging
from typing import Dict, Any, Callable

import numpy as np
import optuna
import pandas as pd

from src.evaluation.metrics import rmsle
from src.evaluation.validation import TimeSeriesSplitWithGap
from src.config import SEED

logger = logging.getLogger(__name__)

# Suppress Optuna's verbose logging
optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_lgbm_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define LightGBM hyperparameter search space.

    Args:
        trial: Optuna trial object.

    Returns:
        Dictionary of sampled hyperparameters.
    """
    return {
        "objective": "tweedie",
        "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.0, 2.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "n_estimators": 2000,
        "verbose": -1,
    }


def get_xgb_search_space(trial: optuna.Trial) -> Dict[str, Any]:
    """Define XGBoost hyperparameter search space.

    Args:
        trial: Optuna trial object.

    Returns:
        Dictionary of sampled hyperparameters.
    """
    return {
        "objective": "reg:squaredlogerror",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 50),
        "n_estimators": 2000,
        "verbosity": 0,
    }


def optimize_model(
    model_class: type,
    search_space_fn: Callable[[optuna.Trial], Dict[str, Any]],
    X: pd.DataFrame,
    y: np.ndarray,
    date_series: pd.Series,
    n_trials: int = 50,
    n_splits: int = 3,
) -> Dict[str, Any]:
    """Run Optuna hyperparameter optimization for a model.

    Args:
        model_class: Model class to optimize (e.g., LGBMModel).
        search_space_fn: Function that returns params dict from an Optuna trial.
        X: Feature DataFrame.
        y: Target array.
        date_series: Date series for time-based splitting.
        n_trials: Number of Optuna trials.
        n_splits: Number of CV folds.

    Returns:
        Best hyperparameters dictionary.
    """
    splitter = TimeSeriesSplitWithGap(n_splits=n_splits)

    # Build a temporary DataFrame for splitting
    split_df = pd.DataFrame({"date": date_series})

    def objective(trial: optuna.Trial) -> float:
        params = search_space_fn(trial)
        scores = []

        for train_idx, val_idx in splitter.split(split_df):
            model = model_class(params=params)
            model.fit(
                X.iloc[train_idx],
                y[train_idx],
                X.iloc[val_idx],
                y[val_idx],
            )
            preds = model.predict(X.iloc[val_idx])
            score = rmsle(y[val_idx], preds)
            scores.append(score)

        return float(np.mean(scores))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    logger.info(
        "Best trial: RMSLE=%.6f, params=%s",
        study.best_trial.value,
        study.best_trial.params,
    )

    return study.best_trial.params
