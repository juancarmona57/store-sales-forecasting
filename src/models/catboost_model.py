"""CatBoost model with RMSE on log1p-transformed target.

Trains on log1p(sales) and transforms predictions back with expm1,
providing ensemble diversity through different loss landscape.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.models.base import BaseModel
from src.config import CATBOOST_PARAMS, SEED

logger = logging.getLogger(__name__)


class CatBoostModel(BaseModel):
    """CatBoost model training on log1p(target).

    Args:
        params: Model hyperparameters. Merged with CATBOOST_PARAMS defaults.
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = {**CATBOOST_PARAMS, **(params or {})}
        super().__init__(name="catboost_log1p", params=merged_params)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "CatBoostModel":
        """Train CatBoost on log1p-transformed target.

        Args:
            X_train: Training features.
            y_train: Training target (original scale).
            X_val: Validation features.
            y_val: Validation target (original scale).

        Returns:
            self
        """
        self.feature_names = list(X_train.columns)

        # Transform target to log1p scale
        y_log = np.log1p(np.clip(y_train, 0, None))

        self.model = CatBoostRegressor(
            **self.params,
            random_seed=SEED,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            y_val_log = np.log1p(np.clip(y_val, 0, None))
            eval_set = (X_val, y_val_log)

        # Detect categorical columns
        cat_features = [
            i for i, col in enumerate(X_train.columns)
            if X_train[col].dtype.name == "category" or X_train[col].dtype == object
        ]

        self.model.fit(
            X_train,
            y_log,
            eval_set=eval_set,
            cat_features=cat_features if cat_features else None,
            early_stopping_rounds=50 if eval_set else None,
        )

        logger.info("CatBoost trained. Best iteration: %s", self.model.best_iteration_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions, transforming back from log1p scale.

        Args:
            X: Feature DataFrame.

        Returns:
            Array of non-negative predictions in original scale.
        """
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        preds_log = self.model.predict(X)
        preds = np.expm1(preds_log)
        return np.clip(preds, 0, None)

    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: File path for the saved model.
        """
        if self.model is None:
            raise RuntimeError("No model to save.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("CatBoost model saved to %s", path)

    def load(self, path: Path) -> "CatBoostModel":
        """Load model from disk.

        Args:
            path: File path of the saved model.

        Returns:
            self
        """
        path = Path(path)
        self.model = CatBoostRegressor()
        self.model.load_model(str(path))
        logger.info("CatBoost model loaded from %s", path)
        return self
