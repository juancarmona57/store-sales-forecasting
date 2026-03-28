"""CatBoost model for sales forecasting.

Uses RMSE on original scale for ensemble diversity alongside
LightGBM (Tweedie) and XGBoost (squared log error).
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
    """CatBoost with RMSE — predicts on original scale."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = {**CATBOOST_PARAMS, **(params or {})}
        super().__init__(name="catboost", params=merged_params)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "CatBoostModel":
        self.feature_names = list(X_train.columns)

        # Detect categorical columns
        cat_features = [
            i for i, col in enumerate(X_train.columns)
            if X_train[col].dtype.name == "category" or X_train[col].dtype == object
        ]

        self.model = CatBoostRegressor(
            **self.params,
            random_seed=SEED,
            early_stopping_rounds=100,
        )

        eval_set = None
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)

        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            cat_features=cat_features if cat_features else None,
        )

        logger.info("CatBoost trained. Best iteration: %s", self.model.best_iteration_)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        logger.info("CatBoost model saved to %s", path)

    def load(self, path: Path) -> "CatBoostModel":
        path = Path(path)
        self.model = CatBoostRegressor()
        self.model.load_model(str(path))
        logger.info("CatBoost model loaded from %s", path)
        return self
