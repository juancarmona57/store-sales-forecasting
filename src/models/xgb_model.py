"""XGBoost model with squared log error objective."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.base import BaseModel
from src.config import XGB_PARAMS, SEED

logger = logging.getLogger(__name__)


class XGBModel(BaseModel):
    """XGBoost with squaredlogerror — optimizes RMSLE directly on original scale."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = {**XGB_PARAMS, **(params or {})}
        super().__init__(name="xgb_rmsle", params=merged_params)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "XGBModel":
        self.feature_names = list(X_train.columns)
        y_train = np.clip(y_train, 0, None)

        self.model = xgb.XGBRegressor(**self.params, random_state=SEED, enable_categorical=True)

        eval_set = []
        if X_val is not None and y_val is not None:
            y_val = np.clip(y_val, 0, None)
            eval_set = [(X_val, y_val)]

        self.model.fit(X_train, y_train, eval_set=eval_set if eval_set else None, verbose=False)

        best_iter = getattr(self.model, "best_iteration", self.params.get("n_estimators"))
        logger.info("XGBoost trained. Best iteration: %s", best_iter)
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
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: Path) -> "XGBModel":
        path = Path(path)
        self.model = xgb.XGBRegressor(**self.params)
        self.model.load_model(str(path))
        logger.info("XGBoost model loaded from %s", path)
        return self
