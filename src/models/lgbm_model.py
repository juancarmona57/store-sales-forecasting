"""LightGBM model with Tweedie objective for sales forecasting."""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.models.base import BaseModel
from src.config import LGBM_PARAMS, SEED

logger = logging.getLogger(__name__)


class LGBMModel(BaseModel):
    """LightGBM model with Tweedie objective."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = {**LGBM_PARAMS, **(params or {})}
        super().__init__(name="lgbm_tweedie", params=merged_params)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "LGBMModel":
        self.feature_names = list(X_train.columns)

        callbacks = [lgb.log_evaluation(period=100)]
        if X_val is not None and y_val is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=50))
            eval_set = [(X_val, y_val)]
        else:
            eval_set = None

        self.model = lgb.LGBMRegressor(**self.params, random_state=SEED)
        self.model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)

        logger.info("LightGBM trained. Best iteration: %s",
                     getattr(self.model, "best_iteration_", self.params.get("n_estimators")))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if isinstance(self.model, lgb.Booster):
            preds = self.model.predict(X)
        else:
            preds = self.model.predict(X)
        return np.clip(preds, 0, None)

    def save(self, path: Path) -> None:
        if self.model is None:
            raise RuntimeError("No model to save. Call fit() first.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.booster_.save_model(str(path))
        logger.info("LightGBM model saved to %s", path)

    def load(self, path: Path) -> "LGBMModel":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        self._booster = lgb.Booster(model_file=str(path))
        self.model = self._booster
        logger.info("LightGBM model loaded from %s", path)
        return self
