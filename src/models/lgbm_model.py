"""LightGBM model with RMSE on log1p-transformed target."""

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
    """LightGBM model training on log1p(target)."""

    def __init__(self, params: Optional[Dict[str, Any]] = None):
        merged_params = {**LGBM_PARAMS, **(params or {})}
        super().__init__(name="lgbm_log1p", params=merged_params)

    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "LGBMModel":
        self.feature_names = list(X_train.columns)

        y_log = np.log1p(np.clip(y_train, 0, None))

        callbacks = [lgb.log_evaluation(period=200)]
        if X_val is not None and y_val is not None:
            y_val_log = np.log1p(np.clip(y_val, 0, None))
            callbacks.append(lgb.early_stopping(stopping_rounds=100))
            eval_set = [(X_val, y_val_log)]
        else:
            eval_set = None

        self.model = lgb.LGBMRegressor(**self.params, random_state=SEED)
        self.model.fit(X_train, y_log, eval_set=eval_set, callbacks=callbacks)

        logger.info("LightGBM trained. Best iteration: %s",
                     getattr(self.model, "best_iteration_", self.params.get("n_estimators")))
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if isinstance(self.model, lgb.Booster):
            preds_log = self.model.predict(X)
        else:
            preds_log = self.model.predict(X)
        return np.clip(np.expm1(preds_log), 0, None)

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
