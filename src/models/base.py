"""Abstract base class for all forecasting models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd


class BaseModel(ABC):
    """Abstract base model for store sales forecasting."""

    def __init__(self, name: str, params: Dict[str, Any]):
        self.name = name
        self.params = params
        self.model = None
        self.feature_names: Optional[list] = None

    @abstractmethod
    def fit(self, X_train: pd.DataFrame, y_train: np.ndarray,
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[np.ndarray] = None) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        ...

    @abstractmethod
    def load(self, path: Path) -> "BaseModel":
        ...
