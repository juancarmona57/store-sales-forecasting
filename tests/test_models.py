"""Tests for src/models/ modules."""

import numpy as np
import pandas as pd
import pytest

from src.models.base import BaseModel


def test_base_model_cannot_be_instantiated():
    """BaseModel is abstract and cannot be instantiated directly."""
    with pytest.raises(TypeError):
        BaseModel()


from src.models.lgbm_model import LGBMModel


class TestLGBMModel:
    """Tests for LightGBM model implementation."""

    @pytest.fixture
    def simple_data(self):
        """Simple training data for model tests."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame({
            "feature_1": np.random.randn(n),
            "feature_2": np.random.randn(n),
            "feature_3": np.random.randint(0, 5, n),
        })
        y = np.abs(np.random.randn(n) * 10 + 50)
        return X, y

    def test_fit_returns_self(self, simple_data):
        """fit() returns the model instance."""
        X, y = simple_data
        model = LGBMModel(params={"n_estimators": 10, "verbose": -1})
        result = model.fit(X, y)
        assert result is model

    def test_predict_returns_array(self, simple_data):
        """predict() returns numpy array of correct length."""
        X, y = simple_data
        model = LGBMModel(params={"n_estimators": 10, "verbose": -1})
        model.fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, np.ndarray)
        assert len(preds) == len(X)

    def test_predictions_non_negative(self, simple_data):
        """Predictions are non-negative (Tweedie link)."""
        X, y = simple_data
        model = LGBMModel(params={"n_estimators": 10, "verbose": -1})
        model.fit(X, y)
        preds = model.predict(X)
        assert (preds >= 0).all()

    def test_save_and_load(self, simple_data, tmp_path):
        """Model can be saved and loaded with same predictions."""
        X, y = simple_data
        model = LGBMModel(params={"n_estimators": 10, "verbose": -1})
        model.fit(X, y)
        preds_before = model.predict(X)

        path = tmp_path / "lgbm_model.txt"
        model.save(path)

        loaded = LGBMModel()
        loaded.load(path)
        preds_after = loaded.predict(X)

        np.testing.assert_array_almost_equal(preds_before, preds_after)
