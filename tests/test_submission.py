"""Tests for src/submission/generator.py."""

import numpy as np
import pandas as pd
import pytest

from src.submission.generator import generate_submission, postprocess_predictions


def test_postprocess_clips_negatives():
    """Negative predictions are clipped to 0."""
    preds = np.array([-1.0, 0.5, 3.0, -0.01])
    result = postprocess_predictions(preds)
    assert (result >= 0).all()


def test_postprocess_rounds_small_to_zero():
    """Very small predictions (<0.01) are rounded to 0."""
    preds = np.array([0.001, 0.005, 0.5, 10.0])
    result = postprocess_predictions(preds)
    assert result[0] == 0.0
    assert result[1] == 0.0
    assert result[2] == 0.5


def test_generate_submission_correct_format():
    """Submission has id and sales columns only."""
    test_df = pd.DataFrame({"id": [0, 1, 2, 3]})
    preds = np.array([1.0, 2.0, 3.0, 4.0])
    result = generate_submission(test_df, preds)
    assert list(result.columns) == ["id", "sales"]
    assert len(result) == 4


def test_generate_submission_saves_csv(tmp_path):
    """Submission is saved to CSV when path is provided."""
    test_df = pd.DataFrame({"id": [0, 1, 2]})
    preds = np.array([1.0, 2.0, 3.0])
    path = tmp_path / "submission.csv"
    generate_submission(test_df, preds, save_path=path)
    assert path.exists()
    loaded = pd.read_csv(path)
    assert list(loaded.columns) == ["id", "sales"]
