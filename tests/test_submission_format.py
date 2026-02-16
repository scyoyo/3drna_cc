"""Tests for submission formatting."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.submission.formatter import format_submission, validate_submission


@pytest.fixture
def sample_submission_csv(tmp_path):
    """Create a minimal sample_submission.csv for testing."""
    rows = []
    for target in ["target_A", "target_B"]:
        for resid in range(1, 4):
            row = {
                "ID": f"{target}_{resid}",
                "resname": "A",
                "resid": resid,
            }
            for i in range(1, 6):
                row[f"x_{i}"] = 0.0
                row[f"y_{i}"] = 0.0
                row[f"z_{i}"] = 0.0
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "sample_submission.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def mock_predictions():
    """Mock predictions: 2 targets, 3 residues each, 5 models."""
    return {
        "target_A": [np.random.randn(3, 3) * 10 for _ in range(5)],
        "target_B": [np.random.randn(3, 3) * 10 for _ in range(5)],
    }


class TestFormatSubmission:
    def test_basic_format(self, sample_submission_csv, mock_predictions, tmp_path):
        sub = format_submission(
            mock_predictions,
            sample_submission_path=sample_submission_csv,
            output_path=tmp_path / "submission.csv",
        )
        assert len(sub) == 6  # 2 targets × 3 residues
        assert "x_1" in sub.columns
        assert "z_5" in sub.columns
        assert not sub.isna().any().any()

    def test_missing_target(self, sample_submission_csv, tmp_path):
        # Only predict target_A, missing target_B
        preds = {
            "target_A": [np.random.randn(3, 3) * 10 for _ in range(5)],
        }
        sub = format_submission(
            preds, sample_submission_path=sample_submission_csv
        )
        assert len(sub) == 6
        # target_B should have zeros
        b_rows = sub[sub["ID"].str.startswith("target_B")]
        assert (b_rows["x_1"] == 0.0).all()


class TestValidateSubmission:
    def test_valid(self, sample_submission_csv, mock_predictions):
        sub = format_submission(
            mock_predictions, sample_submission_path=sample_submission_csv
        )
        assert validate_submission(sub, sample_submission_csv)

    def test_wrong_row_count(self, sample_submission_csv):
        sub = pd.DataFrame({"ID": ["x_1"], "resname": ["A"], "resid": [1],
                           "x_1": [0], "y_1": [0], "z_1": [0],
                           "x_2": [0], "y_2": [0], "z_2": [0],
                           "x_3": [0], "y_3": [0], "z_3": [0],
                           "x_4": [0], "y_4": [0], "z_4": [0],
                           "x_5": [0], "y_5": [0], "z_5": [0]})
        assert not validate_submission(sub, sample_submission_csv)
