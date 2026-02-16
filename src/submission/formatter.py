"""Format predictions into Kaggle submission CSV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import SAMPLE_SUBMISSION, OUTPUT_DIR


def format_submission(
    predictions: dict[str, list[np.ndarray]],
    sample_submission_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build a submission DataFrame from predictions.

    Parameters
    ----------
    predictions : dict mapping target_id → list of 5 (L, 3) coordinate arrays.
    sample_submission_path : path to sample_submission.csv for format reference.
    output_path : if provided, write CSV to this path.

    Returns
    -------
    DataFrame with columns: ID, resname, resid, x_1,y_1,z_1, ..., x_5,y_5,z_5
    """
    if sample_submission_path is None:
        sample_submission_path = SAMPLE_SUBMISSION
    sample = pd.read_csv(sample_submission_path)

    # Build output row-by-row
    rows = []
    for _, ref_row in sample.iterrows():
        row_id = ref_row["ID"]
        # Parse target_id and resid from ID column: "{target_id}_{resid}"
        parts = row_id.rsplit("_", 1)
        target_id = parts[0]
        resid = int(parts[1]) if len(parts) > 1 else 0

        row_data = {
            "ID": row_id,
            "resname": ref_row["resname"],
            "resid": resid,
        }

        if target_id in predictions:
            preds = predictions[target_id]
            # resid is 1-indexed in the data, coords are 0-indexed arrays
            idx = resid - 1  # Convert to 0-indexed

            for model_idx in range(5):
                suffix = model_idx + 1
                if model_idx < len(preds) and idx < len(preds[model_idx]):
                    coord = preds[model_idx][idx]
                    row_data[f"x_{suffix}"] = coord[0]
                    row_data[f"y_{suffix}"] = coord[1]
                    row_data[f"z_{suffix}"] = coord[2]
                else:
                    # Fallback: use first prediction or zeros
                    if preds and idx < len(preds[0]):
                        coord = preds[0][idx]
                        row_data[f"x_{suffix}"] = coord[0]
                        row_data[f"y_{suffix}"] = coord[1]
                        row_data[f"z_{suffix}"] = coord[2]
                    else:
                        row_data[f"x_{suffix}"] = 0.0
                        row_data[f"y_{suffix}"] = 0.0
                        row_data[f"z_{suffix}"] = 0.0
        else:
            # Target not predicted — fill with zeros
            for suffix in range(1, 6):
                row_data[f"x_{suffix}"] = 0.0
                row_data[f"y_{suffix}"] = 0.0
                row_data[f"z_{suffix}"] = 0.0

        rows.append(row_data)

    submission = pd.DataFrame(rows)

    # Ensure column order matches sample submission
    coord_cols = []
    for i in range(1, 6):
        coord_cols.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    submission = submission[["ID", "resname", "resid"] + coord_cols]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        print(f"Submission written to {output_path} ({len(submission)} rows)")

    return submission


def validate_submission(
    submission: pd.DataFrame,
    sample_submission_path: str | Path | None = None,
) -> bool:
    """Validate submission format against sample_submission.csv.

    Returns True if valid.
    """
    if sample_submission_path is None:
        sample_submission_path = SAMPLE_SUBMISSION
    sample = pd.read_csv(sample_submission_path)

    issues = []

    # Check row count
    if len(submission) != len(sample):
        issues.append(
            f"Row count mismatch: {len(submission)} vs expected {len(sample)}"
        )

    # Check IDs match
    if not submission["ID"].equals(sample["ID"]):
        mismatched = (submission["ID"] != sample["ID"]).sum()
        issues.append(f"{mismatched} ID mismatches")

    # Check columns
    expected_cols = set(sample.columns)
    actual_cols = set(submission.columns)
    if expected_cols != actual_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        if missing:
            issues.append(f"Missing columns: {missing}")
        if extra:
            issues.append(f"Extra columns: {extra}")

    # Check for NaN coordinates
    coord_cols = [c for c in submission.columns if c.startswith(("x_", "y_", "z_"))]
    nan_count = submission[coord_cols].isna().sum().sum()
    if nan_count > 0:
        issues.append(f"{nan_count} NaN coordinate values")

    # Check coordinate ranges (sanity: should be within ~±1000 Å)
    for col in coord_cols:
        vals = submission[col].dropna()
        if vals.abs().max() > 1000:
            issues.append(f"Extreme values in {col}: max abs = {vals.abs().max():.1f}")

    if issues:
        print("Submission validation FAILED:")
        for issue in issues:
            print(f"  ✗ {issue}")
        return False
    else:
        print(f"Submission validation PASSED ({len(submission)} rows, "
              f"{len(submission['ID'].str.rsplit('_', n=1).str[0].unique())} targets)")
        return True
