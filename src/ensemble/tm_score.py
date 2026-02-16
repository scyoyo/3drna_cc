"""TM-score computation for RNA 3D structure comparison."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def compute_tm_score(
    coords_pred: np.ndarray,
    coords_ref: np.ndarray,
) -> float:
    """Compute TM-score between predicted and reference C1' coordinates.

    Uses the Zhang & Skolnick (2004) formula adapted for nucleic acids.

    Parameters
    ----------
    coords_pred : (L, 3) predicted coordinates.
    coords_ref  : (L, 3) reference coordinates.

    Returns
    -------
    TM-score in [0, 1].  1.0 = perfect match.
    """
    L = len(coords_ref)
    assert len(coords_pred) == L, (
        f"Length mismatch: pred={len(coords_pred)}, ref={L}"
    )

    if L == 0:
        return 0.0

    # d0 normalization factor (Zhang & Skolnick formula)
    # For RNA, use the same formula as protein with adjusted constant
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)

    # Superpose using Kabsch alignment
    pred_aligned = kabsch_align(coords_pred, coords_ref)

    # Per-residue distances after alignment
    di = np.linalg.norm(pred_aligned - coords_ref, axis=1)  # (L,)

    # TM-score
    tm = np.sum(1.0 / (1.0 + (di / d0) ** 2)) / L

    return float(tm)


def compute_rmsd(coords_pred: np.ndarray, coords_ref: np.ndarray) -> float:
    """Compute RMSD after Kabsch alignment."""
    pred_aligned = kabsch_align(coords_pred, coords_ref)
    diff = pred_aligned - coords_ref
    return float(np.sqrt(np.mean(np.sum(diff ** 2, axis=1))))


def kabsch_align(
    coords_moving: np.ndarray,
    coords_target: np.ndarray,
) -> np.ndarray:
    """Kabsch superposition: align *coords_moving* onto *coords_target*.

    Returns the transformed (rotated + translated) moving coordinates.
    """
    # Center both
    center_moving = coords_moving.mean(axis=0)
    center_target = coords_target.mean(axis=0)
    p = coords_moving - center_moving
    q = coords_target - center_target

    # Cross-covariance matrix
    H = p.T @ q  # (3, 3)

    # SVD
    U, S, Vt = np.linalg.svd(H)

    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])

    # Optimal rotation
    R = Vt.T @ sign_matrix @ U.T

    # Apply
    aligned = (p @ R.T) + center_target
    return aligned


def pairwise_rmsd_matrix(coords_list: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise RMSD matrix between multiple structures.

    Parameters
    ----------
    coords_list : list of (L, 3) arrays (all same L).

    Returns
    -------
    (N, N) symmetric RMSD matrix.
    """
    N = len(coords_list)
    mat = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(i + 1, N):
            rmsd = compute_rmsd(coords_list[i], coords_list[j])
            mat[i, j] = rmsd
            mat[j, i] = rmsd
    return mat


def best_of_n_tm_score(
    predictions: list[np.ndarray],
    reference: np.ndarray,
) -> tuple[float, int]:
    """Compute best-of-N TM-score (competition metric).

    Returns
    -------
    best_tm   : highest TM-score among all predictions.
    best_idx  : index of the best prediction.
    """
    best_tm = -1.0
    best_idx = 0
    for i, pred in enumerate(predictions):
        tm = compute_tm_score(pred, reference)
        if tm > best_tm:
            best_tm = tm
            best_idx = i
    return best_tm, best_idx
