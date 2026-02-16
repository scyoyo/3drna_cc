"""Validate predicted RNA structure geometry."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.config import C1_C1_ADJACENT_MIN, C1_C1_ADJACENT_MAX, CLASH_MIN_DISTANCE


def validate_geometry(
    coords: np.ndarray,
    sequence: str | None = None,
) -> dict:
    """Validate the geometric quality of predicted C1' coordinates.

    Parameters
    ----------
    coords   : (L, 3) float64 array.
    sequence : RNA sequence (optional, for future residue-specific checks).

    Returns
    -------
    Report dict with:
      - ``n_residues``           : int
      - ``backbone_distances``   : (L-1,) array of consecutive C1' distances
      - ``backbone_violations``  : int — count outside [5.0, 7.0] Å
      - ``backbone_breaks``      : list of (i, i+1, distance) tuples for violations
      - ``clash_count``          : int — atom pairs closer than 3.0 Å
      - ``clashes``              : list of (i, j, distance) tuples
      - ``radius_of_gyration``   : float
      - ``is_valid``             : bool — no severe issues
    """
    L = len(coords)
    report: dict = {"n_residues": L}

    # ── Backbone distances ────────────────────────────────────────────
    if L > 1:
        diffs = np.diff(coords, axis=0)
        bb_dists = np.linalg.norm(diffs, axis=1)
    else:
        bb_dists = np.array([], dtype=np.float64)

    report["backbone_distances"] = bb_dists

    violations = []
    for i, d in enumerate(bb_dists):
        if d < C1_C1_ADJACENT_MIN or d > C1_C1_ADJACENT_MAX:
            violations.append((i, i + 1, float(d)))
    report["backbone_violations"] = len(violations)
    report["backbone_breaks"] = violations

    # ── Steric clashes ────────────────────────────────────────────────
    clashes = []
    if L > 2:
        dist_matrix = squareform(pdist(coords))
        for i in range(L):
            for j in range(i + 2, L):  # skip bonded neighbors
                if dist_matrix[i, j] < CLASH_MIN_DISTANCE:
                    clashes.append((i, j, float(dist_matrix[i, j])))
    report["clash_count"] = len(clashes)
    report["clashes"] = clashes

    # ── Radius of gyration ────────────────────────────────────────────
    if L > 0:
        centroid = coords.mean(axis=0)
        rg = np.sqrt(np.mean(np.sum((coords - centroid) ** 2, axis=1)))
    else:
        rg = 0.0
    report["radius_of_gyration"] = float(rg)

    # ── Overall validity ──────────────────────────────────────────────
    severe_bb = sum(1 for _, _, d in violations if d < 3.0 or d > 15.0)
    report["is_valid"] = severe_bb == 0 and len(clashes) < L * 0.1

    return report


def print_geometry_report(report: dict):
    """Pretty-print a geometry validation report."""
    print(f"Residues: {report['n_residues']}")
    print(f"Backbone violations: {report['backbone_violations']}")
    if report["backbone_breaks"]:
        for i, j, d in report["backbone_breaks"][:10]:
            print(f"  residue {i}-{j}: {d:.2f} Å")
        if len(report["backbone_breaks"]) > 10:
            print(f"  ... and {len(report['backbone_breaks']) - 10} more")
    print(f"Steric clashes: {report['clash_count']}")
    if report["clashes"]:
        for i, j, d in report["clashes"][:10]:
            print(f"  residue {i}-{j}: {d:.2f} Å")
    print(f"Radius of gyration: {report['radius_of_gyration']:.2f} Å")
    print(f"Valid: {'YES' if report['is_valid'] else 'NO'}")
