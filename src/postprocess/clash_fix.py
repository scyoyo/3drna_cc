"""Fix steric clashes and backbone breaks in predicted RNA structures."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from src.config import C1_C1_ADJACENT_MIN, C1_C1_ADJACENT_MAX, CLASH_MIN_DISTANCE


def fix_steric_clashes(
    coords: np.ndarray,
    min_distance: float = CLASH_MIN_DISTANCE,
    max_iterations: int = 100,
    step_size: float = 0.1,
) -> np.ndarray:
    """Fix steric clashes by iterative repulsion.

    Pushes atoms apart when they are closer than *min_distance* Angstroms.

    Parameters
    ----------
    coords        : (L, 3) float64 array of C1' coordinates.
    min_distance  : minimum allowed distance between non-bonded atoms.
    max_iterations: maximum iterations of repulsion.
    step_size     : displacement per iteration.

    Returns
    -------
    Fixed (L, 3) coordinate array.
    """
    coords = coords.copy()
    L = len(coords)

    for iteration in range(max_iterations):
        dists = squareform(pdist(coords))
        clashes_found = False

        for i in range(L):
            for j in range(i + 2, L):  # skip consecutive (bonded) pairs
                d = dists[i, j]
                if d < min_distance and d > 1e-6:
                    clashes_found = True
                    # Push apart along the i→j vector
                    direction = coords[j] - coords[i]
                    direction /= np.linalg.norm(direction) + 1e-8
                    displacement = step_size * (min_distance - d) / 2
                    coords[i] -= direction * displacement
                    coords[j] += direction * displacement

        if not clashes_found:
            break

    return coords


def fix_backbone_breaks(
    coords: np.ndarray,
    min_dist: float = C1_C1_ADJACENT_MIN,
    max_dist: float = C1_C1_ADJACENT_MAX,
    max_iterations: int = 200,
) -> np.ndarray:
    """Fix consecutive C1' distance violations.

    Adjusts coordinates so that adjacent C1' atoms are within
    [min_dist, max_dist] Angstroms.

    Parameters
    ----------
    coords : (L, 3) coordinate array.
    """
    coords = coords.copy()
    target_mid = (min_dist + max_dist) / 2.0
    L = len(coords)

    for _ in range(max_iterations):
        violations = 0
        for i in range(L - 1):
            diff = coords[i + 1] - coords[i]
            d = np.linalg.norm(diff)
            if d < 1e-8:
                # Degenerate — add small random displacement
                coords[i + 1] += np.random.randn(3) * 0.1
                violations += 1
                continue

            if d < min_dist:
                # Too close — push apart
                correction = (min_dist - d) / 2.0
                direction = diff / d
                coords[i] -= direction * correction
                coords[i + 1] += direction * correction
                violations += 1
            elif d > max_dist:
                # Too far — pull together
                correction = (d - max_dist) / 2.0
                direction = diff / d
                coords[i] += direction * correction
                coords[i + 1] -= direction * correction
                violations += 1

        if violations == 0:
            break

    return coords


def fix_all(
    coords: np.ndarray,
    sequence: str | None = None,
    fix_backbone: bool = True,
    fix_clashes: bool = True,
) -> np.ndarray:
    """Apply all geometry fixes in sequence.

    Order: backbone breaks first (local), then steric clashes (global).
    """
    if fix_backbone:
        coords = fix_backbone_breaks(coords)
    if fix_clashes:
        coords = fix_steric_clashes(coords)
    return coords
