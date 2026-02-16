"""Secondary structure prediction — auxiliary features for Protenix."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import C1_C1_WC_PAIR


def predict_secondary_structure(
    sequence: str,
    method: str = "eternafold",
) -> dict:
    """Predict RNA secondary structure (base-pair probability matrix).

    Parameters
    ----------
    sequence : RNA sequence (ACGU).
    method   : ``"eternafold"`` or ``"vienna"`` (RNAfold).

    Returns
    -------
    dict with:
      - ``bpp_matrix``      : (L, L) float32 base-pair probability matrix
      - ``dot_bracket``     : str dot-bracket notation
      - ``paired_residues`` : list[tuple[int,int]] paired positions (0-indexed)
      - ``mfe``             : float minimum free energy (kcal/mol)
    """
    if method == "eternafold":
        return _predict_eternafold(sequence)
    elif method == "vienna":
        return _predict_vienna(sequence)
    else:
        raise ValueError(f"Unknown method: {method}")


def _predict_eternafold(sequence: str) -> dict:
    """Use EternaFold for secondary structure prediction."""
    try:
        import eternafold
    except ImportError:
        print("EternaFold not available, falling back to Vienna RNAfold")
        return _predict_vienna(sequence)

    bpp = eternafold.fold(sequence)
    bpp_matrix = np.array(bpp["bpp_matrix"], dtype=np.float32)
    dot_bracket = bpp.get("structure", "." * len(sequence))
    mfe = bpp.get("mfe", 0.0)
    paired = _dot_bracket_to_pairs(dot_bracket)

    return {
        "bpp_matrix": bpp_matrix,
        "dot_bracket": dot_bracket,
        "paired_residues": paired,
        "mfe": mfe,
    }


def _predict_vienna(sequence: str) -> dict:
    """Use ViennaRNA (RNAfold) for secondary structure prediction."""
    try:
        import RNA

        # Use ViennaRNA Python API
        fc = RNA.fold_compound(sequence)
        structure, mfe = fc.mfe()
        bpp = np.array(fc.bpp(), dtype=np.float32)

        # bpp from ViennaRNA is 1-indexed, convert to 0-indexed (L, L)
        L = len(sequence)
        bpp_matrix = np.zeros((L, L), dtype=np.float32)
        for i in range(1, L + 1):
            for j in range(i + 1, L + 1):
                if i < bpp.shape[0] and j < bpp.shape[1]:
                    bpp_matrix[i - 1, j - 1] = bpp[i, j]
                    bpp_matrix[j - 1, i - 1] = bpp[i, j]

        paired = _dot_bracket_to_pairs(structure)
        return {
            "bpp_matrix": bpp_matrix,
            "dot_bracket": structure,
            "paired_residues": paired,
            "mfe": mfe,
        }
    except ImportError:
        pass

    # Fallback: call RNAfold CLI
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False) as f:
        f.write(f">query\n{sequence}\n")
        fa_path = f.name

    try:
        result = subprocess.run(
            ["RNAfold", "--noPS", "-p"],
            input=f">query\n{sequence}\n",
            capture_output=True,
            text=True,
        )
        lines = result.stdout.strip().split("\n")
        # Parse MFE structure from line 1
        dot_bracket = lines[1].split()[0] if len(lines) > 1 else "." * len(sequence)
        mfe_str = lines[1].split()[-1].strip("()") if len(lines) > 1 else "0.0"
        mfe = float(mfe_str) if mfe_str.replace("-", "").replace(".", "").isdigit() else 0.0

        paired = _dot_bracket_to_pairs(dot_bracket)
        L = len(sequence)

        return {
            "bpp_matrix": np.zeros((L, L), dtype=np.float32),  # CLI doesn't easily give BPP
            "dot_bracket": dot_bracket,
            "paired_residues": paired,
            "mfe": mfe,
        }
    finally:
        Path(fa_path).unlink(missing_ok=True)


def _dot_bracket_to_pairs(structure: str) -> list[tuple[int, int]]:
    """Convert dot-bracket notation to list of base pairs (0-indexed)."""
    stack = []
    pairs = []
    for i, ch in enumerate(structure):
        if ch == "(":
            stack.append(i)
        elif ch == ")":
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return sorted(pairs)


# ── Distance constraints from secondary structure ─────────────────────────


def ss_to_distance_constraints(
    paired_residues: list[tuple[int, int]],
    seq_length: int,
    c1_wc_distance: float = C1_C1_WC_PAIR,
    c1_wc_tolerance: float = 1.5,
) -> dict:
    """Convert secondary structure to C1'–C1' distance constraints.

    Parameters
    ----------
    paired_residues : list of (i, j) base-pair indices (0-indexed).
    seq_length      : total sequence length.
    c1_wc_distance  : expected C1'–C1' distance for WC pairs (~10.4 Å).
    c1_wc_tolerance : allowed deviation from expected distance.

    Returns
    -------
    dict with:
      - ``pair_indices``   : (N, 2) int array of paired residue indices
      - ``target_dists``   : (N,) float array of target distances
      - ``tolerances``     : (N,) float array of tolerances
      - ``constraint_mask``: (L, L) bool mask for constrained pairs
    """
    if not paired_residues:
        return {
            "pair_indices": np.zeros((0, 2), dtype=np.int64),
            "target_dists": np.zeros(0, dtype=np.float32),
            "tolerances": np.zeros(0, dtype=np.float32),
            "constraint_mask": np.zeros((seq_length, seq_length), dtype=bool),
        }

    pair_indices = np.array(paired_residues, dtype=np.int64)
    target_dists = np.full(len(paired_residues), c1_wc_distance, dtype=np.float32)
    tolerances = np.full(len(paired_residues), c1_wc_tolerance, dtype=np.float32)

    constraint_mask = np.zeros((seq_length, seq_length), dtype=bool)
    for i, j in paired_residues:
        constraint_mask[i, j] = True
        constraint_mask[j, i] = True

    return {
        "pair_indices": pair_indices,
        "target_dists": target_dists,
        "tolerances": tolerances,
        "constraint_mask": constraint_mask,
    }
