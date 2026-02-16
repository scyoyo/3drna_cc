"""Load and parse competition data (sequences, labels, MSA)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import (
    TRAIN_SEQUENCES,
    TRAIN_LABELS,
    VAL_SEQUENCES,
    VAL_LABELS,
    TEST_SEQUENCES,
    MSA_DIR,
)


# ── Sequences ─────────────────────────────────────────────────────────────


def load_sequences(csv_path: str | Path | None = None, split: str = "train") -> pd.DataFrame:
    """Load a sequences CSV.

    Parameters
    ----------
    csv_path : path to the CSV.  If *None*, inferred from *split*.
    split    : one of ``"train"``, ``"val"``, ``"test"``.

    Returns
    -------
    DataFrame with columns:
      target_id, sequence, stoichiometry, all_sequences, ligand_ids, ligand_smiles
    """
    if csv_path is None:
        csv_path = {
            "train": TRAIN_SEQUENCES,
            "val": VAL_SEQUENCES,
            "validation": VAL_SEQUENCES,
            "test": TEST_SEQUENCES,
        }[split]
    df = pd.read_csv(csv_path)
    # Ensure string columns are str (not NaN)
    for col in ["stoichiometry", "all_sequences", "ligand_ids", "ligand_smiles"]:
        if col in df.columns:
            df[col] = df[col].fillna("")
    return df


# ── Labels ────────────────────────────────────────────────────────────────


def load_labels(csv_path: str | Path | None = None, split: str = "train") -> pd.DataFrame:
    """Load a labels CSV.

    Returns DataFrame with columns:
      ID, resname, resid, x_1..z_5 (or x_1..z_1 for single-model labels),
      chain, copy
    """
    if csv_path is None:
        csv_path = {
            "train": TRAIN_LABELS,
            "val": VAL_LABELS,
            "validation": VAL_LABELS,
        }[split]
    return pd.read_csv(csv_path)


def labels_to_coords(labels_df: pd.DataFrame, target_id: str) -> np.ndarray:
    """Extract C1' coordinate array for a single target.

    Parameters
    ----------
    labels_df : full labels DataFrame
    target_id : target to extract

    Returns
    -------
    coords : (L, 3) float64 array of (x, y, z) from the first model.
    """
    mask = labels_df["ID"].str.startswith(f"{target_id}_")
    sub = labels_df.loc[mask].sort_values("resid")
    coords = sub[["x_1", "y_1", "z_1"]].values.astype(np.float64)
    return coords


# ── Stoichiometry parsing ─────────────────────────────────────────────────


def parse_stoichiometry(stoichiometry: str, all_sequences: str) -> list[dict]:
    """Parse multi-chain stoichiometry into a list of chain definitions.

    Parameters
    ----------
    stoichiometry : e.g. ``"A2B1"`` meaning 2 copies of chain A, 1 of chain B.
    all_sequences : semicolon-separated sequences for each unique chain type.

    Returns
    -------
    List of dicts, each with keys ``chain_id``, ``sequence``, ``copies``.

    Example
    -------
    >>> parse_stoichiometry("A2B1", "GGGAAACCC;UUUAAAGGG")
    [{'chain_id': 'A', 'sequence': 'GGGAAACCC', 'copies': 2},
     {'chain_id': 'B', 'sequence': 'UUUAAAGGG', 'copies': 1}]
    """
    if not stoichiometry:
        # Single chain
        return [{"chain_id": "A", "sequence": all_sequences, "copies": 1}]

    # Parse "A2B1" → [('A', 2), ('B', 1)]
    parts = re.findall(r"([A-Za-z])(\d+)", stoichiometry)
    sequences = all_sequences.split(";") if all_sequences else []

    chains = []
    for i, (chain_id, count) in enumerate(parts):
        seq = sequences[i] if i < len(sequences) else ""
        chains.append({
            "chain_id": chain_id,
            "sequence": seq,
            "copies": int(count),
        })
    return chains


# ── MSA ───────────────────────────────────────────────────────────────────


def load_msa(target_id: str, msa_dir: str | Path | None = None) -> dict:
    """Load pre-computed MSA for a target.

    Parameters
    ----------
    target_id : target identifier.
    msa_dir   : directory containing ``{target_id}.MSA.fasta`` files.

    Returns
    -------
    dict with:
      - ``sequences`` : list[str] — aligned sequences (first = query)
      - ``descriptions`` : list[str] — FASTA headers
      - ``depth`` : int — number of sequences
    """
    if msa_dir is None:
        msa_dir = MSA_DIR
    msa_dir = Path(msa_dir)

    msa_file = msa_dir / f"{target_id}.MSA.fasta"
    if not msa_file.exists():
        return {"sequences": [], "descriptions": [], "depth": 0}

    sequences: list[str] = []
    descriptions: list[str] = []
    current_seq_parts: list[str] = []
    current_desc = ""

    with open(msa_file) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if current_seq_parts:
                    sequences.append("".join(current_seq_parts))
                    descriptions.append(current_desc)
                    current_seq_parts = []
                current_desc = line[1:]
            else:
                current_seq_parts.append(line)
        # Last entry
        if current_seq_parts:
            sequences.append("".join(current_seq_parts))
            descriptions.append(current_desc)

    return {
        "sequences": sequences,
        "descriptions": descriptions,
        "depth": len(sequences),
    }


# ── Utilities ─────────────────────────────────────────────────────────────


def get_target_ids(split: str = "train") -> list[str]:
    """Return list of target_id values for a split."""
    df = load_sequences(split=split)
    return df["target_id"].tolist()


def get_sequence_lengths(split: str = "train") -> pd.Series:
    """Return a Series of sequence lengths indexed by target_id."""
    df = load_sequences(split=split)
    return df.set_index("target_id")["sequence"].str.len()
