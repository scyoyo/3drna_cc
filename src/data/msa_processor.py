"""MSA preprocessing: sub-sampling, depth trimming, gap filtering."""

from __future__ import annotations

import random
from typing import Optional

import numpy as np


# ── Filtering ─────────────────────────────────────────────────────────────


def filter_msa_by_gap_fraction(
    sequences: list[str],
    descriptions: list[str],
    max_gap_fraction: float = 0.5,
) -> tuple[list[str], list[str]]:
    """Remove MSA rows where more than *max_gap_fraction* of positions are gaps.

    The query (first row) is always kept.
    """
    kept_seqs = [sequences[0]]
    kept_descs = [descriptions[0]]
    query_len = len(sequences[0])

    for seq, desc in zip(sequences[1:], descriptions[1:]):
        n_gaps = seq.count("-") + seq.count(".")
        if n_gaps / max(query_len, 1) <= max_gap_fraction:
            kept_seqs.append(seq)
            kept_descs.append(desc)

    return kept_seqs, kept_descs


# ── Sub-sampling ──────────────────────────────────────────────────────────


def subsample_msa(
    sequences: list[str],
    descriptions: list[str],
    max_depth: int = 512,
    seed: Optional[int] = None,
) -> tuple[list[str], list[str]]:
    """Randomly sub-sample MSA to at most *max_depth* sequences.

    The query (first row) is always kept.
    """
    if len(sequences) <= max_depth:
        return sequences, descriptions

    rng = random.Random(seed)
    indices = list(range(1, len(sequences)))
    rng.shuffle(indices)
    selected = sorted(indices[: max_depth - 1])

    kept_seqs = [sequences[0]] + [sequences[i] for i in selected]
    kept_descs = [descriptions[0]] + [descriptions[i] for i in selected]
    return kept_seqs, kept_descs


def subsample_msa_diverse(
    sequences: list[str],
    descriptions: list[str],
    max_depth: int = 512,
    n_variants: int = 4,
    base_seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """Generate *n_variants* different MSA sub-samples for diversity.

    Each variant uses a different random seed so the resulting predictions
    cover different parts of the co-evolutionary signal.
    """
    variants = []
    for i in range(n_variants):
        s, d = subsample_msa(sequences, descriptions, max_depth, seed=base_seed + i)
        variants.append((s, d))
    return variants


# ── One-hot encoding ──────────────────────────────────────────────────────

_RNA_TOKENS = {
    "A": 0, "C": 1, "G": 2, "U": 3, "T": 3,
    "-": 4, ".": 4, "N": 5,
}
_NUM_TOKENS = 6


def msa_to_onehot(sequences: list[str]) -> np.ndarray:
    """Convert MSA sequences to a one-hot array.

    Returns
    -------
    onehot : (depth, length, n_tokens) uint8 array.
    """
    depth = len(sequences)
    length = len(sequences[0]) if sequences else 0

    arr = np.zeros((depth, length, _NUM_TOKENS), dtype=np.uint8)
    for i, seq in enumerate(sequences):
        for j, ch in enumerate(seq):
            tok = _RNA_TOKENS.get(ch.upper(), 5)
            arr[i, j, tok] = 1
    return arr


def compute_msa_covariance(sequences: list[str]) -> np.ndarray:
    """Compute pairwise covariance matrix from MSA (simplified DCA-like).

    Returns
    -------
    cov : (L, L) float32 array where L = alignment length.
    Higher values suggest co-evolving (possibly contacting) residue pairs.
    """
    onehot = msa_to_onehot(sequences).astype(np.float32)  # (D, L, T)
    depth, length, n_tok = onehot.shape

    if depth < 2:
        return np.zeros((length, length), dtype=np.float32)

    # Flatten tokens: (D, L*T)
    flat = onehot.reshape(depth, -1)
    # Mean-center
    flat -= flat.mean(axis=0, keepdims=True)
    # Covariance over depth dimension
    cov_full = (flat.T @ flat) / (depth - 1)  # (L*T, L*T)

    # Aggregate over token pairs to get (L, L) contact-like scores
    cov_full = cov_full.reshape(length, n_tok, length, n_tok)
    # Frobenius norm over token dimensions
    cov = np.sqrt((cov_full ** 2).sum(axis=(1, 3)))

    # Zero out diagonal
    np.fill_diagonal(cov, 0.0)
    return cov
