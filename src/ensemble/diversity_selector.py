"""Diversity-aware selection of best 5 predictions from N candidates."""

from __future__ import annotations

import numpy as np

from src.ensemble.tm_score import pairwise_rmsd_matrix, kabsch_align


def generate_diverse_predictions(
    runner,
    input_dict: dict,
    n_seed_variants: int = 5,
    n_msa_variants: int = 4,
) -> list[dict]:
    """Generate diverse predictions by varying seeds and MSA sub-samples.

    Parameters
    ----------
    runner          : ProtenixRunner instance.
    input_dict      : base Protenix input dict.
    n_seed_variants : number of diffusion seed variants.
    n_msa_variants  : number of MSA sub-sampling variants.

    Returns
    -------
    List of dicts with ``coords`` (L,3) and ``source`` metadata.
    """
    from src.data.featurizer import build_protenix_input
    from src.data.msa_processor import subsample_msa

    predictions = []

    # Strategy 1: different diffusion seeds (primary diversity source)
    for seed in range(1, n_seed_variants + 1):
        inp = input_dict.copy()
        inp["modelSeeds"] = [seed]
        try:
            cif_files = runner.predict_from_dict(inp, n_seeds=1)
            for cif in cif_files:
                coords = runner.extract_c1_prime(cif)
                predictions.append({
                    "coords": coords,
                    "source": f"seed_{seed}",
                    "confidence": 1.0,  # placeholder
                })
        except Exception as e:
            print(f"Warning: seed {seed} failed: {e}")

    # Strategy 2: different MSA sub-samples
    if "msas" in input_dict and input_dict["msas"]:
        for msa_idx in range(n_msa_variants):
            inp = input_dict.copy()
            inp["modelSeeds"] = [100 + msa_idx]
            # MSA sub-sampling is handled by the featurizer with different seeds
            try:
                cif_files = runner.predict_from_dict(inp, n_seeds=1)
                for cif in cif_files:
                    coords = runner.extract_c1_prime(cif)
                    predictions.append({
                        "coords": coords,
                        "source": f"msa_variant_{msa_idx}",
                        "confidence": 1.0,
                    })
            except Exception as e:
                print(f"Warning: MSA variant {msa_idx} failed: {e}")

    return predictions


def select_best_five(
    predictions: list[dict],
    strategy: str = "maxmin_diversity",
    n_select: int = 5,
) -> list[np.ndarray]:
    """Select the best 5 predictions from N candidates.

    Parameters
    ----------
    predictions : list of dicts with ``coords`` key (each (L, 3)).
    strategy    : selection strategy:
        - ``"maxmin_diversity"`` : greedily maximize minimum pairwise RMSD.
        - ``"confidence_first"`` : pick top confidence, then diversify.
        - ``"random"``          : random selection (baseline).
    n_select    : number of predictions to select.

    Returns
    -------
    List of n_select (L, 3) coordinate arrays.
    """
    if len(predictions) <= n_select:
        return [p["coords"] for p in predictions]

    coords_list = [p["coords"] for p in predictions]

    if strategy == "maxmin_diversity":
        selected_indices = _maxmin_diversity_selection(coords_list, n_select)
    elif strategy == "confidence_first":
        selected_indices = _confidence_first_selection(predictions, n_select)
    elif strategy == "random":
        selected_indices = np.random.choice(
            len(predictions), n_select, replace=False
        ).tolist()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return [coords_list[i] for i in selected_indices]


def _maxmin_diversity_selection(
    coords_list: list[np.ndarray],
    n_select: int,
) -> list[int]:
    """Greedily select predictions to maximize minimum pairwise RMSD.

    This ensures the 5 selected predictions are as structurally diverse
    as possible, maximizing the chance that at least one is close to
    the ground truth.
    """
    N = len(coords_list)
    rmsd_mat = pairwise_rmsd_matrix(coords_list)

    # Start with the prediction that has the highest average RMSD to all others
    avg_rmsd = rmsd_mat.sum(axis=1) / (N - 1)
    selected = [int(np.argmax(avg_rmsd))]

    for _ in range(n_select - 1):
        best_candidate = -1
        best_min_dist = -1.0

        for i in range(N):
            if i in selected:
                continue
            # Min RMSD from candidate i to all already-selected
            min_dist = min(rmsd_mat[i, j] for j in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = i

        if best_candidate >= 0:
            selected.append(best_candidate)

    return selected


def _confidence_first_selection(
    predictions: list[dict],
    n_select: int,
) -> list[int]:
    """Select top-confidence prediction first, then diversify the rest."""
    # Sort by confidence
    indexed = sorted(
        enumerate(predictions),
        key=lambda x: x[1].get("confidence", 0.0),
        reverse=True,
    )

    # Pick top confidence
    selected = [indexed[0][0]]

    # Fill remaining with diversity
    coords_list = [p["coords"] for p in predictions]
    rmsd_mat = pairwise_rmsd_matrix(coords_list)

    for _ in range(n_select - 1):
        best_candidate = -1
        best_min_dist = -1.0

        for idx, _ in indexed:
            if idx in selected:
                continue
            min_dist = min(rmsd_mat[idx, j] for j in selected)
            if min_dist > best_min_dist:
                best_min_dist = min_dist
                best_candidate = idx

        if best_candidate >= 0:
            selected.append(best_candidate)

    return selected
