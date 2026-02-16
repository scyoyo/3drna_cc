"""Extract structural features from PDB template CIF files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


def extract_template_features(
    template_hits: list[dict],
    pdb_cif_dir: str | Path | None = None,
) -> dict:
    """Extract template structural features for Protenix input.

    Parameters
    ----------
    template_hits : list of dicts from ``template_search.search_templates()``.
    pdb_cif_dir   : directory containing CIF files (overrides hit cif_path).

    Returns
    -------
    dict with:
      - ``template_coords``   : (N_templates, L_template, 3) C1' coordinates
      - ``template_mask``     : (N_templates, L_query) bool — which query positions are covered
      - ``template_identities``: (N_templates,) sequence identity scores
      - ``alignment_mappings`` : list of {query_pos: template_pos}
    """
    from src.config import PDB_RNA_DIR

    if pdb_cif_dir is None:
        pdb_cif_dir = PDB_RNA_DIR
    pdb_cif_dir = Path(pdb_cif_dir)

    all_coords = []
    all_identities = []
    all_mappings = []

    for hit in template_hits:
        cif_path = hit.get("cif_path")
        if cif_path is None:
            cif_path = pdb_cif_dir / f"{hit['pdb_id']}.cif"
        else:
            cif_path = Path(cif_path)

        if not cif_path.exists():
            continue

        try:
            coords = _extract_c1_from_cif(cif_path, hit.get("chain_id", "A"))
            all_coords.append(coords)
            all_identities.append(hit.get("identity", 0.0))
            all_mappings.append(hit.get("alignment_mapping", {}))
        except Exception as e:
            print(f"Warning: failed to extract template {hit['pdb_id']}: {e}")
            continue

    if not all_coords:
        return {
            "template_coords": np.zeros((0, 0, 3), dtype=np.float32),
            "template_mask": np.zeros((0, 0), dtype=bool),
            "template_identities": np.zeros(0, dtype=np.float32),
            "alignment_mappings": [],
        }

    return {
        "template_coords": all_coords,  # list of variable-length arrays
        "template_mask": None,  # computed per-query in featurizer
        "template_identities": np.array(all_identities, dtype=np.float32),
        "alignment_mappings": all_mappings,
    }


def build_template_pair_features(
    query_length: int,
    template_coords: list[np.ndarray],
    alignment_mappings: list[dict],
    max_templates: int = 4,
) -> np.ndarray:
    """Build (N_templates, L_query, L_query, C) template pair features.

    For each template, computes pairwise distances between mapped query
    positions, encoding structural prior into the pair representation.

    Returns
    -------
    features : (N_templates, L_query, L_query, 1) float32 array.
    Channel 0 = inverse distance (1 / (1 + d)).
    """
    n_templates = min(len(template_coords), max_templates)
    features = np.zeros(
        (max_templates, query_length, query_length, 1), dtype=np.float32
    )

    for t_idx in range(n_templates):
        coords = template_coords[t_idx]
        mapping = alignment_mappings[t_idx]

        for qi, ti in mapping.items():
            for qj, tj in mapping.items():
                qi, qj = int(qi), int(qj)
                ti, tj = int(ti), int(tj)
                if ti < len(coords) and tj < len(coords):
                    if qi < query_length and qj < query_length:
                        dist = np.linalg.norm(coords[ti] - coords[tj])
                        features[t_idx, qi, qj, 0] = 1.0 / (1.0 + dist)

    return features


def _extract_c1_from_cif(cif_path: Path, chain_id: str = "A") -> np.ndarray:
    """Extract C1' coordinates from a CIF file for a specific chain.

    Returns (L, 3) float64 array.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("template", str(cif_path))

    c1_coords = []
    for model in structure:
        for chain in model:
            if chain.id == chain_id or chain_id == "":
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "C1'":
                            c1_coords.append(atom.get_vector().get_array())
                if c1_coords:
                    break
        break  # first model only

    if not c1_coords:
        raise ValueError(f"No C1' atoms found in {cif_path} chain {chain_id}")

    return np.array(c1_coords, dtype=np.float64)
