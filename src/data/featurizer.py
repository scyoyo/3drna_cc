"""Build Protenix-compatible input features from competition data."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import MSA_DIR, PDB_RNA_DIR
from src.data.loader import load_msa, parse_stoichiometry
from src.data.msa_processor import filter_msa_by_gap_fraction, subsample_msa


# ── Protenix JSON input builder ───────────────────────────────────────────
#
# Protenix v1.0.0 JSON format (array of jobs):
# [
#   {
#     "name": "<target_id>",
#     "sequences": [
#       {
#         "rnaSequence": {
#           "sequence": "GGGAAACCC",
#           "count": 2,
#           "unpairedMsaPath": "/path/to/rna_msa.a3m"   # optional
#         }
#       },
#       {"ligand": {"ligand": "CCD_MG", "count": 1}},   # CCD code
#       {"ion": {"ion": "MG", "count": 2}},
#     ]
#   }
# ]
#
# Seeds are specified via CLI (--seeds "101,102"), NOT in the JSON.


def build_protenix_input(
    target_id: str,
    sequence: str,
    stoichiometry: str = "",
    all_sequences: str = "",
    ligand_ids: str = "",
    ligand_smiles: str = "",
    msa_dir: str | Path | None = None,
    msa_max_depth: int = 512,
    msa_seed: Optional[int] = None,
    template_hits: Optional[list[dict]] = None,
    model_seeds: Optional[list[int]] = None,
) -> dict:
    """Build a Protenix-format JSON input dict for one target.

    Parameters
    ----------
    target_id      : target identifier.
    sequence       : primary RNA sequence.
    stoichiometry  : e.g. "A2B1".
    all_sequences  : semicolon-separated per-chain sequences.
    ligand_ids     : comma-separated CCD ligand IDs.
    ligand_smiles  : comma-separated SMILES strings.
    msa_dir        : path to MSA directory containing ``{target_id}.MSA.fasta``.
    msa_max_depth  : cap on MSA depth.
    msa_seed       : random seed for MSA sub-sampling (None = no sub-sample).
    template_hits  : list of template dicts from template_search module.
    model_seeds    : ignored (seeds set via CLI), kept for API compat.

    Returns
    -------
    input_dict suitable for ``json.dumps`` and passing to Protenix CLI.
    """
    # ── Prepare RNA MSA a3m file ──────────────────────────────────────
    # Protenix expects an .a3m file path inside the rnaSequence entity.
    # We convert the competition FASTA MSA to a3m format on disk.
    rna_msa_path = _prepare_rna_msa_a3m(
        target_id, msa_dir, msa_max_depth, msa_seed
    )

    # ── Build sequence entities ───────────────────────────────────────
    chains = parse_stoichiometry(stoichiometry, all_sequences or sequence)
    seq_entities = []
    for chain in chains:
        rna_entity: dict = {
            "sequence": chain["sequence"],
            "count": chain["copies"],
        }
        # Attach MSA path to the first RNA chain (primary query)
        if rna_msa_path and not seq_entities:
            rna_entity["unpairedMsaPath"] = str(rna_msa_path)

        seq_entities.append({"rnaSequence": rna_entity})

    # ── Ligands ───────────────────────────────────────────────────────
    if ligand_ids:
        ccd_list = [s.strip() for s in ligand_ids.split(",") if s.strip()]
        for ccd in ccd_list:
            seq_entities.append({"ligand": {"ligand": f"CCD_{ccd}", "count": 1}})
    elif ligand_smiles:
        smiles_list = [s.strip() for s in ligand_smiles.split(",") if s.strip()]
        for smi in smiles_list:
            seq_entities.append({"ligand": {"ligand": smi, "count": 1}})

    # ── Assemble ──────────────────────────────────────────────────────
    input_dict: dict = {
        "name": target_id,
        "sequences": seq_entities,
    }

    return input_dict


def _prepare_rna_msa_a3m(
    target_id: str,
    msa_dir: str | Path | None,
    max_depth: int,
    msa_seed: Optional[int],
) -> Optional[Path]:
    """Convert competition FASTA MSA to .a3m format for Protenix.

    Protenix expects an a3m file (Stockholm-like: sequences without gap
    characters in the query, lower-case insertions). For simplicity we
    write a standard a3m where the query has no insertions.

    Returns path to the a3m file, or None if no MSA available.
    """
    msa_data = load_msa(target_id, msa_dir)
    if msa_data["depth"] == 0:
        return None

    seqs, descs = filter_msa_by_gap_fraction(
        msa_data["sequences"], msa_data["descriptions"]
    )
    if msa_seed is not None:
        seqs, descs = subsample_msa(seqs, descs, max_depth, seed=msa_seed)
    elif len(seqs) > max_depth:
        seqs, descs = subsample_msa(seqs, descs, max_depth, seed=42)

    if not seqs:
        return None

    # Write a3m file alongside the original FASTA
    if msa_dir is None:
        msa_dir = MSA_DIR
    msa_dir = Path(msa_dir)
    a3m_path = msa_dir / f"{target_id}.a3m"

    # If a3m already exists and no sub-sampling requested, reuse it
    if a3m_path.exists() and msa_seed is None:
        return a3m_path

    # Write fresh a3m
    suffix = f"_seed{msa_seed}" if msa_seed is not None else ""
    a3m_path = msa_dir / f"{target_id}{suffix}.a3m"
    with open(a3m_path, "w") as f:
        for desc, seq in zip(descs, seqs):
            f.write(f">{desc}\n{seq}\n")

    return a3m_path


def save_input_json(input_dict: dict, output_path: str | Path) -> Path:
    """Write the Protenix input dict to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump([input_dict], f, indent=2)
    return output_path


def build_all_inputs(
    sequences_df,
    msa_dir: str | Path | None = None,
    output_dir: str | Path = "inputs",
    n_seeds: int = 5,
) -> list[Path]:
    """Build Protenix input JSONs for all targets in a sequences DataFrame."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    for _, row in sequences_df.iterrows():
        tid = row["target_id"]
        inp = build_protenix_input(
            target_id=tid,
            sequence=row["sequence"],
            stoichiometry=row.get("stoichiometry", ""),
            all_sequences=row.get("all_sequences", ""),
            ligand_ids=row.get("ligand_ids", ""),
            ligand_smiles=row.get("ligand_smiles", ""),
            msa_dir=msa_dir,
        )
        p = save_input_json(inp, output_dir / f"{tid}.json")
        paths.append(p)

    return paths
