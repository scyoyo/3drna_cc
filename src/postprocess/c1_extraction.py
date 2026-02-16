"""Extract C1' atom coordinates from Protenix CIF output files."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def extract_c1_from_cif(cif_path: str | Path) -> np.ndarray:
    """Extract C1' coordinates from a predicted CIF structure.

    Parameters
    ----------
    cif_path : path to the CIF file.

    Returns
    -------
    coords : (L, 3) float64 array ordered by chain then residue ID.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("pred", str(cif_path))

    c1_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "C1'":
                        c1_atoms.append({
                            "chain": chain.id,
                            "resid": residue.id[1],
                            "resname": residue.resname.strip(),
                            "coords": atom.get_vector().get_array(),
                        })
        break  # first model only

    if not c1_atoms:
        raise ValueError(f"No C1' atoms found in {cif_path}")

    # Sort by chain, then residue ID
    c1_atoms.sort(key=lambda x: (x["chain"], x["resid"]))
    coords = np.array([a["coords"] for a in c1_atoms], dtype=np.float64)
    return coords


def extract_c1_with_metadata(cif_path: str | Path) -> tuple[np.ndarray, list[dict]]:
    """Extract C1' coords along with per-residue metadata.

    Returns
    -------
    coords   : (L, 3) coordinate array.
    metadata : list of dicts with chain, resid, resname per residue.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("pred", str(cif_path))

    c1_atoms = []
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.get_name() == "C1'":
                        c1_atoms.append({
                            "chain": chain.id,
                            "resid": residue.id[1],
                            "resname": residue.resname.strip(),
                            "coords": atom.get_vector().get_array(),
                        })
        break

    if not c1_atoms:
        raise ValueError(f"No C1' atoms found in {cif_path}")

    c1_atoms.sort(key=lambda x: (x["chain"], x["resid"]))
    coords = np.array([a["coords"] for a in c1_atoms], dtype=np.float64)
    metadata = [{"chain": a["chain"], "resid": a["resid"], "resname": a["resname"]}
                for a in c1_atoms]
    return coords, metadata


def extract_all_cifs(cif_dir: str | Path) -> list[np.ndarray]:
    """Extract C1' coords from all CIF files in a directory.

    Returns list of (L, 3) arrays.
    """
    cif_dir = Path(cif_dir)
    results = []
    for cif_path in sorted(cif_dir.glob("*.cif")):
        try:
            coords = extract_c1_from_cif(cif_path)
            results.append(coords)
        except (ValueError, Exception) as e:
            print(f"Warning: skipping {cif_path.name}: {e}")
    return results
