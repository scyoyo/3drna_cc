"""Protenix inference wrapper — runs AF3 predictions on RNA targets."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import (
    MODEL_DIR,
    OUTPUT_DIR,
    DEFAULT_N_SEEDS,
    DEFAULT_N_DIFFUSION_SAMPLES,
    PROTENIX_MODEL_NAME,
)


class ProtenixRunner:
    """High-level wrapper around Protenix for RNA 3D structure prediction.

    Protenix is invoked via its CLI (``protenix pred``) or Python API.
    This class handles:
    1. Building the input JSON
    2. Running inference with multiple seeds
    3. Parsing CIF outputs to extract C1' coordinates
    """

    def __init__(
        self,
        model_dir: str | Path | None = None,
        model_name: str = PROTENIX_MODEL_NAME,
        lora_dir: str | Path | None = None,
        device: str = "cuda",
    ):
        self.model_dir = Path(model_dir) if model_dir else MODEL_DIR
        self.model_name = model_name
        self.lora_dir = Path(lora_dir) if lora_dir else None
        self.device = device
        self._output_dir = OUTPUT_DIR / "protenix_out"
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Core prediction ───────────────────────────────────────────────

    def predict(
        self,
        input_json_path: str | Path,
        seeds: list[int] | None = None,
        n_samples: int = DEFAULT_N_DIFFUSION_SAMPLES,
        n_cycle: int = 10,
        n_step: int = 200,
        use_templates: bool = True,
        use_msa: bool = True,
        use_rna_msa: bool = True,
        dtype: str = "bf16",
    ) -> list[Path]:
        """Run Protenix prediction, returning paths to output CIF files.

        Parameters
        ----------
        input_json_path : path to the Protenix-format input JSON.
        seeds           : list of diffusion seeds (default: [101..105]).
        n_samples       : number of diffusion samples per seed.
        n_cycle         : number of Pairformer recycling iterations.
        n_step          : number of diffusion steps.
        use_templates   : whether to use template features.
        use_msa         : whether to use protein MSA.
        use_rna_msa     : whether to use RNA MSA.
        dtype           : ``"bf16"`` or ``"fp32"``.

        Returns
        -------
        List of paths to predicted CIF files.
        """
        input_json_path = Path(input_json_path)

        if seeds is None:
            seeds = list(range(101, 101 + DEFAULT_N_SEEDS))

        seeds_str = ",".join(str(s) for s in seeds)

        # Protenix CLI: protenix pred -i <json> -o <dir> -n <model> ...
        cmd = [
            "protenix", "pred",
            "-i", str(input_json_path),
            "-o", str(self._output_dir),
            "-n", self.model_name,
            "-s", seeds_str,
            "-e", str(n_samples),
            "-c", str(n_cycle),
            "-p", str(n_step),
            "-d", dtype,
            "--use_msa", str(use_msa).lower(),
            "--use_template", str(use_templates).lower(),
            "--use_rna_msa", str(use_rna_msa).lower(),
        ]

        print(f"Running Protenix: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Protenix stderr:\n{result.stderr}")
            raise RuntimeError(
                f"Protenix prediction failed (rc={result.returncode})\n"
                f"stdout:\n{result.stdout[:2000]}"
            )

        # Collect output CIF files — pattern: <name>_<seed>_sample_<N>.cif
        cif_files = sorted(self._output_dir.glob("**/*.cif"))
        if not cif_files:
            raise FileNotFoundError(
                f"No CIF outputs found in {self._output_dir}. "
                f"Protenix stdout:\n{result.stdout[:2000]}"
            )
        return cif_files

    def predict_from_dict(
        self,
        input_dict: dict,
        **kwargs,
    ) -> list[Path]:
        """Run prediction from an in-memory dict (writes temp JSON)."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, dir=str(self._output_dir)
        ) as f:
            json.dump([input_dict], f, indent=2)
            tmp_path = Path(f.name)

        try:
            return self.predict(tmp_path, **kwargs)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ── Confidence parsing ────────────────────────────────────────────

    @staticmethod
    def parse_confidence(json_path: str | Path) -> dict:
        """Parse a Protenix confidence summary JSON.

        Returns dict with plddt, ptm, gpde, ranking_score, has_clash, etc.
        """
        with open(json_path) as f:
            return json.load(f)

    def get_confidence_for_cif(self, cif_path: str | Path) -> dict | None:
        """Find and parse the confidence JSON corresponding to a CIF file.

        Protenix outputs: ``<name>_<seed>_sample_<N>.cif``
        Confidence:       ``<name>_<seed>_summary_confidence_sample_<N>.json``
        """
        cif_path = Path(cif_path)
        stem = cif_path.stem  # e.g. "target_101_sample_0"
        conf_name = stem.replace("_sample_", "_summary_confidence_sample_") + ".json"
        conf_path = cif_path.parent / conf_name
        if conf_path.exists():
            return self.parse_confidence(conf_path)
        return None

    # ── C1' extraction from CIF ───────────────────────────────────────

    @staticmethod
    def extract_c1_prime(cif_path: str | Path) -> np.ndarray:
        """Extract C1' atom coordinates from a Protenix output CIF file.

        Parameters
        ----------
        cif_path : path to the predicted structure CIF.

        Returns
        -------
        coords : (L, 3) float64 array of C1' (x, y, z).
        """
        from Bio.PDB.MMCIFParser import MMCIFParser

        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("pred", str(cif_path))

        c1_coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        if atom.get_name() == "C1'":
                            c1_coords.append(atom.get_vector().get_array())
            break  # first model only

        if not c1_coords:
            raise ValueError(f"No C1' atoms found in {cif_path}")

        return np.array(c1_coords, dtype=np.float64)

    @staticmethod
    def extract_c1_prime_all_models(cif_dir: str | Path) -> list[np.ndarray]:
        """Extract C1' coords from all CIF files in a directory.

        Returns list of (L, 3) arrays, one per predicted structure.
        """
        cif_dir = Path(cif_dir)
        cif_files = sorted(cif_dir.glob("*.cif"))
        results = []
        for cif_path in cif_files:
            try:
                coords = ProtenixRunner.extract_c1_prime(cif_path)
                results.append(coords)
            except (ValueError, Exception) as e:
                print(f"Warning: skipping {cif_path.name}: {e}")
        return results

    # ── Batch inference ───────────────────────────────────────────────

    def predict_all(
        self,
        input_json_dir: str | Path,
        seeds: list[int] | None = None,
    ) -> dict[str, list[np.ndarray]]:
        """Run prediction for all JSON files in a directory.

        Returns
        -------
        dict mapping target_id -> list of (L, 3) C1' coordinate arrays.
        """
        input_dir = Path(input_json_dir)
        results = {}

        for json_path in sorted(input_dir.glob("*.json")):
            target_id = json_path.stem
            print(f"\n{'='*60}\nPredicting {target_id}\n{'='*60}")
            try:
                cif_files = self.predict(json_path, seeds=seeds)
                coords_list = [self.extract_c1_prime(f) for f in cif_files]
                results[target_id] = coords_list
            except Exception as e:
                print(f"ERROR predicting {target_id}: {e}")
                results[target_id] = []

        return results
