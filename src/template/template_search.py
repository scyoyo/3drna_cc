"""PDB template search — find homologous RNA structures in PDB_RNA/."""

from __future__ import annotations

import csv
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import PDB_SEQRES, PDB_DATES, PDB_RNA_DIR


def search_templates(
    target_id: str,
    sequence: str,
    pdb_seqres_path: str | Path | None = None,
    pdb_dates_path: str | Path | None = None,
    pdb_dir: str | Path | None = None,
    temporal_cutoff: Optional[str] = None,
    max_templates: int = 4,
    min_identity: float = 0.3,
    method: str = "mmseqs2",
) -> list[dict]:
    """Search for homologous RNA structures in PDB_RNA/.

    Parameters
    ----------
    target_id       : target identifier.
    sequence        : query RNA sequence.
    pdb_seqres_path : path to ``pdb_seqres_NA.fasta``.
    pdb_dates_path  : path to ``pdb_release_dates_NA.csv``.
    temporal_cutoff : exclude templates released after this date (YYYY-MM-DD).
    max_templates   : maximum number of templates to return.
    min_identity    : minimum sequence identity threshold.
    method          : ``"mmseqs2"`` or ``"blastn"``.

    Returns
    -------
    List of template hit dicts with keys:
      - pdb_id, chain_id, identity, alignment_mapping, cif_path, release_date
    """
    pdb_seqres_path = Path(pdb_seqres_path) if pdb_seqres_path else PDB_SEQRES
    pdb_dates_path = Path(pdb_dates_path) if pdb_dates_path else PDB_DATES
    pdb_dir = Path(pdb_dir) if pdb_dir else PDB_RNA_DIR

    # Load release dates for temporal filtering
    date_map = {}
    if pdb_dates_path.exists():
        date_map = _load_pdb_dates(pdb_dates_path)

    if method == "mmseqs2":
        raw_hits = _search_mmseqs2(sequence, pdb_seqres_path)
    else:
        raw_hits = _search_blastn(sequence, pdb_seqres_path)

    # Filter and rank
    templates = []
    for hit in raw_hits:
        pdb_id = hit["pdb_id"]

        # Identity filter
        if hit["identity"] < min_identity:
            continue

        # Temporal filter
        if temporal_cutoff and pdb_id.lower() in date_map:
            release = date_map[pdb_id.lower()]
            if release > temporal_cutoff:
                continue

        # Check CIF exists
        cif_path = pdb_dir / f"{pdb_id}.cif"
        if not cif_path.exists():
            cif_path = pdb_dir / f"{pdb_id.upper()}.cif"
        if not cif_path.exists():
            continue

        hit["cif_path"] = str(cif_path)
        templates.append(hit)

        if len(templates) >= max_templates:
            break

    return templates


def _load_pdb_dates(csv_path: Path) -> dict[str, str]:
    """Load PDB release dates from CSV → {pdb_id: "YYYY-MM-DD"}."""
    date_map = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb_id = row.get("pdb_id", row.get("PDB_ID", "")).lower()
            date = row.get("release_date", row.get("Release_Date", ""))
            if pdb_id and date:
                date_map[pdb_id] = date
    return date_map


def _search_mmseqs2(
    sequence: str,
    db_fasta: Path,
    evalue: float = 1e-3,
) -> list[dict]:
    """Run MMseqs2 sequence search against PDB RNA sequences."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        query_fa = tmpdir / "query.fasta"
        query_fa.write_text(f">query\n{sequence}\n")

        target_db = tmpdir / "targetDB"
        result_file = tmpdir / "results.m8"

        try:
            # Create target DB
            subprocess.run(
                ["mmseqs", "createdb", str(db_fasta), str(target_db)],
                capture_output=True,
                check=True,
            )
            query_db = tmpdir / "queryDB"
            subprocess.run(
                ["mmseqs", "createdb", str(query_fa), str(query_db)],
                capture_output=True,
                check=True,
            )
            # Search
            result_db = tmpdir / "resultDB"
            subprocess.run(
                [
                    "mmseqs", "search",
                    str(query_db), str(target_db), str(result_db), str(tmpdir),
                    "--search-type", "3",  # nucleotide
                    "-e", str(evalue),
                ],
                capture_output=True,
                check=True,
            )
            # Convert to tabular
            subprocess.run(
                [
                    "mmseqs", "convertalis",
                    str(query_db), str(target_db), str(result_db), str(result_file),
                    "--format-output", "target,fident,alnlen,qstart,qend,tstart,tend,evalue",
                ],
                capture_output=True,
                check=True,
            )

            return _parse_m8(result_file)

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"MMseqs2 search failed: {e}. Returning empty template list.")
            return []


def _search_blastn(sequence: str, db_fasta: Path) -> list[dict]:
    """Fallback: BLAST nucleotide search."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        query_fa = tmpdir / "query.fasta"
        query_fa.write_text(f">query\n{sequence}\n")
        result_file = tmpdir / "results.m8"

        try:
            # Make BLAST DB
            subprocess.run(
                ["makeblastdb", "-in", str(db_fasta), "-dbtype", "nucl",
                 "-out", str(tmpdir / "blastdb")],
                capture_output=True, check=True,
            )
            # Run blastn
            subprocess.run(
                [
                    "blastn", "-query", str(query_fa),
                    "-db", str(tmpdir / "blastdb"),
                    "-outfmt", "6 sseqid pident length qstart qend sstart send evalue",
                    "-out", str(result_file),
                    "-evalue", "1e-3",
                    "-max_target_seqs", "20",
                ],
                capture_output=True, check=True,
            )
            return _parse_m8(result_file)

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"BLAST search failed: {e}. Returning empty template list.")
            return []


def _parse_m8(result_file: Path) -> list[dict]:
    """Parse tabular alignment output (BLAST -outfmt 6 / MMseqs2 convertalis)."""
    hits = []
    if not result_file.exists():
        return hits

    with open(result_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 7:
                continue
            target_name = parts[0]
            identity = float(parts[1]) if "." in parts[1] else float(parts[1]) / 100.0

            # Parse PDB ID and chain from target name (e.g., "1ABC_A" or "1abc|A")
            pdb_id, chain_id = _parse_pdb_chain(target_name)

            # Build alignment mapping (query pos → template pos)
            qstart, qend = int(parts[3]) - 1, int(parts[4]) - 1
            tstart, tend = int(parts[5]) - 1, int(parts[6]) - 1
            mapping = {}
            q_len = qend - qstart + 1
            t_len = tend - tstart + 1
            step = min(q_len, t_len)
            for k in range(step):
                mapping[qstart + k] = tstart + k

            hits.append({
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "identity": identity,
                "alignment_mapping": mapping,
                "release_date": "",
            })

    # Sort by identity descending
    hits.sort(key=lambda x: x["identity"], reverse=True)
    return hits


def _parse_pdb_chain(name: str) -> tuple[str, str]:
    """Extract PDB ID and chain ID from a sequence name."""
    # Try common formats: "1ABC_A", "1abc|A", "1ABC:A"
    for sep in ["_", "|", ":"]:
        if sep in name:
            parts = name.split(sep, 1)
            return parts[0].upper(), parts[1][:1]
    # If no separator, assume 4-char PDB + 1-char chain
    if len(name) >= 5:
        return name[:4].upper(), name[4]
    return name.upper(), "A"
