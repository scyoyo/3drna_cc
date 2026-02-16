"""Tests for data loading modules."""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.data.loader import (
    parse_stoichiometry,
    load_msa,
)
from src.data.msa_processor import (
    filter_msa_by_gap_fraction,
    subsample_msa,
    msa_to_onehot,
    compute_msa_covariance,
)
from src.data.featurizer import build_protenix_input


# ── parse_stoichiometry ───────────────────────────────────────────────────


class TestParseStoichiometry:
    def test_single_chain(self):
        result = parse_stoichiometry("", "GGGAAACCC")
        assert len(result) == 1
        assert result[0]["chain_id"] == "A"
        assert result[0]["sequence"] == "GGGAAACCC"
        assert result[0]["copies"] == 1

    def test_multi_chain(self):
        result = parse_stoichiometry("A2B1", "GGGAAACCC;UUUAAAGGG")
        assert len(result) == 2
        assert result[0]["chain_id"] == "A"
        assert result[0]["sequence"] == "GGGAAACCC"
        assert result[0]["copies"] == 2
        assert result[1]["chain_id"] == "B"
        assert result[1]["sequence"] == "UUUAAAGGG"
        assert result[1]["copies"] == 1

    def test_homodimer(self):
        result = parse_stoichiometry("A2", "ACGU")
        assert len(result) == 1
        assert result[0]["copies"] == 2


# ── MSA loading ──────────────────────────────────────────────────────────


class TestLoadMSA:
    def test_missing_file(self):
        result = load_msa("nonexistent_target", msa_dir="/tmp/nonexistent")
        assert result["depth"] == 0
        assert result["sequences"] == []

    def test_real_fasta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = Path(tmpdir) / "test_target.MSA.fasta"
            fasta_path.write_text(
                ">query\nGGGAAACCC\n>hit1\nGG-AAACCC\n>hit2\nGGGAA-CCC\n"
            )
            result = load_msa("test_target", msa_dir=tmpdir)
            assert result["depth"] == 3
            assert result["sequences"][0] == "GGGAAACCC"
            assert result["sequences"][1] == "GG-AAACCC"


# ── MSA processor ────────────────────────────────────────────────────────


class TestMSAProcessor:
    def test_filter_gaps(self):
        seqs = ["ACGU", "A--U", "----"]
        descs = ["q", "h1", "h2"]
        filtered_seqs, filtered_descs = filter_msa_by_gap_fraction(
            seqs, descs, max_gap_fraction=0.5
        )
        assert len(filtered_seqs) == 2  # query + h1 (50% gaps OK)
        assert "----" not in filtered_seqs

    def test_subsample(self):
        seqs = [f"seq_{i}" for i in range(100)]
        descs = [f"desc_{i}" for i in range(100)]
        sub_seqs, sub_descs = subsample_msa(seqs, descs, max_depth=10, seed=42)
        assert len(sub_seqs) == 10
        assert sub_seqs[0] == "seq_0"  # query preserved

    def test_onehot(self):
        seqs = ["ACGU", "A-GU"]
        onehot = msa_to_onehot(seqs)
        assert onehot.shape == (2, 4, 6)
        assert onehot[0, 0, 0] == 1  # A at position 0
        assert onehot[0, 1, 1] == 1  # C at position 1
        assert onehot[1, 1, 4] == 1  # gap at position 1

    def test_covariance(self):
        seqs = ["ACGU"] * 10
        cov = compute_msa_covariance(seqs)
        assert cov.shape == (4, 4)
        # Diagonal should be zero
        assert np.allclose(np.diag(cov), 0.0)


# ── Featurizer ───────────────────────────────────────────────────────────


class TestFeaturizer:
    def test_build_input_basic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy MSA file
            msa_dir = Path(tmpdir) / "MSA"
            msa_dir.mkdir()
            (msa_dir / "test_id.MSA.fasta").write_text(
                ">query\nGGGAAACCC\n>hit1\nGGGAA-CCC\n"
            )

            inp = build_protenix_input(
                target_id="test_id",
                sequence="GGGAAACCC",
                msa_dir=msa_dir,
            )

            assert inp["name"] == "test_id"
            assert len(inp["sequences"]) == 1
            assert inp["sequences"][0]["rnaSequence"]["sequence"] == "GGGAAACCC"
            # MSA path should be set on the first RNA entity
            assert "unpairedMsaPath" in inp["sequences"][0]["rnaSequence"]

    def test_build_input_multi_chain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            msa_dir = Path(tmpdir) / "MSA"
            msa_dir.mkdir()

            inp = build_protenix_input(
                target_id="multi",
                sequence="ACGU",
                stoichiometry="A2B1",
                all_sequences="ACGU;UGCA",
                msa_dir=msa_dir,
            )

            assert len(inp["sequences"]) == 2
            assert inp["sequences"][0]["rnaSequence"]["count"] == 2
            assert inp["sequences"][1]["rnaSequence"]["count"] == 1
