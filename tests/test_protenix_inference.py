"""Tests for model inference and postprocessing modules."""

import numpy as np
import pytest

from src.postprocess.clash_fix import fix_steric_clashes, fix_backbone_breaks, fix_all
from src.postprocess.geometry_check import validate_geometry
from src.ensemble.tm_score import (
    compute_tm_score,
    compute_rmsd,
    kabsch_align,
    pairwise_rmsd_matrix,
    best_of_n_tm_score,
)
from src.ensemble.diversity_selector import select_best_five


# ── Geometry checks ──────────────────────────────────────────────────────


class TestGeometryCheck:
    def test_valid_structure(self):
        """Create a simple valid backbone."""
        L = 20
        coords = np.zeros((L, 3))
        for i in range(L):
            coords[i] = [i * 6.0, 0, 0]  # 6 Å spacing (within 5-7 range)

        report = validate_geometry(coords)
        assert report["n_residues"] == L
        assert report["backbone_violations"] == 0
        assert report["clash_count"] == 0
        assert report["is_valid"]

    def test_backbone_break(self):
        """A large gap should be detected."""
        coords = np.array([
            [0, 0, 0],
            [6, 0, 0],   # OK: 6 Å
            [30, 0, 0],  # BREAK: 24 Å
            [36, 0, 0],  # OK: 6 Å
        ], dtype=np.float64)

        report = validate_geometry(coords)
        assert report["backbone_violations"] >= 1

    def test_steric_clash(self):
        """Two non-adjacent atoms too close."""
        coords = np.array([
            [0, 0, 0],
            [6, 0, 0],
            [12, 0, 0],
            [0.5, 0, 0],  # clash with residue 0
        ], dtype=np.float64)

        report = validate_geometry(coords)
        assert report["clash_count"] >= 1


# ── Clash fixing ─────────────────────────────────────────────────────────


class TestClashFix:
    def test_fix_clashes(self):
        """Atoms should be pushed apart."""
        coords = np.array([
            [0, 0, 0],
            [6, 0, 0],
            [12, 0, 0],
            [1.0, 0, 0],  # too close to residue 0
        ], dtype=np.float64)

        fixed = fix_steric_clashes(coords, min_distance=3.0)
        d03 = np.linalg.norm(fixed[3] - fixed[0])
        assert d03 >= 2.5  # should be pushed apart (may not reach full 3.0)

    def test_fix_backbone(self):
        """Backbone distances should be corrected."""
        coords = np.array([
            [0, 0, 0],
            [10, 0, 0],  # too far: 10 Å
            [16, 0, 0],
        ], dtype=np.float64)

        fixed = fix_backbone_breaks(coords)
        d01 = np.linalg.norm(fixed[1] - fixed[0])
        assert d01 <= 7.5  # should be pulled closer


# ── TM-score ─────────────────────────────────────────────────────────────


class TestTMScore:
    def test_identical(self):
        coords = np.random.randn(50, 3) * 10
        tm = compute_tm_score(coords, coords)
        assert tm > 0.99

    def test_rmsd_identical(self):
        coords = np.random.randn(30, 3) * 5
        rmsd = compute_rmsd(coords, coords)
        assert rmsd < 1e-6

    def test_kabsch_rotation(self):
        """After rotation, Kabsch should recover alignment."""
        np.random.seed(42)
        coords = np.random.randn(30, 3) * 10

        # Apply a known rotation
        angle = np.pi / 4
        R = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ])
        rotated = coords @ R.T + np.array([5, 10, -3])

        aligned = kabsch_align(rotated, coords)
        rmsd = np.sqrt(np.mean(np.sum((aligned - coords) ** 2, axis=1)))
        assert rmsd < 1e-6

    def test_pairwise_matrix(self):
        coords_list = [np.random.randn(20, 3) for _ in range(4)]
        mat = pairwise_rmsd_matrix(coords_list)
        assert mat.shape == (4, 4)
        assert np.allclose(np.diag(mat), 0.0)
        assert np.allclose(mat, mat.T)  # symmetric

    def test_best_of_n(self):
        ref = np.random.randn(30, 3) * 10
        preds = [
            ref + np.random.randn(30, 3) * 5,  # noisy
            ref + np.random.randn(30, 3) * 0.1,  # very close
            ref + np.random.randn(30, 3) * 10,  # far
        ]
        best_tm, best_idx = best_of_n_tm_score(preds, ref)
        assert best_idx == 1  # the close one should win


# ── Diversity selection ──────────────────────────────────────────────────


class TestDiversitySelector:
    def test_select_five(self):
        preds = [
            {"coords": np.random.randn(20, 3), "source": f"s{i}"}
            for i in range(10)
        ]
        selected = select_best_five(preds, strategy="maxmin_diversity", n_select=5)
        assert len(selected) == 5
        assert all(isinstance(s, np.ndarray) for s in selected)

    def test_fewer_than_five(self):
        preds = [
            {"coords": np.random.randn(20, 3), "source": "s0"},
            {"coords": np.random.randn(20, 3), "source": "s1"},
        ]
        selected = select_best_five(preds, n_select=5)
        assert len(selected) == 2  # returns all if < n_select
