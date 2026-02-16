"""One-click environment setup for Colab / Kaggle."""

import os
import subprocess
import sys
from pathlib import Path

from src.config import (
    IN_COLAB,
    IN_KAGGLE,
    DATA_DIR,
    MODEL_DIR,
    DEPS_DIR,
)


def run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, printing output live."""
    print(f">>> {cmd}")
    return subprocess.run(cmd, shell=True, check=check)


# ── Dependencies ──────────────────────────────────────────────────────────


def install_dependencies():
    """Install Python packages.

    On Kaggle (offline) installs from pre-downloaded wheels.
    On Colab installs via pip from PyPI.
    """
    if IN_KAGGLE:
        wheel_dir = DEPS_DIR
        if wheel_dir.exists():
            run(
                f"{sys.executable} -m pip install --no-index "
                f"--find-links {wheel_dir} protenix biopython einops peft"
            )
        else:
            print(f"WARNING: offline wheel dir {wheel_dir} not found; "
                  "trying online install as fallback")
            run(f"{sys.executable} -m pip install -q protenix biopython einops peft")
    else:
        run(f"{sys.executable} -m pip install -q protenix biopython einops peft")


# ── Competition data ──────────────────────────────────────────────────────


def download_competition_data():
    """Download competition data via Kaggle API (Colab only).

    On Kaggle the data is mounted automatically; this is a no-op.
    Downloads selectively: CSVs + MSA first (small), PDB_RNA on-demand.
    """
    if IN_KAGGLE:
        print("On Kaggle — competition data already mounted.")
        return

    if DATA_DIR.exists() and (DATA_DIR / "train_sequences.csv").exists():
        print("Competition data already present — skipping download.")
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    comp = "stanford-rna-3d-folding-2"

    # Download CSVs + MSA (relatively small)
    run(
        f"kaggle competitions download -c {comp} "
        f"-p {DATA_DIR} "
        f"-f train_sequences.csv"
    )
    run(
        f"kaggle competitions download -c {comp} "
        f"-p {DATA_DIR} "
        f"-f train_labels.csv"
    )
    run(
        f"kaggle competitions download -c {comp} "
        f"-p {DATA_DIR} "
        f"-f validation_sequences.csv"
    )
    run(
        f"kaggle competitions download -c {comp} "
        f"-p {DATA_DIR} "
        f"-f validation_labels.csv"
    )
    run(
        f"kaggle competitions download -c {comp} "
        f"-p {DATA_DIR} "
        f"-f test_sequences.csv"
    )
    run(
        f"kaggle competitions download -c {comp} "
        f"-p {DATA_DIR} "
        f"-f sample_submission.csv"
    )

    # Download MSA directory
    run(
        f"kaggle competitions download -c {comp} "
        f"-p {DATA_DIR} "
        f"-f MSA"
    )

    # Unzip if needed
    for zf in DATA_DIR.glob("*.zip"):
        run(f"unzip -o -q {zf} -d {DATA_DIR}")
        zf.unlink()

    print(f"Competition data ready at {DATA_DIR}")


# ── Protenix weights ─────────────────────────────────────────────────────


def download_protenix_weights():
    """Download Protenix pre-trained weights.

    Protenix auto-downloads weights on first ``protenix pred`` invocation.
    This function pre-downloads them to avoid delays during inference.
    """
    from src.config import PROTENIX_WEIGHTS, PROTENIX_WEIGHT_URL, PROTENIX_MODEL_NAME

    if PROTENIX_WEIGHTS.exists():
        print("Protenix weights already present — skipping download.")
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Method 1: Let Protenix's built-in downloader handle it
    result = subprocess.run(
        [sys.executable, "-c",
         "from protenix.utils.download import download_all_data; download_all_data()"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Protenix weights downloaded via built-in downloader.")
        return

    # Method 2: Direct wget from ByteDance cloud
    print(f"Downloading Protenix weights from {PROTENIX_WEIGHT_URL} ...")
    run(f"wget -q -O {PROTENIX_WEIGHTS} {PROTENIX_WEIGHT_URL}")

    print(f"Protenix weights ready at {MODEL_DIR}")


# ── Verification ─────────────────────────────────────────────────────────


def verify_setup():
    """Verify that all required files and packages are available."""
    issues = []

    # Check packages
    for pkg in ["protenix", "Bio", "einops", "peft"]:
        try:
            __import__(pkg)
        except ImportError:
            issues.append(f"Package '{pkg}' not importable")

    # Check data files
    for name in [
        "train_sequences.csv",
        "train_labels.csv",
        "validation_sequences.csv",
        "validation_labels.csv",
    ]:
        if not (DATA_DIR / name).exists():
            issues.append(f"Missing data file: {name}")

    # Check MSA
    if not (DATA_DIR / "MSA").exists():
        issues.append("MSA directory not found")

    if issues:
        print("Setup issues found:")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("Setup verified — all OK ✓")

    return len(issues) == 0


# ── Main entry point ─────────────────────────────────────────────────────


def setup_environment():
    """One-click setup: install deps → download data → download weights."""
    print("=" * 60)
    print("Setting up RNA 3D folding environment")
    print("=" * 60)

    install_dependencies()
    download_competition_data()
    download_protenix_weights()
    ok = verify_setup()

    if ok:
        print("\n🎉 Environment ready!")
    else:
        print("\n⚠️  Some issues detected — see above.")


if __name__ == "__main__":
    setup_environment()
