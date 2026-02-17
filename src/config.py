"""Global configuration — auto-detects Colab / Kaggle / local environment."""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------
IN_COLAB = "COLAB_GPU" in os.environ or os.path.exists("/content")
IN_KAGGLE = os.path.exists("/kaggle")
GDRIVE_AVAILABLE = os.path.exists("/content/drive")  # Google Drive mounted

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if IN_KAGGLE:
    DATA_DIR = Path("/kaggle/input/stanford-rna-3d-folding-2")
    OUTPUT_DIR = Path("/kaggle/working")
    MODEL_DIR = Path("/kaggle/input/protenix-rna-weights")
    DEPS_DIR = Path("/kaggle/input/protenix-rna-deps")
elif IN_COLAB:
    # Prefer Google Drive for persistent storage if available
    if GDRIVE_AVAILABLE:
        GDRIVE_BASE = Path("/content/drive/MyDrive/3drna_cc")
        DATA_DIR = GDRIVE_BASE / "data" / "stanford-rna-3d-folding-2"
        OUTPUT_DIR = GDRIVE_BASE / "output"
        MODEL_DIR = GDRIVE_BASE / "models" / "protenix"
        DEPS_DIR = GDRIVE_BASE / "deps"
    else:
        # Fallback to Colab temp storage
        DATA_DIR = Path("/content/data/stanford-rna-3d-folding-2")
        OUTPUT_DIR = Path("/content/output")
        MODEL_DIR = Path("/content/models/protenix")
        DEPS_DIR = Path("/content/deps")
else:
    _root = Path(__file__).resolve().parent.parent
    DATA_DIR = _root / "data"
    OUTPUT_DIR = _root / "output"
    MODEL_DIR = _root / "models"
    DEPS_DIR = _root / "deps"

# Competition data sub-paths
TRAIN_SEQUENCES = DATA_DIR / "train_sequences.csv"
TRAIN_LABELS = DATA_DIR / "train_labels.csv"
VAL_SEQUENCES = DATA_DIR / "validation_sequences.csv"
VAL_LABELS = DATA_DIR / "validation_labels.csv"
TEST_SEQUENCES = DATA_DIR / "test_sequences.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"
MSA_DIR = DATA_DIR / "MSA"
PDB_RNA_DIR = DATA_DIR / "PDB_RNA"
PDB_SEQRES = PDB_RNA_DIR / "pdb_seqres_NA.fasta"
PDB_DATES = PDB_RNA_DIR / "pdb_release_dates_NA.csv"

# Model
PROTENIX_MODEL_NAME = "protenix_base_default_v1.0.0"
PROTENIX_WEIGHTS = MODEL_DIR / f"{PROTENIX_MODEL_NAME}.pt"
PROTENIX_WEIGHT_URL = (
    f"https://protenix.tos-cn-beijing.volces.com/checkpoint/{PROTENIX_MODEL_NAME}.pt"
)
LORA_WEIGHTS = MODEL_DIR / "lora_weights"

# ---------------------------------------------------------------------------
# RNA geometry constants (Angstroms)
# ---------------------------------------------------------------------------
C1_C1_ADJACENT_MIN = 5.0      # min C1'–C1' distance for consecutive residues
C1_C1_ADJACENT_MAX = 7.0      # max C1'–C1' distance for consecutive residues
C1_C1_WC_PAIR = 10.4          # typical C1'–C1' for Watson-Crick base pair
CLASH_MIN_DISTANCE = 3.0      # minimum non-bonded atom distance

# ---------------------------------------------------------------------------
# Protenix inference defaults
# ---------------------------------------------------------------------------
DEFAULT_N_SEEDS = 5
DEFAULT_N_DIFFUSION_SAMPLES = 5
MAX_SEQ_CROP = 384             # max residues per crop during training
LORA_RANK = 16
LORA_TARGET_MODULES = [
    "pairformer_stack.blocks.*.pair_attention",
    "diffusion_module.blocks.*.self_attn",
]

# ---------------------------------------------------------------------------
# Kaggle notebook constraints
# ---------------------------------------------------------------------------
KAGGLE_MAX_RAM_GB = 29
KAGGLE_MAX_DISK_GB = 20
KAGGLE_MAX_VRAM_GB = 16
KAGGLE_MAX_HOURS = 12
