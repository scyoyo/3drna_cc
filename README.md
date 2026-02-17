# Stanford RNA 3D Folding — Protenix Competition Framework

[![GitHub](https://img.shields.io/badge/github-3drna_cc-blue?logo=github)](https://github.com/scyoyo/3drna_cc)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)

A comprehensive framework for predicting RNA 3D structures using **Protenix** (open-source AlphaFold3 reimplementation) fine-tuned on competition data. Implements biology-informed strategies including MSA processing, template search, LoRA adaptation, and diversity ensemble selection.

## 🎯 Overview

**Challenge**: Predict C1' atom coordinates for RNA molecules (28 test targets, 9,761 residues total)
**Metric**: Best-of-5 TM-score (task allows 5 structure predictions per target)
**Data**: ~310GB competition dataset with sequences, labels, MSA, PDB structures
**Approach**: Protenix v1.0.0 + LoRA fine-tuning + template search + diversity ensemble
**Environment**: Google Colab (training) → Kaggle Notebook (submission)

## 📋 Quick Start

### Prerequisites

- **Google Colab Pro** (for A100 GPU during fine-tuning)
- **Kaggle Account** with API token (for competition data download)
- Python 3.11+

### 1️⃣ Setup in Colab

Create a new notebook and run:

```python
# Install kaggle and upload credentials
!pip install kaggle -q
from google.colab import files
import os

if not os.path.exists('/root/.kaggle/kaggle.json'):
    print("Upload your kaggle.json file:")
    uploaded = files.upload()
    !mkdir -p /root/.kaggle && mv kaggle.json /root/.kaggle/ && chmod 600 /root/.kaggle/kaggle.json

# Clone and setup
!git clone https://github.com/scyoyo/3drna_cc.git /content/3drna_cc
%cd /content/3drna_cc
!pip install -q -r requirements.txt
```

### 2️⃣ Setup Option A: With Google Drive (Recommended)

**Advantages**:
- ✅ Data persists across Colab sessions
- ✅ Share data between notebooks
- ✅ kaggle.json stored once, reused forever
- ✅ 50GB data stored in your Drive (not deleted after session)

Run `00_colab_setup_with_gdrive.ipynb` first:
```python
# This notebook will:
1. Mount Google Drive
2. Create project directory structure
3. Upload kaggle.json to Drive (one-time)
4. Download competition data to /MyDrive/3drna_cc/data
5. Verify all configurations
```

Then run other notebooks (they auto-detect Google Drive paths via `src/config.py`)

### 2️⃣ Setup Option B: Colab Temporary Storage

Run the standard setup cell from `01_setup_and_explore.ipynb`
```python
# Data stored in /content/ (deleted after session ends)
# Suitable for quick experiments
```

### 3️⃣ Run Notebooks in Order

| # | Notebook | Duration | Purpose |
|---|----------|----------|---------|
| 0 | `00_colab_setup_with_gdrive.ipynb` | 30 min | Setup Google Drive storage (optional but recommended) |
| 1 | `01_setup_and_explore.ipynb` | 10 min | Download competition data, explore format |
| 2 | `02_protenix_baseline.ipynb` | 2h | Run Protenix inference, establish TM baseline |
| 3 | `03_finetune_protenix.ipynb` | 20-40h | LoRA fine-tune on training set |
| 4 | `04_full_pipeline.ipynb` | 3h | End-to-end test on validation set |
| 5 | `kaggle_submission.ipynb` | varies | Submit to Kaggle (offline notebook) |

## 📁 Project Structure

```
3drna_cc/
├── notebooks/                          # Colab entry points
│   ├── 01_setup_and_explore.ipynb      # Data download & exploration
│   ├── 02_protenix_baseline.ipynb      # Baseline inference
│   ├── 03_finetune_protenix.ipynb      # LoRA fine-tuning
│   ├── 04_full_pipeline.ipynb          # Full pipeline test
│   └── kaggle_submission.ipynb         # Final submission
│
├── src/                                # Core modules
│   ├── config.py                       # Global config (auto-detects Colab/Kaggle)
│   ├── setup.py                        # One-click environment setup
│   │
│   ├── data/
│   │   ├── loader.py                   # Load sequences, labels, MSA
│   │   ├── featurizer.py              # Build Protenix JSON inputs
│   │   └── msa_processor.py           # MSA filtering, subsampling, encoding
│   │
│   ├── model/
│   │   ├── protenix_runner.py         # Protenix CLI wrapper
│   │   ├── lora_finetune.py           # LoRA fine-tuning (PEFT + CLI)
│   │   └── secondary_structure.py     # RNA secondary structure prediction
│   │
│   ├── template/
│   │   ├── template_search.py         # PDB template search (MMseqs2/BLAST)
│   │   └── template_featurizer.py     # Extract template features
│   │
│   ├── postprocess/
│   │   ├── clash_fix.py               # Fix steric clashes & backbone breaks
│   │   ├── c1_extraction.py           # Extract C1' from CIF files
│   │   └── geometry_check.py          # Validate geometry
│   │
│   ├── ensemble/
│   │   ├── tm_score.py                # TM-score, RMSD, Kabsch alignment
│   │   └── diversity_selector.py      # Select best 5 via maxmin RMSD
│   │
│   └── submission/
│       └── formatter.py               # Format & validate submission CSV
│
├── tests/                              # Unit tests
│   ├── test_data_loader.py
│   ├── test_protenix_inference.py
│   └── test_submission_format.py
│
├── requirements.txt                    # Python dependencies
└── README.md
```

## 🔬 Core Concepts

### RNA Folding Hierarchy

This framework is grounded in molecular biology:

1. **Primary Structure** → **Secondary Structure** (base pairing, fast ~ms)
2. **Secondary Structure** → **Tertiary Structure** (3D folding, slow ~s-min)
3. **Assembly** (multi-chain, ligands)

**Protenix mirrors this hierarchy**:
- **MSA Module** extracts evolutionary signals → base-pair prediction
- **Pairformer** builds residue-pair interactions
- **Diffusion Module** samples 3D coordinates in energy landscape

### LoRA Fine-tuning Strategy

Insert low-rank matrices into **Pairformer** and **Diffusion attention** layers:
- Trainable params: ~2-5M (vs 368M total) → 95% memory savings
- Freeze MSA Module & Input Embedder → preserve pre-trained knowledge
- Loss: FAPE (Frame-Aligned Point Error) on C1' + backbone continuity constraint
- Expected improvement: TM 0.35→0.50 (baseline) → 0.50-0.60 (fine-tuned)

### Diversity Ensemble

Generate N predictions (10-20), select best 5 via **MaxMin RMSD**:
- Greedily maximize minimum pairwise RMSD among selected structures
- Ensures coverage of conformational space
- Aligns with competition metric (best-of-5 TM-score)

## 📊 Expected Results

| Stage | Mean TM | Notes |
|-------|---------|-------|
| Protenix baseline (v1.0.0, no fine-tune) | 0.35-0.50 | AF3 level on RNA |
| + LoRA fine-tuning | 0.45-0.60 | Adapts to competition data distribution |
| + Template features | +0.05-0.10 | Homologous targets benefit most |
| + Diversity ensemble (best-of-5) | +0.05-0.10 | Multiple predictions improve score |
| **Final (all combined)** | **0.55-0.70** | Target for top-tier ranking |

## 🛠️ Protenix API Reference

### Installation

```bash
pip install protenix  # requires Python >= 3.11
apt-get install -y kalign hmmer  # system deps for MSA
```

### Inference CLI

```bash
protenix pred \
  -i input.json \
  -o ./output \
  -n protenix_base_default_v1.0.0 \
  -s "101,102,103" \
  --use_rna_msa true \
  --use_template false
```

### JSON Input Format (RNA with MSA)

```json
[{
  "name": "target_id",
  "sequences": [
    {
      "rnaSequence": {
        "sequence": "GUACGUAC...",
        "count": 1,
        "unpairedMsaPath": "/path/to/rna_msa.a3m"
      }
    },
    {"ligand": {"ligand": "CCD_MG", "count": 1}}
  ]
}]
```

### Output Files

- **Structures**: `<name>_<seed>_sample_<N>.cif` (mmCIF format)
- **Confidence**: `<name>_<seed>_summary_confidence_sample_<N>.json`
  - pLDDT (per-residue confidence 0-100)
  - pTM (predicted TM-score)
  - has_clash (steric clash detection)

### Available Models

| Model | Size | Features |
|-------|------|----------|
| `protenix_base_default_v1.0.0` ⭐ | 368M | MSA + RNA MSA + Template |
| `protenix_base_20250630_v1.0.0` | 368M | MSA + RNA MSA (newer) |
| v0.5.0 variants | 109-368M | MSA only (no RNA support) |

**Note**: Only v1.0.0 supports RNA MSA feature.

## 💻 Hardware Requirements

### Colab (Training)

| Task | GPU | VRAM | Time |
|------|-----|------|------|
| Data exploration | T4 | 16GB | 10 min |
| Baseline inference | A100 | 16GB | 2h |
| LoRA fine-tuning | A100 | 16GB | 25h |
| Full pipeline test | A100 | 16GB | 3h |

### Kaggle Notebook (Submission)

| Resource | Limit | Usage |
|----------|-------|-------|
| RAM | 29GB | ~20GB |
| Disk | 20GB | ~8GB |
| VRAM | 16GB | ~14GB peak |
| Time | 12h | 4-8h |

## 📖 Workflow

### Phase 1: Exploration (Colab, ~1h)

```python
# In notebook 01_setup_and_explore.ipynb
- Download 50GB of competition data
- Analyze sequence length distribution
- Understand label format (C1' coordinates)
- Explore MSA quality (depth, diversity)
```

### Phase 2: Baseline (Colab, ~2h)

```python
# In notebook 02_protenix_baseline.ipynb
- Build Protenix input JSONs for validation set
- Run inference with pretrained v1.0.0 weights
- Extract C1' from output CIF files
- Compute TM-scores vs ground truth labels
→ Expected baseline: mean TM ≈ 0.40-0.50
```

### Phase 3: Fine-tune (Colab Pro, 20-40h)

```python
# In notebook 03_finetune_protenix.ipynb
- Prepare training dataset
- Inject LoRA adapters (rank=16)
- Train 5 epochs with FAPE + continuity loss
- Save ~100MB LoRA weights
→ Expected improvement: +0.10-0.15 TM
```

### Phase 4: Integration (Colab, ~3h)

```python
# In notebook 04_full_pipeline.ipynb
- End-to-end pipeline on validation
- Generate N diverse predictions per target
- Select best 5 via maxmin RMSD
- Format submission.csv
- Validate format against sample
```

### Phase 5: Submit (Kaggle, varies)

```python
# Upload to kaggle_submission.ipynb (Kaggle Notebooks)
- Auto-download data at submission time
- Run protenix pred on test set (no internet)
- Generate predictions within 12h
- Submit to leaderboard
```

## 🔧 Configuration

All settings in `src/config.py`:

```python
# Auto-detects environment
IN_COLAB = ...              # Colab paths
GDRIVE_AVAILABLE = ...      # Google Drive mounted

# Auto-detects storage location
if GDRIVE_AVAILABLE:
    DATA_DIR = /content/drive/MyDrive/3drna_cc/data
    OUTPUT_DIR = /content/drive/MyDrive/3drna_cc/output
else:
    DATA_DIR = /content/data (Colab temp, deleted after session)
    OUTPUT_DIR = /content/output

# RNA geometry constants (Angstroms)
C1_C1_ADJACENT_MIN = 5.0      # min consecutive C1' distance
C1_C1_ADJACENT_MAX = 7.0      # max consecutive C1' distance
CLASH_MIN_DISTANCE = 3.0      # min non-bonded atom distance

# Protenix inference
DEFAULT_N_SEEDS = 5           # diffusion seeds per target
LORA_RANK = 16                # low-rank adapter rank
MAX_SEQ_CROP = 384            # max residues per crop
```

### Google Drive Setup (Recommended)

Run `00_colab_setup_with_gdrive.ipynb` to:
1. Mount Google Drive
2. Create `/MyDrive/3drna_cc/` directory structure
3. Upload `kaggle.json` once (persists for all sessions)
4. Download 50GB competition data to Drive

**Directory structure**:
```
/MyDrive/3drna_cc/
├── credentials/
│   └── kaggle.json          # One-time upload
├── data/
│   └── stanford-rna-3d-folding-2/
│       ├── train_sequences.csv
│       ├── train_labels.csv
│       ├── MSA/
│       └── ...
├── models/
│   └── protenix/            # Pre-trained weights cached here
└── output/                  # Inference outputs stored here
```

**Config auto-detection**:
- If `/content/drive` exists (Google Drive mounted) → use Drive paths
- Else → use Colab temp storage (`/content/`)
- No code changes needed!

**Speed notes**:
- Drive I/O slower than Colab temp (~100 MB/s vs 1 GB/s)
- Suitable for: storing data (write once), quick reads
- Not suitable for: frequent streaming (training loops)
- Workaround: Copy from Drive to `/tmp/` during training for speed

## 📦 Dependencies

```
protenix>=1.0.0         # AF3 reimplementation
torch>=2.1.0            # Deep learning
biopython>=1.81         # Structure I/O
peft>=0.7.0             # LoRA fine-tuning
einops>=0.7.0           # Tensor operations
kaggle>=1.6.0           # Data download
```

See `requirements.txt` for full list.

## 🧪 Testing

Run unit tests locally:

```bash
pytest tests/
# or individual modules
pytest tests/test_data_loader.py -v
pytest tests/test_protenix_inference.py -v
pytest tests/test_submission_format.py -v
```

## 📊 Key Metrics

- **TM-score**: Template modeling score (0-1, higher better)
- **RMSD**: Root-mean-square deviation after Kabsch alignment
- **pLDDT**: Per-residue confidence from Protenix confidence JSON
- **Backbone violations**: Count of consecutive C1' distances outside [5.0, 7.0] Å
- **Steric clashes**: Count of atom pairs closer than 3.0 Å

## 🚀 Performance Tips

### Memory Optimization
- Use `dtype=bf16` for inference (16-bit precision)
- Crop sequences to 384 residues during fine-tuning
- Enable `enable_cache=true` if memory-constrained

### Speed Optimization
- Run template search in parallel over targets
- Use diverse seed selection (not sequential 1-5)
- Pre-compute MSA a3m files locally before fine-tuning

### Quality Improvement
- Enable `--use_rna_msa true` (critical for RNA)
- Use high-quality templates from PDB_RNA
- Generate 10-20 predictions, select best 5

## 📝 Kaggle Submission Checklist

- [ ] All dependencies installed (`pip install -q -r requirements.txt`)
- [ ] Protenix model weights auto-downloaded on first run
- [ ] Test data loaded: 28 targets, 9,761 rows in submission
- [ ] All 5 coordinate columns (x_1..z_5) populated
- [ ] No NaN values in coordinates
- [ ] submission.csv format matches sample_submission.csv
- [ ] Total runtime < 12 hours
- [ ] Total disk usage < 20GB

## 🐛 Troubleshooting

### Protenix not found
```python
!pip install protenix  # Install in Colab
!apt-get install kalign hmmer  # System deps
```

### Kaggle API authentication fails
```python
# Re-upload kaggle.json
from google.colab import files
files.upload()
!mv kaggle.json /root/.kaggle/
```

### CUDA out of memory
```python
# Reduce batch size or sequence crop
MAX_SEQ_CROP = 256  # from 384
# or use --dtype fp32 instead of bf16
```

### MSA not found
```python
# Ensure MSA files are in correct directory
from src.config import MSA_DIR
import os
print(f"MSA directory: {MSA_DIR}")
print(f"Files: {os.listdir(MSA_DIR)}")
```

## 📚 References

- **Protenix**: [GitHub](https://github.com/bytedance/Protenix) | [Docs](https://github.com/bytedance/Protenix/tree/main/docs)
- **AlphaFold3**: [Nature Biotechnology Paper](https://www.nature.com/articles/s41587-024-02355-4)
- **Competition**: [Kaggle RNA 3D Folding](https://www.kaggle.com/competitions/stanford-rna-3d-folding-2)
- **TM-score**: [Zhang & Skolnick (2004)](https://onlinelibrary.wiley.com/doi/10.1002/prot.20264)

## 📄 License

Apache License 2.0 (matching Protenix)

## 👤 Authors

- Framework: Claude (Anthropic)
- Based on: [Protenix](https://github.com/bytedance/Protenix) by ByteDance

---

**Last Updated**: February 2026
**Status**: Ready for Colab development & Kaggle submission
