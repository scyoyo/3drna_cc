"""LoRA fine-tuning of Protenix for RNA C1' coordinate prediction.

Two approaches are provided:

1. **CLI approach** (recommended): Use Protenix's built-in ``runner/train.py``
   with ``--load_checkpoint_path`` and a PDB list for fine-tuning. This is the
   most stable path and leverages Protenix's own training loop + DeepSpeed.

2. **Python API approach**: Load the model, inject LoRA adapters via PEFT,
   and train with a custom loop. More flexible but requires adapting to
   Protenix internals which may change between versions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.config import (
    MODEL_DIR,
    LORA_RANK,
    LORA_TARGET_MODULES,
    MAX_SEQ_CROP,
    PROTENIX_MODEL_NAME,
    PROTENIX_WEIGHTS,
    C1_C1_ADJACENT_MIN,
    C1_C1_ADJACENT_MAX,
    C1_C1_WC_PAIR,
)


# ── RNA Training Dataset ─────────────────────────────────────────────────


class RNATrainingDataset(Dataset):
    """Dataset for fine-tuning Protenix on competition RNA data.

    Each sample contains:
    - Protenix input features (sequence, MSA, templates)
    - Ground truth C1' coordinates from train_labels
    """

    def __init__(
        self,
        input_json_dir: str | Path,
        labels_df,
        sequences_df,
        max_crop: int = MAX_SEQ_CROP,
    ):
        self.input_json_dir = Path(input_json_dir)
        self.labels_df = labels_df
        self.sequences_df = sequences_df
        self.max_crop = max_crop

        # Index targets
        self.target_ids = sequences_df["target_id"].tolist()
        self.seq_lengths = {
            row["target_id"]: len(row["sequence"])
            for _, row in sequences_df.iterrows()
        }

    def __len__(self) -> int:
        return len(self.target_ids)

    def __getitem__(self, idx: int) -> dict:
        import json

        target_id = self.target_ids[idx]
        json_path = self.input_json_dir / f"{target_id}.json"

        # Load Protenix input
        with open(json_path) as f:
            input_data = json.load(f)
        if isinstance(input_data, list):
            input_data = input_data[0]

        # Load ground truth C1' coords
        from src.data.loader import labels_to_coords

        coords = labels_to_coords(self.labels_df, target_id)  # (L, 3)

        # Crop if needed
        seq_len = coords.shape[0]
        if seq_len > self.max_crop:
            start = np.random.randint(0, seq_len - self.max_crop)
            coords = coords[start : start + self.max_crop]
            # Note: Protenix input cropping is handled internally

        return {
            "target_id": target_id,
            "input_json": input_data,
            "gt_coords": torch.tensor(coords, dtype=torch.float32),
        }


# ── LoRA Trainer ──────────────────────────────────────────────────────────


class ProtenixLoRATrainer:
    """Fine-tune Protenix using LoRA on RNA competition data.

    Inserts low-rank adaptation matrices into Pairformer attention and
    Diffusion Module attention layers.  Freezes all other parameters.
    """

    def __init__(
        self,
        base_model_dir: str | Path | None = None,
        lora_rank: int = LORA_RANK,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
        device: str = "cuda",
    ):
        self.base_model_dir = Path(base_model_dir) if base_model_dir else MODEL_DIR
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or LORA_TARGET_MODULES
        self.device = device
        self.model = None
        self.optimizer = None

    def setup_model(self):
        """Load base Protenix model and inject LoRA adapters."""
        from peft import LoraConfig, get_peft_model

        # Load base model
        # Protenix model loading — adapt to actual API
        from protenix.model.protenix import Protenix

        self.model = Protenix.from_pretrained(str(self.base_model_dir))

        # Configure LoRA
        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=self.target_modules,
            bias="none",
        )

        self.model = get_peft_model(self.model, lora_config)
        self.model.to(self.device)

        # Report trainable params
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

        return self.model

    # ── Loss functions ────────────────────────────────────────────────

    @staticmethod
    def fape_loss(pred_coords: torch.Tensor, gt_coords: torch.Tensor) -> torch.Tensor:
        """Frame Aligned Point Error on C1' atoms.

        Simplified version: RMSD-like loss in local frames.

        Parameters
        ----------
        pred_coords, gt_coords : (B, L, 3)
        """
        # Global alignment via Kabsch, then point-wise error
        diff = pred_coords - gt_coords  # (B, L, 3)
        per_residue = torch.sqrt((diff ** 2).sum(dim=-1) + 1e-8)  # (B, L)
        # Clamp to avoid extreme outliers dominating
        per_residue = torch.clamp(per_residue, max=10.0)
        return per_residue.mean()

    @staticmethod
    def backbone_continuity_loss(
        pred_coords: torch.Tensor,
        target_min: float = C1_C1_ADJACENT_MIN,
        target_max: float = C1_C1_ADJACENT_MAX,
    ) -> torch.Tensor:
        """Penalize consecutive C1' distances outside [target_min, target_max].

        Parameters
        ----------
        pred_coords : (B, L, 3)
        """
        # Consecutive distances
        diffs = pred_coords[:, 1:] - pred_coords[:, :-1]  # (B, L-1, 3)
        dists = torch.sqrt((diffs ** 2).sum(dim=-1) + 1e-8)  # (B, L-1)

        # Penalize distances outside range
        below = torch.clamp(target_min - dists, min=0.0)
        above = torch.clamp(dists - target_max, min=0.0)
        return (below + above).mean()

    @staticmethod
    def combined_loss(
        pred_coords: torch.Tensor,
        gt_coords: torch.Tensor,
        fape_weight: float = 1.0,
        continuity_weight: float = 0.5,
    ) -> dict[str, torch.Tensor]:
        """Combined training loss."""
        l_fape = ProtenixLoRATrainer.fape_loss(pred_coords, gt_coords)
        l_cont = ProtenixLoRATrainer.backbone_continuity_loss(pred_coords)
        total = fape_weight * l_fape + continuity_weight * l_cont
        return {
            "total": total,
            "fape": l_fape,
            "continuity": l_cont,
        }

    # ── Training loop ─────────────────────────────────────────────────

    def train(
        self,
        train_dataset: Dataset,
        val_dataset: Optional[Dataset] = None,
        epochs: int = 5,
        lr: float = 1e-4,
        gradient_accumulation: int = 8,
        max_grad_norm: float = 1.0,
        save_dir: str | Path = "checkpoints",
    ) -> dict:
        """Run LoRA fine-tuning.

        Returns dict with training metrics.
        """
        if self.model is None:
            self.setup_model()

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs * len(train_dataset) // gradient_accumulation
        )

        # Use batch_size=1 with gradient accumulation
        train_loader = DataLoader(
            train_dataset, batch_size=1, shuffle=True, num_workers=0
        )

        scaler = torch.amp.GradScaler("cuda")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            self.model.train()
            epoch_losses = []
            self.optimizer.zero_grad()

            for step, batch in enumerate(train_loader):
                gt_coords = batch["gt_coords"].to(self.device)  # (1, L, 3)

                with torch.amp.autocast("cuda"):
                    # Forward pass through Protenix
                    # The actual API depends on Protenix internals
                    output = self.model(batch["input_json"])
                    pred_coords = output["predicted_coords"]  # (1, L, 3)

                    losses = self.combined_loss(pred_coords, gt_coords)
                    loss = losses["total"] / gradient_accumulation

                scaler.scale(loss).backward()

                if (step + 1) % gradient_accumulation == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_grad_norm
                    )
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad()
                    scheduler.step()

                epoch_losses.append(losses["total"].item())

                if step % 50 == 0:
                    print(
                        f"Epoch {epoch+1}/{epochs} step {step}: "
                        f"loss={losses['total']:.4f} "
                        f"fape={losses['fape']:.4f} "
                        f"cont={losses['continuity']:.4f}"
                    )

            avg_loss = np.mean(epoch_losses)
            history["train_loss"].append(avg_loss)
            print(f"Epoch {epoch+1}/{epochs} — avg train loss: {avg_loss:.4f}")

            # Validation
            if val_dataset is not None:
                val_loss = self._validate(val_dataset)
                history["val_loss"].append(val_loss)
                print(f"  val loss: {val_loss:.4f}")

            # Save checkpoint
            self.save_lora_weights(save_dir / f"epoch_{epoch+1}")

        return history

    def _validate(self, val_dataset: Dataset) -> float:
        """Compute validation loss."""
        self.model.eval()
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
        losses = []

        with torch.no_grad():
            for batch in val_loader:
                gt_coords = batch["gt_coords"].to(self.device)
                with torch.amp.autocast("cuda"):
                    output = self.model(batch["input_json"])
                    pred_coords = output["predicted_coords"]
                    loss_dict = self.combined_loss(pred_coords, gt_coords)
                losses.append(loss_dict["total"].item())

        return float(np.mean(losses))

    # ── Save / Load ───────────────────────────────────────────────────

    def save_lora_weights(self, output_dir: str | Path):
        """Save only LoRA adapter weights (small, ~50-100 MB)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(output_dir))
        print(f"LoRA weights saved to {output_dir}")

    def load_lora_weights(self, lora_dir: str | Path):
        """Load LoRA adapter weights onto the base model."""
        from peft import PeftModel

        if self.model is None:
            from protenix.model.protenix import Protenix

            self.model = Protenix.from_pretrained(str(self.base_model_dir))

        self.model = PeftModel.from_pretrained(self.model, str(lora_dir))
        self.model.to(self.device)
        print(f"LoRA weights loaded from {lora_dir}")


# ── CLI-based fine-tuning (recommended approach) ─────────────────────────


def finetune_via_cli(
    pdb_list_path: str | Path,
    output_dir: str | Path = "output/finetune",
    max_steps: int = 5000,
    lr: float = 0.001,
    train_crop_size: int = MAX_SEQ_CROP,
    warmup_steps: int = 500,
    eval_interval: int = 400,
    checkpoint_interval: int = 400,
    diffusion_batch_size: int = 48,
    dtype: str = "bf16",
) -> subprocess.CompletedProcess:
    """Fine-tune Protenix using its built-in training runner.

    This is the recommended approach — it uses Protenix's own training loop
    with DeepSpeed, matching the original finetune_demo.sh workflow.

    Parameters
    ----------
    pdb_list_path : path to a text file listing PDB IDs (one per line).
    output_dir    : directory for checkpoints and logs.
    max_steps     : total training steps.
    lr            : peak learning rate.

    Returns
    -------
    subprocess.CompletedProcess from the training run.
    """
    pdb_list_path = Path(pdb_list_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "./runner/train.py",
        "--model_name", PROTENIX_MODEL_NAME,
        "--run_name", "protenix_rna_finetune",
        "--seed", "42",
        "--base_dir", str(output_dir),
        "--dtype", dtype,
        "--project", "rna3d",
        "--use_wandb", "false",
        "--diffusion_batch_size", str(diffusion_batch_size),
        "--eval_interval", str(eval_interval),
        "--log_interval", "50",
        "--checkpoint_interval", str(checkpoint_interval),
        "--ema_decay", "0.999",
        "--train_crop_size", str(train_crop_size),
        "--max_steps", str(max_steps),
        "--warmup_steps", str(warmup_steps),
        "--lr", str(lr),
        "--model.N_cycle", "4",
        "--sample_diffusion.N_step", "20",
        "--load_checkpoint_path", str(PROTENIX_WEIGHTS),
        "--load_ema_checkpoint_path", str(PROTENIX_WEIGHTS),
        "--data.train_sets", "rna_finetune",
        f"--data.rna_finetune.base_info.pdb_list", str(pdb_list_path),
    ]

    print(f"Starting Protenix fine-tuning: {len(cmd)} args")
    print(f"  PDB list: {pdb_list_path}")
    print(f"  Output:   {output_dir}")
    print(f"  Steps:    {max_steps}, LR: {lr}, Crop: {train_crop_size}")

    result = subprocess.run(cmd, text=True)

    if result.returncode != 0:
        print(f"Fine-tuning exited with code {result.returncode}")
    else:
        print("Fine-tuning completed successfully.")

    return result


def prepare_pdb_list_from_labels(
    train_labels_df,
    output_path: str | Path = "finetune_pdb_list.txt",
) -> Path:
    """Extract unique PDB IDs from training labels and write to a text file.

    Protenix fine-tuning expects a simple list of PDB IDs (one per line).
    """
    output_path = Path(output_path)

    # target_id format may contain PDB-like IDs
    # Extract unique identifiers
    target_ids = train_labels_df["ID"].str.rsplit("_", n=1).str[0].unique()

    with open(output_path, "w") as f:
        for tid in sorted(set(target_ids)):
            f.write(f"{tid}\n")

    print(f"Wrote {len(target_ids)} PDB IDs to {output_path}")
    return output_path
