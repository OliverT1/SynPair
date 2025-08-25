# dual_encoder_retriever.py – v4 (LoRA option)
"""
Dual‑encoder VH↔VL retriever **with optional LoRA fine‑tuning** of the ESM‑2
backbone via HuggingFace PEFT.

Key additions
-------------
* `--lora` flag enables LoRA on q/k/v/o projections of every transformer layer.
* LoRA hyper‑params adjustable (`--lora_r`, `--lora_alpha`, `--lora_dropout`).
* Only LoRA and projection‑head params are trainable; all other weights stay
  frozen, so VRAM hit is minimal.
* `--data_dir` specifies the root directory for data, expecting:
  `data_dir/train/true.csv`, `data_dir/train/fake.csv`
  `data_dir/val/true.csv` (optional), `data_dir/val/fake.csv` (optional)

Example usage
~~~~~~~~~~~~~
```bash
python train.py \
    --data_dir path/to/your/data \
    --lora \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --epochs 3 --batch 512
```

Dependencies: `pip install peft transformers pytorch_lightning faiss-gpu`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import lightning as L
import wandb
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from synpair.constants import HF_MODEL
from synpair.model import Collater, ContrastiveDualEncoder, DualEncoder, PairDataset

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser("Dual‑encoder (LoRA optional)")
    ap.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="Directory containing train/ and val/ subdirectories with true.csv and fake.csv files",
    )
    ap.add_argument("--output_dir", type=Path, default=Path("./training_outputs"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument(
        "--projection", action="store_true", help="Enable projection layer training"
    )
    ap.add_argument(
        "--proj_dim",
        type=int,
        default=128,
        help="Dimension of the projection layer output",
    )
    ap.add_argument(
        "--compile_model",
        action="store_true",
        help="Compile the model using torch.compile (requires PyTorch >= 2.0)",
    )
    ap.add_argument("--build_index", action="store_true")
    ap.add_argument(
        "--limit_val_batches",
        type=float,
        default=0.1,
        help="Fraction of validation batches to check (float) or number of batches (int). 1.0 means all.",
    )
    ap.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of steps to accumulate gradients before performing an optimizer step. Effectively multiplies batch size.",
    )
    ap.add_argument("--contrastive", action="store_true")
    ap.add_argument("--validate_first", action="store_true")
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.lora and not args.projection:
        raise ValueError(
            "Either --lora or --projection (or both) must be specified for training."
        )

    tok = AutoTokenizer.from_pretrained(HF_MODEL)

    train_true_file = args.data_dir / "train" / "true.csv"
    train_fake_file = args.data_dir / "train" / "fake.csv"

    if not train_true_file.is_file() or not train_fake_file.is_file():
        raise FileNotFoundError(
            f"Training files true.csv and/or fake.csv not found in {args.data_dir / 'train'}"
        )

    train_ds = PairDataset(
        train_true_file, train_fake_file, load_negative_samples=not args.contrastive
    )
    train_ds.shuffle()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch,
        collate_fn=Collater(tok),
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        shuffle=True,
    )

    val_ds = None
    val_true_file = args.data_dir / "val" / "true.csv"
    val_fake_file = args.data_dir / "val" / "fake.csv"

    val_ds = PairDataset(val_true_file, val_fake_file)
    val_ds.shuffle()
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch,
        collate_fn=Collater(tok),
        num_workers=4,
        persistent_workers=True,
    )

    # --- Updated W&B Initialization ---
    wandb_logger_save_dir = args.output_dir / "logs"
    wandb_logger_save_dir.mkdir(parents=True, exist_ok=True)

    # Manually initialize W&B run
    # This ensures the run is started and the ID is available.
    # You can add other wandb.init arguments here like entity, tags, config=vars(args), etc.
    manual_wandb_run = wandb.init(
        project="dual‑encoder",  # Project name for W&B
        dir=str(wandb_logger_save_dir),  # Local directory for W&B files
        config=vars(args),  # Log hyperparameters
    )

    if manual_wandb_run is None:
        # This case should ideally not happen if wandb is set up correctly,
        # as wandb.init() usually raises an error on failure.
        raise ConnectionError(
            "wandb.init() failed. Check W&B setup, credentials, and internet connection."
        )

    # Get the unique W&B run ID to version your local artifacts
    run_id = manual_wandb_run.id  # Use the ID from the manually initialized run

    # Initialize PTL WandbLogger with the existing W&B run
    # The 'project' and 'save_dir' are now handled by the manual wandb.init()
    logger = L.pytorch.loggers.WandbLogger(
        experiment=manual_wandb_run,  # Pass the existing run object
        log_model=False,  # PTL specific: whether to log model checkpoints as W&B artifacts
    )
    # The fallback for run_id is no longer needed as manual_wandb_run.id should be reliable.
    if not args.contrastive:
        model = DualEncoder(
            lr=args.lr,
            use_lora=args.lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            batch_size=args.batch,
            use_projection=args.projection,
            proj_dim=args.proj_dim,
            compile_model=args.compile_model,
        )
    else:
        model = ContrastiveDualEncoder(
            lr=args.lr,
            use_lora=args.lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            batch_size=args.batch,
            use_projection=args.projection,
            proj_dim=args.proj_dim,
            compile_model=args.compile_model,
        )

    # Create versioned directory for checkpoints using the W&B run ID
    checkpoint_dir = args.output_dir / "checkpoints" / str(run_id)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved to: {checkpoint_dir}")
    if val_ds and not args.contrastive:
        monitor_metric = "val_auc"
        mode = "max"
    elif val_ds and args.contrastive:
        monitor_metric = "val_MRR"
        mode = "max"
    else:
        monitor_metric = None

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="best-{epoch}-{val_auc:.4f}" if val_ds else "last-{epoch}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=1 if val_ds else 0,
        save_last=True,
        verbose=True,
    )

    trainer = L.Trainer(
        accelerator="auto",
        precision="16-mixed",
        max_epochs=args.epochs,
        log_every_n_steps=10,
        logger=logger,
        default_root_dir=str(args.output_dir),
        callbacks=[checkpoint_callback],
        limit_val_batches=args.limit_val_batches,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        val_check_interval=1.0,
    )
    # run a validation step
    if args.validate_first:
        trainer.validate(model, dataloaders=val_loader)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if logger:
        # logger.experiment will point to manual_wandb_run, so this is correct.
        logger.experiment.finish()
