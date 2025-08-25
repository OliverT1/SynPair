import random
from pathlib import Path
from typing import List, Tuple

import lightning as L
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset
from transformers import AutoModel

from synpair.constants import HF_MODEL

try:
    from peft import LoraConfig, TaskType, get_peft_model

    _PEFT = True
except ImportError:
    _PEFT = False


# ---------------------------------------------------------------------------
# Dataset / collate
# ---------------------------------------------------------------------------
class PairDataset(Dataset):
    def __init__(
        self, true_file: Path, fake_file: Path, load_negative_samples: bool = True
    ):
        self.samples: List[Tuple[str, str, int]] = []
        self.num_true = 0
        self.num_fake = 0
        self.load_negative_samples = load_negative_samples  # Store the parameter

        with true_file.open() as f:
            for line in f:
                line = line.rstrip()
                vh = " ".join(list(line[:200].replace("-", ""))).strip()
                vl = " ".join(list(line[200:].replace("-", ""))).strip()
                self.samples.append((vh, vl, 1))
                self.num_true += 1

        if self.load_negative_samples:
            with fake_file.open() as f:
                for line in f:
                    line = line.rstrip()
                    vh = " ".join(list(line[:200].replace("-", ""))).strip()
                    vl = " ".join(list(line[200:].replace("-", ""))).strip()
                    self.samples.append((vh, vl, 0))
                    self.num_fake += 1

        print(f"Loaded {len(self.samples):,} samples…")
        print(f"True: {self.num_true:,}")
        if self.load_negative_samples:
            print(f"Fake: {self.num_fake:,}")
        else:
            print("Fake: 0 (not loaded due to contrastive mode)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def shuffle(self):
        random.shuffle(self.samples)


class Collater:
    def __init__(self, tokenizer, max_len=256):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        vhs, vls, labels = zip(*batch)
        vh_tok = self.tok(
            list(vhs),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        vl_tok = self.tok(
            list(vls),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        return vh_tok, vl_tok, torch.tensor(labels, dtype=torch.float32)


class SeqEmbedder(nn.Module):
    def __init__(
        self,
        proj_dim=128,
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        use_projection=False,
    ):
        super().__init__()
        base = AutoModel.from_pretrained(HF_MODEL)
        print(base)
        if use_lora:
            if not _PEFT:
                raise RuntimeError(
                    "peft not installed; install peft or run without --lora"
                )
            target = ["query", "key", "value", "dense"]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target,
                bias="none",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
            base = get_peft_model(base, config)
            print(base.print_trainable_parameters())
        else:
            for p in base.parameters():
                p.requires_grad = False
        self.encoder = base
        self.proj = None
        if use_projection:
            self.proj = nn.Sequential(
                nn.Linear(self.encoder.config.hidden_size, proj_dim, bias=False),
                nn.LayerNorm(
                    proj_dim,
                ),
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim, bias=False),
            )

    def forward(self, tok):
        out = self.encoder(**tok).last_hidden_state
        mask = tok.attention_mask.unsqueeze(-1)
        pooled = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
        if self.proj is not None:
            pooled = self.proj(pooled)
        return nn.functional.normalize(pooled, dim=-1)


class DualEncoder(L.LightningModule):
    def __init__(
        self,
        lr=1e-3,
        pos_weight=3.0,
        use_lora=False,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        batch_size=512,
        use_projection=False,
        proj_dim=128,
        compile_model=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.vh_enc = SeqEmbedder(
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_projection=use_projection,
            proj_dim=proj_dim,
        )
        self.vl_enc = SeqEmbedder(
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_projection=use_projection,
            proj_dim=proj_dim,
        )

        if compile_model:
            print("Compiling model with torch.compile...")
            self.vh_enc = torch.compile(self.vh_enc, mode="reduce-overhead")
            self.vl_enc = torch.compile(self.vl_enc, mode="reduce-overhead")

        self.scale = nn.Parameter(torch.tensor(1.0))
        # self.supcon = SupConLoss(temperature=0.07)
        self.pos_weight = pos_weight
        self.batch_size = batch_size

        self.validation_step_outputs = []

    def forward(self, vh_tok, vl_tok):
        return self.vh_enc(vh_tok), self.vl_enc(vl_tok)

    def training_step(self, batch, _):
        vh_tok, vl_tok, labels = batch
        vh_vec, vl_vec = self(vh_tok, vl_tok)
        logits = (vh_vec * vl_vec).sum(-1) * self.scale
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits, labels.to(self.device)
        )
        self.log("train_loss", loss, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        vh_tok, vl_tok, labels = batch
        vh_vec, vl_vec = self(vh_tok, vl_tok)
        scores = (vh_vec * vl_vec).sum(-1) * self.scale
        loss = nn.functional.binary_cross_entropy_with_logits(
            scores,
            labels.to(self.device),
        )
        self.log("val_loss", loss)
        self.validation_step_outputs.append({"scores": scores, "labels": labels})
        return loss

    def on_validation_epoch_end(self):
        all_scores = torch.cat(
            [o["scores"] for o in self.validation_step_outputs]
        ).cpu()
        all_labels = torch.cat(
            [o["labels"] for o in self.validation_step_outputs]
        ).cpu()

        # apply sigmoid to scores, currently they are cosine similarity scores [-1, 1]
        all_scores = torch.sigmoid(all_scores)
        if all_labels.numel() == 0:
            print("Validation epoch end: No labels found in outputs.")
            return

        # Ensure labels are suitable for roc_auc_score (e.g., binary)
        # and scores are continuous probabilities or decision values.
        auc = roc_auc_score(all_labels.numpy(), all_scores.numpy())
        # Accuracy calculation assumes scores are logits; threshold at 0.5 for binary classification
        acc = ((all_scores > 0.5).float() == all_labels).float().mean().item()

        self.log("val_auc", auc, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)
        # Optional: print for immediate feedback, complementing the logger
        print(f"Validation epoch end: val_auc: {auc:.4f}, val_acc: {acc:.4f}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        # CosineAnnealingLR gradually decreases the learning rate following a cosine curve.
        # T_max: Maximum number of iterations for one cycle. Here, it's set
        #        to self.trainer.estimated_stepping_batches for per-step scheduling.
        # eta_min: Minimum learning rate. Here, it's 1% of the initial learning rate.
        scheduler_config = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.trainer.estimated_stepping_batches,  # Updated for per-step
                eta_min=self.hparams.lr * 0.01,
            ),
            "interval": "step",  # Step the scheduler at the end of each training step
            "frequency": 1,  # Step the scheduler every 1 interval (every step)
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_config}

    @torch.no_grad()
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # The InferenceCollater provides: vh_tok, vl_tok, original_vhs_batch, original_vls_batch
        # The Trainer will automatically move vh_tok and vl_tok to the correct device.
        vh_tok, vl_tok, original_vhs_batch, original_vls_batch = batch

        # self() calls the forward method of your LightningModule
        vh_vec, vl_vec = self(vh_tok, vl_tok)

        # Calculate raw scores (logits)
        # self.scale should be on the same device as vh_vec and vl_vec
        # because it's an nn.Parameter and the model has been moved to the device by the Trainer.
        batch_scores_raw = (vh_vec * vl_vec).sum(-1)  # * self.scale

        # Convert logits to probabilities
        probabilities = torch.sigmoid(batch_scores_raw)

        # Return the original sequences and their scores for this batch
        # It's good practice to move results to CPU if they are not needed on GPU anymore,
        # especially before converting to Python lists.
        return original_vhs_batch, original_vls_batch, probabilities.cpu().tolist()


class ContrastiveDualEncoder(DualEncoder):
    def __init__(self, temperature=0.07, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_tau = nn.Parameter(torch.log(torch.tensor(temperature)))

    def _contrastive(self, q, k, labels):
        # check all labels are 1
        assert (labels == 1).all(), (
            "ContrastiveDualEncoder only supports positive pairs"
        )
        sim = torch.einsum("id,jd->ij", q, k) / self.log_tau.exp()
        tgt = torch.arange(q.size(0), device=q.device)
        loss_i = nn.functional.cross_entropy(sim, tgt, reduction="none")
        loss_j = nn.functional.cross_entropy(sim.t(), tgt, reduction="none")
        # w = torch.where(labels > 0, self.pos_weight, 1.0)
        return (loss_i).mean() + (loss_j).mean()

    def training_step(self, batch, _):
        vh_tok, vl_tok, labels = batch
        vh_vec, vl_vec = self(vh_tok, vl_tok)
        loss = self._contrastive(vh_vec, vl_vec, labels)
        self.log("train_loss", loss, prog_bar=True)
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("learning_rate", lr, prog_bar=True)
        return loss

    # ---------- validation: collect embeddings ----------
    def validation_step(self, batch, _):
        vh_tok, vl_tok, labels = batch
        vh_vec, vl_vec = self(vh_tok, vl_tok)

        # Calculate scores (raw logits) as used for loss and AUC
        scores = (vh_vec * vl_vec).sum(-1)

        # Calculate loss for this batch using
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            scores, labels.to(self.device)
        )
        # calculate contrastive loss, using only positive pairs
        pos_idx = torch.nonzero(labels).squeeze(-1)
        sim_loss = self._contrastive(vh_vec[pos_idx], vl_vec[pos_idx], labels[pos_idx])
        # Store outputs for epoch-level metrics
        output_data = {
            "vh_vec": vh_vec.detach().cpu(),
            "vl_vec": vl_vec.detach().cpu(),
            "labels": labels.detach().cpu(),
            "scores": scores.detach().cpu(),
            "bce_loss": bce_loss.detach().cpu(),
        }
        self.validation_step_outputs.append(output_data)
        self.log("val_loss", sim_loss, prog_bar=False)
        return sim_loss

    # ---------- epoch‑end: AUROC + retrieval ----------
    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            print(
                f"Epoch {self.current_epoch}: No validation outputs to process for ContrastiveDualEncoder."
            )
            # Log default metrics if validation loader was empty or no outputs produced
            self.log_dict(
                {"val_auc": 0.0, "val_MRR": 0.0, "val_R@20": 0.0},
                prog_bar=True,
                sync_dist=True,
            )
            self.validation_step_outputs.clear()  # Ensure cleared even if logic exits early
            return

        # Aggregate all outputs
        vh_all = torch.cat([o["vh_vec"] for o in self.validation_step_outputs])
        vl_all = torch.cat([o["vl_vec"] for o in self.validation_step_outputs])
        labels_all = torch.cat([o["labels"] for o in self.validation_step_outputs])
        bce_losses = torch.stack([o["bce_loss"] for o in self.validation_step_outputs])
        self.log("val_bce_loss", bce_losses.mean(), prog_bar=True, sync_dist=True)
        # Use the per-batch scores (raw logits) that were stored
        all_scores_for_auc = torch.cat(
            [o["scores"] for o in self.validation_step_outputs]
        )

        # Probabilities for AUC (scores were already on CPU)
        probs = torch.sigmoid(all_scores_for_auc)

        # Pairwise AUROC
        if labels_all.numel() == 0:
            print(
                f"Epoch {self.current_epoch}: No labels in validation outputs for ContrastiveDualEncoder."
            )
            auc = 0.0  # Default AUC if no labels
        else:
            auc = roc_auc_score(labels_all.numpy(), probs.numpy())
        self.log("val_auc", auc, prog_bar=True, sync_dist=True)

        # Retrieval metrics computed only on TRUE rows
        true_idx = torch.nonzero(labels_all).squeeze(-1)

        if true_idx.numel() > 0:
            # vh_all and vl_all are already on CPU
            sims = vh_all @ vl_all.T  # (N, N) cosine (because L2‑normed)

            ranks_list = []
            for i_tensor in true_idx:  # true_idx contains 0-dim tensors
                i = i_tensor.item()  # Get scalar Python int index
                current_sim_val = sims[i, i]
                better_count = (sims[i] > current_sim_val).sum().item()
                ranks_list.append(better_count + 1)

            # This check is technically redundant if true_idx.numel() > 0,
            # but kept for safety in case ranks_list could be empty by other logic paths.
            if ranks_list:
                ranks = torch.tensor(ranks_list, dtype=torch.float32)
                mrr = (1.0 / ranks).mean().item()
                r20 = (ranks <= 20).float().mean().item()
                self.log_dict(
                    {"val_MRR": mrr, "val_R@20": r20}, prog_bar=True, sync_dist=True
                )
            else:  # Should not be reached if true_idx.numel() > 0
                self.log_dict(
                    {"val_MRR": 0.0, "val_R@20": 0.0}, prog_bar=True, sync_dist=True
                )
        else:  # No true samples for retrieval metrics
            self.log_dict(
                {"val_MRR": 0.0, "val_R@20": 0.0}, prog_bar=True, sync_dist=True
            )

        # Reset buffer
        self.validation_step_outputs.clear()
