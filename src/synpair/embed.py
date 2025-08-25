#!/usr/bin/env python
"""
Embed VH or VL sequences with a trained projection model.

Example
-------
python embed.py \
    --csv vh.csv \
    --ckpt checkpoints/best.ckpt \
    --out_dir embeds/vh \
    --batch 512
"""
# TODO Make sure to deduplicate the light chain sequences (and maybe VH)

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from synpair.constants import HF_MODEL
from synpair.model import ContrastiveDualEncoder, DualEncoder


class EmbedDataset(Dataset):
    def __init__(
        self, csv_path: Path, debug_mode: bool = False, max_sequences: int | None = None
    ):
        try:
            df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input CSV not found at {csv_path}")

        if df.shape[1] != 2:
            raise ValueError(
                f"Input CSV at {csv_path} must contain exactly two columns: id and sequence. Found {df.shape[1]} columns."
            )

        df.columns = ["id", "sequence"]
        original_ids_full = df["id"].tolist()
        sequences_original_full = df["sequence"].astype(str).tolist()

        sequences_processed_full = [
            " ".join(list(str(seq).replace("-", ""))).strip()
            for seq in sequences_original_full
        ]

        # Deduplicate sequences
        dedup_df = pd.DataFrame(
            {"id": original_ids_full, "processed_sequence": sequences_processed_full}
        )

        initial_count = len(dedup_df)
        dedup_df.drop_duplicates(
            subset=["processed_sequence"], keep="first", inplace=True
        )
        final_count = len(dedup_df)
        num_duplicates_removed = initial_count - final_count

        if num_duplicates_removed > 0:
            print(
                f"INFO: Removed {num_duplicates_removed} duplicate sequences. {final_count} unique sequences remain before any debug mode limiting."
            )

        current_ids = dedup_df["id"].tolist()
        current_sequences_processed = dedup_df["processed_sequence"].tolist()

        if debug_mode and max_sequences is not None and max_sequences > 0:
            if max_sequences < len(current_ids):
                print(
                    f"DEBUG MODE: Limiting to first {max_sequences} unique sequences from {csv_path} (out of {len(current_ids)} unique sequences)."
                )
                self.ids = current_ids[:max_sequences]
                self.sequences_processed = current_sequences_processed[:max_sequences]
            else:
                print(
                    f"DEBUG MODE: max_sequences ({max_sequences}) is >= number of unique sequences ({len(current_ids)}). Using all {len(current_ids)} unique sequences."
                )
                self.ids = current_ids
                self.sequences_processed = current_sequences_processed
        else:
            self.ids = current_ids
            self.sequences_processed = current_sequences_processed

        print(f"Loaded {len(self.ids)} sequences for processing.")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.sequences_processed[idx]


def embed(model, tokenizer, dataset: EmbedDataset, batch=512, use_vh=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=lambda s: tokenizer(
            s, return_tensors="pt", padding=True, truncation=True, max_length=256
        ),
    )
    vecs = []
    model.eval().cuda()
    with torch.no_grad():
        for tok_batch in tqdm(loader, desc="Embedding sequences", unit="batch"):
            tok_on_gpu = tok_batch.to("cuda", non_blocking=True)

            encoder = model.vh_enc if use_vh else model.vl_enc

            with autocast():
                result = encoder(tok_on_gpu)

            vecs.append(result.cpu())
    return torch.cat(vecs).half().numpy()  # float16


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="CSV: id,sequence (first col must be id, second col must be sequence)",
    )
    ap.add_argument(
        "--ckpt",
        required=True,
        type=Path,
        help="Lightning checkpoint (.ckpt) for projection model",
    )
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--batch", default=512, type=int)
    ap.add_argument(
        "--use_vh",
        action="store_true",
        help="If set, use VH encoder.",
    )
    ap.add_argument(
        "--use_vl",
        action="store_true",
        help="If set, use VL encoder.",
    )

    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode. Limits the number of sequences processed if --max_sequences is set.",
    )
    ap.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to process in debug mode. Ignored if --debug is not set, or if set to 0 or None.",
    )
    ap.add_argument(
        "--contrastive",
        action="store_true",
        help="If set, use contrastive model.",
    )
    args = ap.parse_args()
    if args.use_vh and args.use_vl:
        raise ValueError("Cannot use both VH and VL encoders.")
    if not args.use_vh and not args.use_vl:
        raise ValueError("Must use either VH or VL encoder.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not args.debug and args.max_sequences is not None:
        print(
            "WARNING: --max_sequences is set but --debug is not. --max_sequences will be ignored."
        )
    if args.debug and (args.max_sequences is None or args.max_sequences <= 0):
        print(
            "DEBUG MODE: --max_sequences is not set to a positive value. All sequences from the input will be processed."
        )

    # Instantiate EmbedDataset
    dataset = EmbedDataset(
        csv_path=args.csv, debug_mode=args.debug, max_sequences=args.max_sequences
    )

    print(f"Embedding {len(dataset):,} sequences …")
    if args.contrastive:
        model = ContrastiveDualEncoder.load_from_checkpoint(args.ckpt)
    else:
        model = DualEncoder.load_from_checkpoint(args.ckpt)
    tok = AutoTokenizer.from_pretrained(HF_MODEL)

    t0 = time.time()
    # Pass the dataset instance to the embed function
    vecs = embed(model, tok, dataset, batch=args.batch, use_vh=args.use_vh)
    print(f"Done in {time.time() - t0:.1f}s  →  {vecs.shape}")

    # save
    np.save(args.out_dir / "embeddings.fp16.npy", vecs)

    # Get IDs from the dataset instance and save using pandas
    ids_df = pd.DataFrame(dataset.ids)
    ids_df.to_csv(args.out_dir / "ids.csv", index=False, header=False)

    meta = {"model_ckpt": str(args.ckpt), "dim": int(vecs.shape[1])}
    (args.out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
