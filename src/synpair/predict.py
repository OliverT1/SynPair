import argparse
import os

import lightning as L
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from synpair.constants import HF_MODEL
from synpair.model import ContrastiveDualEncoder, DualEncoder


class InferenceDataset(Dataset):
    def __init__(self, csv_path: str, vh_col_name: str = "VH", vl_col_name: str = "VL"):
        try:
            self.df = pd.read_csv(csv_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Input CSV not found at {csv_path}")
        if vh_col_name not in self.df.columns or vl_col_name not in self.df.columns:
            raise ValueError(
                f"Input CSV must contain '{vh_col_name}' and '{vl_col_name}' columns."
            )

        self.vh_sequences_original = self.df[vh_col_name].astype(str).tolist()
        self.vl_sequences_original = self.df[vl_col_name].astype(str).tolist()

        # remove any alignment tokens if they exist, and add spaces fro tokeniser

        self.vh_sequences_processed = [
            " ".join(list(seq.replace("-", ""))).strip()
            for seq in self.vh_sequences_original
        ]
        self.vl_sequences_processed = [
            " ".join(list(seq.replace("-", ""))).strip()
            for seq in self.vl_sequences_original
        ]
        print(f"Loaded {len(self.df)} pairs from {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.vh_sequences_processed[idx],
            self.vl_sequences_processed[idx],
            self.vh_sequences_original[idx],
            self.vl_sequences_original[idx],
        )


class InferenceCollater:
    def __init__(self, tokenizer, max_len=256):
        self.tok = tokenizer
        self.max_len = max_len

    def __call__(self, batch):
        vhs_proc, vls_proc, vhs_orig, vls_orig = zip(*batch)

        vh_tok = self.tok(
            list(vhs_proc),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        vl_tok = self.tok(
            list(vls_proc),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_len,
        )
        return vh_tok, vl_tok, list(vhs_orig), list(vls_orig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VH/VL pairing scores using a trained DualEncoder model."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to the input CSV file with 'VH' and 'VL' columns.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to save the output CSV file with 'VH', 'VL', and 'pairing_score'. "
        "Defaults to the input CSV directory with '_synpair_preds.csv' appended to the input filename.",
    )
    parser.add_argument(
        "--vh_col_name",
        type=str,
        default="VH",
        help="Name of the column containing VH sequences. Defaults to 'VH'.",
    )
    parser.add_argument(
        "--vl_col_name",
        type=str,
        default="VL",
        help="Name of the column containing VL sequences. Defaults to 'VL'.",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt) file.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for inference."
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=256,
        help="Maximum sequence length for tokenizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference ('cuda', 'cpu', or 'mps'). Autodetects if None. Note: For Lightning, this influences accelerator choice.",
    )
    parser.add_argument(
        "--contrastive",
        action="store_true",
    )

    args = parser.parse_args()

    # Determine accelerator and devices for Lightning Trainer
    trainer_args = {"logger": False, "enable_checkpointing": False}

    if args.output_csv is None:
        input_dir = os.path.dirname(args.input_csv)
        input_filename_without_ext = os.path.splitext(os.path.basename(args.input_csv))[
            0
        ]
        args.output_csv = os.path.join(
            input_dir, f"{input_filename_without_ext}_synpair_preds.csv"
        )

    if args.device:
        if args.device == "cuda" and torch.cuda.is_available():
            trainer_args["accelerator"] = "gpu"
            trainer_args["devices"] = 1  # Or specific GPU IDs e.g., [0]
        elif args.device == "mps" and torch.backends.mps.is_available():
            trainer_args["accelerator"] = "mps"
            trainer_args["devices"] = 1
        else:
            trainer_args["accelerator"] = "cpu"
    else:  # Autodetect
        if torch.cuda.is_available():
            trainer_args["accelerator"] = "gpu"
            trainer_args["devices"] = 1
        elif torch.backends.mps.is_available():  # Added MPS autodetect
            trainer_args["accelerator"] = "mps"
            trainer_args["devices"] = 1
        else:
            trainer_args["accelerator"] = "cpu"

    print(f"Using accelerator: {trainer_args.get('accelerator', 'cpu')}")

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)

    print(f"Loading dataset from {args.input_csv}...")
    inference_dataset = InferenceDataset(
        csv_path=args.input_csv,
        vh_col_name=args.vh_col_name,
        vl_col_name=args.vl_col_name,
    )

    inference_collater = InferenceCollater(tokenizer=tokenizer, max_len=args.max_len)

    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle for inference
        collate_fn=inference_collater,
        num_workers=4,  # Adjust based on your system
        pin_memory=True if trainer_args.get("accelerator") in ["gpu", "mps"] else False,
    )

    print(f"Loading model from checkpoint: {args.checkpoint_path}...")
    if args.contrastive:
        model = ContrastiveDualEncoder.load_from_checkpoint(
            args.checkpoint_path, map_location="cpu"
        )
    else:
        model = DualEncoder.load_from_checkpoint(
            args.checkpoint_path, map_location="cpu"
        )

    trainer = L.Trainer(**trainer_args)

    print("Running inference using trainer.predict()...")
    # trainer.predict() returns a list of outputs from predict_step (one item per batch)
    # Each item will be (original_vhs_batch, original_vls_batch, probabilities_batch)
    results_list = trainer.predict(model=model, dataloaders=inference_dataloader)

    all_original_vhs = []
    all_original_vls = []
    all_scores = []

    if results_list:
        for vhs_batch, vls_batch, scores_batch in results_list:
            all_original_vhs.extend(vhs_batch)
            all_original_vls.extend(vls_batch)
            all_scores.extend(scores_batch)

    print("Saving results...")
    results_df = pd.DataFrame(
        {
            args.vh_col_name: all_original_vhs,
            args.vl_col_name: all_original_vls,
            "pairing_score": all_scores,
        }
    )

    results_df.to_csv(args.output_csv, index=False)
    print(f"Results saved to {args.output_csv}")
    print("Done.")
