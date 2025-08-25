#!/usr/bin/env python
"""
Build a FAISS IVF-PQ index from saved embeddings.

Example
-------
python build_index.py --embed_dir embeds/vh
"""

import argparse
import time
from pathlib import Path

import faiss
import numpy as np


def build_index(vecs: np.ndarray, nlist=4096, m=16, use_gpu=False):
    """Builds a FAISS IVF-PQ index, optionally using GPU."""
    d = vecs.shape[1]
    quant = faiss.IndexFlatIP(d)
    # The last argument is bits_per_code, 8 means 8 bits per sub-quantizer
    idx_cpu = faiss.IndexIVFPQ(quant, d, nlist, m, 8)

    if use_gpu:
        print("Using GPU for FAISS indexing.")
        # Check if GPU resources are available
        if faiss.get_num_gpus() == 0:
            print("Warning: No GPU detected by FAISS. Falling back to CPU.")
            idx = idx_cpu
        else:
            res = faiss.StandardGpuResources()  # Use default GPU resources
            # Co-options for IVF-PQ index on GPU
            co = faiss.GpuClonerOptions()
            co.useFloat16 = True  # Use float16 for faster computation if possible
            # Transfer the CPU index to GPU
            idx = faiss.index_cpu_to_gpu(res, 0, idx_cpu, co)  # Use GPU 0
    else:
        print("Using CPU for FAISS indexing.")
        idx = idx_cpu

    print(f"Training FAISS index (nlist={nlist}, m={m}) on {vecs.shape} vectors...")
    t0 = time.time()
    idx.train(vecs)
    print(f"Training done in {time.time() - t0:.1f}s")
    print("Adding vectors to index...")
    t0 = time.time()
    idx.add(vecs)
    print(f"Adding done in {time.time() - t0:.1f}s")

    # If we used GPU, transfer the index back to CPU for saving
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Transferring index from GPU back to CPU...")
        idx_cpu = faiss.index_gpu_to_cpu(idx)
        return idx_cpu
    else:
        return idx


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embed_dir",
        required=True,
        type=Path,
        help="Directory containing embeddings.fp16.npy",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=None,
        help="Directory to save the index (defaults to embed_dir)",
    )
    ap.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="Number of centroids for IVF",
    )
    ap.add_argument(
        "--m",
        type=int,
        default=16,
        help="Number of sub-quantizers for PQ",
    )
    ap.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for FAISS indexing if available.",
    )
    args = ap.parse_args()

    embed_file = args.embed_dir / "embeddings.fp16.npy"
    if not embed_file.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embed_file}")

    out_dir = args.out_dir if args.out_dir else args.embed_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    index_file = out_dir / "faiss_ivfpq.index"

    print(f"Loading embeddings from {embed_file}...")
    vecs = np.load(embed_file)
    # FAISS requires float32 for CPU training, but GPU can leverage float16 if available
    if vecs.dtype == np.float16 and not args.gpu:
        print("Converting embeddings from float16 to float32 for FAISS CPU...")
        vecs = vecs.astype(np.float32)
    elif vecs.dtype == np.float16 and args.gpu:
        # Keep as float16 if using GPU, FAISS GPU cloner options can handle it
        print("Using float16 embeddings with FAISS GPU.")
    elif vecs.dtype != np.float32 and not args.gpu:
        print(f"Converting embeddings from {vecs.dtype} to float32 for FAISS CPU...")
        vecs = vecs.astype(np.float32)

    idx = build_index(vecs, nlist=args.nlist, m=args.m, use_gpu=args.gpu)

    print(f"Saving index to {index_file}...")
    faiss.write_index(idx, str(index_file))  # Save the CPU index
    print("Index saved.")
