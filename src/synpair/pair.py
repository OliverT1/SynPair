#!/usr/bin/env python
"""
Pair VH embeddings with VL index / embeddings.

Example
-------
python pair.py \
   --vh_dir embeds/vh \
   --vl_dir embeds/vl \
   --top_k 20 \
   --out pairs.tsv
"""

import argparse
import csv
import json
import time  # Add timing functionality
from pathlib import Path

import faiss
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def load_embeddings(dir_: Path):
    vecs = np.load(dir_ / "embeddings.fp16.npy")
    ids = (dir_ / "ids.csv").read_text().splitlines()
    meta = json.loads((dir_ / "meta.json").read_text())
    return ids, vecs, meta


def search_faiss(index, queries, k):
    D, I = index.search(queries, k)
    return D, I


def brute_cosine(queries, vecs_vl, k, use_gpu=False):
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Attempting brute-force search on GPU using Faiss IndexFlatIP...")
        try:
            dim = vecs_vl.shape[1]
            queries_f32 = queries.astype(np.float32)
            vecs_vl_f32 = vecs_vl.astype(np.float32)

            index_cpu = faiss.IndexFlatIP(dim)
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, index_cpu)
            gpu_index.add(vecs_vl_f32)

            D, I = gpu_index.search(queries_f32, k)

            print("Successfully performed brute-force search on GPU using Faiss.")

            # Add heatmap plotting
            print("Generating and saving heatmap...")
            vl_sims_sample = vecs_vl_f32[:200] @ vecs_vl_f32[:200].T
            # If vl_sims_sample is a Faiss GPU tensor, it needs to be moved to CPU
            if hasattr(
                vl_sims_sample, "get"
            ):  # Check if it's a GPU tensor from Faiss (heuristic)
                vl_sims_sample_cpu = vl_sims_sample.get()
            elif hasattr(vl_sims_sample, "cpu"):  # Check for PyTorch-like .cpu()
                vl_sims_sample_cpu = (
                    vl_sims_sample.cpu().numpy()
                )  # Assuming it might be a torch tensor
            else:  # Assuming it's already a NumPy array on CPU or can be directly used
                vl_sims_sample_cpu = vl_sims_sample

            plt.figure()
            sns.heatmap(vl_sims_sample_cpu)
            plt.title("Similarity Heatmap of VL Embeddings (GPU Brute-Force)")
            plt.xlabel("VL Index")
            plt.ylabel("VL Index")
            heatmap_filename = "brute_force_gpu_heatmap.png"
            plt.savefig(heatmap_filename)
            plt.close()  # Close the figure to free memory
            print(f"Heatmap saved to {heatmap_filename}")

            return D, I
        except Exception as e:
            print(
                f"Failed to run brute-force on GPU with Faiss: {e}. Falling back to CPU implementation."
            )
            # Fall through to CPU implementation below

    print("Performing brute-force search on CPU using NumPy.")
    sims = queries @ vecs_vl.T
    idx_unsorted = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
    D_unsorted = np.take_along_axis(sims, idx_unsorted, axis=1)

    order_in_k = np.argsort(-D_unsorted, axis=1)

    I_sorted = np.take_along_axis(idx_unsorted, order_in_k, axis=1)
    D_sorted = np.take_along_axis(D_unsorted, order_in_k, axis=1)

    return D_sorted, I_sorted


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vh_dir", required=True, type=Path)
    ap.add_argument("--vl_dir", required=True, type=Path)
    ap.add_argument("--top_k", default=20, type=int)
    ap.add_argument(
        "--max_queries",
        type=int,
        default=None,
        help="Limit the number of query vectors (VH) to the first N.",
    )
    ap.add_argument(
        "--sigmoid",
        action="store_true",
        help="Apply sigmoid to similarities before output.",
    )
    ap.add_argument(
        "--no_faiss_index",
        action="store_true",
        help="Do not use FAISS index even if available, use brute-force instead.",
    )
    ap.add_argument(
        "--gpu_brute_force",
        action="store_true",
        help="Use GPU for brute-force calculation if FAISS index is not used.",
    )
    ap.add_argument(
        "--out", required=True, type=Path, help="CSV output: VH_id, VL_id, rank, score"
    )
    args = ap.parse_args()

    vh_ids, vh_vecs, _ = load_embeddings(args.vh_dir)
    vl_ids, vl_vecs, _ = load_embeddings(args.vl_dir)

    # Convert to bytes for exact comparison
    as_bytes = vl_vecs.view(dtype="S" + str(vl_vecs.shape[1] * vl_vecs.itemsize))
    unique_vecs, counts = np.unique(as_bytes, return_counts=True)

    num_total = len(vl_vecs)
    num_unique = len(unique_vecs)
    num_dups = num_total - num_unique
    print(f"Total VL embeddings: {num_total}")
    print(f"Unique embeddings: {num_unique}")
    print(f"Duplicate embeddings: {num_dups}")
    if args.max_queries is not None and args.max_queries > 0:
        print(f"Limiting query vectors to the first {args.max_queries}.")
        if args.max_queries < len(vh_ids):
            indices = np.random.choice(len(vh_ids), args.max_queries, replace=False)
            vh_ids = [vh_ids[i] for i in indices]
            vh_vecs = vh_vecs[indices]
        else:
            print(
                f"Warning: max_queries ({args.max_queries}) is greater than or equal to the total number of queries ({len(vh_ids)}). Using all queries."
            )

    print(
        f"Processing {len(vh_ids):,} heavy chain and {len(vl_ids):,} light chain sequences"
    )

    # try to load FAISS
    faiss_idx_path = args.vl_dir / "faiss_ivfpq.index"
    if not args.no_faiss_index and faiss_idx_path.exists():
        print("Loading FAISS index …")
        cpu_index = faiss.read_index(str(faiss_idx_path))
        index = cpu_index  # Default to CPU index

        # Attempt to move the index to GPU
        if faiss.get_num_gpus() > 0:
            print(f"Found {faiss.get_num_gpus()} GPU(s). Attempting to use GPU 0...")
            try:
                res = (
                    faiss.StandardGpuResources()
                )  # Initialize GPU resources for a single GPU

                # --- Optional: Configure GpuClonerOptions for more control ---
                # The provided documentation mentions GpuClonerOptions, which can control
                # aspects like using float16 for storage or computations.
                # Example:
                # cloner_options = faiss.GpuClonerOptions()
                # cloner_options.useFloat16 = True # For GpuIndexIVFFlat storage or GpuIndexIVFPQ intermediate calcs
                # cloner_options.useFloat16CoarseQuantizer = False
                # gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index, cloner_options)
                # --- End Optional ---

                # Transfer the CPU index to GPU 0 (the first GPU)
                gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
                index = gpu_index  # Use the GPU index for subsequent operations
                print("Successfully moved FAISS index to GPU.")
            except RuntimeError as e:  # Catch specific Faiss/CUDA errors if possible
                print(
                    f"Failed to move FAISS index to GPU: {e}. Using CPU index instead."
                )
            except Exception as e:  # Catch any other unexpected errors
                print(
                    f"An unexpected error occurred while moving index to GPU: {e}. Using CPU index instead."
                )
        else:
            print("No GPUs found by FAISS. Using CPU index.")

        search_start = time.time()
        D, I = search_faiss(index, vh_vecs, args.top_k)
    else:
        if args.no_faiss_index:
            print(
                "Skipping FAISS index by user request – brute‑force cosine (may be slow)."
            )
        else:
            print("FAISS index not found – brute‑force cosine (may be slow).")
        search_start = time.time()

        D, I = brute_cosine(vh_vecs, vl_vecs, args.top_k, use_gpu=args.gpu_brute_force)
    search_time = time.time() - search_start
    print(f"Search completed in {search_time:.2f} seconds")

    # optional sigmoid (monotonic, not needed for ranking)
    if args.sigmoid:
        import scipy.special

        D = scipy.special.expit(D)

    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh, delimiter=",")
        writer.writerow(["vh_id", "vl_id", "rank", "score"])
        for q, (row_i, row_d) in enumerate(zip(I, D)):
            for rank, (j, score) in enumerate(zip(row_i, row_d), 1):
                if j < 0:
                    # FAISS returns -1 for no match, prevents erroneously writing out final vl_id as match
                    continue
                writer.writerow([vh_ids[q], vl_ids[j], rank, f"{score:.4f}"])

    print(f"Saved top‑{args.top_k} pairs to {args.out} (CSV format)")
