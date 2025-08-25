
uv run src/synpair/train.py --data_dir data/paired --output_dir results --projection --lora --lr 1e-3 --compile --epochs 100  --batch 512 --proj_dim 128 --contrastive --gradient_accumulation_steps 8 

uv run src/synpair/embed.py \
    --csv data/paired/test/heavies.csv \
    --ckpt results/checkpoints/fe8mx1p0/last.ckpt \
    --out_dir data/paired/test/heavies_embeddings_contrastive \
    --use_vh --contrastive

uv run src/synpair/embed.py \
    --csv data/paired/test/lights.csv \
    --ckpt results/checkpoints/fe8mx1p0/last.ckpt \
    --out_dir data/paired/test/light_embeddings_contrastive \
    --use_vl --contrastive



mamba activate faiss

python src/synpair/build_index.py \
    --embed_dir data/paired/test/light_embeddings_contrastive \
    --out_dir data/paired/test/light_embeddings_contrastive \
    --gpu

python src/synpair/pair.py \
    --vh_dir data/paired/test/heavies_embeddings_contrastive \
    --vl_dir data/paired/test/light_embeddings_contrastive \
    --out data/paired/test/synthetic_pairs_20k_exact_contrastive.csv --top_k 5 --max_queries 20000 


