# SynPair

Code to train and evaluate contrastive learning model for synthetically pairing unpaired OAS.

Training scripts can be run using UV environment for ease.

Indexing building and pairing requires the FAISS library, which is system specific and can be installed via conda [[https://github.com/facebookresearch/faiss/blob/main/INSTALL.md]]

### Using uv (recommended)

This project is set up to use `uv` for fast, reproducible Python execution without manually managing virtual environments.

- Install `uv`:
  - macOS (Homebrew): `brew install uv`
  - Pip: `pip install uv`

- Run any script (non-Faiss) with project dependencies resolved automatically:

```bash
uv run src/synpair/train.py --help
uv run src/synpair/embed.py --help
```

- Optional: create a local venv managed by uv and activate it:

```bash
uv venv
source .venv/bin/activate
```

Note: FAISS is system-specific. Install via conda as per the FAISS docs if you plan to use indexing/search.

### Base model

You can change the base masked protein LM by editing `HF_MODEL` in `src/synpair/constants.py` to any Hugging Face masked protein language model.

### Examples

See `pairing_pipeline.sh` for end-to-end example commands covering training, embedding, indexing (FAISS), and pairing.