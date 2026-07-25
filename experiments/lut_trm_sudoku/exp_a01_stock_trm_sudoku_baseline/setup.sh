#!/usr/bin/env bash
# exp_a01 — stock TRM Sudoku-Extreme baseline: SETUP (network steps).
#
# These steps CROSS THE SANDBOX (git clone / pip install / dataset build all need network,
# which the sbox cage does not have) and therefore require Anatoli's approval. Run them
# UNCAGED (do NOT wrap in sbox). After this completes, launch.sh runs the training in-cage.
#
# One intentional deviation from the reference README: we KEEP the box's torch 2.9.1+cu130
# (Blackwell sm_120-capable) instead of the reference's torch-nightly-cu126. Everything else
# is stock. flash-attn is intentionally NOT installed (unnecessary for the MLP/Sudoku path).
set -euo pipefail

REPO_DIR="$HOME/projects/TinyRecursiveModels"
DATA_DIR="data/sudoku-extreme-1k-aug-1000"

# 1) Get the reference implementation (MIT, archived read-only).
if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/SamsungSAILMontreal/TinyRecursiveModels "$REPO_DIR"
fi
cd "$REPO_DIR"

# 2) Dedicated venv, but reuse the box's Blackwell-capable torch rather than cu126 nightly.
#    (uv is available on the box; falls back to python -m venv + pip if preferred.)
uv venv .venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel setuptools
pip install "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu130   # Blackwell sm_120
# Repo deps MINUS torch (already installed above). Edit requirements.txt to drop the torch line,
# or install the known set explicitly:
pip install -r requirements.txt || true
pip install einops omegaconf hydra-core pydantic argdantic coolname wandb huggingface_hub tqdm
pip install --no-cache-dir --no-build-isolation adam-atan2   # builds a CUDA ext for sm_120

# 3) Build the Sudoku-Extreme dataset (1000 train + 1000 aug; ~423K test).
python dataset/build_sudoku_dataset.py --output-dir "$DATA_DIR" --subsample-size 1000 --num-aug 1000

echo "SETUP DONE. Repo: $REPO_DIR ; dataset: $REPO_DIR/$DATA_DIR"
echo "Next: run launch.sh (in-cage GPU training, frictionless)."
