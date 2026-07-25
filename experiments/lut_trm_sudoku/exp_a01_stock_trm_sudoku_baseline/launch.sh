#!/usr/bin/env bash
# exp_a01 — stock TRM Sudoku-Extreme baseline: LAUNCH (in-cage GPU training).
#
# Prereqs done in setup: repo cloned, isolated venv (torch 2.9.1+cu130, all deps), Sudoku-Extreme
# dataset built, and the pure-PyTorch AdamATan2 drop-in (adam_atan2.py) installed into the venv
# (the fused CUDA kernel is unbuildable/unrunnable on sm_120 — see README).
#
# In-cage GPU work, frictionless. Launched detached in the PERSISTENT host tmux server (the caged
# tmux client reaches it via the shared /tmp socket; the host server runs the payload uncaged &
# persistent, so it survives session rotation — a bare `&` inside sbox dies with --die-with-parent).
#
# wandb runs OFFLINE with all dirs redirected to writable paths (the cage's HOME is read-only
# except ~/projects, /tmp, ~/.cache). expandable_segments avoids fragmentation OOM.
set -euo pipefail

REPO_DIR="$HOME/projects/TinyRecursiveModels"
EXP_DIR="$HOME/projects/spiky/experiments/lut_trm_sudoku/exp_a01_stock_trm_sudoku_baseline"
LOG="$EXP_DIR/stdout.log"
RUN_NAME="exp_a01_trm_mlp_sudoku"

# global_batch_size=512 (stock is 768; reduced to fit the 32GB RTX 5090 — 768 OOMs. No grad-accum
# in the harness, so this is the largest stable batch: ~22GB peak, ~9GB headroom). Documented in
# config.json / README as a hardware-forced deviation.
CMD="cd $REPO_DIR && \
export WANDB_MODE=offline WANDB_SILENT=true WANDB_DIR=/tmp/wandb-a01 WANDB_DATA_DIR=/tmp/wandb-a01/data WANDB_CACHE_DIR=\$HOME/.cache/wandb \
       MPLCONFIGDIR=/tmp/mpl-$RUN_NAME OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
mkdir -p /tmp/wandb-a01/data \$HOME/.cache/wandb && \
.venv/bin/python -u pretrain.py \
 arch=trm data_paths='[data/sudoku-extreme-1k-aug-1000]' evaluators='[]' \
 global_batch_size=512 epochs=50000 eval_interval=5000 \
 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
 arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
 +run_name=$RUN_NAME ema=True"

sbox tmux new-session -d -s "$RUN_NAME" "bash -lc \"$CMD > $LOG 2>&1; echo === EXIT=\$? === >> $LOG\""
echo "Launched tmux session '$RUN_NAME'. Track:  tail -f $LOG   |   sbox tmux attach -t $RUN_NAME"
