#!/usr/bin/env bash
# exp_a02 — fast-iteration TRM Sudoku baseline: LAUNCH (in-cage GPU, frictionless).
# Prereqs: TRM repo + venv + full dataset from exp_a01, AdamATan2 drop-in in the venv, and the 5k
# test subset built by subsample_test.py. Detached in the persistent host tmux server.
set -euo pipefail
REPO_DIR="$HOME/projects/TinyRecursiveModels"
EXP_DIR="$HOME/projects/spiky/experiments/lut_trm_sudoku/exp_a02_fast_baseline"
LOG="$EXP_DIR/stdout.log"
RUN_NAME="exp_a02_fast"

CMD="cd $REPO_DIR && \
export WANDB_MODE=offline WANDB_SILENT=true WANDB_DIR=/tmp/wandb-a02 WANDB_DATA_DIR=/tmp/wandb-a02/data WANDB_CACHE_DIR=\$HOME/.cache/wandb \
       MPLCONFIGDIR=/tmp/mpl-$RUN_NAME OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
mkdir -p /tmp/wandb-a02/data \$HOME/.cache/wandb && \
.venv/bin/python -u pretrain.py \
 arch=trm data_paths='[data/sudoku-extreme-1k-aug-1000-testsub5k]' evaluators='[]' \
 global_batch_size=768 epochs=10000 eval_interval=2000 \
 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
 arch.mlp_t=True arch.pos_encodings=none arch.L_layers=2 arch.H_cycles=2 arch.L_cycles=3 \
 +run_name=$RUN_NAME ema=True"

sbox tmux new-session -d -s "$RUN_NAME" "bash -lc \"$CMD > $LOG 2>&1; echo === EXIT=\$? === >> $LOG\""
echo "Launched tmux '$RUN_NAME'. Track: tail -f $LOG | sbox tmux attach -t $RUN_NAME"
