#!/usr/bin/env bash
# exp_a01 — stock TRM Sudoku-Extreme baseline: LAUNCH (in-cage GPU training).
#
# Assumes setup.sh has completed (repo cloned, venv built, dataset built). This is in-cage
# GPU work and is frictionless. It launches the stock TRM-MLP training detached in a persistent
# tmux session with its own log, so it survives session rotation (per feedback-launching-experiments:
# a bare `&` inside sbox dies with --die-with-parent; the persistent host tmux server keeps it alive).
set -euo pipefail

REPO_DIR="$HOME/projects/TinyRecursiveModels"
EXP_DIR="$HOME/projects/spiky/experiments/lut_trm_sudoku/exp_a01_stock_trm_sudoku_baseline"
LOG="$EXP_DIR/stdout.log"
RUN_NAME="exp_a01_trm_mlp_sudoku"

# Stock TRM-MLP Sudoku command (unmodified except run_name); H_cycles=T=3, L_cycles=n=6, EMA on.
CMD="cd $REPO_DIR && source .venv/bin/activate && \
MPLCONFIGDIR=/tmp/mpl-$RUN_NAME python -u pretrain.py \
 arch=trm \
 data_paths='[data/sudoku-extreme-1k-aug-1000]' \
 evaluators='[]' \
 epochs=50000 eval_interval=5000 \
 lr=1e-4 puzzle_emb_lr=1e-4 weight_decay=1.0 puzzle_emb_weight_decay=1.0 \
 arch.mlp_t=True arch.pos_encodings=none \
 arch.L_layers=2 arch.H_cycles=3 arch.L_cycles=6 \
 +run_name=$RUN_NAME ema=True"

# Launch inside the persistent host tmux server via the cage (frictionless GPU passthrough).
sbox tmux new-session -d -s "$RUN_NAME" "bash -lc \"$CMD > $LOG 2>&1; echo === EXIT=\$? === >> $LOG\""
echo "Launched tmux session '$RUN_NAME'. Track with:  tail -f $LOG   |   sbox tmux attach -t $RUN_NAME"
