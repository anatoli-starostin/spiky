#!/usr/bin/env bash
# VM launcher for exp_n_0239 (pure-LUT control, 48k). Runs in a tmux session on the VM.
set -u
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export NANOCHAT_ROOT="$HOME/projects/nanochat"
export MPLCONFIGDIR="$HOME/.cache/mpl_0239"
export TZ=Asia/Jerusalem
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
RUN="$HOME/projects/spiky/experiments/ffn_replacement/runs_corrected/exp_n_0239_pureLUT0193_control_48k_seed1"
cd "$RUN" || exit 97
echo "[vm-0239] starting at $(date)"
"$HOME/venv/bin/python" train.py > "$RUN/train.log" 2>&1
printf '\n=== EXIT %s\n' "$?" >> "$RUN/train.log"
echo "[vm-0239] done at $(date)"
