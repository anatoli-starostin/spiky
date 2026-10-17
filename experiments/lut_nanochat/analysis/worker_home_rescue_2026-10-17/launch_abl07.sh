#!/usr/bin/env bash
# Live launch: exp_n_abl_07 (Gen-2 fast hard, row 2.1) on the worker, BARE (no sbox).
set -e
. "$HOME/.wandb_env"                     # WANDB_BASE_URL / WANDB_ENTITY -> tracker online (worker reaches nucstar directly over tailnet)
RUN="$HOME/projects/spiky/experiments/ffn_replacement/lut_ablation/exp_n_abl_07_B16k_fast_hard_topk0_nap8_tph128_seed1"
cd "$RUN"
# --- guards ---
[ -e metrics.csv ] && { echo "GUARD: metrics.csv exists, abort"; exit 3; }
[ -e train.log ] && { echo "GUARD: train.log exists, abort"; exit 3; }
if nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; then
  echo "GUARD: a process is already on the GPU, abort"; exit 4
fi
# --- env (no PYTHONPATH: the worker's editable spiky already = the branch checkout at 40d604ab) ---
export NANOCHAT_ROOT="$HOME/projects/nanochat"
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export MPLCONFIGDIR="$HOME/.cache/mpl_abl07"
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
exec "$HOME/venv/bin/python" -u train.py > train.log 2>&1
