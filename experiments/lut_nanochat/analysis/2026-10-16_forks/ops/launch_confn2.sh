#!/bin/bash
# Launch confn2 (v2 config, read_top_n=2) OUT of cage: wandb ONLINE, nohup-detached, horizon pinned 5568.
set -u
NC=/home/astarostin/projects/lut_nanochat_wt/experiments/lut_nanochat/nanochat
cd "$NC" || exit 2
export PATH="$NC/.venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda
unset WANDB_MODE
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
echo "GPU_NOW_MiB=$U"; [ "$U" -gt 3000 ] && { echo "ABORT: GPU busy"; exit 3; }
"$NC/.venv/bin/python" -I - <<'PY' || { echo "ABORT: wandb name collision / check failed"; exit 4; }
import wandb
runs = wandb.Api().runs("anatoli-starostin-relocation/spiky-nanochat")
hits = [r.id for r in runs if r.name == "d24-lutffn-confn2-1xh100-s0" or "confn2" in r.id]
print("WANDB_COLLISIONS =", hits)
raise SystemExit(1 if hits else 0)
PY
LOG="$NC/../results/confn2.nohup.log"
NUM_ITERATIONS=5568 nohup bash runs/lut_ffn_d24_confn2_1xh100.sh > "$LOG" 2>&1 &
PID=$!; disown "$PID" 2>/dev/null || true
echo "CONFN2_PID=$PID"; echo "CONFN2_LOG=$LOG"
