#!/bin/bash
# Worker: launch d24-lutffn-confn1-projmuonfp8-lrrs-1xh100-s0 (v2 + LR rescale on + fp8/Muon projections).
# nohup-detached, wandb ONLINE, NUM_ITERATIONS=5568 in env (also pinned in the launcher). Then wait for step 27
# and dump the verification facts. KILLS the run immediately if the horizon auto-derives.
N=$HOME/projects/spiky/experiments/lut_nanochat/nanochat
cd "$N" || exit 2
[ "$(git -C "$N" rev-parse --short HEAD)" = "0bdac787" ] || { echo "ABORT: repo not at 0bdac787"; exit 2; }
export PATH="$N/.venv/bin:/usr/local/cuda/bin:$PATH" CUDA_HOME=/usr/local/cuda TORCH_EXTENSIONS_DIR="$HOME/.cache/torch_extensions_lutorch_ex"
unset WANDB_MODE
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); echo "GPU_NOW_MiB=$U"
[ "$U" -gt 3000 ] && { echo "ABORT: GPU busy"; exit 3; }
pgrep -f "d24_lutffn_confn1_projmuonfp8_lrrs" >/dev/null && { echo "ABORT: a projmuon run is already running"; exit 3; }
"$N/.venv/bin/python" -I - <<'PY' || { echo "ABORT: wandb collision/check failed"; exit 4; }
import wandb
runs = wandb.Api().runs("anatoli-starostin-relocation/spiky-nanochat")
hits = [r.id for r in runs if r.name == "d24-lutffn-confn1-projmuonfp8-lrrs-1xh100-s0" or "projmuon" in r.id]
print("WANDB_COLLISIONS =", hits); raise SystemExit(1 if hits else 0)
PY
LOG=$HOME/projects/spiky/experiments/lut_nanochat/results/projmuonfp8_lrrs.nohup.log
NUM_ITERATIONS=5568 nohup bash runs/lut_ffn_d24_projmuonfp8_lrrs_1xh100.sh > "$LOG" 2>&1 &
PID=$!; disown "$PID" 2>/dev/null || true
echo "PID=$PID"; echo "LOG=$LOG"; date -u +"START_UTC=%FT%TZ"
PEAK=0; ST=""
for i in $(seq 1 180); do
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$u" -gt "$PEAK" ] && PEAK=$u
  if grep -q "Calculated number of iterations" "$LOG"; then ST=HORIZON_BUG; break; fi
  if ! kill -0 $PID 2>/dev/null; then ST=PROCESS_GONE; break; fi
  if grep -qE "Traceback|loss: (nan|inf)|out of memory" "$LOG"; then ST=TROUBLE; break; fi
  if grep -qE "^step 00027/" "$LOG"; then ST=REACHED_27; break; fi
  sleep 5
done
echo "STATUS=$ST"
if [ "$ST" = HORIZON_BUG ]; then
  echo "HORIZON AUTO-DERIVED -> KILLING"; kill -TERM $PID; BP=$(pgrep -f "scripts.base_train.*projmuonfp8" | head -1); kill -INT $BP; sleep 60; kill -KILL $BP 2>/dev/null
fi
echo "=HORIZON="; grep -E "number of iterations" "$LOG"
echo "=BANNER="; grep -E "LUT-FFN enabled|^total |FP8 training|Scaling LRs|batch_lr_scale|Scaling weight decay|Scaling the LR" "$LOG" | cut -c1-170
echo "=WANDB="; grep -oE "https://wandb.ai/[^ ]*runs/[^ ]*" "$LOG" | head -1
echo "=STEPS 0-5="; grep -E "^step 0000[0-5]/" "$LOG" | cut -c1-110
echo "=DT steps 7-27="; grep -E "^step 000(0[7-9]|1[0-9]|2[0-7])/" "$LOG" | grep -oE "dt: [0-9.]+" | grep -oE "[0-9.]+" | tr "\n" " "; echo
echo "=LAST="; grep -E "^step [0-9]+/" "$LOG" | tail -1 | cut -c1-175
echo "=PEAK_NVSMI_MiB=$PEAK"; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
echo "=PROCS="; ps -p $PID -o pid,etime --no-headers; pgrep -af "scripts.base_train.*projmuonfp8" | cut -c1-90
