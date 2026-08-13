#!/usr/bin/env bash
# Pure evaluation of the SHIPPED w=0.3 checkpoint at several output-quantiser resolutions.
# No training. Headline tier throughout: 1024 envs x 2000 steps, matched physics 100/50,
# deterministic mean action.
#
# The question: this policy was TRAINED at 22 levels. Does emitting it at 8 cost anything?
# If not, the spiking readout can keep TAU_M_OUT=10 (~296-tick episode) instead of 31.26
# (~609 ticks, dmax 236 against a 255 cap).
#
# Each invocation also re-runs the two out-quant-OFF arms, which gives a repeated-control
# noise floor for free.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant/qat_n22_l2/levels
CK=/home/astarostin/projects/ckpt_backups/exp23_qat/l2w0p3_s0.pt
PY=/home/astarostin/projects/spiky/.venv/bin/python
PROG=/home/astarostin/work/slack-facade/progress.py
TASK=3498f639

cd "$SRC"
export PYTHONPATH=/home/astarostin/projects/spiky/experiments/walker2d-lut/_qat_deps
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p "$OUT"

LEVELS=(8 12 16 22 6)
H=$(python3 "$PROG" start --task "$TASK" --label "exp23 output-level eval (w=0.3 ckpt)" \
      --stats "N in ${LEVELS[*]}")
start=$SECONDS
i=0
for N in "${LEVELS[@]}"; do
  $PY -u eval_qat_ckpt.py --ckpt "$CK" --envs 1024 --steps 2000 \
      --quant-ticks 128 --quant-sigma 1.0 --out-quant-levels "$N" \
      > "$OUT/eval_n$N.log" 2>&1
  i=$((i+1))
  echo "--- N=$N (rc=$?)"
  grep -E "out-quant" "$OUT/eval_n$N.log" | grep -v "load on device"
  python3 "$PROG" update "$H" --step "$i" --total "${#LEVELS[@]}" \
      --stats "N=$N done · $((SECONDS-start))s" >/dev/null 2>&1
done
python3 "$PROG" done "$H" --text "output-level sweep complete in $((SECONDS-start))s" >/dev/null 2>&1
echo "ALL_DONE $((SECONDS - start))s"
