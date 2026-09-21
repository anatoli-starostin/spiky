#!/usr/bin/env bash
# Depth scaling under plain SGD (task 2661acbe piece B). The hypothesis this tests: BP+SGD degrades with
# depth because its interior update is attenuated by the Jacobian product, while PC+SGD stays flat because
# its per-layer targets are local and never compose. BP+Adam at each L is the reference ceiling, so we can
# see how much of BP's depth robustness is purely Adam.
#
# L in {2,4,8} x {BP, PC-A, PC-ALM-B} x lr in {1e-2, 3e-2} + BP-Adam@1e-3 per L = 21 runs.
# T = 2L, so the PC cost grows with depth -- that is the honest accounting, not a handicap.
# N=64, 32 tables, seed 0, 500 steps, pinned, table dropout 0, local logging only, nothing at L=16.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_MODE=disabled
unset WANDB_BASE_URL
mkdir -p runs_depth_logs

run () {
  local name=$1; shift
  if [ -e "runs_depth/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date +%T)"
  $PY -u train_paired.py --width 64 --tables 32 --steps 500 --batch 128 --probe-every 25 \
      --seed 0 --table-dropout 0 --clamp pinned --train-eval --out-dir runs_depth --name "$name" "$@" \
      > "runs_depth_logs/$name.log" 2>&1
  local rc=$?
  echo "== $name exit $rc $(date +%T)"
  if grep -qiE "nan|traceback" "runs_depth_logs/$name.log" 2>/dev/null; then
    echo "!! $name reported nan/traceback -- check its log"
  fi
}

for L in 3 4 8; do
  T=$((2 * L))
  run "depth-L$L-bp-adam1e-3"  --depth $L --arm bp     --optimizer adam --lr 1e-3
  for lr in 1e-2 3e-2; do
    run "depth-L$L-bp-sgd$lr"      --depth $L --arm bp      --optimizer sgd --lr $lr
    run "depth-L$L-pcA-sgd$lr"     --depth $L --arm pcA     --optimizer sgd --lr $lr --T $T
    run "depth-L$L-pcalmB-sgd$lr"  --depth $L --arm pcalmB  --optimizer sgd --lr $lr --T $T
  done
done
echo "== DEPTH SWEEP DONE $(date +%T)"
