#!/usr/bin/env bash
# SGD vs Adam lr probe (task 27f350f3), under the standing debug rule: 500 steps, L=4, N=64, 32 tables,
# T=8, seed 0, table dropout p=0.25, pinned clamping, local logging only, nothing at L=16.
#
# Plain SGD (momentum 0, no weight decay) on the WEIGHTS only -- the dual ascent on lambda keeps its own
# alpha rule and the inner relaxation keeps its derived eta_h. BP is run at every lr as the matched
# control, so optimiser is never confounded with method. The Adam 1e-3 rows are re-run here rather than
# reused, so that every row carries the new applied-update instrumentation.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_MODE=disabled
unset WANDB_BASE_URL
mkdir -p runs_sgd_logs

run () {
  local name=$1; shift
  if [ -e "runs_sgd/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date +%T)"
  $PY -u train_paired.py --depth 4 --width 64 --tables 32 --steps 500 --batch 128 --probe-every 25 \
      --seed 0 --table-dropout 0.25 --clamp pinned --train-eval --out-dir runs_sgd --name "$name" "$@" \
      > "runs_sgd_logs/$name.log" 2>&1
  local rc=$?
  echo "== $name exit $rc $(date +%T)"
  # stop early on a run that blew up rather than burning the budget on the rest of that arm
  if grep -qiE "nan|inf" "runs_sgd_logs/$name.log" 2>/dev/null; then
    echo "!! $name produced nan/inf -- see its log"
  fi
}

for lr in 3e-4 1e-3 3e-3 1e-2 3e-2; do
  run "sgd-bp-lr$lr"       --arm bp      --optimizer sgd --lr $lr
  run "sgd-pcA-lr$lr"      --arm pcA     --optimizer sgd --lr $lr --T 8
  run "sgd-pcalmB-lr$lr"   --arm pcalmB  --optimizer sgd --lr $lr --T 8
done
# the Adam reference row, same config, new instrumentation
run "adam-bp-lr1e-3"       --arm bp      --optimizer adam --lr 1e-3
run "adam-pcA-lr1e-3"      --arm pcA     --optimizer adam --lr 1e-3 --T 8
run "adam-pcalmB-lr1e-3"   --arm pcalmB  --optimizer adam --lr 1e-3 --T 8
echo "== SGD PROBE DONE $(date +%T)"
