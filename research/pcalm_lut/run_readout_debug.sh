#!/usr/bin/env bash
# Readout-scale debugging (task dabc7a53), under the standing debug rule: 500 steps, L=4, T=8, seed 0,
# table dropout p=0.25, eval every 25, local logging only, no W&B, nothing at L=16.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_MODE=disabled
unset WANDB_BASE_URL
mkdir -p runs_debug_logs

run () {
  local name=$1; shift
  if [ -e "runs_debug/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date +%T)"
  $PY -u train_paired.py --depth 4 --width 64 --tables 32 --steps 500 --batch 128 --probe-every 25 \
      --seed 0 --table-dropout 0.25 --train-eval --out-dir runs_debug --name "$name" "$@" \
      > "runs_debug_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

run "dbg-bp"        --arm bp      --clamp pinned
run "dbg-pcA"       --arm pcA     --clamp pinned --T 8
run "dbg-pcalmB"    --arm pcalmB  --clamp pinned --T 8
# the control that separates "the readout's gradient path is broken" from "the states carry no signal":
# readout trained by BP, interior by PC-A.
run "dbg-hybrid"    --arm hybrid  --clamp pinned --T 8
echo "== READOUT DEBUG DONE $(date +%T)"
