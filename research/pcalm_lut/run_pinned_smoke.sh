#!/usr/bin/env bash
# L=4 smoke of OUTPUT-PINNED clamping (task abc84212): all three arms, table dropout p=0.25 with the mask
# pinned across the inner loop, seed 0. Local logging only -- no WANDB_BASE_URL in the environment means
# the tracker turns itself off -- so this runs INSIDE the cage.
#
# The two no-pin (clamp=data) cells are the matched controls for the collapse question: table norms and
# the residual-branch ratio were never logged in the finished L=16 runs, so the only honest way to say
# whether arm A was collapsing is to run the same instrumentation at this scale under both clampings.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_MODE=disabled
unset WANDB_BASE_URL
mkdir -p runs_pinned_logs

run () {
  local name=$1; shift
  if [ -e "runs_pinned/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date +%T)"
  $PY -u train_paired.py --depth 4 --width 64 --tables 32 --steps 2000 --batch 128 --probe-every 25 \
      --seed 0 --table-dropout 0.25 --train-eval --out-dir runs_pinned --name "$name" "$@" \
      > "runs_pinned_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

run "pin-bp-L4"        --arm bp     --clamp pinned
run "pin-pcA-L4"       --arm pcA    --clamp pinned --T 8
run "pin-pcalmB-L4"    --arm pcalmB --clamp pinned --T 8
# controls under the OLD clamping, same everything else
run "nopin-pcA-L4"     --arm pcA    --clamp data   --T 8
run "nopin-pcalmB-L4"  --arm pcalmB --clamp data   --T 8
echo "== PINNED SMOKE DONE $(date +%T)"
