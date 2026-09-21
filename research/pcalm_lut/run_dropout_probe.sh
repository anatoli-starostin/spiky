#!/usr/bin/env bash
# BP-only dropout probe (task 5fd246bb). N=64 / 32 tables / 2000 steps, L in {4, 16}, using LightMHL's
# EXISTING table dropout (head_dropout_rate) at p in {0.1, 0.25} plus the p=0 baselines, and the
# residual-stream variant as a secondary control. Local logging only: no WANDB_BASE_URL in the
# environment means the tracker turns itself off and history still lands in runs_dropout/<name>/run.json,
# so this runs INSIDE the cage.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_MODE=disabled
unset WANDB_BASE_URL
mkdir -p runs_dropout_logs

# seeds to run; seed 0 keeps the original unsuffixed names. A cell's effect here is ~0.005 test acc and
# one 2,000-row accuracy estimate has sd ~0.0075, so a single seed cannot answer the depth question.
SEEDS=${SEEDS:-"0 1 2"}

run () {
  local name=$1; shift
  if [ -e "runs_dropout/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date +%T)"
  $PY -u train_paired.py --arm bp --width 64 --tables 32 --steps 2000 --batch 128 --probe-every 50 \
      --train-eval --out-dir runs_dropout --name "$name" "$@" > "runs_dropout_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

for s in $SEEDS; do
  sfx=""; [ "$s" = 0 ] || sfx="-s$s"
  for L in 4 16; do
    run "drop-none-L$L$sfx"       --depth $L --seed $s
    for p in 0.1 0.25; do
      run "drop-table-p$p-L$L$sfx"    --depth $L --seed $s --table-dropout $p
      run "drop-resid-p$p-L$L$sfx"    --depth $L --seed $s --residual-dropout $p
    done
  done
done
echo "== DROPOUT PROBE DONE $(date +%T)"
