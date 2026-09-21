#!/usr/bin/env bash
# Main sweep: L=16, N=64, 32 tables (2T = 64 = N, deliberately at the full-rank boundary), 256 cells, nap 8,
# read_top_n 2, tau 0.5 trainable. a_i = b_i = 1/sqrt(L*N) = 0.03125 (computed in the model, never hardcoded).
# Three arms x 3 seeds at T = 2L = 32, plus a T in {L, 2L, 4L} sweep at seed 0, plus the warm-start blend
# diagnostics for arm B. Online W&B, group pcalm-lut-paired. Runs OUTSIDE the cage (the cage has no network).
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_BASE_URL=https://api.wandb.ai WANDB_PROJECT=Spiky WANDB_ENTITY=anatoli-starostin-relocation
export WANDB_DIR=$HOME/.cache/wandb

run () {  # run <name> <extra args...>
  local name=$1; shift
  if [ -e "runs/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date)"
  $PY -u train_paired.py --depth 16 --width 64 --tables 32 --steps 2000 --batch 128 --probe-every 50 \
      --name "$name" "$@" > "runs_logs/$name.log" 2>&1
  echo "== $name exit $? $(date)"
}
mkdir -p runs_logs

for s in 0 1 2; do run "bp-L16-N64-s$s"            --arm bp     --seed $s; done
for s in 0 1 2; do run "pcA-L16-N64-T32-s$s"       --arm pcA    --T 32 --seed $s; done
for s in 0 1 2; do run "pcalmB-L16-N64-T32-s$s"    --arm pcalmB --T 32 --seed $s; done
# T sweep (the direct test of the routing-conditioning argument), seed 0
run "pcA-L16-N64-T16-s0"      --arm pcA    --T 16 --seed 0
run "pcA-L16-N64-T64-s0"      --arm pcA    --T 64 --seed 0
run "pcalmB-L16-N64-T16-s0"   --arm pcalmB --T 16 --seed 0
run "pcalmB-L16-N64-T64-s0"   --arm pcalmB --T 64 --seed 0
echo "== SWEEP DONE $(date)"
