#!/usr/bin/env bash
# L=16, N=64, 32 tables (2T = 64 = N, deliberately at the full-rank boundary of the input Jacobian),
# 256 cells, nap 8, read_top_n 2, tau 0.5 trainable, a_i = b_i = 1/sqrt(L*N) computed in the model.
#
# 2026-09-21, second configuration (task 8f6c0c37): SEED 0 ONLY -- the seed-1/seed-2 runs are dropped
# from the queue, not merely skipped -- and TABLE dropout p=0.25 on by default for every new run, with
# the mask PINNED for the whole weight update (one mask per update, held across all T inner relaxation
# steps and the vjp, f and g alike). train_paired.py verifies the pin at run start and re-checks the mask
# fingerprint on EVERY step, raising if anything resampled.
#
# The dropout-free runs already finished (bp s0/s1/s2, pcA s0/s1/s2 and pcalmB s0) stay as the matched
# no-dropout controls; the dropout runs carry the wandb tag td025 so the two families are separable
# inside the group. BP is re-run WITH dropout because the finished BP runs are dropout-free and the
# comparison has to be like-for-like.
#
# Runs OUTSIDE the cage because online W&B needs network. Idempotent: a run whose runs/<name>/run.json
# exists is skipped, so this is resumable.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_BASE_URL=https://api.wandb.ai WANDB_PROJECT=Spiky WANDB_ENTITY=anatoli-starostin-relocation
export WANDB_DIR=$HOME/.cache/wandb
TD=${TD:-0.25}                 # table dropout for every run queued below

run () {  # run <name> <extra args...>
  local name=$1; shift
  if [ -e "runs/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date)"
  $PY -u train_paired.py --depth 16 --width 64 --tables 32 --steps 2000 --batch 128 --probe-every 50 \
      --seed 0 --table-dropout "$TD" --tags "td025,dropout" --name "$name" "$@" \
      > "runs_logs/$name.log" 2>&1
  echo "== $name exit $? $(date)"
}
mkdir -p runs_logs

# --- the three main dropout runs, matched at T = 2L = 32, seed 0 -----------------------------------------
run "bp-L16-N64-s0-td025"            --arm bp
run "pcA-L16-N64-T32-s0-td025"       --arm pcA    --T 32
run "pcalmB-L16-N64-T32-s0-td025"    --arm pcalmB --T 32
echo "== MAIN DROPOUT RUNS DONE $(date)"

# --- T sweep (the direct test of the routing-conditioning argument), same dropout, seed 0 ----------------
run "pcA-L16-N64-T16-s0-td025"       --arm pcA    --T 16
run "pcA-L16-N64-T64-s0-td025"       --arm pcA    --T 64
run "pcalmB-L16-N64-T16-s0-td025"    --arm pcalmB --T 16
run "pcalmB-L16-N64-T64-s0-td025"    --arm pcalmB --T 64
echo "== SWEEP DONE $(date)"
