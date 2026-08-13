#!/usr/bin/env bash
# Retrain the exp19 policy under DEPLOYMENT-MATCHED physics, so the exported actor is
# trained on the same observations and dynamics the walker2d-viz server produces.
#
# Two gaps between our training env and any gymnasium deployment, both measured:
#   1. OBSERVATION. gymnasium's Walker2d clips qvel to [-10, 10] before returning the obs;
#      warp_env did not. On a trained exp19 policy 9.0% of velocity components exceed |10|
#      (peak 73.9), so the deployed policy sees a vector it never saw in training, and its
#      normalisation statistics were fitted to the unclipped distribution. --obs-clip-vel 10
#   2. SOLVER. warp_env used iterations=10, ls_iterations=8; stock MuJoCo (what gymnasium
#      uses) defaults to 100 and 50. Training physics were substantially softer than
#      deployment physics. --solver-iters 100 --ls-iters 50
#
# Everything else is exp19's, verbatim. Both flags default to the old behaviour, so
# exp00-19 remain bit-reproducible.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp19_lut-lse-expmlpcrit-t32/deploy_matched
cd "$SRC"
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=/home/astarostin/projects/spiky/.venv/bin/python

mkdir -p "$OUT"

start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u train.py --algo ppo --arch fastlut_lse_sum_expmlpcrit --tables-per-head 32 \
      --envs 8192 --graph --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 \
      --logstd-min -1.897 --ent-coef 0.0 --target-kl 0.02 --norm-returns \
      --obs-clip-vel 10.0 --solver-iters 100 --ls-iters 50 \
      --out "$OUT/ppo_s${s}.json" --save-model "$OUT/actor_s${s}.pt" \
      > "$OUT/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
echo "PARALLEL_WALL_S=$((SECONDS - start))"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
