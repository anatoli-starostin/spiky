#!/usr/bin/env bash
# Retrain the exp05 plain-MLP baseline under DEPLOYMENT-MATCHED physics, so the exported
# actor is trained on the same observations and dynamics the walker2d-viz server produces.
# This is exp19's run_deploy_matched.sh with `--arch mlp` and no --tables-per-head.
#
# Two gaps between our training env and any gymnasium deployment, both measured — and both
# already paid for once, on exp19's first deploy artifact (see exp19's deploy/README.md:
# 4845.9 +- 1372.2 with a worst episode of 1097.5, against 6284.7 +- 319.5 after the fix):
#
#   1. OBSERVATION. gymnasium's Walker2d clips qvel to [-10, 10] before returning the obs;
#      warp_env does not. Measured on the exp05 checkpoints trained WITHOUT the flag,
#      9.96-14.87% of velocity components exceed |10| (peak 84.2) and more than half of all
#      timesteps have at least one clipped component, worth up to 6 sigma of normalised
#      input shift against stats fitted on the unclipped distribution. --obs-clip-vel 10
#   2. SOLVER. warp_env defaults to iterations=10, ls_iterations=8; stock MuJoCo (what
#      gymnasium uses) defaults to 100 and 50. --solver-iters 100 --ls-iters 50
#
# Everything else is exp05's config.json, verbatim. Both flags default to the old
# behaviour, so exp00-19 remain bit-reproducible.
#
# OUT may be overridden to keep checkpoints out of a deployment checkout:
#     OUT=/somewhere/durable ./run_deploy_matched.sh
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$HERE/../src" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT=${OUT:-$HERE/deploy_matched}
cd "$SRC"
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=${PY:-$REPO/.venv/bin/python}

mkdir -p "$OUT"
echo "SRC=$SRC"
echo "OUT=$OUT"

start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u ppo.py --arch mlp \
      --envs 8192 --graph --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 \
      --logstd-min -1.897 --ent-coef 0.0 --target-kl 0.02 --norm-returns \
      --obs-clip-vel 10.0 --solver-iters 100 --ls-iters 50 \
      --out "$OUT/ppo_s${s}.json" --save-model "$OUT/actor_s${s}.pt" \
      > "$OUT/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
echo "PARALLEL_WALL_S=$((SECONDS - start))"
ls -la "$OUT"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
