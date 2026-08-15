#!/usr/bin/env bash
# Re-run exp05's 3 seeds with --save-model, so the plain-MLP arm has actual WEIGHTS.
#
# WHY THIS IS NEEDED. `ppo.py` saved no checkpoint at the time exp05 ran, by the repo's own
# policy (walker2d-lut/README.md: "Checkpoints (*.pt) — never tracked; reproduce from each
# config.json"). exp05 therefore left learning curves but no model, and no
# observation-normalisation statistics — and the policy is trained on normalised
# observations, so weights alone would not be a usable model. `--save-model` writes both:
# the full state_dict plus obs_mean / obs_var / obs_count.
#
# Flags are exactly exp05's config.json, which is bench7 == the recipe exp10/exp17/exp19
# also use. The ONLY difference from run_exp19.sh is `--arch mlp` and no
# --tables-per-head: that is the whole point — a matched MLP arm.
#
# exp05's originals, as the sanity bar: 5392.0 / 6387.5 / 6076.8 (mean 5952.1).
#
# Output goes to its own directory so exp05's original records are untouched. OUT may be
# overridden to keep checkpoints out of a deployment checkout:
#     OUT=/somewhere/durable ./rerun_for_checkpoints.sh
set -uo pipefail
# Resolve everything relative to this script, so a clone or a worktree anywhere runs
# against its own tree. Override PY if your interpreter is not the repo venv.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$HERE/../src" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT=${OUT:-$HERE/rerun_ckpt}
cd "$SRC"
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=${PY:-$REPO/.venv/bin/python}

mkdir -p "$OUT"
echo "SRC=$SRC"
echo "OUT=$OUT"
echo "PY=$PY"

start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u ppo.py --arch mlp \
      --envs 8192 --graph --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 \
      --logstd-min -1.897 --ent-coef 0.0 --target-kl 0.02 --norm-returns \
      --out "$OUT/ppo_s${s}.json" --save-model "$OUT/actor_s${s}.pt" \
      > "$OUT/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
echo "PARALLEL_WALL_S=$((SECONDS - start))"
ls -la "$OUT"
echo "exp05 originals for comparison: 5392.0 / 6387.5 / 6076.8  (mean 5952.1)"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
