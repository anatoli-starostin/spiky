#!/usr/bin/env bash
# Re-run exp19's 3 seeds with --save-model so there are actual WEIGHTS to deploy.
#
# WHY THIS IS NEEDED. `ppo.py` never saved a checkpoint (no torch.save anywhere), by the
# repo's own policy -- experiments/walker2d-lut/README.md: "Checkpoints (*.pt) — never
# tracked; reproduce from each config.json". So exp19 produced learning curves but no
# model. The observation-normalisation statistics were not saved either, and the policy is
# trained on normalised observations, so weights alone would not be a usable model.
#
# Flags are IDENTICAL to run_exp19.sh; the only additions are --save-model (new, default
# off, so exp00-19 are unaffected) and a separate output directory so the original run's
# records are not overwritten.
#
# This doubles as a reproducibility check: same seeds, same config, same host -> the finals
# should land on exp19's 5400.5 / 5869.2 / 5389.6.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp19_lut-lse-expmlpcrit-t32/rerun_ckpt
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
      --out "$OUT/ppo_s${s}.json" --save-model "$OUT/actor_s${s}.pt" \
      > "$OUT/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
echo "PARALLEL_WALL_S=$((SECONDS - start))"
ls -la "$OUT"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
