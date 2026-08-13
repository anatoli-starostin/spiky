#!/usr/bin/env bash
# exp16 — exp10 + a trainable exponential output transform on the actor mean:
#     mean -> c + exp(mean / t)     c free, t > 0 (softplus)
#
# Fork of exp10_lut-anchor-pair-t32. EVERY other flag is exp10's, verbatim from its
# config.json: fastlut anchor-pair actor tph=32 + MLP critic, bench7 recipe (truncation
# bootstrap, return-norm, KL early-stop, cosine LR -> 3e-5, log_std floor), 8192 envs,
# 768 updates, 3 seeds in parallel. Only --arch changes: fastlut -> fastlut_exp.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp16_lut-anchor-pair-t32-expout
cd "$SRC"
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=/home/astarostin/projects/spiky/.venv/bin/python

mkdir -p "$OUT"

( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done > "$OUT/agg.gpu" 2>/dev/null ) &
SAMP=$!

start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u train.py --algo ppo --arch fastlut_exp --tables-per-head 32 --envs 8192 --graph \
      --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897 \
      --ent-coef 0.0 --target-kl 0.02 --norm-returns --out "$OUT/ppo_s${s}.json" \
      > "$OUT/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
wall=$((SECONDS - start))
kill $SAMP 2>/dev/null

echo "PARALLEL_WALL_S=${wall}"
awk -F, '{u+=$1;n++;if($1>mu)mu=$1;if($2>mem)mem=$2} END{printf "AGG_GPU mean=%.0f%% max=%.0f%% maxmem=%.0fMB n=%d\n",u/n,mu,mem,n}' "$OUT/agg.gpu"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
