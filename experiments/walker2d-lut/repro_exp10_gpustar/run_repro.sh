#!/usr/bin/env bash
# Reproduction of exp10_lut-anchor-pair-t32 on gpustar (RTX 5090).
#
# Flags are taken VERBATIM from exp10/config.json (provenance bench12/t32), i.e. the same
# line src/run_bench12.sh runs for tph=32:
#   arch fastlut, tables-per-head 32, envs 8192, rollout 32, updates 768,
#   cosine LR -> 3e-5, logstd-min -1.897, ent-coef 0, target-kl 0.02, --norm-returns, --graph
# 3 seeds IN PARALLEL, matching the original protocol.
#
# NOTE: ppo.py resolves --out relative to dirname(ppo.py) (= src/), so absolute paths are
# passed here; that is the only deviation from run_bench12.sh and it is purely I/O routing.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/repro_exp10_gpustar
cd "$SRC"
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=/home/astarostin/projects/spiky/.venv/bin/python

mkdir -p "$OUT"

# GPU sampler (util,mem), 2s cadence — same as run_bench12.sh
( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done > "$OUT/agg.gpu" 2>/dev/null ) &
SAMP=$!

start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u train.py --algo ppo --arch fastlut --tables-per-head 32 --envs 8192 --graph \
      --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897 \
      --ent-coef 0.0 --target-kl 0.02 --norm-returns --out "$OUT/ppo_s${s}.json" \
      > "$OUT/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"                     # ONLY the 3 training PIDs — sampler excluded
wall=$((SECONDS - start))
kill $SAMP 2>/dev/null

echo "PARALLEL_WALL_S=${wall}"
awk -F, '{u+=$1;n++;if($1>mu)mu=$1;if($2>mem)mem=$2} END{printf "AGG_GPU mean=%.0f%% max=%.0f%% maxmem=%.0fMB n=%d\n",u/n,mu,mem,n}' "$OUT/agg.gpu"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
