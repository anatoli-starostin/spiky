#!/usr/bin/env bash
# fastlut2 (FastMultiHeadLut anchor-pair actor + anchor-pair LUT critic) sweep: tph 32/64/128.
# Anchor-pair analogue of the exp10-12 runs but with a LUT critic instead of MLP.
# 3 SEQUENTIAL groups; within each group the 3 seeds run in PARALLEL (bench12 pattern).
# Outputs land directly in the convention exp folders (exp13/14/15).
set -uo pipefail
cd "$(dirname "$0")"                       # experiments/walker2d-lut/src
BASE=..                                     # experiments/walker2d-lut
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=~/projects/spiky/.venv/bin/python

declare -A DIR=( [32]=exp13_lut-anchor-pair-lutcrit-t32 \
                 [64]=exp14_lut-anchor-pair-lutcrit-t64 \
                 [128]=exp15_lut-anchor-pair-lutcrit-t128 )

total_start=$SECONDS
for tph in 32 64 128; do
  D="$BASE/${DIR[$tph]}"
  mkdir -p "$D"
  ( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done > "$D/agg.gpu" 2>/dev/null ) &
  SAMP=$!
  gstart=$SECONDS
  pids=()
  for s in 0 1 2; do
    $PY -u train.py --algo ppo --arch fastlut2 --tables-per-head "$tph" --envs 8192 --graph \
        --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897 \
        --ent-coef 0.0 --target-kl 0.02 --norm-returns --out "$D/ppo_s${s}.json" \
        > "$D/ppo_s${s}.log" 2>&1 &
    pids+=("$!")
  done
  wait "${pids[@]}"
  kill $SAMP 2>/dev/null
  echo "GROUP_t${tph}_WALL_S=$((SECONDS - gstart))"
  awk -F, -v t="$tph" '{u+=$1;n++;if($1>mu)mu=$1;if($2>mem)mem=$2} END{printf "GROUP_t%s_GPU mean=%.0f%% max=%.0f%% maxmem=%.0fMB\n",t,u/n,mu,mem}' "$D/agg.gpu"
  echo "=== group t${tph} done $(date +%T) ==="
done
echo "TOTAL_WALL_S=$((SECONDS - total_start))"
echo "ALL DONE $(date +%T)"
