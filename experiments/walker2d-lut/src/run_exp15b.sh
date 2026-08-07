#!/usr/bin/env bash
# exp15b: MIXED arch — anchor-pair LUT actor (tph=128) + hyperplane LUT critic (tph=64).
# 3 seeds in PARALLEL. Results -> exp15b_anchor-t128-actor_hyperplane-t64-critic/.
set -uo pipefail
cd "$(dirname "$0")"                        # experiments/walker2d-lut/src
BASE=..
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache MPLCONFIGDIR=/tmp/mpl
PY=~/projects/spiky/.venv/bin/python
D="$BASE/exp15b_anchor-t128-actor_hyperplane-t64-critic"
mkdir -p "$D"

( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done > "$D/agg.gpu" 2>/dev/null ) &
SAMP=$!
start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u train.py --algo ppo --arch fastlut_hypcrit --tables-per-head 128 --envs 8192 --graph \
      --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897 \
      --ent-coef 0.0 --target-kl 0.02 --norm-returns --out "$D/ppo_s${s}.json" \
      > "$D/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
kill $SAMP 2>/dev/null
echo "WALL_S=$((SECONDS - start))"
awk -F, '{u+=$1;n++;if($1>mu)mu=$1;if($2>mem)mem=$2} END{printf "GPU mean=%.0f%% max=%.0f%% maxmem=%.0fMB\n",u/n,mu,mem}' "$D/agg.gpu"
echo "ALL DONE $(date +%T)"
