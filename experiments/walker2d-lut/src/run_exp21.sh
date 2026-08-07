#!/usr/bin/env bash
# exp21: STACKED LIFLayer actor (17->32->32->6) + exp19-style MLP-exp critic. 3 seeds in PARALLEL.
# Results -> exp21_liflayer-32-32-mlpexpcrit/.  (arch takes no --tables-per-head.)
set -uo pipefail
cd "$(dirname "$0")"                        # experiments/walker2d-lut/src
BASE=..
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache MPLCONFIGDIR=/tmp/mpl
PY=~/projects/spiky/.venv/bin/python
D="$BASE/exp21_liflayer-32-32-mlpexpcrit"
mkdir -p "$D"

( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done > "$D/agg.gpu" 2>/dev/null ) &
SAMP=$!
start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u train.py --algo ppo --arch liflayer_mlpexpcrit --envs 8192 --graph \
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
