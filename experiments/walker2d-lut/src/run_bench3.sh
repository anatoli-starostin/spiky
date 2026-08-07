#!/usr/bin/env bash
# 3 PPO seeds IN PARALLEL on one H100 — parallel-vs-sequential wall-clock test.
# Same config as bench2 PPO: MLP, N=8192, --graph, 384 updates. Writes bench3/.
set -uo pipefail
cd /home/astarostin/projects/walker2d_gpu
export WARP_CACHE_PATH=/tmp/warp_cache
PY=~/projects/spiky/.venv/bin/python
mkdir -p bench3

# ONE global GPU sampler (util + mem) for the whole concurrent run
( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 1; done > bench3/agg.gpu 2>/dev/null ) &
SAMP=$!

start=$SECONDS
for s in 0 1 2; do
  $PY -u train.py --algo ppo --arch mlp --envs 8192 --graph --updates 384 --seed "$s" \
      --out "bench3/ppo_s${s}.json" > "bench3/ppo_s${s}.log" 2>&1 &
done
wait                                   # all 3 train.py finish
parallel_wall=$((SECONDS - start))
kill $SAMP 2>/dev/null

echo "PARALLEL_WALL_S=${parallel_wall}"
awk '{u+=$1;n++; if($1>mu)mu=$1; if($2>mem)mem=$2} END{printf "AGG_GPU mean=%.0f%% max=%.0f%% maxmem=%.0fMB (n=%d)\n",u/n,mu,mem,n}' bench3/agg.gpu
echo "ALL DONE $(date +%T)"
