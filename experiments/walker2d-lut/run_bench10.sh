#!/usr/bin/env bash
# 3 PPO seeds IN PARALLEL, 768 updates (2x bench3). Writes bench10/.
# FIX vs run_bench3.sh: wait ONLY on the 3 train.py PIDs (not bare `wait`, which also
# blocked on the infinite GPU sampler) — so the total-wall timer is captured correctly.
set -uo pipefail
cd /home/astarostin/projects/walker2d_gpu
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=~/projects/spiky/.venv/bin/python
mkdir -p bench10

# global GPU sampler (util,mem; nounits so awk -F, is clean)
( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 1; done > bench10/agg.gpu 2>/dev/null ) &
SAMP=$!

start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u train.py --algo ppo --arch hyperlut2_t64 --envs 8192 --graph --updates 768 --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897 --ent-coef 0.0 --target-kl 0.02 --norm-returns --seed "$s" \
      --out "bench10/ppo_s${s}.json" > "bench10/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"                      # ONLY the 3 training PIDs — sampler excluded
parallel_wall=$((SECONDS - start))
kill $SAMP 2>/dev/null

echo "PARALLEL_WALL_S=${parallel_wall}"
awk -F, '{u+=$1;n++; if($1>mu)mu=$1; if($2>mem)mem=$2} END{printf "AGG_GPU mean=%.0f%% max=%.0f%% maxmem=%.0fMB n=%d\n",u/n,mu,mem,n}' bench10/agg.gpu
echo "ALL DONE $(date +%T)"
