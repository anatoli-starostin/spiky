#!/usr/bin/env bash
# Fair equal-data PPO-vs-SAC benchmark, N=8192, physics graph on, MLP.
# PPO 384 updates (~100M env-steps) vs SAC 10000 updates (~82M env-steps) — matched data budget.
set -uo pipefail
cd /home/astarostin/projects/walker2d_gpu
export WARP_CACHE_PATH=/tmp/warp_cache
PY=~/projects/spiky/.venv/bin/python
mkdir -p bench2

run() {  # algo updates seed extra...
  local algo=$1 upd=$2 seed=$3; shift 3
  local tag="bench2/${algo}_s${seed}"
  ( while true; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; sleep 2; done > "${tag}.gpu" 2>/dev/null ) &
  local samp=$!
  echo "=== ${algo} seed=${seed} updates=${upd} start $(date +%T) ==="
  $PY -u train.py --algo "$algo" --arch mlp --envs 8192 --graph --updates "$upd" --seed "$seed" \
      --out "${tag}.json" "$@" > "${tag}.log" 2>&1
  local ec=$?
  kill $samp 2>/dev/null
  awk '{s+=$1;n++;if($1>m)m=$1} END{if(n>0)printf "   GPU mean=%.0f%% max=%.0f%%\n",s/n,m}' "${tag}.gpu"
  echo "=== ${algo} seed=${seed} exit=${ec} $(date +%T) ==="
}

for s in 0 1 2; do run ppo 384 $s; done
for s in 0 1 2; do run sac 10000 $s --utd 4; done
echo "ALL DONE $(date +%T)"
