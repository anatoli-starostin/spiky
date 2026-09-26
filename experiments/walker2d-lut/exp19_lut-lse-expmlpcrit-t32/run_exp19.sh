#!/usr/bin/env bash
# exp19 — exp17's log-sum-exp ACTOR + the exp10 MLP critic with its FINAL LINEAR READOUT
# replaced by the matching sum-scaled log-sum-exp over the 256 penultimate units:
#
#   plain (exp10, exp17):  value = sum_i w_i*h_i + b
#   exp19:                 value = T*tau_c*log( (1/T) sum_i exp(w_i*h_i / tau_c) ) + b
#
# The critic BACKBONE is untouched (obs -> [256,256] Tanh, same orthogonal init, same RNG
# draw), so the control for this experiment is exactly exp17 (same actor, plain linear
# critic readout, 5403.8 +- 34.4). The only difference between exp19 and exp17 is the
# critic's readout.
#
# This is deliberately NOT exp18: there the critic was swapped for a LUT, which is unstable
# on its own (exp13-15, exp18: seed sd ~1000+) and confounded the measurement. Here the
# strong MLP backbone is held fixed.
#
# tau_c init 0.25 — chosen by measurement (design_tau_critic.py): the value function stays
# 97.6% correlated with exp17's while the exponential remains measurably live, so tau_c has
# real gradient. tau_c >= 1 would be inert (a null result by construction).
#
# Every other flag is exp10's, verbatim: 8192 envs, 768 updates, bench7 recipe, 3 seeds.
set -uo pipefail
# Resolve everything relative to this script, so a clone or a worktree anywhere runs
# against its own tree. Override PY if your interpreter is not the repo venv.
OUT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$OUT/../src" && pwd)
REPO=$(cd "$OUT/../../.." && pwd)
cd "$SRC"
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=${PY:-$REPO/.venv/bin/python}

mkdir -p "$OUT"

( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done > "$OUT/agg.gpu" 2>/dev/null ) &
SAMP=$!

start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u ppo.py --arch fastlut_lse_sum_expmlpcrit --tables-per-head 32 \
      --envs 8192 --graph --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 \
      --logstd-min -1.897 --ent-coef 0.0 --target-kl 0.02 --norm-returns \
      --out "$OUT/ppo_s${s}.json" > "$OUT/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
wall=$((SECONDS - start))
kill $SAMP 2>/dev/null

echo "PARALLEL_WALL_S=${wall}"
awk -F, '{u+=$1;n++;if($1>mu)mu=$1;if($2>mem)mem=$2} END{printf "AGG_GPU mean=%.0f%% max=%.0f%% maxmem=%.0fMB n=%d\n",u/n,mu,mem,n}' "$OUT/agg.gpu"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
