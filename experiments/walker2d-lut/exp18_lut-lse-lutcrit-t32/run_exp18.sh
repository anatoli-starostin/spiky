#!/usr/bin/env bash
# exp18 — does giving the CRITIC the same exponential geometry make the actor's tau use
# the max-like regime? Two arms, 3 seeds each, run as 2 SEQUENTIAL groups (3 seeds in
# parallel within a group), so per-group throughput stays comparable to exp10/exp17.
#
#   TREATMENT  exp18_lut-lse-lutcrit-t32      actor LSE-sum + critic LSE-sum  (two taus)
#   CONTROL    exp18ctl_lut-lse-plaincrit-t32 actor LSE-sum + critic PLAIN sum (one tau)
#
# The control is the point of this experiment. exp13-15 already established that a plain
# LUT critic costs a lot against an MLP critic (tph32: 2358.6 vs 5488.4), so comparing the
# treatment to exp17 alone would just re-measure "LUT critic vs MLP critic". Holding the
# LUT critic fixed and toggling ONLY its readout isolates the exponential.
#
# Critic tph = 32, matching the actor and matching exp13's config exactly, so exp13
# (plain actor + plain critic) is directly comparable as the both-plain corner.
#
# Every other flag is exp10's, verbatim: 8192 envs, 768 updates, bench7 recipe.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
BASE=/home/astarostin/projects/spiky/experiments/walker2d-lut
cd "$SRC"
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=/home/astarostin/projects/spiky/.venv/bin/python

run_group () {                     # $1 = arch, $2 = output dir
  local ARCH="$1" OUT="$BASE/$2"
  mkdir -p "$OUT"
  ( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done > "$OUT/agg.gpu" 2>/dev/null ) &
  local SAMP=$!
  local start=$SECONDS
  local pids=()
  for s in 0 1 2; do
    $PY -u train.py --algo ppo --arch "$ARCH" --tables-per-head 32 --envs 8192 --graph \
        --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897 \
        --ent-coef 0.0 --target-kl 0.02 --norm-returns --out "$OUT/ppo_s${s}.json" \
        > "$OUT/ppo_s${s}.log" 2>&1 &
    pids+=("$!")
  done
  wait "${pids[@]}"
  kill $SAMP 2>/dev/null
  echo "GROUP ${ARCH} WALL_S=$((SECONDS - start))"
  awk -F, '{u+=$1;n++;if($1>mu)mu=$1;if($2>mem)mem=$2} END{printf "  GPU mean=%.0f%% max=%.0f%% maxmem=%.0fMB\n",u/n,mu,mem}' "$OUT/agg.gpu"
}

total=$SECONDS
run_group fastlut_lse_sum2            exp18_lut-lse-lutcrit-t32
run_group fastlut_lse_sum2_plaincrit  exp18ctl_lut-lse-plaincrit-t32
echo "TOTAL_WALL_S=$((SECONDS - total))"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
