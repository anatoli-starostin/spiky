#!/usr/bin/env bash
# exp_c22 — the LUT-vs-MLP verdict at MATCHED PARAMETERS and n=12 (#75).
#
# exp_c19 found the LUT far less seed-sensitive than a 2x256 MLP (variance ratio 0.11x),
# but that comparison had two weaknesses: the MLP had 2.6x the actor parameters, and n=6
# per arm gives almost no power. This fixes both.
#
# PART (a) PARAM-MATCHED MLP. Target: the LUT actor's 28,032 params.
#     2x153  ->  17*153 + 153 + 153^2 + 153 + 153*12 + 12  =  28,164   (+0.47%)
#     1x934  ->  17*934 + 934 + 934*12 + 12                =  28,032   (exact)
#   Chosen: 2x153. The exact one-layer match changes DEPTH as well as width, so it would
#   differ from exp_c19's 2x256 in two ways at once; 2x153 changes only the width, making
#   this a clean capacity control against exp_c19 rather than a new architecture. 0.47% of
#   parameters is not a capacity story. Everything else is exp_c19's setup verbatim.
#
# PART (b) n=12 BOTH ARMS. Six new LUT seeds (6-11) on exp_c18's exact config, joining the
# six already run; twelve MLP seeds (0-11) at the matched width.
#
# ORDER IS LUT-FIRST, deliberately: it is only 6 runs and it completes the LUT arm to n=12,
# which stands on its own as a spread measurement even if the sweep is cut short. The MLP
# arm needs all 12 to be worth anything.
#
# ~18 runs at 3 concurrent: roughly 2 waves of LUT (~90 min) + 4 waves of MLP (~130 min).
set -u
cd "$(dirname "$0")"

C21=../exp_c21_seed4_long/run_long.log
PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
LUT="../exp_c09_lut_sac/lut_sac.py"
MLP="../exp_c19_mlp_sac_control/mlp_sac.py"

echo "waiting for exp_c21 to finish ($(date -u +%FT%TZ))"
until grep -q "LONG RUN EVAL DONE" "$C21" 2>/dev/null; do sleep 60; done
echo "exp_c21 done — starting $(date -u +%FT%TZ)"

SLACK_TASK="${SLACK_TASK:-16660dd4}"
nohup setsid "$PY" -u slack_bar.py --task "$SLACK_TASK" --interval 150 \
      > slack_bar.log 2>&1 &
BAR_PID=$!

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

MAXJOBS=3
HIDDEN=153

LUT_COMMON="--addressing hyperplane --hyperplane-init anchor_pairs \
        --hyperplane-anchor-policy canonical_full_coverage \
        --forward-mode hard --nap 6 --tph 32 --heads 1 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

MLP_COMMON="--iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --hidden $HIDDEN --eval-every 500 --eval-episodes 20"

pids=()
throttle() {
  while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
    wait -n 2>/dev/null || true
    alive=()
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    pids=("${alive[@]}")
  done
}

echo "=== PART b1: LUT seeds 6-11 (extends exp_c18's arm to n=12) ==="
for seed in 6 7 8 9 10 11; do
  throttle
  echo "=== launch LUT seed $seed  $(date -u +%FT%TZ) ==="
  nohup $PY -u "$LUT" --seed "$seed" $LUT_COMMON \
        --tag "_c22_lut_s${seed}" > "cell_lut_s${seed}.log" 2>&1 &
  pids+=($!)
  sleep 25
done

echo "=== PART a: param-matched MLP 2x${HIDDEN} (28,164 params), seeds 0-11 ==="
for seed in 0 1 2 3 4 5 6 7 8 9 10 11; do
  throttle
  echo "=== launch MLP seed $seed  $(date -u +%FT%TZ) ==="
  nohup $PY -u "$MLP" --seed "$seed" $MLP_COMMON \
        --tag "_c22_mlp${HIDDEN}_s${seed}" > "cell_mlp_s${seed}.log" 2>&1 &
  pids+=($!)
  sleep 25
done

# Trainer PIDs only -- a bare `wait` would also wait on the Slack bar, which does not exit
# until the evals below have run. That deadlock cost exp_c19 2.5 h of idle GPU.
if [ "${#pids[@]}" -gt 0 ]; then wait "${pids[@]}" 2>/dev/null || true; fi
echo "ALL 18 RUNS DONE $(date -u +%FT%TZ)"

echo "evaluating $(date -u +%FT%TZ)"
$PY -u collect.py
echo "MATCHED POWER EVAL DONE $(date -u +%FT%TZ)"
