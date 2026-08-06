#!/usr/bin/env bash
# exp_c30 — 3 seeds of Walker2d SAC with the JAX LIF-detector actor (#75).
#
# One arm, three seeds. There is no none/grid style contrast here: the comparison is
# against exp_c18's hyperplane cell at the SAME nap6/tph32 shape (6 seeds, deterministic,
# 4308.0 +/- 500.1), which is already on disk. Running our own baseline again would spend
# 1.5 GPU-hours to reproduce a number we have.
#
# NOT PARAM-MATCHED, on purpose and stated up front: 87,361 actor params against 49,152
# for the LUT actors. The ordered-pair channel P alone is 55,488 of that.
#
# Determinism on, as everywhere in this chapter since exp_c17: a fixed seed reproduces
# bit-for-bit, so the seed spread is the only noise left.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="./lif_sac.py"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Same SAC recipe as exp_c29/exp_c09 -- only the actor front-end differs.
COMMON="--nap 6 --tph 32 --heads 1 --iters 10000 --envs 64 --rollout 1 --updates 32 \
        --batch 512 --warmup 500 --row-clip 1.0 --eval-every 500 --eval-episodes 20 \
        --eps-start 2.0 --eps-end 0.3 --eval-eps 0.3"

SEEDS="0 1 2"
MAXJOBS=3

echo "LIF-SAC 3-seed sweep starting $(date -u +%FT%TZ)"
echo "XLA_FLAGS=$XLA_FLAGS  CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

pids=()
for seed in $SEEDS; do
  while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
    wait -n 2>/dev/null || true
    alive=()
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    pids=("${alive[@]}")
  done
  echo "=== launch seed=$seed  $(date -u +%FT%TZ) ==="
  nohup $PY -u "$TRAIN" --seed "$seed" $COMMON --tag "_c30_s${seed}" \
        > "cell_s${seed}.log" 2>&1 &
  pids+=($!)
  sleep 25
done

wait
echo "ALL 3 LIF RUNS DONE $(date -u +%FT%TZ)"

for seed in $SEEDS; do
  A="lif_sac_c30_s${seed}_actor.npz"
  [ -f "$A" ] || { echo "MISSING $A"; continue; }
  $PY -u eval_lif_cpu.py "$A" --episodes 100 2>&1 | grep -v "^Failed to import"
done
echo "LIF EVALS DONE $(date -u +%FT%TZ)"
date -u +%FT%TZ > SWEEP_DONE_LIF
