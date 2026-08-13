#!/usr/bin/env bash
# exp_c29 wave 1 — RESUME after the arm set was cut from four to two mid-flight (#75).
#
# WHY THIS FILE EXISTS. run_sweep.sh was executing when `random` and `clumped` were
# dropped. Editing a running bash script is not safe -- bash reads the file lazily by
# byte offset, so changing its length under itself can land execution in the middle of a
# token. So the original driver was killed and this one took over. Its background
# trainers were nohup'd children and survived the kill, which is exactly what was wanted:
# clumped s0, none s1 and grid s1 were already burning GPU and are allowed to finish.
#
# WHAT IT DOES
#   1. waits for the three in-flight trainers to exit (by CHECKPOINT + "done:" marker,
#      never by pgrep -- a -f pattern matches this script's own command line, which is
#      how exp_c25's watchdog span for nine hours over finished work);
#   2. runs the only two wave-1 cells that had not started: none s2 and grid s2;
#   3. CPU-evaluates and instruments every wave-1 checkpoint that exists, including the
#      salvaged random s0 and clumped s0 -- compute already spent is compute worth
#      measuring, even for a dropped arm;
#   4. writes SWEEP_DONE, which releases wave 2, which releases wave 3.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="./const_lut_sac.py"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

COMMON="--addressing anchors --anchor-policy balanced --forward-mode hard \
        --nap 6 --tph 64 --heads 1 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

INFLIGHT="clumped_s0 none_s1 grid_s1"
TODO="none_s2 grid_s2"
ALL="none_s0 grid_s0 random_s0 clumped_s0 none_s1 grid_s1 none_s2 grid_s2"

echo "resume driver up $(date -u +%FT%TZ); waiting on:$INFLIGHT"
MAXWAIT=$((6 * 3600))
waited=0
while true; do
  pending=""
  for c in $INFLIGHT; do
    grep -q "^done:" "cell_${c}.log" 2>/dev/null || pending="$pending $c"
  done
  [ -z "$pending" ] && break
  if [ "$waited" -ge "$MAXWAIT" ]; then
    echo "WARNING: still waiting on$pending after ${MAXWAIT}s; proceeding anyway"
    break
  fi
  sleep 60
  waited=$((waited + 60))
done
echo "in-flight runs finished $(date -u +%FT%TZ); launching:$TODO"

MAXJOBS=3
pids=()
for c in $TODO; do
  arm="${c%_s*}"; seed="${c##*_s}"
  while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
    wait -n 2>/dev/null || true
    alive=()
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    pids=("${alive[@]}")
  done
  echo "=== launch arm=$arm seed=$seed  $(date -u +%FT%TZ) ==="
  nohup $PY -u "$TRAIN" --seed "$seed" --constants "$arm" $COMMON \
        --tag "_c29_${arm}_s${seed}" > "cell_${arm}_s${seed}.log" 2>&1 &
  pids+=($!)
  sleep 25
done

wait
echo "ALL WAVE-1 RUNS DONE $(date -u +%FT%TZ)"

for c in $ALL; do
  A="lut_sac_c29_${c}_actor.npz"
  [ -f "$A" ] || { echo "MISSING $A"; continue; }
  $PY -u eval_const_cpu.py "$A" --episodes 100 2>&1 | grep -v "^Failed to import"
  $PY -u bit_usage.py "$A" --episodes 100 2>&1 | grep -v "^Failed to import"
done
echo "WAVE-1 EVALS DONE $(date -u +%FT%TZ)"
date -u +%FT%TZ > SWEEP_DONE
