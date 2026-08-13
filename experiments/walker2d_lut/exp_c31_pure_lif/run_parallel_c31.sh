#!/usr/bin/env bash
# exp_c31 — the 3 seeds run CONCURRENTLY on the one 5090, replacing run_sweep_c31.sh.
#
# WHY THIS EXISTS. run_sweep_c31.sh queued the seeds because "1 GPU -> sequential" is the
# standing convention in this chapter. Measured, that convention is wrong for this model:
#
#   peak GPU memory, whole trainer process   1.84 GB   (3 seeds = 5.5 GB of 32 GB)
#   3-way concurrent throughput              2.33x     (controlled 1-vs-3 test)
#
# The utilisation figure that would have talked us out of it (~70% from nvidia-smi) is
# misleading: `utilization.gpu` reports the fraction of TIME at least one kernel was
# resident, not SM occupancy, so a stream of small kernels reads as "70% busy" while most
# of the card sits idle. The throughput test measures the thing we actually care about.
#
# SEED 0 IS NOT RESTARTED. It was ~1,700 iterations in when this switch was made, and
# discarding that to make the launch tidy would cost more than the tidiness is worth. The
# original driver (the bash script, PID 32151) was killed; its python child was left alive
# and is adopted here by polling its log for the completion line. Seed 0 therefore keeps
# JAX's default 75% preallocation while seeds 1 and 2 run with preallocation OFF -- a
# memory-management difference only, with no effect on any number either produces.
#
# Concurrency does not touch determinism: the three processes are independent, and each is
# still bit-reproducible from its own seed under the XLA flags below.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# One seed, end to end. Seeds 1 and 2 train here; seed 0 is already training elsewhere and
# only needs its eval, so `train` is skipped for it.
cell () {
  local S=$1 TRAIN=$2
  local TAG="_c31_s${S}"
  if [ "$TRAIN" = train ]; then
    echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
      $PY -u pure_lif_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  else
    echo "=== seed $S: adopted, waiting for the running trainer ($(date -u +%H:%M:%SZ)) ==="
    # The inherited process writes "done: best MJX ..." as its last line. Poll for it
    # rather than for the PID: a PID can be reused, and the log line is the real event.
    until grep -q "^done: best MJX" "cell_s${S}.log" 2>/dev/null; do sleep 60; done
  fi
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    $PY -u eval_pure_cpu.py "pure_lif_sac${TAG}_actor.npz" --episodes 100 \
    >> "cell_s${S}.log" 2>&1
  echo "=== seed $S: done ($(date -u +%H:%M:%SZ)) ==="
}

cell 0 adopt &
cell 1 train &
cell 2 train &
wait

touch SWEEP_DONE_C31
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
