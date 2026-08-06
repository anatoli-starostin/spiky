#!/usr/bin/env bash
# exp_c29 — does giving frozen anchors a block of fixed CONSTANTS to compare against
# cure their magnitude blindness? (#75)
#
# FOUR ARMS x THREE SEEDS = 12 runs. Everything is held fixed except the sixteen numbers
# appended to the LUT's addressing input:
#
#   none     17-dim input, ordinary anchors. The magnitude-blind baseline, and the only
#            arm whose architecture is exp_c09's published one (verify_const.py check A
#            proves w and b come out bit-identical).
#   grid     33-dim; 16 evenly tiled thresholds -- exp_c28's "16 levels are enough" set.
#   random   33-dim; 16 irregular thresholds over the same range.
#   clumped  33-dim; 16 thresholds inside the central fifth of that range only.
#
# grid / random / clumped share IDENTICAL wiring at a given seed (check E), so those
# three differ in the constants and in nothing else whatsoever.
#
# CONFIG. anchors x hard, nap6/tph64/1 head -- the mid-capacity anchors cell. Chosen over
# nap6/tph32, which would match exp_c18's hyperplane reference exactly, because tph32
# returned 918 on one of exp_c13's three seeds: a config that collapses on a third of its
# draws is a bad instrument for an A/B, whatever else it is. Everything else follows
# exp_c13/exp_c18: 10,000 iters, envs 64, rollout 1, updates 32, batch 512, warmup 500,
# --row-clip 1.0, eval every 500.
#
# DETERMINISM IS ON. exp_c16 measured 999 return of run-to-run spread at a FIXED seed;
# exp_c17 traced it to the atomics scatter-add in the table-weight backward and killed it
# with these two environment variables (checkpoints came out bit-for-bit identical). The
# differences this experiment is looking for are far smaller than 999, so without these
# flags it could not resolve anything at all. Cost is ~+27% wall clock.
#
# ORDER IS SEED-MAJOR: all four arms at seed 0, then seed 1, then seed 2. If the sweep is
# cut short, what survives is complete arm coverage at fewer seeds rather than a couple of
# arms with three seeds and the rest with none -- the first is still an experiment.
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

# HISTORICAL NOTE -- this file no longer describes all of wave 1. It ran with
# ARMS="none grid random clumped" and launched six runs in that order before being
# stopped on 2026-08-02 at 13:0x UTC:
#   1 none s0 (finished 4431.1)   2 grid s0 (finished 4548.1)   3 random s0 (finished 4326.7)
#   4 clumped s0 (in flight)      5 none s1 (in flight)         6 grid s1 (in flight)
# `random` and `clumped` were then dropped. The three in-flight runs were left to finish
# rather than waste the compute already spent; the remaining none/grid seeds and all the
# CPU evals were completed by run_sweep_resume.sh, which is what wrote SWEEP_DONE.
# The list below is updated so a re-run of this file cannot resurrect the dropped arms.
ARMS="none grid"
SEEDS="0 1 2"
MAXJOBS=3            # ~7.5 GB each of 32 GB, as in exp_c13/exp_c18; 4-way crowds the box

echo "XLA_FLAGS=$XLA_FLAGS"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

pids=()
n=0
for seed in $SEEDS; do
  for arm in $ARMS; do
    # Throttle by WAITING ON A PID (the exp_c13 form) -- `pgrep | wc -l` mis-parses.
    while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
      wait -n 2>/dev/null || true
      alive=()
      for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
      pids=("${alive[@]}")
    done
    n=$((n + 1))
    echo "=== launch $n/12  arm=$arm seed=$seed  $(date -u +%FT%TZ) ==="
    nohup $PY -u "$TRAIN" --seed "$seed" --constants "$arm" $COMMON \
          --tag "_c29_${arm}_s${seed}" > "cell_${arm}_s${seed}.log" 2>&1 &
    pids+=($!)
    sleep 25   # stagger so the JIT compiles do not collide
  done
done

wait
echo "ALL 12 RUNS DONE $(date -u +%FT%TZ)"

# The CPU-reference eval and the bit-usage instrumentation, sequentially -- both are
# CPU-bound MuJoCo rollouts, so running them concurrently would only fight for cores.
for seed in $SEEDS; do
  for arm in $ARMS; do
    A="lut_sac_c29_${arm}_s${seed}_actor.npz"
    [ -f "$A" ] || { echo "MISSING $A"; continue; }
    $PY -u eval_const_cpu.py "$A" --episodes 100 2>&1 | grep -v "^Failed to import"
    $PY -u bit_usage.py "$A" --episodes 100 2>&1 | grep -v "^Failed to import"
  done
done
echo "EVALS DONE $(date -u +%FT%TZ)"
# Sentinel for slack_bar.py. A file, not a pgrep check: exp_c25's bar refresher matched
# its OWN command line with pgrep -f and span for 9 hours over finished work.
date -u +%FT%TZ > SWEEP_DONE
