#!/usr/bin/env bash
# exp_c29, WAVE 3 — the same four arms at a DIFFERENT nap/tph split, param-matched (#75).
#
# THE INVARIANT. Learnable params = tph * 2^nap * 12 (the table is the whole policy; the
# anchor w/b are frozen buffers). Holding that equal to nap6/tph64's 49,152 forces
# tph * 2^nap = 4,096 -- which is ALSO the total row count. So every param-matched
# config owns exactly 4,096 rows; only the PARTITION changes:
#
#   nap5/tph128 128 tables x   32 rows   2,816 active/step   640 comparators
#   nap6/tph64   64 tables x   64 rows   1,536               384   <- waves 1 and 2
#   nap7/tph32   32 tables x  128 rows     832               224
#   nap8/tph16   16 tables x  256 rows     448               128
#   nap9/tph8     8 tables x  512 rows     240                72
#   nap10/tph4    4 tables x 1024 rows     128                40
#
# The two directions ask opposite questions. UP the ladder (higher nap, fewer tables)
# buys finer per-table resolution and a weaker ensemble; DOWN it (lower nap, more tables)
# buys more comparators and more tables voting, at more active work per step. Nothing
# buys more rows -- that total is pinned by the parameter match.
#
# Override with NAP= and TPH= in the environment. The script REFUSES to run a config
# whose table size differs from the reference, because the entire point is the match.
#
# WAITS on every sentinel in WAIT= so waves never contend for the GPU. Each newly queued
# wave must list the sentinels of all waves queued before it, or two of them will wake up
# together and fight over the card.
set -u
cd "$(dirname "$0")"

NAP="${NAP:-8}"
TPH="${TPH:-16}"
POLICY="${POLICY:-balanced}"     # match wave 1 by default, so the only change is nap/tph
TAGP="${TAGP:-c29k}"
LOGP="${LOGP:-cellk}"
SENT="${SENT:-SWEEP_DONE_CAPACITY}"
WAIT="${WAIT:-SWEEP_DONE SWEEP_DONE_CANONICAL}"

REF=$((64 * 64 * 12))
HAVE=$((TPH * (1 << NAP) * 12))
if [ "$HAVE" -ne "$REF" ]; then
  echo "REFUSING: nap$NAP/tph$TPH gives $HAVE learnable table params, not $REF."
  echo "Param-matched options: nap5/tph128 nap6/tph64 nap7/tph32 nap8/tph16 nap9/tph8 nap10/tph4"
  exit 2
fi
echo "nap$NAP/tph$TPH -> $HAVE learnable table params (matched), "\
     "$((TPH * (1 << NAP))) rows, $((TPH * NAP)) comparators, "\
     "$((TPH * 12 + TPH * NAP * 2)) active values/step"

MAXWAIT=$((24 * 3600))
waited=0
while true; do
  pending=""
  for s in $WAIT; do [ -f "$s" ] || pending="$pending $s"; done
  [ -z "$pending" ] && break
  if [ "$waited" -ge "$MAXWAIT" ]; then
    echo "ABORT: still waiting on$pending after ${MAXWAIT}s; not launching $TAGP"
    echo "aborted $(date -u +%FT%TZ)" > "$SENT"
    exit 1
  fi
  sleep 60
  waited=$((waited + 60))
done
echo "waited on [$WAIT]; starting $TAGP at $(date -u +%FT%TZ)"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="./const_lut_sac.py"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

COMMON="--addressing anchors --anchor-policy $POLICY --forward-mode hard \
        --nap $NAP --tph $TPH --heads 1 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

# Two arms only -- see the note in run_sweep_canonical.sh. `random` and `clumped` were
# dropped on 2026-08-02 before this wave started, so it never ran them at all.
ARMS="none grid"
SEEDS="0 1 2"
MAXJOBS=3

pids=()
n=0
for seed in $SEEDS; do
  for arm in $ARMS; do
    while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
      wait -n 2>/dev/null || true
      alive=()
      for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
      pids=("${alive[@]}")
    done
    n=$((n + 1))
    echo "=== launch $n/12  arm=$arm seed=$seed  nap$NAP/tph$TPH  $(date -u +%FT%TZ) ==="
    nohup $PY -u "$TRAIN" --seed "$seed" --constants "$arm" $COMMON \
          --tag "_${TAGP}_${arm}_s${seed}" > "${LOGP}_${arm}_s${seed}.log" 2>&1 &
    pids+=($!)
    sleep 25
  done
done

wait
echo "ALL 12 CAPACITY RUNS DONE $(date -u +%FT%TZ)"

for seed in $SEEDS; do
  for arm in $ARMS; do
    A="lut_sac_${TAGP}_${arm}_s${seed}_actor.npz"
    [ -f "$A" ] || { echo "MISSING $A"; continue; }
    $PY -u eval_const_cpu.py "$A" --episodes 100 2>&1 | grep -v "^Failed to import"
    $PY -u bit_usage.py "$A" --episodes 100 2>&1 | grep -v "^Failed to import"
  done
done
echo "CAPACITY EVALS DONE $(date -u +%FT%TZ)"
date -u +%FT%TZ > "$SENT"
