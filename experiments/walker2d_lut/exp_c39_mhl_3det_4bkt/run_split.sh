#!/usr/bin/env bash
# exp_c39 diagnosis, round 2 — WHICH HALF of the init carries the outcome?
#
# Round 1 settled that the outcome follows the actor init wholesale: the winner's init
# rescued both losing RL streams (4002, 3455) and a loser's init destroyed the winning
# stream (971). But "the init" is two very different things bundled together:
#
#   the FRONT-END init   delay (half-normal, scale 4) and w_raw (N(-2.2, 0.5)) -- what the
#                        addressing function does, i.e. which observations land in which cell
#   the TABLE init       0.1*randn -- the initial action values sitting in those cells
#
# Every aggregate measure of the ADDRESSING was indistinguishable between seeds (no-spike
# rate, detector entropy, effective cells, pair agreement, zero dead detectors in all
# three), which points at the table -- but that is an argument from absence, and the
# addressing measures are aggregates that could hide fine structure. So measure it.
#
#   D  front-end from the WINNER, table from loser 0, on loser 0's stream
#   E  front-end from loser 0,     table from the WINNER, on loser 0's stream
#   F  front-end from the WINNER, table from loser 1, on loser 1's stream   (replicate of D)
#
# Read-off:
#   D and F take off, E does not  -> the FRONT-END init decides; fix delay/w initialisation.
#   E takes off, D and F do not   -> the TABLE init decides; the addressing is a red herring
#                                    and the fix is cheap (restart on table init alone).
#   all flat                      -> the two halves only work together; they must be
#                                    transplanted as a pair.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local NAME=$1 SEED=$2 ASEED=$3 TSEED=$4 TAG="_sp_$1"
  echo "=== $NAME: stream $SEED, front-end $ASEED, table $TSEED ($(date -u +%H:%M:%SZ)) ==="
  $PY -u mhl_sac_transplant.py --seed "$SEED" --actor-seed "$ASEED" \
      --table-seed "$TSEED" --tag "$TAG" > "sp_${NAME}.log" 2>&1
  $PY -u eval_mhl_cpu.py "mhl_sac${TAG}_actor.npz" --episodes 100 \
      >> "sp_${NAME}.log" 2>&1
  echo "=== $NAME: done ($(date -u +%H:%M:%SZ)) ==="
}

cell D 0 2 0 &
cell E 0 0 2 &
cell F 1 2 1 &
wait

touch SPLIT_DONE
echo "=== split done $(date -u +%H:%M:%SZ) ==="
