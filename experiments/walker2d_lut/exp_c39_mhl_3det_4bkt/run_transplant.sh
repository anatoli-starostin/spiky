#!/usr/bin/env bash
# exp_c39 diagnosis — the decisive test: was it the INIT or the RL TRAJECTORY?
#
# Nothing in the finished checkpoints can separate these two, because by the end the init
# has been overwritten and the trajectory has been run. But the trainer is deterministic
# from PRNGKey(seed), and the actor's init consumes exactly one key (`ka`), so the two can
# be exchanged independently:
#
#   A  --seed 0 --actor-seed 2   the WINNER's starting parameters on a LOSER's RL stream
#   B  --seed 2 --actor-seed 0   a LOSER's starting parameters on the WINNER's RL stream
#   C  --seed 1 --actor-seed 2   the winner's init on the other loser's stream (replicate)
#
# Read-off:
#   A and C take off, B does not   -> the init is what matters; fix initialisation.
#   B takes off, A and C do not    -> the init is irrelevant; the RL trajectory decides.
#   all three flat / all take off  -> neither alone; it is the interaction.
#
# Everything else -- critic init, env resets, exploration noise, replay sampling -- stays
# attached to --seed, so each of these is a real, self-consistent run and not a Frankenstein
# of two checkpoints.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local NAME=$1 SEED=$2 ASEED=$3 TAG="_tp_$1"
  echo "=== $NAME: rl-stream seed $SEED, actor init seed $ASEED ($(date -u +%H:%M:%SZ)) ==="
  $PY -u mhl_sac_transplant.py --seed "$SEED" --actor-seed "$ASEED" --tag "$TAG" \
      > "tp_${NAME}.log" 2>&1
  $PY -u eval_mhl_cpu.py "mhl_sac${TAG}_actor.npz" --episodes 100 \
      >> "tp_${NAME}.log" 2>&1
  echo "=== $NAME: done ($(date -u +%H:%M:%SZ)) ==="
}

cell A 0 2 &
cell B 2 0 &
cell C 1 2 &
wait

touch TRANSPLANT_DONE
echo "=== transplant done $(date -u +%H:%M:%SZ) ==="
