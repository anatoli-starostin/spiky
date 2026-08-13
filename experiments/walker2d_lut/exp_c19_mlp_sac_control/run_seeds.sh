#!/usr/bin/env bash
# exp_c19 — MLP-actor SAC control, 6 seeds, determinism on (#75).
#
# QUEUED, NOT IMMEDIATE. It blocks until exp_c18's whole pipeline reports DIAG DONE, so it
# cannot steal GPU from the study it is a control for. Waiting on exp_c18's *chain* rather
# than just its training also keeps the CPU-reference evals off a busy GPU, which would
# otherwise slow them without changing their (deterministic) result.
#
# If exp_c18 fails outright, this waits forever by design: a control measured while the
# thing it controls for is broken is worse than no control. Kill it deliberately instead.
set -u
cd "$(dirname "$0")"

C18=../exp_c18_seed_variance/chain_eval.log
PY="$HOME/projects/walker2d_mjx/.venv/bin/python"

echo "waiting for exp_c18 to finish ($(date -u +%FT%TZ))"
until grep -q "DIAG DONE" "$C18" 2>/dev/null; do sleep 60; done
echo "exp_c18 done — starting the MLP control $(date -u +%FT%TZ)"

# The bar is raised HERE, not at queue time: a bar that says "waiting" for an hour is
# noise, and the owner asked for it once the control starts.
SLACK_TASK="${SLACK_TASK:-da6cc26a}"
nohup setsid "$PY" -u slack_bar.py --task "$SLACK_TASK" --interval 150 \
      > slack_bar.log 2>&1 &

export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Identical to exp_c18: without BOTH of these a fixed seed does not reproduce (exp_c17).
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Every knob here is exp_c18's value. The actor architecture is the only difference.
COMMON="--iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --hidden 256 --eval-every 500 --eval-episodes 20"

SEEDS="0 1 2 3 4 5"
MAXJOBS=3

echo "XLA_FLAGS=$XLA_FLAGS"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

pids=()
for seed in $SEEDS; do
  while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
    wait -n 2>/dev/null || true
    alive=()
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    pids=("${alive[@]}")
  done
  echo "=== launch MLP seed $seed  $(date -u +%FT%TZ) ==="
  nohup $PY -u mlp_sac.py --seed "$seed" $COMMON \
        --tag "_c19_seed${seed}" > "cell_seed${seed}.log" 2>&1 &
  pids+=($!)
  sleep 25
done

# Wait on the TRAINER pids only. A bare `wait` here DEADLOCKED this run: it also waits for
# the Slack bar launched above, and the bar does not exit until every eval JSON exists --
# but the evals run below, after this line. Training finished at 10:10Z and the script then
# sat idle for 2.5 h with the bar frozen at 90%. (An empty array is guarded because
# `wait "${pids[@]}"` on an empty array expands to a bare `wait`, reintroducing the bug.)
if [ "${#pids[@]}" -gt 0 ]; then wait "${pids[@]}" 2>/dev/null || true; fi
echo "ALL 6 MLP SEEDS DONE $(date -u +%FT%TZ)"

echo "evaluating $(date -u +%FT%TZ)"
$PY -u collect.py
echo "MLP EVAL DONE $(date -u +%FT%TZ)"
