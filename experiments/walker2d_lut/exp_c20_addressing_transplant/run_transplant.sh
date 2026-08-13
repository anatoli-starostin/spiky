#!/usr/bin/env bash
# exp_c20 — does seed 4's LEARNED ADDRESSING carry its win? (#75)
#
# exp_c18 found seed 4 at 5286.6 against a pack at 4112 +/- 160, reproducibly, and the
# behaviour deep-dive showed the entire advantage is forward velocity (4.29 vs 3.20 m/s).
# What remains unknown is WHERE that solution lives. This transplants seed 4's FINAL
# trained (w, b) into fresh runs, FREEZES it (no gradient, exactly as anchors mode), and
# learns only the table content, the critic and the temperature, at 3 seeds outside the
# original 0-5 range.
#
# TWO ARMS, and the second is not optional. Freezing the addressing removes the joint
# optimisation that every exp_c18 run had, so freezing may cost return all by itself. With
# only the seed-4 arm, a result of (say) 4600 could mean "seed 4's routing helps a little"
# or "freezing costs everyone ~700" -- indistinguishable. Arm B repeats the identical
# procedure with a PACK seed's routing (seed 5: 4102.1, never falls, closest to the pack
# median) so the freezing penalty is measured rather than assumed. The comparison that
# answers the question is A vs B, not A vs 5286.6.
#
# Queued behind exp_c19 so it takes no GPU from the MLP control.
set -u
cd "$(dirname "$0")"

C19=../exp_c19_mlp_sac_control/run_seeds.log
PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"
CKPT_DIR="../exp_c09_lut_sac"

echo "waiting for exp_c19 to finish ($(date -u +%FT%TZ))"
until grep -q "MLP EVAL DONE" "$C19" 2>/dev/null; do sleep 60; done
echo "exp_c19 done — starting the transplant $(date -u +%FT%TZ)"

SLACK_TASK="${SLACK_TASK:-ab676142}"
nohup setsid "$PY" -u slack_bar.py --task "$SLACK_TASK" --interval 150 \
      > slack_bar.log 2>&1 &
BAR_PID=$!     # tracked so the `wait` below can EXCLUDE it -- see the note there

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# Identical to exp_c18 in every knob. --freeze-wb-from overrides the init branch and zeroes
# the addressing gradient, so --hyperplane-init is inert here; it is left in place so the
# command line stays diffable against exp_c18's.
COMMON="--addressing hyperplane --hyperplane-init anchor_pairs \
        --hyperplane-anchor-policy canonical_full_coverage \
        --forward-mode hard --nap 6 --tph 32 --heads 1 \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

SEEDS="100 101 102"
MAXJOBS=3

echo "XLA_FLAGS=$XLA_FLAGS"
echo "CUBLAS_WORKSPACE_CONFIG=$CUBLAS_WORKSPACE_CONFIG"

pids=()
launch() {   # launch <arm> <source-seed> <fresh-seed>
  local arm="$1" src="$2" seed="$3"
  while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
    wait -n 2>/dev/null || true
    alive=()
    for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
    pids=("${alive[@]}")
  done
  echo "=== launch arm $arm (routing from seed $src) fresh seed $seed  $(date -u +%FT%TZ) ==="
  nohup $PY -u "$TRAIN" --seed "$seed" $COMMON \
        --freeze-wb-from "$CKPT_DIR/lut_sac_c18_seed${src}_actor.npz" \
        --tag "_c20_${arm}_s${seed}" > "cell_${arm}_s${seed}.log" 2>&1 &
  pids+=($!)
  sleep 25
}

# Arm-major order: if the sweep is cut short, arm A is complete rather than both halves
# being partial and neither comparable.
for seed in $SEEDS; do launch from4 4 "$seed"; done
for seed in $SEEDS; do launch from5 5 "$seed"; done

# Wait on the TRAINER pids only, never a bare `wait`. A bare `wait` also waits for the
# Slack bar, and the bar does not exit until every eval JSON exists -- but the evals run
# after this line. That is a deadlock, and exp_c19 hit it: its training finished and then
# sat idle for 2.5 h with the bar frozen at 90%.
# (and `wait "${pids[@]}"` with an EMPTY array expands to a bare `wait`, so guard it.)
if [ "${#pids[@]}" -gt 0 ]; then wait "${pids[@]}" 2>/dev/null || true; fi
echo "ALL 6 TRANSPLANT RUNS DONE $(date -u +%FT%TZ)"

echo "evaluating $(date -u +%FT%TZ)"
$PY -u collect.py
echo "TRANSPLANT EVAL DONE $(date -u +%FT%TZ)"
