#!/usr/bin/env bash
# exp_c13 — the anchors x hard nap x tph capacity sweep, rerun under the NEW lutorch
# `balanced` sampler and at THREE SEEDS (#75).
#
# This supersedes exp_c12, which used the home-grown anchor draw and one seed. It is a
# NEW directory rather than a rerun in place so exp_c12's logs and numbers stay intact
# and the two samplers can be compared.
#
# Grid: nap {6,7,8} x tph {32,64,128} x seed {0,1,2} = 27 runs. The nap6/tph32 baseline
# is INCLUDED this time (exp_c12 reused it from exp_c11), so every config is on equal
# footing under identical settings.
#
# ORDER IS SEED-MAJOR, deliberately: all 9 configs at seed 0, then seed 1, then seed 2.
# If the sweep is cut short you are left with complete single-seed coverage of the whole
# grid rather than a few configs with 3 seeds and the rest with none.
#
# Everything else matches exp_c12 exactly: anchors frozen, hard forward, ratio 0.5,
# 10,000 iterations, same env and optimizer. Only the sampler and the seed vary.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

COMMON="--addressing anchors --anchor-policy balanced --forward-mode hard \
        --iters 10000 --envs 64 --rollout 1 --updates 32 --batch 512 --warmup 500 \
        --row-clip 1.0 --eval-every 500 --eval-episodes 20"

CELLS="6:32 6:64 6:128 7:32 7:64 7:128 8:32 8:64 8:128"
SEEDS="0 1 2"
MAXJOBS=3            # ~7.5 GB each of 32 GB; 4-way would crowd the desktop's 2.7 GB

pids=()

launched=0
total=0
for seed in $SEEDS; do for cell in $CELLS; do total=$((total + 1)); done; done

for seed in $SEEDS; do
  for cell in $CELLS; do
    nap="${cell%%:*}"; tph="${cell##*:}"
    # Throttle by WAITING ON A PID, not by counting pgrep hits: the old exp_c12 loop
    # used `pgrep -f ... | wc -l` and that mis-parsed ("integer expected" in its log).
    while [ "${#pids[@]}" -ge "$MAXJOBS" ]; do
      wait -n 2>/dev/null || true
      alive=()
      for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive+=("$p"); done
      pids=("${alive[@]}")
    done
    launched=$((launched + 1))
    echo "=== launch $launched/$total  nap$nap tph$tph seed$seed  $(date -u +%FT%TZ) ==="
    nohup $PY -u "$TRAIN" --nap "$nap" --tph "$tph" --seed "$seed" $COMMON \
          --tag "_c13_nap${nap}_tph${tph}_s${seed}" \
          > "cell_nap${nap}_tph${tph}_s${seed}.log" 2>&1 &
    pids+=($!)
    sleep 25   # stagger so the JIT compiles don't collide
  done
done

wait
echo "ALL 27 CELLS DONE $(date -u +%FT%TZ)"
