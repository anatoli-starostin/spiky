#!/usr/bin/env bash
# exp_c11 — the real-training 2x2: {hyperplane, anchors} x {hard, hybrid_smooth}.
# All four cells trained with LUT-SAC at update-to-data ratio 0.5, same per-row trust
# region, same nap6/tph32 config as v3, each trained in the mode it is evaluated in.
# Sequential so the cells don't contend on the GPU (and so timings stay comparable).
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
TRAIN="../exp_c09_lut_sac/lut_sac.py"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# ratio 0.5 = 64 env-steps/iter, 32 updates x batch 512
COMMON="--nap 6 --tph 32 --iters 10000 --envs 64 --rollout 1 --updates 32 \
        --batch 512 --warmup 500 --row-clip 1.0 --eval-every 500 --eval-episodes 20"

# wait for any earlier LUT-SAC run to finish so the cells get the GPU to themselves
while pgrep -f "lut_sac.py .*--tag _v3" > /dev/null; do sleep 30; done

for cell in "hyperplane hard" "hyperplane hybrid_smooth" \
            "anchors hard" "anchors hybrid_smooth"; do
  set -- $cell
  tag="_c11_$1_$2"
  echo "=== $1 x $2  $(date -u +%FT%TZ) ==="
  $PY -u "$TRAIN" --addressing "$1" --forward-mode "$2" $COMMON \
      --tag "$tag" > "cell$tag.log" 2>&1
  echo "  rc=$?"
done
echo "ALL CELLS DONE $(date -u +%FT%TZ)"
