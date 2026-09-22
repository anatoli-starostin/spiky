#!/usr/bin/env bash
# Pure DTP grid (task b77c2cd7). L=4, N=64, 32 tables, seed 0, 500 steps, no dropout, pinned output,
# local logging only, nothing at L=16.
#
# The BP controls are not optional decoration: the MLP DTP arm looked dead in the smoke test, and only a
# matched BP run on the same model says whether that is DTP's doing or the model's.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_MODE=disabled
unset WANDB_BASE_URL
mkdir -p runs_dtp_logs

run () {
  local name=$1; shift
  if [ -e "runs_dtp/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date +%T)"
  $PY -u run_dtp.py --depth 4 --width 64 --tables 32 --steps 500 --batch 128 --probe-every 25 \
      --seed 0 --out-dir runs_dtp --name "$name" "$@" > "runs_dtp_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

for m in lut mlp; do
  # the rule under test, at both optimisers -- the SGD/Adam split decided the last study
  run "$m-dtp-adam1e-3"     --model $m --rule dtp       --optimizer adam --lr 1e-3
  run "$m-dtp-sgd1e-2"      --model $m --rule dtp       --optimizer sgd  --lr 1e-2
  run "$m-dtp-sgd3e-2"      --model $m --rule dtp       --optimizer sgd  --lr 3e-2
  # matched BP controls
  run "$m-bp-adam1e-3"      --model $m --rule bp        --optimizer adam --lr 1e-3
  run "$m-bp-sgd3e-2"       --model $m --rule bp        --optimizer sgd  --lr 3e-2
  # falsify the difference form
  run "$m-dtpplain-adam1e-3" --model $m --rule dtp-plain --optimizer adam --lr 1e-3
done
# does the inverse get better with more g steps, or a larger perturbation?
run "lut-dtp-adam1e-3-g3"    --model lut --rule dtp --optimizer adam --lr 1e-3 --g-steps 3
run "lut-dtp-adam1e-3-sig03" --model lut --rule dtp --optimizer adam --lr 1e-3 --sigma 0.3
run "mlp-dtp-adam1e-3-g3"    --model mlp --rule dtp --optimizer adam --lr 1e-3 --g-steps 3
echo "== DTP GRID DONE $(date +%T)"
