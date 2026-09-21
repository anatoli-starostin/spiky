#!/usr/bin/env bash
# Trust region on the relaxation (task 0cb457ba piece 2), under the standing debug rule: 500 steps, L=4,
# N=64, 32 tables, T=8, seed 0, pinned, local logging only, nothing at L=16.
#
# No dropout: the p=0 vs p=0.25 comparison showed dropout does nothing at this scale, so it is dropped
# rather than carried along as an uncontrolled knob. Adam 1e-3, i.e. the setting in which the decay
# actually appears -- a trust region that only works under SGD would prove nothing.
set -u
D=/home/astarostin/projects/spiky/research/pcalm_lut
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$D" || exit 1
export TRITON_CACHE_DIR=$HOME/.cache/triton_distill
export WANDB_MODE=disabled
unset WANDB_BASE_URL
mkdir -p runs_trust_logs

run () {
  local name=$1; shift
  if [ -e "runs_trust/$name/run.json" ]; then echo "== $name exists, skipping"; return; fi
  echo "== $name start $(date +%T)"
  $PY -u train_paired.py --depth 4 --width 64 --tables 32 --steps 500 --batch 128 --probe-every 25 \
      --seed 0 --table-dropout 0 --clamp pinned --train-eval --out-dir runs_trust --name "$name" "$@" \
      > "runs_trust_logs/$name.log" 2>&1
  echo "== $name exit $? $(date +%T)"
}

run "trust-bp"  --arm bp                       # the reference; the trust region does not apply to BP
for r in 0 0.02 0.05 0.1 0.2; do
  run "trust-pcA-r$r"     --arm pcA    --T 8 --trust-radius $r
  run "trust-pcalmB-r$r"  --arm pcalmB --T 8 --trust-radius $r
done
echo "== TRUST PROBE DONE $(date +%T)"
