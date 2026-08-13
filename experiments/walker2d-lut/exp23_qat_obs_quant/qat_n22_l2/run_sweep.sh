#!/usr/bin/env bash
# Pick the out-of-band penalty weight by measurement, not by guessing. 20 updates, 1 seed
# per weight. The signal is `out_oob` (fraction of raw mean components outside [-1,1]) vs
# `ep_ret` -- we want the first to fall hard without the second following it down.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant/qat_n22_l2/sweep
CK=/home/astarostin/projects/ckpt_backups/exp19_walker2d_lut/deploy_matched/actor_s2.pt
PY=/home/astarostin/projects/spiky/.venv/bin/python
cd "$SRC"
export PYTHONPATH=/home/astarostin/projects/spiky/experiments/walker2d-lut/_qat_deps
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p "$OUT"

COMMON=(--arch fastlut_lse_sum_expmlpcrit --tables-per-head 32 --envs 8192 --graph
        --updates 20 --seed 0 --lr 3e-4 --lr-schedule cosine --lr-min 3e-5
        --init-lr-mode cosine --logstd-min -1.897 --ent-coef 0.0 --target-kl 0.02
        --norm-returns --obs-clip-vel 10 --solver-iters 100 --ls-iters 50
        --init-from "$CK" --quant-ticks 128 --quant-sigma 1.0
        --out-quant-levels 22 --out-quant-clip 1.0)

for W in 0.003 0.01 0.03 0.1 0.3; do
  tag=$(echo "$W" | tr . p)
  $PY -u ppo_qat_obs.py "${COMMON[@]}" --oob-penalty "$W" \
      --out "$OUT/w$tag.json" > "$OUT/w$tag.log" 2>&1
  echo "w=$W rc=$?"
done
echo SWEEP_DONE
