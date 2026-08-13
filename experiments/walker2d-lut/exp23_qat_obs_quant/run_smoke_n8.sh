#!/usr/bin/env bash
# Adds the arms that actually match the deployed readout. The shipped Stage-3 step is
# 0.276-0.283 action units, i.e. N = 2/step + 1 = 8.1 levels across [-1,1]. N in {16,22,32}
# are all 2-4x FINER than the hardware, so they cannot answer "what does the real readout
# cost". N=8 (step 0.2857) is the faithful case; N=6 (step 0.400) brackets it from below.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant/outq
CK=/home/astarostin/projects/ckpt_backups/exp19_walker2d_lut/deploy_matched/actor_s2.pt
PY=/home/astarostin/projects/spiky/.venv/bin/python

cd "$SRC"
export PYTHONPATH=/home/astarostin/projects/spiky/experiments/walker2d-lut/_qat_deps
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p "$OUT"

COMMON=(--arch fastlut_lse_sum_expmlpcrit --tables-per-head 32 --envs 8192 --graph
        --updates 20 --seed 2 --lr 3e-4 --lr-schedule cosine --lr-min 3e-5
        --init-lr-mode cosine --logstd-min -1.897 --ent-coef 0.0 --target-kl 0.02
        --norm-returns --obs-clip-vel 10 --solver-iters 100 --ls-iters 50
        --init-from "$CK" --quant-ticks 128 --quant-sigma 1.0)

for N in 8 6; do
  echo "=== ARM N=$N ==="
  $PY -u ppo_qat_obs.py "${COMMON[@]}" --out-quant-levels "$N" \
      --out "$OUT/n$N.json" > "$OUT/n$N.log" 2>&1
  echo "  rc=$?"
done

for N in 8 6; do
  $PY -u eval_qat_ckpt.py --ckpt "$CK" --envs 1024 --steps 2000 \
      --quant-ticks 128 --quant-sigma 1.0 --out-quant-levels "$N" \
      > "$OUT/eval_n$N.log" 2>&1
  echo "--- eval out-quant N=$N (rc=$?)"
  grep -E "quant" "$OUT/eval_n$N.log" | grep -v "load on device"
done
echo "N8_DONE"
