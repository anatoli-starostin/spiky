#!/usr/bin/env bash
# exp23b SMOKE ONLY — 20 updates per arm, NOT the full run.
#
# Control (output-quant OFF, input-quant ON) vs output-quant at N in {16, 22, 32}.
# All arms: same checkpoint, same seed, matched physics, frozen normaliser, FULL cosine LR
# 3e-4 -> 3e-5 (--init-lr-mode cosine, as requested for this smoke), input quantizer on.
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

start=$SECONDS
echo "=== ARM ctl: output-quant OFF (input-quant ON) ==="
$PY -u ppo_qat_obs.py "${COMMON[@]}" --out "$OUT/ctl.json" > "$OUT/ctl.log" 2>&1
echo "  rc=$?"
for N in 16 22 32; do
  echo "=== ARM N=$N ==="
  $PY -u ppo_qat_obs.py "${COMMON[@]}" --out-quant-levels "$N" \
      --out "$OUT/n$N.json" > "$OUT/n$N.log" 2>&1
  echo "  rc=$?"
done
echo "TRAIN_DONE $((SECONDS - start))s"

echo "=== deterministic EVAL, 1024 envs x 2000 steps ==="
for N in 0 16 22 32; do
  $PY -u eval_qat_ckpt.py --ckpt "$CK" --envs 1024 --steps 2000 \
      --quant-ticks 128 --quant-sigma 1.0 --out-quant-levels "$N" \
      > "$OUT/eval_n$N.log" 2>&1
  echo "--- eval out-quant N=$N (rc=$?)"
  grep -E "quant" "$OUT/eval_n$N.log" | grep -v "load on device"
done
echo "ALL_DONE $((SECONDS - start))s"
