#!/usr/bin/env bash
# exp23 SMOKE ONLY — 20 updates per arm, NOT the full fine-tune.
#
# Two arms, both resuming from the SAME checkpoint under the SAME deployment-matched
# physics the checkpoint was trained with (--obs-clip-vel 10 --solver-iters 100
# --ls-iters 50). The only difference is the observation quantizer, so any gap between
# them is attributable to quantization and nothing else.
#
#   ARM A (control) : quantizer OFF  -> confirms the resume path itself is sound
#   ARM B (QAT)     : quantizer ON   -> 128 ticks, sigma 1.0
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant
CK=/home/astarostin/projects/ckpt_backups/exp19_walker2d_lut/deploy_matched/actor_s2.pt
PY=/home/astarostin/projects/spiky/.venv/bin/python

cd "$SRC"
export PYTHONPATH=/home/astarostin/projects/spiky/experiments/walker2d-lut/_qat_deps
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p "$OUT"

COMMON=(--arch fastlut_lse_sum_expmlpcrit --tables-per-head 32 --envs 8192 --graph
        --updates 20 --seed 2 --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897
        --ent-coef 0.0 --target-kl 0.02 --norm-returns
        --obs-clip-vel 10 --solver-iters 100 --ls-iters 50 --init-from "$CK")

start=$SECONDS
echo "=== ARM A: control, quantizer OFF ==="
$PY -u ppo_qat_obs.py "${COMMON[@]}" --out "$OUT/smoke_ctl.json" > "$OUT/smoke_ctl.log" 2>&1
echo "ARM A rc=$? wall=$((SECONDS - start))s"

mid=$SECONDS
echo "=== ARM B: quantizer ON, 128 ticks, sigma 1.0 ==="
$PY -u ppo_qat_obs.py "${COMMON[@]}" --quant-ticks 128 --quant-sigma 1.0 \
    --out "$OUT/smoke_qat.json" > "$OUT/smoke_qat.log" 2>&1
echo "ARM B rc=$? wall=$((SECONDS - mid))s"
echo "ALL_SMOKE_DONE total=$((SECONDS - start))s"
