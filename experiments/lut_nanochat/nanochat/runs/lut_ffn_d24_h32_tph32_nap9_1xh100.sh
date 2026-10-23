#!/bin/bash
# [lut_nanochat] FORK of runs/lut_ffn_d24_h32_tph32_1xh100.sh: SAME geometry (h=32, tph=32, d=48, r=1536),
# ONLY CHANGE is nap 8 -> 9 (256 -> 512 cells per table). "Wider tables" experiment.
#
# cells = 2^nap, so nap 8->9 DOUBLES the per-table weight count; the cartridge param count roughly doubles
# (nap=8 cartridge was 12,582,914/layer -> ~25,165,826/layer at nap=9). Projections are UNCHANGED (square
# Linear(1536->1536), 4,721,664/layer). So LUT-FFN total ~= 24*(25,165,826 + 4,721,664) ~= 717M (vs the nap=8
# fork's 415,309,872), whole model ~= 1.65B (vs 1,346,447,162). Everything else is identical to the running
# nap=8 fork (d24-lutffn-confn1-h32-tph32-1xh100-s0): fused ConfidenceLUT, read_top_n=1, anchor_mode pairs,
# arm B (fp8 backbone + bf16 projections), --no-lr-rescale (LR 1.0x, v2 parity), 5568 steps, seed 42 /
# cartridge seed 1, total batch 2^20, grad-accum 32, max_seq_len 2048, table_dropout 0.2, tv_lambda 10, fp8
# tensorwise, FA3. Distinct RUN_NAME/MODEL_TAG so it does not collide with the nap=8 fork's run or checkpoints.
#
# NOT launched automatically. Usage (from experiments/lut_nanochat/nanochat):
#   bash runs/lut_ffn_d24_h32_tph32_nap9_1xh100.sh            # fresh start / resume
#   DBS=8 bash runs/lut_ffn_d24_h32_tph32_nap9_1xh100.sh      # if batch 16 does not fit (nap=9 tables are bigger)
set -euo pipefail
cd "$(dirname "$0")/.."

export RUN_NAME=${RUN_NAME:-d24-lutffn-confn1-h32-tph32-nap9-1xh100-s0}
export MODEL_TAG=${MODEL_TAG:-d24_lutffn_confn1_h32_tph32_nap9_1xh100}
export WANDB_GROUP=${WANDB_GROUP:-nanochat_lut_ffn}
export ARCH_TAG=${ARCH_TAG:-lut_ffn_confidence_n1_h32_tph32_nap9}
export EXTRA_TAGS="${EXTRA_TAGS:-},ffn=confidenceLUT,read_top_n=1,h=32,tph=32,d=48,r=1536,cells=512,nap=9"
# --lut-ffn + --no-lr-rescale (v2 parity) + h32/tph32 geometry + bf16 projections + nap=9 (512 cells).
export TRAIN_EXTRA="--lut-ffn --no-lr-rescale --lut-ffn-h 32 --lut-ffn-tph 32 --lut-ffn-proj-bf16 --lut-ffn-nap 9 ${TRAIN_EXTRA:-}"
# Pin the training horizon to v2 parity (5568) exactly as the nap=8 fork does; env-overridable (e.g. NUM_ITERATIONS=3 for smoke).
export NUM_ITERATIONS=${NUM_ITERATIONS:-5568}

export LAUNCHER=${LAUNCHER:-runs/lut_ffn_d24_h32_tph32_nap9_1xh100.sh}   # recorded in pin_config.json and the train.log header
exec bash runs/baseline_d24_1xh100.sh "$@"
