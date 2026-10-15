#!/bin/bash
# [lut_nanochat] FORK of runs/lut_ffn_d24_1xh100.sh: LUT geometry h=32, tph=32 (d stays 48).
#
# r = h*d = 32*48 = 1536 = d_model, restoring the champion ratio r/d_model = 1.0 (small-scale champion was
# 8 heads * 48 = 384 = d_model; d_model is now 4x -> heads 4x: 8->32). The current h16/d48 run was ratio 0.5.
# The read width DOUBLES 768 -> 1536, so ProjectionMHL compress/decompress become SQUARE Linear(1536->1536):
#   projection params/layer 2,361,600 -> 4,721,664 (x24: 56,678,400 -> 113,319,936, +56,641,536).
# Total tables h*tph = 32*32 = 16*64 = 1024 is HELD, so the cartridge param count is UNCHANGED (12,582,914/layer).
#   LUT-FFN total 358,668,336 -> 415,309,872 ; whole model 1,289,805,626 -> 1,346,447,162 (+56,641,536, +4.4%).
# Everything else identical to d24-lutffn-confn1-1xh100-s0-full-v2: fused ConfidenceLUT, read_top_n=1,
# --no-lr-rescale, 5568 steps, seed 42 / cartridge seed 1, total batch 2^20, grad-accum 32, max_seq_len 2048,
# table_dropout 0.2, tv_lambda 10, nap 8 / cells 256, anchor_mode pairs, fp8 tensorwise, FA3.
#
# Projections default to the fp32 AdamW island (parity with v2). For the fp8 + Muon projection variant,
# add --lut-ffn-proj-fp8-muon to TRAIN_EXTRA (orthogonal to this geometry change; see base_train).
#
# NOT launched automatically. Usage (from experiments/lut_nanochat/nanochat):
#   bash runs/lut_ffn_d24_h32_tph32_1xh100.sh            # fresh start / resume
#   DBS=8 bash runs/lut_ffn_d24_h32_tph32_1xh100.sh      # if batch 16 does not fit (r=1536 island is bigger)
set -euo pipefail
cd "$(dirname "$0")/.."

export RUN_NAME=${RUN_NAME:-d24-lutffn-confn1-h32-tph32-1xh100-s0}
export MODEL_TAG=${MODEL_TAG:-d24_lutffn_confn1_h32_tph32_1xh100}
export WANDB_GROUP=${WANDB_GROUP:-nanochat_lut_ffn}
export ARCH_TAG=${ARCH_TAG:-lut_ffn_confidence_n1_h32_tph32}
export EXTRA_TAGS="${EXTRA_TAGS:-},ffn=confidenceLUT,read_top_n=1,h=32,tph=32,d=48,r=1536,cells=256,nap=8"
# --lut-ffn + --no-lr-rescale (v2 parity) + the h32/tph32 geometry (d defaults to 48 -> r=1536).
export TRAIN_EXTRA="--lut-ffn --no-lr-rescale --lut-ffn-h 32 --lut-ffn-tph 32 --lut-ffn-proj-bf16 ${TRAIN_EXTRA:-}"

exec bash runs/baseline_d24_1xh100.sh "$@"
