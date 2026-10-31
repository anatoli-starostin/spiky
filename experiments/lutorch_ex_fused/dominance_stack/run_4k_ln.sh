#!/bin/bash
# LN follow-up: B1 (LN affine), B2 (LN no affine), D control (fmh->fmh), B rerun with hidden-std logging; sequential on one GPU.
set -o pipefail
cd "$(dirname "$0")"; source env.sh
export WANDB_API_KEY=$WANDB_SPIKY_API_KEY
for A in B1_dom_ln_fmh B2_dom_lnna_fmh D_fmh_fmh B_dom_fmh_hstd; do
  CONFIG=configs/${A}_4k.json OUT_DIR=runs/${A}_4k python train.py 2>&1 | tee runs/${A}_4k.log
done
