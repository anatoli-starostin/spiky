#!/bin/bash
# The three 4000-step runs, sequentially on one GPU (clean s/step): A baseline, B dominance stack, C conf n=2.
set -o pipefail
cd "$(dirname "$0")"; source env.sh
for A in A_fmh_rational B_dom_fmh C_conf_n2; do
  CONFIG=configs/${A}_4k.json OUT_DIR=runs/${A}_4k python train.py 2>&1 | tee runs/${A}_4k.log
done
