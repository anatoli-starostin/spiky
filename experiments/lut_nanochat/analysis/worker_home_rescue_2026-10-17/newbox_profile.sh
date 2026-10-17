#!/bin/bash
set -x
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
REPO=$HOME/projects/spiky; NC=$REPO/experiments/lut_nanochat/nanochat
cd $REPO || exit 2
git checkout research/lut_nanochat 2>&1 | tail -1
echo "HEAD=$(git rev-parse HEAD) BRANCH=$(git branch --show-current)"
rm -rf src/spiky/lutorch_ex/cartridges/wandb
cd $NC || exit 2
export PATH=$NC/.venv/bin:$PATH
echo "=== REBUILD (restore 783010da ext) ==="
bash runs/setup_unified_env.sh 2>&1 | grep -iE "builds OK|torch unchanged|ABORT|UNIFIED ENV|error"; echo "SETUP_RC=${PIPESTATUS[0]}"
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export OMP_NUM_THREADS=1
echo "=== CONFIRM FUSED r1536 ==="
PYTHONPATH=$NC .venv/bin/python -c "
import torch
from nanochat.lut_ffn import LUTFFNConfig, LUTFeedForward
from spiky.lutorch_ex import FusedConfidenceLUT
m=LUTFeedForward(1536, LUTFFNConfig(enabled=True,h=32,tph=32), device=\"cuda\")
print(\"CART=\"+type(m.mhl.cartridge).__name__+\" IS_FUSED=\"+str(isinstance(m.mhl.cartridge,FusedConfidenceLUT)))
"
echo "=== PROFILE_LUT_FFN fork geometry r1536 (high / TF32) ==="
LUT_PROFILE_H=32 LUT_PROFILE_TPH=32 PYTHONPATH=$NC .venv/bin/python $REPO/experiments/lut_nanochat/profiling/profile_lut_ffn.py --variants lut-fp32 --passes env block components profile memory --batch 16 --fp32-matmul-precision high --out-dir $HOME/prof_r1536_high --allow-dirty-lib > $HOME/prof_r1536_high.out 2>&1
echo "PROF_HIGH_RC=$?"
echo "=== H1: same block pass, highest (TF32 OFF) ==="
LUT_PROFILE_H=32 LUT_PROFILE_TPH=32 PYTHONPATH=$NC .venv/bin/python $REPO/experiments/lut_nanochat/profiling/profile_lut_ffn.py --variants lut-fp32 --passes block --batch 16 --fp32-matmul-precision highest --out-dir $HOME/prof_r1536_highest --allow-dirty-lib > $HOME/prof_r1536_highest.out 2>&1
echo "PROF_HIGHEST_RC=$?"
echo "=== SUMMARY (high) ==="; cat $HOME/prof_r1536_high/summary.md 2>/dev/null
echo "NEWBOX_PROFILE_FINISHED"
