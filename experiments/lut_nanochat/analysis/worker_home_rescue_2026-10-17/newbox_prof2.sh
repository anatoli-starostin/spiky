#!/bin/bash
set -x
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
NC=$HOME/projects/spiky/experiments/lut_nanochat/nanochat; REPO=$HOME/projects/spiky
cd $NC; export PATH=$NC/.venv/bin:$PATH; export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat OMP_NUM_THREADS=1
LUT_PROFILE_H=32 LUT_PROFILE_TPH=32 PYTHONPATH=$NC .venv/bin/python $REPO/experiments/lut_nanochat/profiling/profile_lut_ffn.py --variants lut-fp32 --passes profile memory --batch 16 --fp32-matmul-precision high --out-dir $HOME/prof_r1536_km --allow-dirty-lib > $HOME/prof_r1536_km.out 2>&1
echo "PROF_KM_RC=$?"
echo "=== profile buckets ==="; sed -n "/profile/,/done/p" $HOME/prof_r1536_km/summary.md 2>/dev/null | head -60
echo "NEWBOX_PROF2_FINISHED"
