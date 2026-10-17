#!/bin/bash
set -x
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
REPO=$HOME/projects/spiky; NC=$REPO/experiments/lut_nanochat/nanochat
cd $REPO || exit 2
echo "HEAD_BEFORE=$(git rev-parse HEAD)"
git branch -f backup/pre-f1d5d03a-d7a96c28 HEAD
git fetch origin 2>&1 | tail -2
echo "=== CHERRY-PICK f1d5d03a (LOCAL ONLY, no push) ==="
git cherry-pick f1d5d03a; CP_RC=$?
echo "CHERRY_PICK_RC=$CP_RC"
if [ $CP_RC -ne 0 ]; then echo "CHERRY_PICK_CONFLICT"; git diff --name-only --diff-filter=U; git cherry-pick --abort; echo "STOPPED_NO_BENCH"; exit 3; fi
echo "HEAD_AFTER=$(git rev-parse HEAD)"
echo "=== diff --stat 783010da..HEAD (expect only f1d5d03a lutorch_ex files) ==="
git diff --stat 783010da..HEAD
echo "=== experiments/ delta (expect empty) ==="
git diff --stat 783010da..HEAD -- experiments/lut_nanochat
rm -rf src/spiky/lutorch_ex/cartridges/wandb
cd $NC || exit 2
export PATH=$NC/.venv/bin:$PATH
UV=$(command -v uv); [ -z "$UV" ] && UV=$(ls $HOME/.local/bin/uv $HOME/.cargo/bin/uv /usr/local/bin/uv 2>/dev/null|head -1)
echo "=== REBUILD (CUDA sources changed) ==="
bash runs/setup_unified_env.sh 2>&1 | grep -iE "warning|error|builds OK|torch unchanged|ABORT|UNIFIED ENV"; echo "SETUP_RC=${PIPESTATUS[0]}"
echo "=== .cu compiler warnings (verbose rebuild of fused_confidence) ==="
PYTHONPATH=$NC .venv/bin/python -c "
import os,torch
from torch.utils.cpp_extension import load
from spiky.lutorch_ex.cartridges._cuda_ext import CSRC
load(name=\"warncheck_fc\", sources=[os.path.join(CSRC,\"fused_confidence.cu\")], extra_cflags=[\"-std=c++20\",\"-O3\"], extra_cuda_cflags=[\"-std=c++20\",\"-O3\"], verbose=True)
print(\"WARNCHECK_BUILD_OK\")
" 2>&1 | grep -iE "warning|error|WARNCHECK_BUILD_OK" | head -25
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export NANOCHAT_FA3_REVISION=$(python3 -c "import json;print(json.load(open(\"../pins.json\"))[\"fa3\"][\"revision\"])")
export OMP_NUM_THREADS=1
echo "=== RESOLVED FORWARD CTA ORDER (CONF, this device) ==="
PYTHONPATH=$NC .venv/bin/python -c "
import torch
from spiky.lutorch_ex.cartridges._cuda_ext import resolve_cta_order, forward_group_major
cap=torch.cuda.get_device_capability(0)
print(\"DEVICE_CC=\"+str(cap)+\" CONF_DEFAULT_ORDER=\"+resolve_cta_order(cap)+\" CONF_FWD_GROUP_MAJOR=\"+str(forward_group_major(\"CONF\", torch.device(\"cuda:0\"))))
"
echo "=== FUSED VERIFY both geometries ==="
PYTHONPATH=$NC .venv/bin/python -c "
import torch
from nanochat.lut_ffn import LUTFFNConfig, LUTFeedForward
from spiky.lutorch_ex import FusedConfidenceLUT
for tag,kw in [(\"r768\",dict()),(\"r1536\",dict(h=32,tph=32))]:
    m=LUTFeedForward(1536, LUTFFNConfig(enabled=True,**kw), device=\"cuda\")
    c=m.mhl.cartridge
    fin=bool(torch.isfinite(m(torch.randn(1,8,1536,device=\"cuda\")).float()).all().item())
    print(tag+\" CART=\"+type(c).__name__+\" IS_FUSED=\"+str(isinstance(c,FusedConfidenceLUT))+\" FWD_FINITE=\"+str(fin))
"
echo "=== FULL lutorch_ex TEST SUITE (gate) ==="
"$UV" pip install --python .venv/bin/python pytest safetensors >/dev/null 2>&1
PYTHONPATH=$NC .venv/bin/python -m pytest $REPO/src/spiky/lutorch_ex/tests -q -rf 2>&1 | tee $HOME/ab_fulltests.log | tail -8
echo "FULLTESTS_SUMMARY=$(grep -E "passed|failed|error" $HOME/ab_fulltests.log | tail -1)"
echo "FAILED_LINES_BELOW:"; grep -E "^FAILED|^ERROR" $HOME/ab_fulltests.log
BADFAIL=$(grep -E "^FAILED|^ERROR" $HOME/ab_fulltests.log | grep -vc "test_deploy")
echo "NON_SAFETENSORS_FAILURES=$BADFAIL"
if [ "$BADFAIL" -ne 0 ]; then echo "REGRESSION beyond known safetensors -- NOT benchmarking"; echo "STOPPED_NO_BENCH"; exit 4; fi
run_ab() {
  TAG=$1; ORDER=$2; shift 2
  LOG=$HOME/ab_${TAG}_${ORDER}.log
  echo "=== BENCH $TAG order=$ORDER START $(date -Is) ==="
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader > $HOME/ab_${TAG}_${ORDER}.clk
  LUTORCH_EX_CONF_CUDA_CTA_ORDER=$ORDER torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --fp8 --max-seq-len=2048 --total-batch-size=1048576 --num-iterations=5568 --max-steps=30 --no-lr-rescale --lut-ffn --lut-ffn-table-dropout=0.2 --lut-ffn-tv-lambda=10 --lut-ffn-read-top-n=1 --lut-ffn-nap=8 --device-batch-size=16 --run=dummy --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --log-every=1 --model-tag=ab_${TAG}_${ORDER} "$@" > $LOG 2>&1
  echo "BENCH_${TAG}_${ORDER}_RC=$?"
  echo "DTVALS_${TAG}_${ORDER}: $(grep -hoE "step (0001[0-9]|0002[0-9])/.{0,80}dt: [0-9.]+ms" $LOG | grep -oE "[0-9.]+ms" | tr "\n" " ")"
  grep -E "Peak memory" $LOG
  echo "LOSSNAN_${TAG}_${ORDER}: $(grep -cE "loss: (nan|inf|-inf)" $LOG)"
}
echo "########## CELL C (r1536) A/B ##########"
run_ab C interleaved --lut-ffn-h 32 --lut-ffn-tph 32
run_ab C group --lut-ffn-h 32 --lut-ffn-tph 32
echo "########## CELL A (r768) A/B ##########"
run_ab A interleaved
run_ab A group
echo "NEWBOX_AB_FINISHED"
