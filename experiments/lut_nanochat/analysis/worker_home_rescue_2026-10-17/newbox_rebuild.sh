#!/bin/bash
set -x
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
REPO=$HOME/projects/spiky
NC=$REPO/experiments/lut_nanochat/nanochat
cd $REPO || exit 2
echo "=== UPDATE REPO to rebased tip ==="
git fetch origin research/lut_nanochat
git reset --hard origin/research/lut_nanochat
echo "HEAD=$(git rev-parse HEAD)"
rm -rf src/spiky/lutorch_ex/cartridges/wandb
cd $NC || exit 2
export PATH=$NC/.venv/bin:$PATH
UV=$(command -v uv); [ -z "$UV" ] && UV=$(ls $HOME/.local/bin/uv $HOME/.cargo/bin/uv /usr/local/bin/uv 2>/dev/null | head -1)
echo "=== REBUILD ENV (fresh lutorch_ex) ==="
"$UV" sync --extra gpu >/dev/null 2>&1; echo "SYNC_RC=$?"
bash runs/setup_unified_env.sh; echo "SETUP_RC=$?"
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export NANOCHAT_FA3_REVISION=$(python3 -c "import json;print(json.load(open(\"../pins.json\"))[\"fa3\"][\"revision\"])")
export OMP_NUM_THREADS=1
echo "=== CONFIRM fused class instantiated (not reference) ==="
PYTHONPATH=$NC .venv/bin/python -c "
import torch
from nanochat.lut_ffn import LUTFFNConfig, LUTFeedForward
from spiky.lutorch_ex import FusedConfidenceLUT
m=LUTFeedForward(1536, LUTFFNConfig(enabled=True), device=\"cuda\")
cart=m.mhl.cartridge
print(\"CARTRIDGE_CLASS=\"+type(cart).__name__)
print(\"IS_FUSED=\"+str(isinstance(cart, FusedConfidenceLUT)))
y=m(torch.randn(1,8,1536,device=\"cuda\")).float()
print(\"FWD_FINITE=\"+str(bool(torch.isfinite(y).all().item())))
"
echo "CONFIRM_RC=$?"
echo "=== LUTORCH_EX TEST SUITE (torch pinned 2.9.1) ==="
PYTHONPATH=$NC .venv/bin/python -m pytest $REPO/src/spiky/lutorch_ex/tests -q 2>&1 | tail -40
echo "TESTS_DONE"
echo "=== CALIBRATION CELL A (r768 fp32+AdamW) on UPDATED lutorch_ex ==="
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --fp8 --max-seq-len=2048 --total-batch-size=1048576 --num-iterations=5568 --max-steps=30 --no-lr-rescale --lut-ffn --lut-ffn-table-dropout=0.2 --lut-ffn-tv-lambda=10 --lut-ffn-read-top-n=1 --lut-ffn-nap=8 --device-batch-size=16 --run=dummy --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --log-every=1 --model-tag=bench_A_postrebase > $HOME/newbox_calibA_postrebase.log 2>&1
echo "CALIB_A_RC=$?"
echo "=== CALIB A dt (steps>=10) ==="
grep -hoE "step (0001[0-9]|0002[0-9])/.{0,80}dt: [0-9.]+ms" $HOME/newbox_calibA_postrebase.log | grep -oE "dt: [0-9.]+" | tr "\n" " "
echo ""
grep -E "Peak memory" $HOME/newbox_calibA_postrebase.log
echo "loss_nan_inf: $(grep -cE "loss: (nan|inf|-inf)" $HOME/newbox_calibA_postrebase.log)"
echo "NEWBOX_REBUILD_CALIB_FINISHED"
