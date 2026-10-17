#!/bin/bash
set -x
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
REPO=$HOME/projects/spiky; NC=$REPO/experiments/lut_nanochat/nanochat
cd $REPO; git fetch origin research/lut_nanochat; git reset --hard origin/research/lut_nanochat; echo "HEAD=$(git rev-parse HEAD)"
rm -rf src/spiky/lutorch_ex/cartridges/wandb
cd $NC; export PATH=$NC/.venv/bin:$PATH
UV=$(command -v uv); [ -z "$UV" ] && UV=$(ls $HOME/.local/bin/uv $HOME/.cargo/bin/uv /usr/local/bin/uv 2>/dev/null|head -1)
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export NANOCHAT_FA3_REVISION=$(python3 -c "import json;print(json.load(open(\"../pins.json\"))[\"fa3\"][\"revision\"])")
export OMP_NUM_THREADS=1
echo "=== install pytest ==="; "$UV" pip install --python .venv/bin/python pytest >/dev/null 2>&1; echo "PYTEST_INSTALL_RC=$?"
echo "=== LUTORCH_EX TESTS ==="; PYTHONPATH=$NC .venv/bin/python -m pytest $REPO/src/spiky/lutorch_ex/tests -q 2>&1 | tail -45; echo "TESTS_DONE"
echo "=== CONFIRM fused h32/tph32 ==="
PYTHONPATH=$NC .venv/bin/python -c "
import torch
from nanochat.lut_ffn import LUTFFNConfig, LUTFeedForward
from spiky.lutorch_ex import FusedConfidenceLUT
m=LUTFeedForward(1536, LUTFFNConfig(enabled=True, h=32, tph=32), device=\"cuda\")
print(\"CELLC_CART=\"+type(m.mhl.cartridge).__name__+\" IS_FUSED=\"+str(isinstance(m.mhl.cartridge, FusedConfidenceLUT)))
print(\"CELLC_FWD_FINITE=\"+str(bool(torch.isfinite(m(torch.randn(1,8,1536,device=\"cuda\")).float()).all().item())))
"
echo "CELLC_CONFIRM_RC=$?"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader > $HOME/cellC_post.clk_start
echo "=== CELL C re-measure (r1536 fp32+AdamW) ==="
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --fp8 --max-seq-len=2048 --total-batch-size=1048576 --num-iterations=5568 --max-steps=30 --no-lr-rescale --lut-ffn --lut-ffn-h 32 --lut-ffn-tph 32 --lut-ffn-table-dropout=0.2 --lut-ffn-tv-lambda=10 --lut-ffn-read-top-n=1 --lut-ffn-nap=8 --device-batch-size=16 --run=dummy --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --log-every=1 --model-tag=bench_C_postrebase > $HOME/newbox_cellC_post.log 2>&1
echo "CELLC_RC=$?"
nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader > $HOME/cellC_post.clk_end
echo "=== CELL C dt (steps>=10) ==="
grep -hoE "step (0001[0-9]|0002[0-9])/.{0,80}dt: [0-9.]+ms" $HOME/newbox_cellC_post.log | grep -oE "dt: [0-9.]+" | tr "\n" " "; echo ""
grep -E "Peak memory" $HOME/newbox_cellC_post.log
echo "loss_nan_inf: $(grep -cE "loss: (nan|inf|-inf)" $HOME/newbox_cellC_post.log)"
echo "CLK:"; cat $HOME/cellC_post.clk_start $HOME/cellC_post.clk_end
echo "NEWBOX_CELLC_FINISHED"
