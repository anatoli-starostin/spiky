#!/bin/bash
set -x
export CUDA_HOME=/usr/local/cuda; export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
NC=$HOME/projects/spiky/experiments/lut_nanochat/nanochat; cd $NC || exit 2
export PATH=$NC/.venv/bin:$PATH NANOCHAT_BASE_DIR=$HOME/.cache/nanochat OMP_NUM_THREADS=1
export NANOCHAT_FA3_REVISION=$(python3 -c "import json;print(json.load(open(\"../pins.json\"))[\"fa3\"][\"revision\"])")
python3 ~/patch_bf16.py
echo "=== CONFIRM proj_bf16 flag builds + fused active (r1536) ==="
PYTHONPATH=$NC .venv/bin/python -c "
import torch
from nanochat.lut_ffn import LUTFFNConfig, LUTFeedForward
from spiky.lutorch_ex import FusedConfidenceLUT
m=LUTFeedForward(1536, LUTFFNConfig(enabled=True,h=32,tph=32,proj_bf16=True), device=\"cuda\")
print(\"BF16FLAG CART=\"+type(m.mhl.cartridge).__name__+\" IS_FUSED=\"+str(isinstance(m.mhl.cartridge,FusedConfidenceLUT))+\" FWD_FINITE=\"+str(bool(torch.isfinite(m(torch.randn(1,8,1536,device=\"cuda\")).float()).all().item())))
"
echo "CONFIRM_RC=$?"
run_arm() { TAG=$1; DBS=$2; shift 2; LOG=$HOME/bf16_${TAG}.log
  echo "=== ARM $TAG dbs=$DBS START $(date -Is) ==="
  nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader > $HOME/bf16_${TAG}.clk
  torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --fp8 --max-seq-len=2048 --total-batch-size=1048576 --num-iterations=5568 --max-steps=30 --no-lr-rescale --lut-ffn --lut-ffn-h 32 --lut-ffn-tph 32 --lut-ffn-table-dropout=0.2 --lut-ffn-tv-lambda=10 --lut-ffn-read-top-n=1 --lut-ffn-nap=8 --device-batch-size=$DBS --run=dummy --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --log-every=1 --model-tag=bf16_${TAG} "$@" > $LOG 2>&1
  echo "ARM_${TAG}_RC=$?"
  echo "DT_${TAG}: $(grep -hoE "step (0001[0-9]|0002[0-9])/.{0,80}dt: [0-9.]+ms" $LOG | grep -oE "[0-9.]+ms" | tr "\n" " ")"
  echo "LOSS_${TAG}: $(grep -hoE "step 000[0-2][0-9]/.{0,40}loss: [0-9.]+" $LOG | grep -oE "loss: [0-9.]+" | grep -oE "[0-9.]+" | tr "\n" " ")"
  grep -E "Peak memory" $LOG
  echo "NAN_${TAG}: $(grep -cE "loss: (nan|inf|-inf)" $LOG)"
  cat $HOME/bf16_${TAG}.clk
}
echo "########## ARM fp32 dbs16 (baseline) ##########"; run_arm fp32_dbs16 16
echo "########## ARM bf16 dbs16 ##########"; run_arm bf16_dbs16 16 --lut-ffn-proj-bf16
echo "########## ARM bf16 dbs32 (memory/microstep) ##########"; run_arm bf16_dbs32 32 --lut-ffn-proj-bf16
echo "NEWBOX_BF16_FINISHED"
