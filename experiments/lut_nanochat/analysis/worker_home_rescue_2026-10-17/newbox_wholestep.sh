#!/bin/bash
set -x
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
NC=$HOME/projects/spiky/experiments/lut_nanochat/nanochat
cd $NC || exit 2
export PATH=$NC/.venv/bin:$PATH NANOCHAT_BASE_DIR=$HOME/.cache/nanochat OMP_NUM_THREADS=1
export NANOCHAT_FA3_REVISION=$(python3 -c "import json;print(json.load(open(\"../pins.json\"))[\"fa3\"][\"revision\"])")
python3 ~/patch_profile.py
echo "=== whole-step profile (fork geometry, profile steps 12-13) ==="
torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --fp8 --max-seq-len=2048 --total-batch-size=1048576 --num-iterations=5568 --max-steps=20 --profile-steps=12 --no-lr-rescale --lut-ffn --lut-ffn-h 32 --lut-ffn-tph 32 --lut-ffn-table-dropout=0.2 --lut-ffn-tv-lambda=10 --lut-ffn-read-top-n=1 --lut-ffn-nap=8 --device-batch-size=16 --run=dummy --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --log-every=1 --model-tag=wholestep_prof > $HOME/newbox_wholestep_run.log 2>&1
echo "WHOLESTEP_RC=$?"
git -C $HOME/projects/spiky checkout scripts/base_train.py 2>/dev/null; git -C $NC checkout scripts/base_train.py 2>/dev/null
echo "NEWBOX_WHOLESTEP_FINISHED"
