#!/bin/bash
set -x
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
REPO=$HOME/projects/spiky
NC=$REPO/experiments/lut_nanochat/nanochat
mkdir -p $HOME/newbox_bench2
rm -f $HOME/newbox_bench2/dbs_used.txt
cd $REPO || exit 2
echo "=== UPDATE REPO ==="
git fetch origin research/lut_nanochat
git reset --hard origin/research/lut_nanochat
echo "HEAD=$(git rev-parse HEAD)"
rm -rf src/spiky/lutorch_ex/cartridges/wandb
cd $NC || exit 2
export PATH=$NC/.venv/bin:$PATH
UV=$(command -v uv); [ -z "$UV" ] && UV=$(ls $HOME/.local/bin/uv $HOME/.cargo/bin/uv /usr/local/bin/uv 2>/dev/null | head -1)
echo "=== BUILD/VERIFY ENV ==="
"$UV" sync --extra gpu >/dev/null 2>&1; echo "SYNC_RC=$?"
bash runs/setup_unified_env.sh; echo "SETUP_RC=$?"
export NANOCHAT_BASE_DIR=$HOME/.cache/nanochat
export NANOCHAT_FA3_REVISION=$(python3 -c "import json;print(json.load(open(\"../pins.json\"))[\"fa3\"][\"revision\"])")
export OMP_NUM_THREADS=1
echo "FA3REV=$NANOCHAT_FA3_REVISION"
run_cell() {
  TAG=$1; shift
  for DBS in 16 8; do
    LOG=$HOME/newbox_bench2/${TAG}_dbs${DBS}.log
    echo "=== CELL $TAG dbs=$DBS START $(date -Is) ==="
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader > $HOME/newbox_bench2/${TAG}_dbs${DBS}.clk_start 2>/dev/null
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --fp8 --max-seq-len=2048 --total-batch-size=1048576 --num-iterations=5568 --max-steps=30 --no-lr-rescale --lut-ffn --lut-ffn-table-dropout=0.2 --lut-ffn-tv-lambda=10 --lut-ffn-read-top-n=1 --lut-ffn-nap=8 --device-batch-size=$DBS --run=dummy --eval-every=-1 --core-metric-every=-1 --sample-every=-1 --save-every=-1 --log-every=1 --model-tag=bench_${TAG} "$@" > $LOG 2>&1
    RC=$?
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader > $HOME/newbox_bench2/${TAG}_dbs${DBS}.clk_end 2>/dev/null
    if [ $RC -eq 0 ] && grep -q "Peak memory reserved" $LOG; then
      echo "=== CELL $TAG OK dbs=$DBS $(date -Is) ==="
      echo "$TAG dbs=$DBS accum=$((1048576/(DBS*2048))) rc=$RC OK" >> $HOME/newbox_bench2/dbs_used.txt
      return 0
    fi
    if grep -qiE "out of memory|CUDA out of memory|OutOfMemoryError" $LOG; then
      echo "=== CELL $TAG OOM dbs=$DBS -> retry lower ==="
      echo "$TAG dbs=$DBS OOM" >> $HOME/newbox_bench2/dbs_used.txt
      continue
    fi
    echo "=== CELL $TAG FAILED dbs=$DBS rc=$RC (not OOM) ==="
    echo "$TAG dbs=$DBS rc=$RC FAIL" >> $HOME/newbox_bench2/dbs_used.txt
    tail -25 $LOG
    return 1
  done
  echo "=== CELL $TAG could not run even at dbs=8 ==="
  return 1
}
echo "########## CELL A r768 fp32+AdamW ##########"
run_cell A
echo "########## CELL B r768 fp8+Muon ##########"
run_cell B --lut-ffn-proj-fp8-muon
echo "########## CELL C r1536 fp32+AdamW ##########"
run_cell C --lut-ffn-h 32 --lut-ffn-tph 32
echo "########## CELL D r1536 fp8+Muon ##########"
run_cell D --lut-ffn-h 32 --lut-ffn-tph 32 --lut-ffn-proj-fp8-muon
echo "NEWBOX_BENCH2_FINISHED"
