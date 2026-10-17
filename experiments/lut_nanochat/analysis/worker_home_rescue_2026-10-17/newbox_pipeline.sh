#!/bin/bash
set -x
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$HOME/.local/bin:$HOME/.cargo/bin:$PATH
REPO=$HOME/projects/spiky
NC=$REPO/experiments/lut_nanochat/nanochat
cd $REPO || exit 2
echo "=== UPDATE REPO ==="
git fetch origin research/lut_nanochat
git reset --hard origin/research/lut_nanochat
echo "HEAD=$(git rev-parse HEAD)"
rm -rf src/spiky/lutorch_ex/cartridges/wandb
UV=$(command -v uv); [ -z "$UV" ] && UV=$(ls $HOME/.local/bin/uv $HOME/.cargo/bin/uv /usr/local/bin/uv 2>/dev/null | head -1)
echo "USING_UV=$UV"
cd $NC || exit 2
echo "=== BUILD: uv sync --extra gpu ==="
"$UV" sync --extra gpu; echo "SYNC_RC=$?"
echo "=== BUILD: setup_unified_env.sh ==="
bash runs/setup_unified_env.sh; echo "SETUP_RC=$?"
echo "=== TEST: ensure pytest present ==="
"$UV" pip install --python .venv/bin/python pytest >/dev/null 2>&1; echo "PYTEST_INSTALL_RC=$?"
echo "=== TEST: lutorch_ex suite ==="
PYTHONPATH=$NC .venv/bin/python -m pytest $REPO/src/spiky/lutorch_ex/tests -q 2>&1 | tail -45
echo "TESTS_DONE"
echo "=== BENCH: current geometry r768 (h16/tph64) ==="
PYTHONPATH=$NC LUT_PROFILE_H=16 LUT_PROFILE_TPH=64 .venv/bin/python $REPO/experiments/lut_nanochat/profiling/profile_lut_ffn.py --variants lut-fp32 lut-fp8lib-c32 --passes env block memory --batch 8 16 --out-dir $HOME/newbox_bench/r768 --allow-dirty-lib
echo "BENCH_R768_RC=$?"
echo "=== BENCH: new geometry r1536 (h32/tph32) ==="
PYTHONPATH=$NC LUT_PROFILE_H=32 LUT_PROFILE_TPH=32 .venv/bin/python $REPO/experiments/lut_nanochat/profiling/profile_lut_ffn.py --variants lut-fp32 lut-fp8lib-c32 --passes env block memory --batch 8 16 --out-dir $HOME/newbox_bench/r1536 --allow-dirty-lib
echo "BENCH_R1536_RC=$?"
echo "=== SUMMARY r768 ==="; cat $HOME/newbox_bench/r768/summary.md 2>/dev/null
echo "=== SUMMARY r1536 ==="; cat $HOME/newbox_bench/r1536/summary.md 2>/dev/null
echo "NEWBOX_PIPELINE_FINISHED"
