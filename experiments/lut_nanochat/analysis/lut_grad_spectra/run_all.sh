#!/usr/bin/env bash
# Full pipeline: gradients/spectra/hits/redundancy/FD/lookahead for every checkpoint, then the report (plots + tables).
# usage: bash run_all.sh [OUT_DIR]     (run from anywhere; needs the GPU, ~35 min on one H100)
set -o pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NANO="$HERE/../../nanochat"
OUT="${1:-$HERE/../../results/lut-grad-spectra-analysis}"
RAW="${RAW:-$HOME/work/lut_grad_spectra_raw}"      # raw per-model json (not committed)
mkdir -p "$RAW" "$OUT"
cd "$NANO"
export NANOCHAT_BASE_DIR=$HOME/work/nanochat_base TORCH_EXTENSIONS_DIR=$HOME/work/torch_ext/lut_nanochat CUDA_HOME=/usr/local/cuda \
       PYTHONPATH=$PWD HF_HOME=$HOME/work/.cache/huggingface PATH=$PWD/.venv/bin:/usr/local/cuda/bin:$PATH
for tag in dense2000 dense5568 lut1500 lut500 lut500_nodrop; do
  python "$HERE/analyze_model.py" --tag $tag --out "$RAW" --n-held 8 2>&1 | grep -v "^20" | tee "$RAW/$tag.log"
done
# plots/tables use matplotlib (pip --target dir envs/extra-matplotlib, built for uv CPython 3.12)
PY312=$(ls -d $HOME/.local/share/uv/python/cpython-3.12*/bin/python3 | head -1)
MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl PYTHONPATH=$HOME/work/envs/extra-matplotlib "$PY312" "$HERE/make_report.py" --raw "$RAW" --out "$OUT"
