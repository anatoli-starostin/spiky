#!/usr/bin/env bash
# VM launcher: run the two SOM arms sequentially on 89.169.110.97 (single H100). Each writes
# its own train.log with a line-start `=== EXIT <code>` marker. Detached on the VM.
set -u
export TRITON_CACHE_DIR="$HOME/.cache/triton"
export NANOCHAT_ROOT="$HOME/projects/nanochat"
export MPLCONFIGDIR="$HOME/.cache/mpl_som"
export TZ=Asia/Jerusalem
mkdir -p "$TRITON_CACHE_DIR" "$MPLCONFIGDIR"
BASE="$HOME/projects/spiky/experiments/ffn_replacement/runs_corrected"
for RUN in exp_n_0235_pureLUT0193_SOMannealed_16k_seed1 exp_n_0236_pureLUT0193_SOMexpstrong_16k_seed1; do
    cd "$BASE/$RUN" || continue
    echo "[vm-som] starting $RUN at $(date)"
    "$HOME/venv/bin/python" train.py > "$BASE/$RUN/train.log" 2>&1
    printf '\n=== EXIT %s\n' "$?" >> "$BASE/$RUN/train.log"
    echo "[vm-som] done $RUN at $(date)"
done
echo "[vm-som] both SOM arms complete"
