#!/bin/bash
# [lut_nanochat] Dense d24 nanochat baseline on ONE H100 80GB.
#
# Same recipe as runs/speedrun.sh at 92d63d4 (d24, --target-param-data-ratio=8, --fp8, ClimbMix),
# with --nproc_per_node=8 -> 1. Global batch stays 2^20 tokens, 5,568 steps, 5.84B tokens:
# base_train derives batch / LR / WD / steps from param count and token ratio, never from GPU count,
# and only grad_accum_steps changes (dbs 16: 4 -> 32; dbs 8: 64). See ../RUNBOOK.md.
#
# Differences from speedrun.sh, all deliberate:
#   - 1 GPU; periodic checkpoints (--save-every) and resume-from-latest (one command restarts it);
#   - wandb group/tags/notes + full pin set in the config; train metrics logged every step;
#   - stops after base_train + standalone base_eval (no SFT / chat_eval: out of scope).
# Data / tokenizer / eval bundle must already be staged by runs/stage_data.sh.
#
# Usage (from experiments/lut_nanochat/nanochat):
#   bash runs/baseline_d24_1xh100.sh            # fresh start, or resume from the latest checkpoint
#   DBS=8 bash runs/baseline_d24_1xh100.sh      # fallback if the memory probe says batch 16 does not fit
# Resume is automatic: rerun the same command. NOTE: the dataloader resume is APPROXIMATE
# (nanochat/dataloader.py: "approximate resume"), so a resumed run is not bit-identical to an
# uninterrupted one. Every resume is logged to $RESULTS/resumes.log; report it.

set -euo pipefail
cd "$(dirname "$0")/.."                                   # vendored nanochat root
source runs/lut_env.sh                                    # NANOCHAT_BASE_DIR, pins, run name, results dir

DBS=${DBS:-16}
SAVE_EVERY=${SAVE_EVERY:-250}                             # ~35 min of 1xH100 time between checkpoints
KEEP_CKPTS=${KEEP_CKPTS:-2}                               # each checkpoint ~11 GB (5.5 model + 5.7 optim)
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
mkdir -p "$RESULTS" "$CKPT_DIR"
source .venv/bin/activate

# --- the newest COMPLETE checkpoint (save order is model -> meta -> optim; a kill mid-save leaves a
# truncated zip, which zipfile rejects without reading the whole file) ---
latest_complete_step() {
    python - "$CKPT_DIR" <<'PY'
import glob, os, re, sys, zipfile
d = sys.argv[1]
steps = sorted({int(m.group(1)) for p in glob.glob(os.path.join(d, "optim_*_rank0.pt"))
                if (m := re.search(r"optim_(\d+)_rank0\.pt$", p))}, reverse=True)
for s in steps:
    files = [f"model_{s:06d}.pt", f"optim_{s:06d}_rank0.pt"]
    try:
        assert os.path.exists(os.path.join(d, f"meta_{s:06d}.json"))
        for f in files:
            zipfile.ZipFile(os.path.join(d, f)).close()
        print(s); break
    except Exception:
        continue
PY
}

# --- keep only the newest $KEEP_CKPTS step sets (never touches the one being written: it is newest) ---
prune_checkpoints() {
    ls "$CKPT_DIR"/model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | head -n -"$KEEP_CKPTS" |
    while read -r s; do rm -f "$CKPT_DIR/model_$s.pt" "$CKPT_DIR/meta_$s.json" "$CKPT_DIR"/optim_"$s"_rank*.pt; done
}

FINAL_STEP=5568
RESUME_ARGS=()
LATEST=$(latest_complete_step || true)
if [ -n "$LATEST" ]; then
    if [ "$LATEST" -ge "$FINAL_STEP" ]; then
        echo "Training already complete (step $LATEST); skipping to base_eval."
    else
        echo "$(date -Is) resuming from step $LATEST (dataloader resume is approximate)" | tee -a "$RESULTS/resumes.log"
        RESUME_ARGS=(--resume-from-step="$LATEST")
    fi
fi

python runs/make_pin_config.py --dbs "$DBS" --save-every "$SAVE_EVERY" --out "$RESULTS/pin_config.json"

if [ -z "$LATEST" ] || [ "$LATEST" -lt "$FINAL_STEP" ]; then
    ( while sleep 300; do prune_checkpoints; done ) &
    PRUNER=$!
    trap 'kill $PRUNER 2>/dev/null || true' EXIT
    # WANDB_RUN_ID + WANDB_RESUME=allow (from lut_env.sh) make a restart continue the same wandb run.
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
        --depth=24 --target-param-data-ratio=8 --device-batch-size="$DBS" --fp8 \
        --save-every="$SAVE_EVERY" --model-tag="$MODEL_TAG" \
        --run="$RUN_NAME" --wandb-project="$WANDB_PROJECT_NAME" --wandb-group="$WANDB_GROUP_NAME" \
        --wandb-tags="arch=dense,precision=fp8,seed=42,gpus=1xH100,dbs=$DBS" \
        --wandb-notes-file=runs/lut_wandb_notes.md --pin-config="$RESULTS/pin_config.json" \
        --log-every=1 "${RESUME_ARGS[@]}" 2>&1 | tee -a "$RESULTS/train.log"
    kill $PRUNER 2>/dev/null || true
    prune_checkpoints
fi

# --- standalone base_eval: the number we report (full CORE, not the 500-per-task in-training subsample) ---
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- \
    --device-batch-size="$DBS" --model-tag="$MODEL_TAG" 2>&1 | tee "$RESULTS/base_eval.log"
python runs/report_results.py --results "$RESULTS"
