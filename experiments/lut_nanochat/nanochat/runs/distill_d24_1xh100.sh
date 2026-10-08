#!/bin/bash
# [lut_nanochat] Online logits-DISTILLATION: fresh d24 student trained on the frozen d24 baseline's logits,
# on ONE H100 80GB. Warm-up for LUT students; proves the distillation harness end-to-end.
#
# Identical machinery to runs/baseline_d24_1xh100.sh (same data, checkpoint/symlink discipline, resume,
# wandb, standalone base_eval) EXCEPT base_train runs with the distillation flags, so the loss is the
# temperature-scaled KL( teacher || student ) (+ optional hard-CE blend) instead of plain cross-entropy.
#   Teacher: model-tag $TEACHER_TAG @ step $TEACHER_STEP (frozen, eval, no-grad) — the trained baseline.
#   Student: fresh d24, identical architecture, random init (base_train builds it as usual).
#
# Usage (from experiments/lut_nanochat/nanochat), AFTER runs/stage_data.sh and with the teacher checkpoint
# present at $NANOCHAT_BASE_DIR/base_checkpoints/$TEACHER_TAG (the symlink into the baseline run is fine):
#   bash runs/distill_d24_1xh100.sh              # fresh start / resume-from-latest
#   DBS=8 bash runs/distill_d24_1xh100.sh        # if the memory probe says batch 16 does not fit (two fwd passes)
# Resume is automatic (rerun the same command). Every resume is logged to $RESULTS/resumes.log.

set -euo pipefail
cd "$(dirname "$0")/.."                                   # vendored nanochat root

# Student run identity (distinct from the baseline so artifacts / checkpoints / wandb never collide).
export RUN_NAME=${RUN_NAME:-distill_d24_from_d24_1xh100}
export MODEL_TAG=${MODEL_TAG:-distill_d24_from_d24_1xh100}
# Teacher = the trained dense baseline.
TEACHER_TAG=${TEACHER_TAG:-d24_1xh100}
TEACHER_STEP=${TEACHER_STEP:-5568}
DISTILL_T=${DISTILL_T:-1.0}
DISTILL_ALPHA=${DISTILL_ALPHA:-0.0}
EARLY_STOP_BPB=${EARLY_STOP_BPB:-0.719}        # stop once val bpb <= this (teacher's val bpb) ...
EARLY_STOP_PATIENCE=${EARLY_STOP_PATIENCE:-2}  # ... for this many CONSECUTIVE val evals

source runs/lut_env.sh                                    # NANOCHAT_BASE_DIR, pins, RUN_NAME/MODEL_TAG, results dir

DBS=${DBS:-16}
SAVE_EVERY=${SAVE_EVERY:-250}
KEEP_CKPTS=${KEEP_CKPTS:-2}
# Checkpoints live in $RESULTS/checkpoints/ (git-ignored); the canonical cache path is symlinked into it (same
# convention as the baseline launcher). Shared inputs stay in the $NANOCHAT_BASE_DIR cache.
RUN_CKPT_DIR="$RESULTS/checkpoints"
CKPT_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$MODEL_TAG"
mkdir -p "$RESULTS" "$RUN_CKPT_DIR" "$(dirname "$CKPT_DIR")"
if [ -L "$CKPT_DIR" ]; then
    if [ "$(readlink -f "$CKPT_DIR")" != "$(readlink -f "$RUN_CKPT_DIR")" ]; then
        echo "STOP: $CKPT_DIR is a symlink to $(readlink -f "$CKPT_DIR"), not to this run's $RUN_CKPT_DIR"; exit 1
    fi
elif [ -d "$CKPT_DIR" ]; then
    echo "NOTE: $CKPT_DIR is a real directory (old layout); checkpoints stay there for this run"
else
    ln -s "$RUN_CKPT_DIR" "$CKPT_DIR"
fi

# Teacher must exist (its checkpoint is read via $NANOCHAT_BASE_DIR/base_checkpoints/$TEACHER_TAG).
TEACHER_DIR="$NANOCHAT_BASE_DIR/base_checkpoints/$TEACHER_TAG"
if [ ! -f "$TEACHER_DIR/model_$(printf '%06d' "$TEACHER_STEP").pt" ]; then
    echo "STOP: teacher checkpoint not found: $TEACHER_DIR/model_$(printf '%06d' "$TEACHER_STEP").pt"; exit 1
fi
source .venv/bin/activate

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
prune_checkpoints() {
    ls "$CKPT_DIR"/model_*.pt 2>/dev/null | sed -E 's/.*model_([0-9]+)\.pt/\1/' | sort -n | head -n -"$KEEP_CKPTS" |
    while read -r s; do rm -f "$CKPT_DIR/model_$s.pt" "$CKPT_DIR/meta_$s.json" "$CKPT_DIR"/optim_"$s"_rank*.pt; done
}

# Smoke-run hooks (runs/smoke_run_distill.sh); default off so the real run is unaffected.
FINAL_STEP=${NUM_ITERATIONS:-5568}
ITER_ARGS=()
[ -n "${NUM_ITERATIONS:-}" ] && ITER_ARGS=(--num-iterations="$NUM_ITERATIONS")
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
    torchrun --standalone --nproc_per_node=1 -m scripts.base_train -- \
        --depth=24 --target-param-data-ratio=8 --device-batch-size="$DBS" --fp8 \
        --save-every="$SAVE_EVERY" --model-tag="$MODEL_TAG" \
        --distill-from="$TEACHER_TAG" --distill-step="$TEACHER_STEP" \
        --distill-temperature="$DISTILL_T" --distill-alpha="$DISTILL_ALPHA" \
        --distill-early-stop-bpb="$EARLY_STOP_BPB" --distill-early-stop-patience="$EARLY_STOP_PATIENCE" \
        --run="$RUN_NAME" --wandb-project="$WANDB_PROJECT_NAME" --wandb-group="$WANDB_GROUP_NAME" \
        --wandb-tags="arch=distill-student,teacher=$TEACHER_TAG,precision=fp8,seed=42,gpus=1xH100,dbs=$DBS,T=$DISTILL_T,alpha=$DISTILL_ALPHA${EXTRA_TAGS:-}" \
        --wandb-notes-file=runs/lut_wandb_notes.md --pin-config="$RESULTS/pin_config.json" \
        --log-every=1 --core-metric-every=500 "${ITER_ARGS[@]}" "${RESUME_ARGS[@]}" ${TRAIN_EXTRA:-} 2>&1 | tee -a "$RESULTS/train.log"
    kill $PRUNER 2>/dev/null || true
    prune_checkpoints
fi

# Standalone base_eval of the trained student (full CORE), same as the baseline.
torchrun --standalone --nproc_per_node=1 -m scripts.base_eval -- \
    --device-batch-size="$DBS" --model-tag="$MODEL_TAG" ${EVAL_EXTRA:-} 2>&1 | tee "$RESULTS/base_eval.log"
python runs/report_results.py --results "$RESULTS"
