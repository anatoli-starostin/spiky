# [lut_nanochat] Shared environment for every lut_nanochat script. Sourced, not executed.
# Override any variable by exporting it before running a script.
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}   # needs >=100 GB free (RUNBOOK §3)
export NANOCHAT_FA3_REVISION=${NANOCHAT_FA3_REVISION:-$(python3 -c "import json;print(json.load(open('../pins.json'))['fa3']['revision'])")}
export RUN_NAME=${RUN_NAME:-d24-dense-1xh100-s0}
export MODEL_TAG=${MODEL_TAG:-d24_1xh100}
export RESULTS=${RESULTS:-$(cd .. && pwd)/results/$RUN_NAME}
export WANDB_PROJECT_NAME=${WANDB_PROJECT:-nanochat}
mkdir -p "$RESULTS"
# One wandb run id per experiment, persisted so a resume continues the same wandb run.
[ -f "$RESULTS/wandb_run_id" ] || python3 -c "import uuid;print('$RUN_NAME-'+uuid.uuid4().hex[:8])" > "$RESULTS/wandb_run_id"
export WANDB_RUN_ID=$(cat "$RESULTS/wandb_run_id")
export WANDB_RESUME=allow
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -q "api.wandb.ai" "$HOME/.netrc" 2>/dev/null; then
    echo "WARNING: no WANDB_API_KEY in env and no ~/.netrc entry; the run will train but wandb logging will fail (it warns, never aborts)."
fi
