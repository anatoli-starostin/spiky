#!/bin/bash
# Watch-and-stop for the nap=9 worker run (BODY_TASK 73a24af2). Runs ON the worker under nohup.
# Polls the train log every 60 s (local greps only). Stops the run cleanly once the step-3750 val bpb is
# logged, the step-3750 checkpoint is written, and training has moved past it (the "step 03750/" line prints
# only after eval + save). Stops early if the run dies, goes non-finite, or errors. Never deletes anything.
L=$HOME/projects/spiky/experiments/lut_nanochat/results/nap9_h32tph32.nohup.log
CK=$HOME/.cache/nanochat/base_checkpoints/d24_lutffn_confn1_h32_tph32_nap9_1xh100
N=$HOME/projects/spiky/experiments/lut_nanochat/nanochat
RD=$HOME/projects/spiky/experiments/lut_nanochat/results/d24-lutffn-confn1-h32-tph32-nap9-1xh100-s0
WD=$N/wandb/run-20261016_073329-d24-lutffn-confn1-h32-tph32-nap9-1xh100-s0-04a80c7b
W=$HOME/nap9_watch.log
PARENT=181713; TR=181753; TARGET=3750
RUNID=d24-lutffn-confn1-h32-tph32-nap9-1xh100-s0-04a80c7b
say(){ echo "$(date -u +%FT%TZ) $*" >> "$W"; }
laststep(){ grep -E "^step [0-9]+/" "$L" | tail -1 | cut -c1-80; }

START_EPOCH=$(date -d "$(ps -o lstart= -p $PARENT)" +%s 2>/dev/null)
BP=$(pgrep -P $TR -f "scripts.base_train" | head -1)
say "WATCH START target=$TARGET parent=$PARENT torchrun=$TR base_train=$BP start_epoch=$START_EPOCH | $(laststep)"

REASON=""; i=0
while true; do
  if ! kill -0 $TR 2>/dev/null; then REASON=DEAD; break; fi
  if grep -qE "loss: (nan|inf|-inf)" "$L"; then REASON=NONFINITE; break; fi
  if grep -qE "Traceback|CUDA error|out of memory" "$L"; then REASON=ERROR; break; fi
  if grep -qE "^Step 0${TARGET} \| Validation bpb" "$L" && grep -qE "^step 0${TARGET}/" "$L" \
     && [ -s "$CK/model_00${TARGET}.pt" ] && [ -s "$CK/meta_00${TARGET}.json" ] && [ -s "$CK/optim_00${TARGET}_rank0.pt" ]; then
    REASON=TARGET; break
  fi
  if [ $((i % 40)) -eq 0 ]; then say "STATUS $(laststep) | $(grep 'Validation bpb' "$L" | tail -1)"; fi
  i=$((i+1)); sleep 60
done
say "TRIGGER=$REASON | $(laststep)"
if [ "$REASON" = NONFINITE ]; then say "NONFINITE LINE: $(grep -nE 'loss: (nan|inf|-inf)' "$L" | head -1 | cut -c1-120)"; fi
if [ "$REASON" = ERROR ]; then grep -nE "Traceback|CUDA error|out of memory" "$L" | head -5 >> "$W"; fi

# ---- stop ----
if [ "$REASON" != DEAD ]; then
  kill -TERM $PARENT 2>/dev/null && say "TERM -> launcher bash $PARENT (prevents post-train base_eval)"
  BP=$(pgrep -P $TR -f "scripts.base_train" | head -1)
  WC=$(pgrep -P "$BP" -f wandb-core | head -1)
  kill -INT "$BP" && say "INT -> base_train $BP (wandb-core $WC)"
  for k in $(seq 1 48); do kill -0 "$BP" 2>/dev/null || { say "base_train exited after ~$((k*5))s"; break; }; sleep 5; done
  if kill -0 "$BP" 2>/dev/null; then say "base_train alive after 240s -> TERM"; kill -TERM "$BP"; sleep 30; fi
  if kill -0 "$BP" 2>/dev/null; then say "-> KILL"; kill -KILL "$BP"; fi
  for k in $(seq 1 36); do { [ -n "$WC" ] && kill -0 "$WC" 2>/dev/null; } || break; sleep 5; done
  if [ -n "$WC" ] && kill -0 "$WC" 2>/dev/null; then say "wandb-core STILL running after 180s"; else say "wandb-core exited"; fi
else
  kill -TERM $PARENT 2>/dev/null && say "TERM -> launcher bash $PARENT"
fi
sleep 5
if pgrep -f "scripts.base_eval.*d24_lutffn_confn1_h32_tph32_nap9" >/dev/null; then
  say "post-train base_eval had started -> stopping it"; pkill -INT -f "scripts.base_eval.*d24_lutffn_confn1_h32_tph32_nap9"; sleep 20
fi
END_EPOCH=$(date +%s)
say "PROCS_LEFT: [$(ps -p $PARENT,$TR,${BP:-1} -o pid=,cmd= 2>/dev/null | grep -v '^ *1 ' | cut -c1-60 | tr '\n' ';')]"
say "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) | apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')]"
say "LAST: $(laststep)"
say "WALL: start=$START_EPOCH end=$END_EPOCH hours=$(awk -v a=$START_EPOCH -v b=$END_EPOCH 'BEGIN{printf "%.2f",(b-a)/3600}')"

# ---- wandb tag (TARGET only) + state ----
if [ "$REASON" = TARGET ]; then
  sleep 20
  cd "$N" && .venv/bin/python - "$RUNID" <<'PY' >> "$W" 2>&1
import sys, time, wandb
rid = sys.argv[1]
api = wandb.Api()
r = api.run(f"anatoli-starostin-relocation/spiky-nanochat/{rid}")
assert r.id == rid
r.tags = sorted(set(list(r.tags) + ["stopped-early-at-3750"]))
note = ("Stopped deliberately at step 3750 (after the step-3750 val bpb + checkpoint) for a same-step comparison "
        "against the nap=8 fork (d24-lutffn-confn1-h32-tph32-1xh100-s0, itself stopped at 3778 to free the local GPU). "
        "NOT a crash.")
r.notes = (r.notes + " | " if r.notes else "") + note
r.update()
time.sleep(5)
r2 = api.run(f"anatoli-starostin-relocation/spiky-nanochat/{rid}")
print("WANDB state=", r2.state, "| tags=", list(r2.tags), "| _step=", r2.summary.get("_step"))
PY
fi

# ---- data for the report ----
say "=== DATA ==="
{ echo "--- val bpb ---"; grep -E "Validation bpb" "$L"
  echo "--- CORE ---"; grep -E "CORE metric" "$L"
  echo "--- train loss every 250 ---"
  for s in $(seq 250 250 3750); do grep -E "^step $(printf %05d $s)/" "$L" | grep -oE "^step [0-9]+|loss: [0-9.]+" | tr '\n' ' '; echo; done
  echo "--- dt last 20 train steps ---"; grep -E "^step [0-9]+/" "$L" | tail -20 | grep -oE "dt: [0-9.]+" | grep -oE "[0-9.]+" | tr '\n' ' '; echo
  echo "--- sizes ---"; du -sh "$RD" "$L" "$WD" 2>/dev/null
  echo "ckpt dir resolves to: $(readlink -f "$CK")"
  echo "model ckpts: $(ls "$CK"/model_*.pt 2>/dev/null | wc -l)  optim ckpts: $(ls "$CK"/optim_*.pt 2>/dev/null | wc -l)  newest: $(ls "$CK"/model_*.pt 2>/dev/null | tail -1)"
  echo "disk: $(df -h "$HOME" | tail -1)"
} >> "$W" 2>&1
say "WATCH_FINAL $REASON"
