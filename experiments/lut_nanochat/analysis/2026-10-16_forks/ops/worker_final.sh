#!/bin/bash
# Worker final: stop projmuon cleanly, final pruner pass + stop pruner, tag wandb, cloud-sync check,
# git state, and the worker-only data at risk. Deletes NOTHING except via the pruner's own KEEP=2 rule.
R=$HOME/projects/spiky/experiments/lut_nanochat
N=$R/nanochat
L=$R/results/projmuonfp8_lrrs.nohup.log
RD=$R/results/d24-lutffn-confn1-projmuonfp8-lrrs-1xh100-s0
PARENT=550170; TR=550208; BP=550218
RUNID=d24-lutffn-confn1-projmuonfp8-lrrs-1xh100-s0-036e09af
NAP9ID=d24-lutffn-confn1-h32-tph32-nap9-1xh100-s0-04a80c7b
echo "######## STOP projmuon ########"
echo "PRE: $(grep -E '^step [0-9]+/' "$L" | tail -1 | cut -c1-80)"; date -u +%FT%TZ
START=$(date -d "$(ps -o lstart= -p $PARENT)" +%s 2>/dev/null)
WC=$(pgrep -P $BP -f wandb-core | head -1); echo "base_train=$BP torchrun=$TR wandb-core=$WC"
kill -TERM $PARENT 2>/dev/null && echo "TERM -> launcher $PARENT"
kill -INT $BP 2>/dev/null && echo "INT -> base_train $BP"
for k in $(seq 1 48); do kill -0 $BP 2>/dev/null || { echo "base_train exited after ~$((k*5))s"; break; }; sleep 5; done
kill -0 $BP 2>/dev/null && { echo "alive after 240s -> TERM"; kill -TERM $BP; sleep 30; }
kill -0 $BP 2>/dev/null && { echo "-> KILL"; kill -KILL $BP; }
for k in $(seq 1 36); do { [ -n "$WC" ] && kill -0 $WC 2>/dev/null; } || break; sleep 5; done
{ [ -n "$WC" ] && kill -0 $WC 2>/dev/null; } && echo "wandb-core STILL running" || echo "wandb-core exited"
END=$(date +%s); echo "WALL hours=$(awk -v a=$START -v b=$END 'BEGIN{printf "%.2f",(b-a)/3600}')"
sleep 3
echo "PROCS_LEFT: [$(ps -p $PARENT,$TR,$BP -o pid=,cmd= 2>/dev/null | cut -c1-60 | tr '\n' ';')]"
echo "GPU: $(nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader) | apps: [$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ';')]"
echo "LAST: $(grep -E '^step [0-9]+/' "$L" | tail -1 | cut -c1-80)"
STOPSTEP=$(grep -E '^step [0-9]+/' "$L" | tail -1 | grep -oE '^step [0-9]+' | grep -oE '[0-9]+' | sed 's/^0*//')
echo "######## PRUNER: final pass, then stop ########"
bash $HOME/projmuon_sidecar_pruner.sh --once
kill -TERM 571238 2>/dev/null && echo "TERM -> pruner loop 571238"; sleep 2
pgrep -af "[p]rojmuon_sidecar_pruner.sh" && echo "PRUNER STILL ALIVE" || echo "pruner loop gone"
tail -4 $HOME/projmuon_sidecar_pruner.log
ls -la "$RD/checkpoints" | grep -E "model_|meta_|optim_"
echo "######## WANDB tag + cloud state ########"
sleep 15
cd "$N" && .venv/bin/python - "$RUNID" "$STOPSTEP" "$NAP9ID" <<'PY' 2>&1 | grep -v "Currently logged in"
import sys, time, wandb
rid, step, nap9 = sys.argv[1], sys.argv[2], sys.argv[3]
api = wandb.Api()
r = api.run(f"anatoli-starostin-relocation/spiky-nanochat/{rid}")
assert r.id == rid
r.tags = sorted(set(list(r.tags) + [f"stopped-early-at-{step}"]))
note = f"Stopped deliberately at step {step} to free the machines (VM shutdown). NOT a crash."
r.notes = (r.notes + " | " if r.notes else "") + note
r.update(); time.sleep(5)
for i in (rid, nap9):
    x = api.run(f"anatoli-starostin-relocation/spiky-nanochat/{i}")
    print(f"CLOUD {i}: state={x.state} _step={x.summary.get('_step')} val/bpb={x.summary.get('val/bpb')} "
          f"tags_has_stop={[t for t in x.tags if t.startswith('stopped-early')]} url={x.url}")
PY
echo "######## DATA ########"
echo "--- val ---"; grep -E "Validation bpb" "$L" | tr "\n" ";"; echo
echo "--- core ---"; grep -E "CORE metric" "$L" | tr "\n" ";"; echo
echo "--- loss ---"; for s in $(seq 250 250 5500); do grep -E "^step $(printf %05d $s)/" "$L" | grep -oE "^step [0-9]+|loss: [0-9.]+" | tr "\n" " "; done; echo
echo "--- dt last 20 ---"; grep -E "^step [0-9]+/" "$L" | tail -20 | grep -oE "dt: [0-9.]+" | grep -oE "[0-9.]+" | awk '{s+=$1;n++} END{printf "%.3f s\n", s/n/1000}'
echo "######## WORKER GIT ########"
cd $HOME/projects/spiky
echo "HEAD=$(git rev-parse --short HEAD) branch=$(git branch --show-current)"; git fetch origin research/lut_nanochat -q 2>&1 | tail -1
echo "ahead of origin: $(git rev-list --count origin/research/lut_nanochat..HEAD)  behind: $(git rev-list --count HEAD..origin/research/lut_nanochat)"
echo "-- tracked modifications --"; git status --porcelain --untracked-files=no
echo "-- untracked (top level, with sizes) --"; git status --porcelain | grep '^??' | cut -c4- | while read -r p; do echo "  $(du -sh "$p" 2>/dev/null | cut -f1)  $p"; done
echo "-- hand-written files in ~ --"; ls -la $HOME/*.sh $HOME/*.py $HOME/*.log $HOME/*.out 2>/dev/null
echo "######## WORKER-ONLY DATA AT RISK ########"
du -sh "$RD" "$R/results/d24-lutffn-confn1-h32-tph32-nap9-1xh100-s0" "$L" "$R/results/nap9_h32tph32.nohup.log" $N/wandb/run-*036e09af $N/wandb/run-*04a80c7b 2>/dev/null
du -sh $HOME/.cache/nanochat 2>/dev/null; df -h /home | tail -1
echo "WORKER_FINAL_DONE"
