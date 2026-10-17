#!/bin/bash
# Sidecar checkpoint pruner for ONE run: d24-lutffn-confn1-projmuonfp8-lrrs-1xh100-s0 (worker VM).
# Re-applies the designed KEEP_CKPTS=2 behaviour that the launcher's in-script pruner fails to (it dies under
# set -euo pipefail on its first pass, before any checkpoint exists).
# Deliberately NO set -e / pipefail: every failure is tolerated and the loop keeps going.
# Usage: sidecar_pruner.sh --once | --dry | --loop
set +e +u +o pipefail 2>/dev/null

CK="/home/nebius-h100/projects/spiky/experiments/lut_nanochat/results/d24-lutffn-confn1-projmuonfp8-lrrs-1xh100-s0/checkpoints"
KEEP=2
MIN_AGE=60                       # never touch a file modified in the last 60 s
LAUNCHER_PID=550170              # loop exits (after a final pass) once the run's launcher is gone
LOG="/home/nebius-h100/projmuon_sidecar_pruner.log"

log(){ echo "$(date -u +%FT%TZ) $*" >> "$LOG"; }

# ---- scope guard: structurally refuse anything but this one directory ----
case "$CK" in *nap9*|*v2*|*full*) log "REFUSE: path guard tripped"; exit 2;; esac
if [ ! -d "$CK" ]; then
  if [ "$1" = "--loop" ]; then
    log "WAIT: $CK does not exist yet (no checkpoint written so far); polling every 60s"
    while [ ! -d "$CK" ]; do kill -0 "$LAUNCHER_PID" 2>/dev/null || { log "LOOP END: launcher gone before any checkpoint"; exit 0; }; sleep 60; done
  else
    log "NOTE: $CK does not exist yet -> nothing to do"; exit 0
  fi
fi
real=$(readlink -f "$CK" 2>/dev/null)
if [ "$real" != "$CK" ]; then
  log "REFUSE: checkpoint dir '$CK' resolves to '$real' (not the expected real path)"; echo "REFUSE (see $LOG)"; exit 2
fi

now(){ date +%s; }
age_ok(){ local f="$1" m; m=$(stat -c %Y "$f" 2>/dev/null) || return 1; [ $(( $(now) - m )) -ge $MIN_AGE ]; }

# a step's set is COMPLETE when model + meta + optim rank0 all exist and none is younger than MIN_AGE
complete(){ local s="$1"
  local fm="$CK/model_$s.pt" fj="$CK/meta_$s.json" fo="$CK/optim_${s}_rank0.pt"
  [ -s "$fm" ] && [ -s "$fj" ] && [ -s "$fo" ] && age_ok "$fm" && age_ok "$fj" && age_ok "$fo"; }

pass(){ local mode="$1"
  # every step that has ANY file of a set (model / meta / optim), numerically sorted
  local steps; steps=$(ls -1 "$CK" 2>/dev/null | sed -nE 's/^(model|meta|optim)_([0-9]+)(_rank[0-9]+)?\.(pt|json)$/\2/p' | sort -un)
  local comp=(); local s
  for s in $steps; do complete "$s" && comp+=("$s"); done
  local n=${#comp[@]}
  if [ "$n" -le "$KEEP" ]; then
    log "PASS($mode): ${n} complete set(s) [${comp[*]}], all steps [$(echo $steps)] -> nothing to delete"; return 0
  fi
  local keep=("${comp[@]: -$KEEP}"); local oldest_kept=${keep[0]}
  log "PASS($mode): complete [${comp[*]}] keep [${keep[*]}] -> delete sets with step < $oldest_kept"
  for s in $steps; do
    [ "$((10#$s))" -lt "$((10#$oldest_kept))" ] || continue
    local f freed=0 files=()
    for f in "$CK/model_$s.pt" "$CK/meta_$s.json" "$CK"/optim_"$s"_rank*.pt; do
      [ -e "$f" ] || continue
      if ! age_ok "$f"; then log "  SKIP $f (modified < ${MIN_AGE}s ago)"; continue; fi
      files+=("$f"); freed=$(( freed + $(stat -c %s "$f" 2>/dev/null || echo 0) ))
    done
    [ ${#files[@]} -gt 0 ] || continue
    if [ "$mode" = dry ]; then log "  WOULD DELETE step $s: ${files[*]##*/} (${freed} bytes)"
    else rm -f -- "${files[@]}"; log "  DELETED step $s: ${files[*]##*/} freed=${freed} bytes ($(awk -v b=$freed 'BEGIN{printf "%.2f GB",b/1e9}'))"; fi
  done
  log "  df after: $(df -h /home | tail -1)"
}

case "$1" in
  --dry)  pass dry ;;
  --once) pass once ;;
  --loop)
    log "LOOP START pid=$$ every 300s, keep=$KEEP, dir=$CK"
    while true; do
      pass loop
      if ! kill -0 "$LAUNCHER_PID" 2>/dev/null; then pass final; log "LOOP END: launcher $LAUNCHER_PID gone"; exit 0; fi
      sleep 300
    done ;;
  *) echo "usage: $0 --dry|--once|--loop"; exit 1 ;;
esac
