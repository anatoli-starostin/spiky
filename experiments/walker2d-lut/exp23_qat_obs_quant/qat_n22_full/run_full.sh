#!/usr/bin/env bash
# exp23 — THE FULL COMBINED QAT FINE-TUNE (option 2), 384 updates x 3 seeds.
#
# Parent: deploy_matched/actor_s2.pt (ret 5966.3, seed 2, tau_actor 0.08646934), loaded
# strict together with its obs stats, normaliser frozen at them.
#
# Both quantizers active:
#   INPUT   128-bucket Gaussian companding, sigma 1.0, ONE shared monotone map over all 17
#           coords (a per-coord map would break the LUT's x[a] > x[b] address bits).
#   OUTPUT  22 uniform levels on [-1,1] with clip + STE, applied to the action MEAN only via
#           the forward hook, so log_std and the sampled action stay continuous and the PPO
#           importance ratio remains a density.
#
# LR replays the FULL cosine 3e-4 -> 3e-5 (--init-lr-mode cosine), as specified.
# target-kl stays 0.02: the smoke showed N=22 completes all 4 epochs at this threshold.
#
# 3 seeds because the measured seed-to-seed spread (~+-250) swamps any single-seed delta.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant/qat_n22_full
CK=/home/astarostin/projects/ckpt_backups/exp19_walker2d_lut/deploy_matched/actor_s2.pt
PY=/home/astarostin/projects/spiky/.venv/bin/python
PROG=/home/astarostin/work/slack-facade/progress.py
TASK=8349fa8c
UPDATES=384

cd "$SRC"
export PYTHONPATH=/home/astarostin/projects/spiky/experiments/walker2d-lut/_qat_deps
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
mkdir -p "$OUT"

COMMON=(--arch fastlut_lse_sum_expmlpcrit --tables-per-head 32 --envs 8192 --graph
        --updates "$UPDATES" --lr 3e-4 --lr-schedule cosine --lr-min 3e-5
        --init-lr-mode cosine --logstd-min -1.897 --ent-coef 0.0 --target-kl 0.02
        --norm-returns --obs-clip-vel 10 --solver-iters 100 --ls-iters 50
        --init-from "$CK" --quant-ticks 128 --quant-sigma 1.0
        --out-quant-levels 22 --out-quant-clip 1.0)

H=$(python3 "$PROG" start --task "$TASK" --label "exp23 QAT n22 · 384 upd x 3 seeds" \
      --stats "starting")
echo "progress handle: $H"

start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u ppo_qat_obs.py "${COMMON[@]}" --seed "$s" \
      --out "$OUT/qat_s${s}.json" --save-model "$OUT/qat_s${s}.pt" \
      > "$OUT/qat_s${s}.log" 2>&1 &
  pids+=("$!")
done
echo "launched: ${pids[*]}"

# ---- progress loop: report the SLOWEST seed, so the bar never runs ahead of the work ----
while true; do
  alive=0
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
  mn=$UPDATES; rets=""
  for s in 0 1 2; do
    u=$(grep -o "\[upd *[0-9]*/" "$OUT/qat_s${s}.log" 2>/dev/null | tail -1 \
        | grep -o "[0-9]*" | tail -1); u=${u:-0}
    [ "$u" -lt "$mn" ] && mn=$u
    r=$(grep -o "ep_ret *[0-9.]*" "$OUT/qat_s${s}.log" 2>/dev/null | tail -1 \
        | grep -o "[0-9.]*$"); rets="$rets s$s=${r:-—}"
  done
  el=$((SECONDS - start))
  if [ "$alive" -eq 0 ]; then break; fi
  eta="?"
  if [ "$mn" -gt 0 ]; then eta=$(( el * (UPDATES - mn) / mn / 60 ))m; fi
  python3 "$PROG" update "$H" --step "$mn" --total "$UPDATES" \
      --stats "ep_ret${rets} · ${alive}/3 running · eta ~${eta}" >/dev/null 2>&1
  sleep 20
done
wait "${pids[@]}"
wall=$((SECONDS - start))
echo "TRAIN_DONE ${wall}s"

finals=""
for s in 0 1 2; do
  f=$(grep -o "ep_ret [0-9]* -> [0-9]*" "$OUT/qat_s${s}.log" | tail -1 | awk '{print $NF}')
  finals="$finals s$s=${f:-?}"
done
python3 "$PROG" update "$H" --step "$UPDATES" --total "$UPDATES" \
    --stats "training done (${wall}s):$finals · evaluating" >/dev/null 2>&1

# ---- deterministic eval of every fine-tuned seed, WITH both quantizers active ----
echo "=== EVAL (1024 envs x 2000 steps, deterministic mean action) ==="
for s in 0 1 2; do
  $PY -u eval_qat_ckpt.py --ckpt "$OUT/qat_s${s}.pt" --envs 1024 --steps 2000 \
      --quant-ticks 128 --quant-sigma 1.0 --out-quant-levels 22 \
      > "$OUT/eval_s${s}.log" 2>&1
  echo "--- seed $s (rc=$?)"
  grep -E "quant" "$OUT/eval_s${s}.log" | grep -v "load on device"
done

python3 "$PROG" done "$H" --text "384 upd x 3 seeds in ${wall}s; eval written" >/dev/null 2>&1
echo "ALL_DONE $((SECONDS - start))s"
