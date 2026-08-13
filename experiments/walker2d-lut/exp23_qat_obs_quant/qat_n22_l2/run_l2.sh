#!/usr/bin/env bash
# exp23 — L2 OUT-OF-BAND PENALTY variant. Identical to qat_n22_full except for --oob-penalty.
#
# WHY. The no-penalty run raised return by +411 but left the RAW readout exactly as sprawled
# as the parent (51.6% -> 51.1% outside [-1,1]) and made the LUT weights WIDER (|w|max 0.455
# -> 0.586), so the spiking Stage-3 delay span grew (dmax 84 -> 96, episode 302 -> 314 ticks).
# Nothing was pulling it in: the clamp's gradient is zero outside the band, and warp_env
# clamps in Python before ctrl_cost, so out-of-band is free in physics AND reward. This adds
# the missing live gradient:  sum_o relu(|mu_o| - 1)^2, averaged over the batch.
#
# WEIGHT CHOSEN BY MEASUREMENT (sweep/, 20 updates each, seed 0):
#     w      out_oob@20   ep_ret@20   kl@20   epochs
#   0.003      50.88%       2368.8    0.0094    4
#   0.010      48.76%       2376.3    0.0099    4
#   0.030      44.17%       2355.2    0.0120    4
#   0.100      39.30%       2297.6    0.0149    4     <- chosen
#   0.300      36.00%       2296.3    0.0285    4        (KL near the 0.03 stop; diminishing)
#   none       51.78%          --     0.0308   2-3
# 0.1 gives most of the achievable reduction while leaving plenty of KL headroom; 0.3 is
# carried as a single-seed bracket to check whether more is actually better over 384 updates.
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp23_qat_obs_quant/qat_n22_l2
CK=/home/astarostin/projects/ckpt_backups/exp19_walker2d_lut/deploy_matched/actor_s2.pt
PY=/home/astarostin/projects/spiky/.venv/bin/python
PROG=/home/astarostin/work/slack-facade/progress.py
TASK=07088000
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

H=$(python3 "$PROG" start --task "$TASK" --label "exp23 L2 oob-penalty · 384 upd" \
      --stats "starting (w=0.1 x3 seeds + w=0.3 bracket)")
echo "progress handle: $H"

start=$SECONDS
pids=(); names=()
for s in 0 1 2; do
  $PY -u ppo_qat_obs.py "${COMMON[@]}" --seed "$s" --oob-penalty 0.1 \
      --out "$OUT/l2w0p1_s${s}.json" --save-model "$OUT/l2w0p1_s${s}.pt" \
      > "$OUT/l2w0p1_s${s}.log" 2>&1 &
  pids+=("$!"); names+=("l2w0p1_s${s}")
done
$PY -u ppo_qat_obs.py "${COMMON[@]}" --seed 0 --oob-penalty 0.3 \
    --out "$OUT/l2w0p3_s0.json" --save-model "$OUT/l2w0p3_s0.pt" \
    > "$OUT/l2w0p3_s0.log" 2>&1 &
pids+=("$!"); names+=("l2w0p3_s0")
echo "launched: ${pids[*]}"

# The bar is driven entirely by the training logs -- on-policy ep_ret and out_oob are already
# printed every 10 updates at ZERO extra physics cost, so no interim eval is run at all.
while true; do
  alive=0
  for p in "${pids[@]}"; do kill -0 "$p" 2>/dev/null && alive=$((alive+1)); done
  mn=$UPDATES; st=""
  for n in "${names[@]}"; do
    u=$(grep -o "\[upd *[0-9]*/" "$OUT/$n.log" 2>/dev/null | tail -1 | grep -o "[0-9]*" | tail -1)
    u=${u:-0}; [ "$u" -lt "$mn" ] && mn=$u
    r=$(grep -o "ep_ret *[0-9.]*" "$OUT/$n.log" 2>/dev/null | tail -1 | grep -o "[0-9.]*$")
    st="$st ${n#l2}=${r:-—}"
  done
  el=$((SECONDS - start))
  [ "$alive" -eq 0 ] && break
  eta="?"; [ "$mn" -gt 0 ] && eta=$(( el * (UPDATES - mn) / mn / 60 ))m
  python3 "$PROG" update "$H" --step "$mn" --total "$UPDATES" \
      --stats "ep_ret$st · ${alive}/4 running · eta ~${eta}" >/dev/null 2>&1
  sleep 20
done
wait "${pids[@]}"
wall=$((SECONDS - start))
echo "TRAIN_DONE ${wall}s"

# ---- CHEAP screening pass first: 256 envs x 600 steps at DEFAULT physics (solver 10/8) ----
python3 "$PROG" update "$H" --step "$UPDATES" --total "$UPDATES" \
    --stats "training done (${wall}s) · cheap screen" >/dev/null 2>&1
echo "=== CHEAP SCREEN (256 envs x 600 steps, default physics 10/8 — indicative only) ==="
for n in "${names[@]}"; do
  $PY -u eval_qat_ckpt.py --ckpt "$OUT/$n.pt" --envs 256 --steps 600 \
      --solver-iters 10 --ls-iters 8 --quant-ticks 128 --quant-sigma 1.0 \
      --out-quant-levels 22 > "$OUT/screen_$n.log" 2>&1
  echo "--- $n"; grep "out-quant 22" "$OUT/screen_$n.log" | grep -v "load on device"
done

# ---- the ONE high-fidelity headline eval: 1024 envs x 2000 steps, MATCHED physics ----
python3 "$PROG" update "$H" --step "$UPDATES" --total "$UPDATES" \
    --stats "headline eval (1024x2000, matched physics)" >/dev/null 2>&1
echo "=== HEADLINE EVAL (1024 envs x 2000 steps, matched physics 100/50) ==="
for n in "${names[@]}"; do
  $PY -u eval_qat_ckpt.py --ckpt "$OUT/$n.pt" --envs 1024 --steps 2000 \
      --quant-ticks 128 --quant-sigma 1.0 --out-quant-levels 22 \
      > "$OUT/eval_$n.log" 2>&1
  echo "--- $n"; grep -E "quant" "$OUT/eval_$n.log" | grep -v "load on device"
done

echo "=== RAW READOUT PROBE (the point of the whole variant) ==="
mkdir -p "$OUT/probe"
for n in "${names[@]}"; do
  $PY -u probe_raw_readout.py --ckpt "$OUT/$n.pt" --label "$n" \
      --out "$OUT/probe/$n.json" 2>&1 | grep -vE "load on device|Warp|CUDA|Devices|cpu|cuda:0|Kernel|/tmp/warp"
done

python3 "$PROG" done "$H" --text "L2 oob-penalty run complete in ${wall}s" >/dev/null 2>&1
echo "ALL_DONE $((SECONDS - start))s"
