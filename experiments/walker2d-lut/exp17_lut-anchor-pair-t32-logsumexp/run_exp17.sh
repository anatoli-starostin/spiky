#!/usr/bin/env bash
# exp17 — exp10 with the table reduction replaced by a SUM-SCALED temperature-tau
# log-sum-exp (the mean-normalised form, which generalises the SUM rather than the mean):
#
#     out = T * tau * log( (1/T) * sum_tables exp( w_selected / tau ) )    tau > 0 (softplus)
#
#     tau -> inf  =>  sum_t w_t   (exactly exp10's readout)
#     tau -> 0    =>  T * max(w)
#
# Uses the PLAIN ADDITIVE weight init -- at large tau the weights are additive
# contributions again, so no special init is needed (verified: output matches exp10's at
# init to 5.4% of its own std, the residual being the Jensen gap).
#
# Fork of exp10_lut-anchor-pair-t32. Every other flag is exp10's, verbatim from its
# config.json: anchor-pair actor tph=32 + MLP critic, bench7 recipe (truncation bootstrap,
# return-norm, KL early-stop, cosine LR -> 3e-5, log_std floor), 8192 envs, 768 updates,
# 3 seeds in parallel. Only --arch changes: fastlut -> fastlut_lse.
#
# TWO EARLIER ATTEMPTS at the PLAIN (non-sum-scaled) readout are preserved and were both
# abandoned; neither is this experiment, but both are kept because they localise the cause:
#   attempt1_additive_init/          tau=0.1, additive init -> head pinned at the constant
#                                    tau*log(32)=0.347, 32x-too-small spread, uniform 1/32
#                                    gradients. Final 495. (3/3 JSONs.)
#   attempt2_plain_lse_logspace_init/ tau=0.05, log-space init that restored exp10's exact
#                                    starting statistics (mean +0.000256 vs +0.000259, std
#                                    ratio 0.987) -- and STILL plateaued at ~350, which is
#                                    what proved the fault was the readout, not the init.
#                                    Stopped at ~update 430; logs only, no JSONs.
# Both are reproducible via --arch fastlut_lse with exp_outputs_init="additive"/"logspace".
set -uo pipefail
SRC=/home/astarostin/projects/spiky/experiments/walker2d-lut/src
OUT=/home/astarostin/projects/spiky/experiments/walker2d-lut/exp17_lut-anchor-pair-t32-logsumexp
cd "$SRC"
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=/home/astarostin/projects/spiky/.venv/bin/python

mkdir -p "$OUT"

( while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits; sleep 2; done > "$OUT/agg.gpu" 2>/dev/null ) &
SAMP=$!

start=$SECONDS
pids=()
for s in 0 1 2; do
  $PY -u train.py --algo ppo --arch fastlut_lse_sum --tables-per-head 32 --envs 8192 --graph \
      --updates 768 --seed "$s" --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897 \
      --ent-coef 0.0 --target-kl 0.02 --norm-returns --out "$OUT/ppo_s${s}.json" \
      > "$OUT/ppo_s${s}.log" 2>&1 &
  pids+=("$!")
done
wait "${pids[@]}"
wall=$((SECONDS - start))
kill $SAMP 2>/dev/null

echo "PARALLEL_WALL_S=${wall}"
awk -F, '{u+=$1;n++;if($1>mu)mu=$1;if($2>mem)mem=$2} END{printf "AGG_GPU mean=%.0f%% max=%.0f%% maxmem=%.0fMB n=%d\n",u/n,mu,mem,n}' "$OUT/agg.gpu"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
