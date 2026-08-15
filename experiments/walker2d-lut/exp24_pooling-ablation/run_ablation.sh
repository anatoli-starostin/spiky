#!/usr/bin/env bash
# exp24 — does the log-sum-exp table pooling change the learned WEIGHT DISTRIBUTION?
#
# The question is exp19's Constraint 2 in the walker2d-spiking post: the soft (log-sum-exp)
# pooling is claimed to be what makes the readout buildable in spikes. This asks the narrower
# empirical question underneath it — with output quantisation and the out-of-band penalty
# BOTH ON, does the pooling operator change where the weights end up?
#
#   arm A  fastlut_lse_sum_expmlpcrit   out = T*tau*log((1/T) sum_t exp(w_t/tau))   (exp19)
#   arm B  fastlut_sum_expmlpcrit       out = sum_t w_t                             (ablation)
#
# The two arches differ in exactly one constructor argument (`exp_outputs`), verified: with
# the same seed every shared tensor is bit-identical, anchors and critic included, and the
# only extra parameter in A is the actor's tau. The sum-scaled log-sum-exp was designed so
# that tau -> inf recovers the plain sum, so B is the tau -> inf limit of A, not a different
# architecture.
#
# Everything else is exp19's recipe under exp23's quantisation regime:
#   --quant-ticks 128 --quant-sigma 1.0     input quantiser (post's Constraint 1)
#   --out-quant-levels 22 --out-quant-clip 1.0   output quantisation (Constraint 3)
#   --oob-penalty 0.1                       the out-of-band penalty (Constraint 4)
#   --obs-clip-vel 10 --solver-iters 100 --ls-iters 50   deployment-matched physics
# trained FROM SCRATCH (no --init-from), because initialising both arms from an LSE-trained
# checkpoint would hand arm B weights already shaped by the operator under test.
#
# NOTE ON `spiky`: the venv resolves `spiky` to the primary checkout, which sits on
# live/walker2d-viz and predates `exp_outputs` — the exp19-family arches cannot even be
# constructed against it. PYTHONPATH must point at this branch's own src/.
#
# OUT may be overridden to keep checkpoints out of a checkout.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SRC=$(cd "$HERE/../src" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
OUT=${OUT:-$HERE/runs}
cd "$SRC"
export PYTHONPATH="$REPO/src"
export WARP_CACHE_PATH=/tmp/warp_cache TRITON_CACHE_DIR=/tmp/triton_cache
PY=${PY:-$REPO/.venv/bin/python}
UPDATES=${UPDATES:-768}
SEEDS=${SEEDS:-"0 1 2"}

mkdir -p "$OUT"
echo "SRC=$SRC"; echo "OUT=$OUT"; echo "PYTHONPATH=$PYTHONPATH"

COMMON=(--tables-per-head 32 --envs 8192 --graph --updates "$UPDATES"
        --lr 3e-4 --lr-schedule cosine --lr-min 3e-5 --logstd-min -1.897
        --ent-coef 0.0 --target-kl 0.02 --norm-returns
        --obs-clip-vel 10 --solver-iters 100 --ls-iters 50
        --quant-ticks 128 --quant-sigma 1.0
        --out-quant-levels 22 --out-quant-clip 1.0 --oob-penalty 0.1)

start=$SECONDS
pids=()
for s in $SEEDS; do
  for arm in A B; do
    if [ "$arm" = "A" ]; then ARCH=fastlut_lse_sum_expmlpcrit; else ARCH=fastlut_sum_expmlpcrit; fi
    $PY -u ppo_qat_obs.py --arch "$ARCH" "${COMMON[@]}" --seed "$s" \
        --out "$OUT/${arm}_s${s}.json" --save-model "$OUT/${arm}_s${s}.pt" \
        > "$OUT/${arm}_s${s}.log" 2>&1 &
    pids+=("$!")
  done
done
wait "${pids[@]}"
echo "PARALLEL_WALL_S=$((SECONDS - start))"
ls -la "$OUT"
echo "ALL DONE $(date -u +%H:%M:%SZ)"
