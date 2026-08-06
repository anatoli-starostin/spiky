#!/usr/bin/env bash
# exp_c53 - the DETACHED-HARD crossing: exp_c50 with SPIKE_FORM="detach_hard".
#
# Everything else is exp_c50 verbatim: 1 head x 128 tables x 1 detector x 16 buckets,
# per-table betas, stock 0.1 table init, delay_init_std=0, freeze_temperature=False, the
# delay clamp with its LOWER bound removed and the upper t_window cap kept, SORT_FORM=rank,
# seeds 0/1/2 - the same three c36, c48, c49 and c50 ran.
#
# THE CHANGE. `t_soft := t_hard`. The soft bucket partition is fed the ACTUAL first-crossing
# arrival instead of the T_cross-weighted expectation over all N arrivals. The crossing
# index is an argmax over a boolean and carries no gradient; the arrival VALUE still does,
# so delay/w/tau keep their route through the membrane kernel and the buckets stay soft
# under T_bkt.
#
# WHAT THE CPU CHECKS ALREADY ESTABLISHED, before any GPU time:
#   * "stop_gradient on the ordering" is a NO-OP here. Wrapping the permutation in an
#     explicit stop_gradient changes no gradient in either variant (0.000e+00 on all 8
#     parameters). The reorder decision was never differentiable - `rank` builds its
#     permutation from integer comparisons, `argsort` from integer indices. What this
#     variant actually removes is the SOFT CROSSING.
#   * The soft surrogate becomes far more faithful: argmax of the soft partition agrees
#     with the hard digit it stands in for 40.6% of the time under SPIKE_FORM="soft" and
#     99.2% under "detach_hard". t_soft sat a mean of 2.60 (max 23.2) from the real
#     crossing.
#   * It costs 2,432 parameters. `w_raw` (2,176) and `tau_raw` (128) reach the output ONLY
#     through the membrane potential V, and V now only picks a detached index - so the
#     synaptic weights and time constants stop learning entirely. `log_T_cross` (128) goes
#     unused. Parity asserts all three are dead on BOTH sides: 122/122.
#
# So this run asks whether a faithful address gradient over a crippled front-end beats an
# unfaithful one over a whole front-end.
set -u
cd "$(dirname "$0")"

PY="$HOME/projects/walker2d_mjx/.venv/bin/python"
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cell () {
  local S=$1 TAG="_c53_s$1"
  echo "=== seed $S: training ($(date -u +%H:%M:%SZ)) ==="
  $PY -u mhl_sac.py --seed "$S" --tag "$TAG" > "cell_s${S}.log" 2>&1
  echo "=== seed $S: CPU reference ($(date -u +%H:%M:%SZ)) ==="
  $PY -u eval_mhl_cpu.py "mhl_sac${TAG}_actor.npz" --episodes 100 >> "cell_s${S}.log" 2>&1
  echo "=== seed $S: done ($(date -u +%H:%M:%SZ)) ==="
}

for S in 0 1 2; do cell "$S" & done
wait

touch SWEEP_DONE_C53
echo "=== sweep done $(date -u +%H:%M:%SZ) ==="
