#!/usr/bin/env bash
# exp_c18 — wait for all 6 seeds to finish training, then evaluate and diagnose (#75).
#
# Chained rather than run by hand so the Slack bar's eval stage completes on its own; a
# bar that stalls at "trained, eval pending" for want of a manual step is worse than none.
set -u
cd "$(dirname "$0")"

until grep -q "ALL 6 SEEDS DONE" run_seeds.log; do sleep 60; done
echo "training done $(date -u +%FT%TZ) — evaluating"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/projects/walker2d_mjx/.venv/bin/python" -u collect.py
echo "EVAL DONE $(date -u +%FT%TZ)"

# Diagnostics run unconditionally: they are cheap (read-only, no training) and the
# question "is the addressing still moving at 10k?" is worth answering whether or not the
# spread turns out to be large.
XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/projects/walker2d_mjx/.venv/bin/python" -u diag_seeds.py
echo "DIAG DONE $(date -u +%FT%TZ)"
