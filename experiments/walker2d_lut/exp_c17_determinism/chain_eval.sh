#!/usr/bin/env bash
# exp_c17 — wait for both replicates to finish training, then evaluate them (#75).
#
# Chained rather than run by hand so the live Slack bar's eval stage actually completes
# on its own; a bar that stalls at "trained, eval pending" for want of a manual step is
# worse than no bar.
set -u
cd "$(dirname "$0")"

until grep -q "BOTH DETERMINISTIC RUNS DONE" run_determinism.log; do sleep 60; done
echo "training done $(date -u +%FT%TZ) — evaluating"
XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$HOME/projects/walker2d_mjx/.venv/bin/python" -u collect.py
echo "EVAL DONE $(date -u +%FT%TZ)"
