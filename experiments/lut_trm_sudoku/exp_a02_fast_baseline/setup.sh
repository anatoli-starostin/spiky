#!/usr/bin/env bash
# exp_a02 one-time prep. Deps + full dataset already exist from exp_a01 (NO network needed).
# Just builds the 5k test subset. Then run launch.sh.
set -euo pipefail
~/projects/TinyRecursiveModels/.venv/bin/python "$(dirname "$0")/subsample_test.py"
