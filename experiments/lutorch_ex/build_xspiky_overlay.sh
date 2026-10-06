#!/usr/bin/env bash
# Build the import overlay train_multi.py needs (see TRAIN_MULTI_LAUNCH.md). Usage:
#   RESEARCH=<checkout of research/ffn_replacement_fix>/src  LUTORCH_EX_COMMIT=<main commit>  XT=<overlay dir> \
#     ./build_xspiky_overlay.sh
# Defaults are gpustar's paths for the Quantised n=1 run.
set -eu
REPO=${REPO:-$HOME/projects/spiky}                                    # any spiky clone that has LUTORCH_EX_COMMIT
RESEARCH=${RESEARCH:-$HOME/projects/spiky-ffnfix-tools/src}          # research/ffn_replacement_fix @ c4a750b2
LUTORCH_EX_COMMIT=${LUTORCH_EX_COMMIT:-e45fd05e}                     # main, with lutorch_ex as merged by #147
S=${S:-$HOME/projects/lx_quant_n1_scratch}
STABLE=${STABLE:-$S/stable_lutorch_ex_$LUTORCH_EX_COMMIT}            # a COPIED snapshot, decoupled from branch switches
XT=${XT:-$S/xspiky_quant_n1}

# 1. snapshot of main's src/spiky/lutorch_ex at a fixed commit (copied, not symlinked into a live checkout)
rm -rf "$STABLE" && mkdir -p "$STABLE"
git -C "$REPO" archive "$LUTORCH_EX_COMMIT" src/spiky/lutorch_ex | tar -x -C "$STABLE"
mv "$STABLE/src/spiky" "$STABLE/spiky" && rmdir "$STABLE/src"

# 2. one synthetic spiky/ package of symlinks: everything from the research checkout, lutorch_ex from the snapshot
rm -rf "$XT"; mkdir -p "$XT/spiky"
for e in "$RESEARCH"/spiky/*; do ln -s "$e" "$XT/spiky/$(basename "$e")"; done
ln -sfn "$STABLE/spiky/lutorch_ex" "$XT/spiky/lutorch_ex"
# spiky must stay a NAMESPACE package: an __init__.py here would stop the cross-checkout submodules resolving.
[ ! -e "$XT/spiky/__init__.py" ] || { echo "ERROR: $XT/spiky/__init__.py must not exist" >&2; exit 1; }
ls -l "$XT/spiky"
