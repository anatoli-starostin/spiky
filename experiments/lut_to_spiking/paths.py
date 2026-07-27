"""Shared path resolution for the lut_to_spiking probe series.

Everything the series writes (captures, figures, result JSONs) lands next to the
scripts; everything it reads from outside the repo is overridable by env var so the
series is runnable on another machine without editing code.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# nanochat checkout — supplies the tokenizer, val data loader and bpb eval.
NANOCHAT_ROOT = os.environ.get(
    "NANOCHAT_ROOT", os.path.expanduser("~/projects/nanochat"))

# exp025 checkpoint (single-stream, Linear unembedder, FIXED FastMHL anchors,
# val bpb 1.2408). NOT in the repo — checkpoints are never committed.
EXP025_CKPT = os.environ.get(
    "EXP025_CKPT",
    os.path.expanduser("~/Downloads/lut_checkpoints/exp025_checkpoint.pt"))

# In-repo trained HyperplaneMHL checkpoint used by the synthetic study (t05-t09).
# Also not in git (experiments/**/*.pt is ignored) — regenerate by rerunning exp011.
EXP011_CKPT = os.path.join(
    REPO, "experiments", "hyperplane_ffn",
    "exp011_hyperplane_mhl_ffn_nap6_tph256_stack2_ln_resid", "checkpoint.pt")


def out(name):
    """Absolute path for an artefact this series writes."""
    return os.path.join(HERE, name)
