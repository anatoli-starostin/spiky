"""exp_c11 — draw anchor pairs with lutorch's OWN sampler and cache them (#75). SPIKY venv.

Why a separate process: the LUT-SAC trainer runs in the MJX venv
(~/projects/walker2d_mjx/.venv), which deliberately has NO torch. lutorch's sampler is
torch code, and torch's RNG stream cannot be reproduced in numpy — `torch.rand(...)
.argsort()` and `torch.randperm` draw from a Philox/MT19937 stream that numpy does not
implement. So "port it to numpy" cannot be made EXACT; the only way to get the identical
draw is to run the real thing and hand the indices across as data.

This script does that: it calls `get_balanced_anchor_pairs` for the requested policy and
writes anchor_a/anchor_b (both [n_tables, nap], long) to a cache .npz keyed by every
argument that changes the draw. `jax_lut_ext.anchor_pair_wb_lutorch` reads that file
(and shells out to this script through the spiky venv if it is missing).

Policies, all dispatched by the same lutorch entry point:
  balanced                 — the one this task asked for. Balances each coordinate's
                             usage GLOBALLY across the whole n_tables*nap stream; a and
                             b drawn independently, a==b repaired by up to 10 rounds.
  canonical_full_coverage  — WHAT FastMultiHeadLut ACTUALLY USES (its default; it raises
                             on `balanced`). Draws distinct canonical pairs a<b, covering
                             all C(input_dim,2) when there are enough slots.
  canonical_distinct       — per-table distinct canonical pairs, no coverage guarantee.
  connected                — balanced draw for a, b = roll(a, -1).
"""
import argparse, os

import numpy as np
import torch

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs

POLICIES = {p.value: p for p in AnchorSamplingPolicy}


def cache_name(policy, n_tables, nap, input_dim, heads, seed, device):
    """Every argument that can change the draw is in the filename. `heads` matters
    only for the canonical policies (they draw per head), but it is included for all
    of them so a stale file can never be picked up after a shape change.

    DEVICE IS PART OF THE KEY, and that is not defensive over-keying: lutorch seeds a
    `torch.Generator(device=...)`, and CUDA and CPU generators produce DIFFERENT
    streams from the same seed. So the same (policy, shape, seed) genuinely yields
    different anchors on cpu and cuda. A module built on the GPU — as
    FastMultiHeadLut is when device="cuda" — must be reproduced with device=cuda."""
    return (f"anchors_{policy}_t{n_tables}_nap{nap}_d{input_dim}"
            f"_h{heads}_s{seed}_{device}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tables", type=int, required=True)
    ap.add_argument("--nap", type=int, required=True)
    ap.add_argument("--input-dim", type=int, required=True)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="balanced", choices=sorted(POLICIES))
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"],
                    help="which torch generator draws — CHANGES THE RESULT (see "
                         "cache_name); use cuda to reproduce a GPU-built module")
    ap.add_argument("--cache-dir",
                    default=os.path.expanduser("~/.cache/spiky_anchors"))
    a = ap.parse_args()

    os.makedirs(a.cache_dir, exist_ok=True)
    idx_a, idx_b = get_balanced_anchor_pairs(
        n_tables=a.n_tables, n_anchor_pairs=a.nap, input_dim=a.input_dim,
        device=torch.device(a.device), random_seed=a.seed,
        policy=POLICIES[a.policy], n_heads=a.heads)

    a_np = idx_a.cpu().numpy().astype(np.int64)
    b_np = idx_b.cpu().numpy().astype(np.int64)
    # A degenerate pair would silently turn a comparator bit into a constant, so
    # refuse to cache one rather than let it reach a training run.
    if (a_np == b_np).any():
        raise SystemExit(
            f"{int((a_np == b_np).sum())} degenerate pair(s) with a == b survived "
            f"lutorch's repair for policy={a.policy}; refusing to cache.")

    path = os.path.join(a.cache_dir, cache_name(a.policy, a.n_tables, a.nap,
                                                a.input_dim, a.heads, a.seed,
                                                a.device))
    np.savez(path, anchor_a=a_np, anchor_b=b_np, policy=a.policy, seed=a.seed,
             n_tables=a.n_tables, nap=a.nap, input_dim=a.input_dim, heads=a.heads,
             device=a.device, torch_version=torch.__version__)
    uniq = len(np.unique(np.stack([np.minimum(a_np, b_np),
                                   np.maximum(a_np, b_np)], -1).reshape(-1, 2), axis=0))
    print(f"{a.policy} seed={a.seed} {a.n_tables}x{a.nap} over d={a.input_dim}: "
          f"{uniq} distinct unordered pairs -> {path}", flush=True)


if __name__ == "__main__":
    main()
