"""Read primitives of the reference cartridges: the flat cell index, and the embedding_bag reads.

The gather + sum over a group's ``tph`` tables is one ``torch.nn.functional.embedding_bag`` call (itself a fused
kernel): :func:`fused_hard_read` for the hard value, :func:`fused_blend_read` for the two-cell blend,
:func:`fused_surrogate_read` for the hard cartridge's surrogate. Plain autograd gives their backward. Used by
ManifestoHardLUT / ManifestoSoftLUT; ConfidenceLUT builds on :func:`_global_cells`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# PyTorch's embedding_bag backward size limit: a known upstream constraint, documented here and not enforced.
# The training reads below, ConfidenceLUT's eager read (confidence.py; on CUDA with LUTORCH_EX_NO_COMPILE=1) and
# QuantisedConfidenceLUT's reads (quantised_confidence.py) get their table gradient on CUDA from PyTorch's
# aten::_embedding_bag_dense_backward (kernel compute_grad_weight_bags), which indexes its launch threads in 32 bits.
# It fails when
#     (floor(numel / 10) + min(numel, num_weights)) * 32 * ceil(d_out / 32) > 2^31
# with numel the index-tensor element count of one call, num_weights the table rows and d_out the row width, for int32
# and int64 indices alike. The symptom is an opaque "CUDA error: an illegal memory access was encountered" in the
# training step; stray reads could in principle return wrong gradients instead of crashing.
# It bites far below 2^31 entries: at h=16, tph=64, nap=8, d_out=48 (262,144 rows) the cliff is at ~2^28.3 entries
# (332,922,889 run, 332,922,890 crash; torch 2.9.1, RTX 5090), i.e. ~325k tokens per micro-batch at read_top_n=1 and
# ~163k at read_top_n=2, against a live micro-batch of 32,768 (~10x and ~5x headroom). The remedy is a smaller
# micro-batch with more gradient accumulation.
# Not exposed: ConfidenceLUT's compiled read (its table gradient is an index_add_, _fused_read.py; verified correct on
# the compiled n=2 path past the cliff, to 819M entries where memory ran out first, tests/test_fused_read.py), the
# fused classes' kernels, the CPU, and every eval / no_grad path.

# Low-precision reductions accumulate in float32 and the result is cast back to the caller's dtype
# (the cartridge does that in _route). For bfloat16 / float16 weights the embedding_bag reduce runs on an
# float32 view of the table (torch's embedding_bag has no accumulate-dtype knob), so summing a
# group's tph rows does not lose precision. float32 and float64 keep their own dtype (so the float64
# equivalence tests stay bit-exact — upcasting float32 to float64 or downcasting float64 to float32 would
# both be wrong).
def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """float32 for bfloat16 / float16, else the dtype itself (float32->float32, float64->float64)."""
    return torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype


def _global_cells(c: torch.Tensor, G: int, tph: int, K: int) -> torch.Tensor:
    """Map per-(group,table) cell index c[B,G,tph] -> flat index into a [G*tph*K, d_out] table, in c's dtype (the
    cartridge's ``index_dtype``: int32 by default -- half the saved index bytes and narrower radix-sort keys -- or
    int64). The values fit c's dtype by construction (resolve_index_dtype checks n_groups * tph * 2^nap)."""
    base = (torch.arange(G, device=c.device, dtype=c.dtype).view(G, 1) * tph
            + torch.arange(tph, device=c.device, dtype=c.dtype).view(1, tph)) * K   # [G, tph]
    return c + base  # [B, G, tph]


def fused_hard_read(weights: torch.Tensor, c: torch.Tensor, drop_mask=None) -> torch.Tensor:
    """y[b,g] = sum_t W[g,t,c[b,g,t]]  via one embedding_bag. weights [G,tph,K,d_out] -> [B,G,d_out] float32.
    ``drop_mask`` [B,G,tph] (optional): per-table keep weights folded in as per_sample_weights, so
    dropped tables contribute 0 — table dropout honoured inside the fused read (no new kernel)."""
    G, tph, K, d_out = weights.shape
    B = c.shape[0]
    acc = _acc_dtype(weights.dtype)
    W2 = weights.reshape(G * tph * K, d_out).to(acc)
    gc = _global_cells(c, G, tph, K).reshape(B * G, tph)
    psw = None if drop_mask is None else drop_mask.reshape(B * G, tph).to(acc)
    return F.embedding_bag(gc, W2, mode="sum", per_sample_weights=psw).reshape(B, G, d_out)


def fused_blend_read(weights: torch.Tensor, c: torch.Tensor, c_alt: torch.Tensor,
                     U: torch.Tensor, drop_mask=None) -> torch.Tensor:
    """Soft blend y[b,g] = sum_t (1-U)W[g,t,c] + U W[g,t,c_alt] via one embedding_bag with
    per_sample_weights (float32 accumulation). Fully differentiable — matches pure ManifestoSoftLUT.
    ``drop_mask`` [B,G,tph] (optional): per-table keep weights multiplied into BOTH cells' weights
    (one Bernoulli per table), so dropped tables contribute 0 — table dropout, fused in.
    -> [B,G,d_out] float32 (the cartridge casts to the caller dtype in _route)."""
    G, tph, K, d_out = weights.shape
    B = c.shape[0]
    acc = _acc_dtype(weights.dtype)
    W2 = weights.reshape(G * tph * K, d_out).to(acc)
    gc = _global_cells(c, G, tph, K)          # [B, G, tph]
    gca = _global_cells(c_alt, G, tph, K)     # [B, G, tph]
    idx = torch.cat([gc, gca], dim=2).reshape(B * G, 2 * tph)
    w0, w1 = 1.0 - U, U
    if drop_mask is not None:
        w0, w1 = w0 * drop_mask, w1 * drop_mask
    psw = torch.cat([w0, w1], dim=2).reshape(B * G, 2 * tph).to(acc)
    return F.embedding_bag(idx, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)


def fused_surrogate_read(weights: torch.Tensor, c: torch.Tensor, c_alt: torch.Tensor, w: torch.Tensor,
                         drop_mask=None) -> torch.Tensor:
    """The hard cartridge's surrogate s[b,g] = sum_t w_t (W[g,t,c_alt] - W[g,t,c]) via one embedding_bag over the
    DETACHED table, per_sample_weights (-w, +w) on (c, c_alt): no weight gradient, and the gradient to w
    (embedding_bag's per-sample-weight backward, dot(go, W[idx])) without materialising the [B, G, tph, d_out] cell rows
    -- the two gathers it replaces took ~6 GiB each at 32,768 vectors. drop_mask [B,G,tph] (optional) multiplies w
    (dropped tables give no surrogate gradient). -> [B,G,d_out] in the accumulation dtype."""
    G, tph, K, d_out = weights.shape
    B = c.shape[0]
    acc = _acc_dtype(weights.dtype)
    W2 = weights.detach().reshape(G * tph * K, d_out).to(acc)
    idx = torch.cat([_global_cells(c, G, tph, K), _global_cells(c_alt, G, tph, K)], dim=2).reshape(B * G, 2 * tph)
    ww = w if drop_mask is None else w * drop_mask
    psw = torch.cat([-ww, ww], dim=2).reshape(B * G, 2 * tph).to(acc)
    return F.embedding_bag(idx, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)
