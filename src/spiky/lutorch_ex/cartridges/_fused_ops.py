"""Fused read/backward primitives for the fused Manifesto cartridges.

Pure-PyTorch tier: the gather+sum over a group's ``tph`` tables is fused into a single
``torch.nn.functional.embedding_bag`` call (itself a fused CUDA kernel), and the hard
cartridge's straight-through backward is a custom ``torch.autograd.Function`` with a
hand-written scatter backward (which also avoids the inductor mis-schedule of the STE
composite). This is the graceful fallback path; an optional native fast-path (reusing the
``lutorch_cuda`` lprojection kernels) can sit behind an availability check later.

Numerically identical to the pure ManifestoHardLUT / ManifestoSoftLUT (the oracle).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# Low-precision reductions accumulate in fp32 and the result is cast back to the caller's dtype
# (the cartridge does that in _route). For bf16/fp16 weights the embedding_bag reduce runs on an
# fp32 view of the table (torch's embedding_bag has no accumulate-dtype knob), so summing a
# group's tph rows does not lose precision. fp32 and fp64 keep their own dtype (so the fp64
# equivalence tests stay bit-exact — upcasting fp32 to fp64 or downcasting fp64 to fp32 would
# both be wrong).
def _acc_dtype(dtype: torch.dtype) -> torch.dtype:
    """fp32 for bf16/fp16, else the dtype itself (fp32->fp32, fp64->fp64)."""
    return torch.float32 if dtype in (torch.bfloat16, torch.float16) else dtype


def _global_cells(c: torch.Tensor, G: int, tph: int, K: int) -> torch.Tensor:
    """Map per-(group,table) cell index c[B,G,tph] -> flat index into a [G*tph*K, d_out] table."""
    base = (torch.arange(G, device=c.device).view(G, 1) * tph
            + torch.arange(tph, device=c.device).view(1, tph)) * K   # [G, tph]
    return c + base  # [B, G, tph]


def fused_hard_read(weights: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """y[b,g] = sum_t W[g,t,c[b,g,t]]  via one embedding_bag. weights [G,tph,K,d_out] -> [B,G,d_out] fp32."""
    G, tph, K, d_out = weights.shape
    B = c.shape[0]
    W2 = weights.reshape(G * tph * K, d_out).to(_acc_dtype(weights.dtype))
    gc = _global_cells(c, G, tph, K).reshape(B * G, tph)
    return F.embedding_bag(gc, W2, mode="sum").reshape(B, G, d_out)


def fused_blend_read(weights: torch.Tensor, c: torch.Tensor, c_alt: torch.Tensor,
                     U: torch.Tensor) -> torch.Tensor:
    """Soft blend y[b,g] = sum_t (1-U)W[g,t,c] + U W[g,t,c_alt] via one embedding_bag with
    per_sample_weights (fp32 accumulation). Fully differentiable — matches pure ManifestoSoftLUT.
    -> [B,G,d_out] fp32 (the cartridge casts to the caller dtype in _route)."""
    G, tph, K, d_out = weights.shape
    B = c.shape[0]
    acc = _acc_dtype(weights.dtype)
    W2 = weights.reshape(G * tph * K, d_out).to(acc)
    gc = _global_cells(c, G, tph, K)          # [B, G, tph]
    gca = _global_cells(c_alt, G, tph, K)     # [B, G, tph]
    idx = torch.cat([gc, gca], dim=2).reshape(B * G, 2 * tph)
    psw = torch.cat([1.0 - U, U], dim=2).reshape(B * G, 2 * tph).to(acc)
    return F.embedding_bag(idx, W2, per_sample_weights=psw, mode="sum").reshape(B, G, d_out)


class FusedHardSTE(torch.autograd.Function):
    """Hard forward (value = sum_t W[c_t]) with the gen-1 straight-through backward:
    weight gradient is HARD (only the addressed cell c_t), input gradient flows through the
    two-alternative rational-uncertainty surrogate. Hand-written scatter backward.
    """

    @staticmethod
    def forward(ctx, weights, z, c, c_alt, a_star, b_star, single):
        # weights [G,tph,K,d_out]; z [B,G,d_in]; c/c_alt/a_star/b_star [B,G,tph] (long for c/*).
        # single: when True the margin is z[a_star] (anchor vs zero) and the input grad
        # scatters to a_star only; b_star is ignored (may be a placeholder).
        G, tph, K, d_out = weights.shape
        B, _, d_in = z.shape
        W2 = weights.reshape(G * tph * K, d_out)
        gc = _global_cells(c, G, tph, K)        # [B,G,tph]
        gca = _global_cells(c_alt, G, tph, K)
        val = F.embedding_bag(gc.reshape(B * G, tph), W2.to(_acc_dtype(weights.dtype)),
                              mode="sum").reshape(B, G, d_out)
        gdiff = W2[gc] - W2[gca]                 # [B,G,tph,d_out] = W[c_t] - W[c_t'] (weights dtype)
        delta = z.gather(2, a_star)              # [B,G,tph] signed deciding margin
        if not single:
            delta = delta - z.gather(2, b_star)
        ctx.save_for_backward(gc, gdiff, delta, a_star, b_star)
        ctx.dims = (G, tph, K, d_out, B, d_in)
        ctx.single = single
        ctx.wdtype = weights.dtype
        return val                              # acc dtype; the cartridge casts to the caller dtype in _route

    @staticmethod
    def backward(ctx, grad_out):                 # grad_out [B,G,d_out] — already in the acc dtype
        gc, gdiff, delta, a_star, b_star = ctx.saved_tensors
        G, tph, K, d_out, B, d_in = ctx.dims
        acc = _acc_dtype(ctx.wdtype)
        # grad_out and delta are already in `acc`: the cartridge upcasts bf16/fp16 x before
        # addressing (so delta = z[.] is fp32) and the forward value is produced in `acc` (so its
        # grad is too). The weight grad accumulates in `acc` and casts back to the weight dtype.
        grad_W2 = torch.zeros(G * tph * K, d_out, device=grad_out.device, dtype=acc)
        go = grad_out.unsqueeze(2).expand(B, G, tph, d_out).reshape(-1, d_out)
        grad_W2.index_add_(0, gc.reshape(-1), go)
        grad_weights = grad_W2.reshape(G, tph, K, d_out).to(ctx.wdtype)
        # Input gradient via the surrogate: du = (grad_out . (W[c]-W[c'])) * (-dU/ddelta),
        # with U = 0.5/(1+|delta|) -> -dU/ddelta = 0.5*sign(delta)/(1+|delta|)^2. gdiff (weight
        # dtype) is the only operand upcast to `acc`.
        # Pairs: delta = z[a]-z[b] -> +du to a, -du to b. Single: delta = z[a] -> +du to a.
        gd = (grad_out.unsqueeze(2) * gdiff.to(acc)).sum(-1)   # [B,G,tph]
        coeff = 0.5 * torch.sign(delta) / (1.0 + delta.abs()) ** 2
        du = gd * coeff
        grad_z = torch.zeros(B, G, d_in, device=grad_out.device, dtype=acc)
        grad_z.scatter_add_(2, a_star, du)
        if not ctx.single:
            grad_z.scatter_add_(2, b_star, -du)
        return grad_weights, grad_z, None, None, None, None, None
