"""Tier-2 native fast path for the Gen-2 soft-sign cartridges.

The expensive, must-be-fused part of the Gen-2 train step is the *backward* two-cell read:
to form the weight gradient and the input/temperature gradients you need, per table, the dot
products ``grad . W[c_t]`` and ``grad . W[c_t']`` and (for smooth) a weighted scatter into the
weight table — all WITHOUT ever materialising the ``[B, G, tph, d_out]`` cell tensors.

The Gen-1 lprojection kernels (``_native_ops``) already do exactly that, and the parts we need are
**uncertainty-agnostic**, so we reuse them unchanged:

* ``lprojection_backward_na1_nonsmooth`` -> (hard 1-row weight grad, carriers gc_main/gc_alt),
* ``lprojection_backward_na1_smooth``    -> (weighted 2-row weight grad for EXPLICIT per-table
  ``main_weight``/``alt_weight`` we pass in, + the same carriers),

where ``gc_main = grad . W[c]`` and ``gc_alt = grad . W[c']`` depend only on the indices, not on
the blend weight. The ONLY Gen-1-specific kernel is ``anchor_pairs_lookup_backward_all``, which
folds in Gen-1's inverse-L1 ``dU/dmargin`` — that is exactly the soft-sign-specific step, so we do
NOT use it and instead compute the Gen-2 surrogate's input/temperature gradient here.

Gen-2 surrogate recap: the output gradient behaves as if the value were the blend
``(1-w) W[c] + w W[c']`` (hard: straight-through; smooth: true value), so for both cartridges
``dL/dw = <grad, W[c'] - W[c]> = gc_alt - gc_main``. The weight ``w = sigmoid(-2 rho / T_sel)``,
``rho = |m| / (T_soft + |m|)`` depends on the deciding margin ``m`` and the two learned
temperatures; its derivatives wrt ``m``, ``T_soft`` and ``T_sel`` are the "soft-sign-specific"
parts. We compute them from ``dL/dw`` and the saved signed margin with a tiny autograd tail
(O(B*n_tables), no ``d_out`` dimension, no weight reads) and scatter the margin gradient into the
input (``+`` to anchor a, ``-`` to anchor b; single mode keeps the ``+a`` half). This matches the
pure cartridge by construction (same ``w`` expression) and never stashes the cell indices.

Forward value: hard = ``sum_t W[c_t]`` (fused via ``embedding_bag``); smooth = the fused
``(1-w)/w`` blend (``embedding_bag`` with per-sample weights). Neither materialises
``[B, G, tph, d_out]``.

fp32/fp64 are the oracle (the native kernels dispatch Double too); bf16/fp16 run with fp32
addressing + fp32 weight-grad accumulation (the kernels' ``at::acc_type`` buffers). Falls back to
the pure cartridge on CPU or when the native extension is unavailable (handled by the cartridge).
"""
from __future__ import annotations

import os

import torch

from ._fused_ops import _acc_dtype, fused_blend_read, fused_hard_read
from ._native_ops import _THREADS, _table_indices, native_manager

# Standalone JIT kernel for the soft-sign surrogate tail (one fused launch: w + its margin/temp
# derivatives + the input-grad scatter). Built lazily; None -> eager fallback (identical result).
# Separate from the lutorch_ex_lprojection extension.
_SS_EXT = None
_SS_TRIED = False


def _ss_ext():
    global _SS_EXT, _SS_TRIED
    if _SS_TRIED:
        return _SS_EXT
    _SS_TRIED = True
    if os.environ.get("LUTORCH_EX_NO_CUDA_EXT", "0") == "1":
        return None
    try:
        from torch.utils.cpp_extension import load

        _csrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc")
        src = os.path.join(_csrc, "softsign_surrogate_grad.cu")
        _SS_EXT = load(name="lutorch_ex_softsign_surrogate_grad", sources=[src], verbose=False)
    except Exception:
        _SS_EXT = None
    return _SS_EXT


def _softsign_grads_from_dLdw(dLdw, delta, t_soft, t_select, a_glob, b_glob, single, B, G, d_in, nt):
    """Soft-sign-specific tail: given dL/dw [B,nt] and the signed deciding margin delta [B,nt],
    return (grad_z [B,G,d_in], grad_t_soft, grad_t_select).

    Computes the soft-sign weight ``w(|delta|, T_soft, T_select)`` and its derivatives wrt the
    margin and the two temperatures, scatters the margin gradient into the input (+a / -b; single
    keeps +a), and reduces the temperature gradients. All O(B*nt): no ``d_out``, no weight reads.

    On CUDA (fp32/fp64) this is the one-launch native kernel ``softsign_surrogate_grad``; on CPU or
    if the kernel can't be built it falls back to a tiny autograd graph over the same ``w``
    expression — bit-identical algebra, just more kernel launches.
    """
    work = delta.dtype
    dLdw = dLdw.to(work)
    ext = _ss_ext() if delta.is_cuda else None
    if ext is not None and work in (torch.float32, torch.float64):
        width = G * d_in
        row_off = torch.arange(B, device=delta.device).view(B, 1) * width
        a_flat = (a_glob + row_off).contiguous()
        b_flat = a_flat if single else (b_glob + row_off).contiguous()
        grad_zf, dts_buf, dtl_buf = ext.softsign_surrogate_grad(
            dLdw.contiguous(), delta.contiguous(), a_flat, b_flat,
            float(t_soft), float(t_select), width, bool(single), _THREADS)
        return (grad_zf.view(B, G, d_in),
                dts_buf.sum().to(t_soft.dtype), dtl_buf.sum().to(t_select.dtype))
    # Eager fallback (CPU / no kernel): autograd through the same w, identical result.
    with torch.enable_grad():
        d_leaf = delta.detach().requires_grad_(True)
        ts = t_soft.detach().to(work).requires_grad_(True)
        tl = t_select.detach().to(work).requires_grad_(True)
        s = d_leaf.abs()
        rho = s / (ts + s)
        w = torch.sigmoid(-2.0 * rho / tl)
        dL_ddelta, dL_dts, dL_dtl = torch.autograd.grad(
            w, [d_leaf, ts, tl], grad_outputs=dLdw, allow_unused=True
        )
    grad_zf = torch.zeros(B, G * d_in, device=delta.device, dtype=work)
    grad_zf.scatter_add_(1, a_glob, dL_ddelta)
    if not single:
        grad_zf.scatter_add_(1, b_glob, -dL_ddelta)
    return grad_zf.view(B, G, d_in), dL_dts.to(t_soft.dtype), dL_dtl.to(t_select.dtype)


def _prep(weights, c, c_alt, z):
    """Common reshapes to the kernels' flat native layout. Returns tensors + dims."""
    G, tph, K, d_out = weights.shape
    B = c.shape[0]
    nt = G * tph
    d_in = z.shape[2]
    W = weights.reshape(nt, K, d_out).contiguous()
    li = c.reshape(B, nt).contiguous()
    lai = c_alt.reshape(B, nt, 1).contiguous()
    tif = _table_indices(B, nt, z.device)
    return W, li, lai, tif, (G, tph, K, d_out, B, d_in, nt)


class NativeSoftSignHard(torch.autograd.Function):
    """Gen-2 hard: fused ``sum_t W[c_t]`` forward; native nonsmooth weight grad + carriers, then
    the soft-sign tail for the input/temperature gradient (no two-cell tensor materialised)."""

    @staticmethod
    def forward(ctx, weights, z, c, c_alt, u_signed_star, a_glob, b_glob, single, t_soft, t_select,
                drop_mask=None):
        with torch.no_grad():
            val = fused_hard_read(weights, c, drop_mask=drop_mask).to(weights.dtype)  # sum_t keep_t W[c_t]
        W, li, lai, tif, dims = _prep(weights, c, c_alt, z)
        G, tph, K, d_out, B, d_in, nt = dims
        ctx.save_for_backward(W, li, lai, tif, u_signed_star.reshape(B, nt),
                              a_glob.reshape(B, nt), b_glob.reshape(B, nt), t_soft, t_select)
        ctx.dims = dims
        ctx.single = bool(single)
        ctx.wdtype = weights.dtype
        ctx.drop_mask = drop_mask
        return val

    @staticmethod
    def backward(ctx, grad_grp):
        W, li, lai, tif, delta, a_g, b_g, t_soft, t_select = ctx.saved_tensors
        G, tph, K, d_out, B, d_in, nt = ctx.dims
        mgr = native_manager()
        grad_pt = grad_grp.to(ctx.wdtype).unsqueeze(2).expand(B, G, tph, d_out)
        if ctx.drop_mask is not None:                 # scale each table's grad by its keep-mask
            grad_pt = grad_pt * ctx.drop_mask.unsqueeze(-1).to(grad_pt.dtype)
        grad_pt = grad_pt.reshape(B, nt, d_out).contiguous()
        wgrad, gc_main, gc_alt = mgr.lprojection_backward_na1_nonsmooth(
            grad_pt, W, li, lai, tif, tif, _THREADS)
        dLdw = gc_alt.reshape(B, nt) - gc_main.reshape(B, nt)            # dL/dw = <grad, W[c']-W[c]>
        grad_z, dts, dtl = _softsign_grads_from_dLdw(
            dLdw, delta, t_soft, t_select, a_g, b_g, ctx.single, B, G, d_in, nt)
        return (wgrad.reshape(G, tph, K, d_out).to(ctx.wdtype), grad_z.to(ctx.wdtype),
                None, None, None, None, None, None, dts, dtl, None)


class NativeSoftSignSmooth(torch.autograd.Function):
    """Gen-2 smooth: fused ``(1-w)/w`` blend forward; native SMOOTH weight grad with the Gen-2
    per-table ``main_weight=1-w`` / ``alt_weight=w`` + carriers, then the soft-sign tail."""

    @staticmethod
    def forward(ctx, weights, z, c, c_alt, u_signed_star, a_glob, b_glob, single, t_soft, t_select, w,
                drop_mask=None):
        # w [B,G,tph] is the Gen-2 blend weight (computed by the cartridge in the addressing dtype).
        with torch.no_grad():
            val = fused_blend_read(weights, c, c_alt, w, drop_mask=drop_mask).to(weights.dtype)
        W, li, lai, tif, dims = _prep(weights, c, c_alt, z)
        G, tph, K, d_out, B, d_in, nt = dims
        wt = w.reshape(B, nt).to(weights.dtype)
        ctx.save_for_backward(W, li, lai, tif, u_signed_star.reshape(B, nt),
                              a_glob.reshape(B, nt), b_glob.reshape(B, nt), t_soft, t_select, wt)
        ctx.dims = dims
        ctx.single = bool(single)
        ctx.wdtype = weights.dtype
        ctx.drop_mask = drop_mask
        return val

    @staticmethod
    def backward(ctx, grad_grp):
        W, li, lai, tif, delta, a_g, b_g, t_soft, t_select, wt = ctx.saved_tensors
        G, tph, K, d_out, B, d_in, nt = ctx.dims
        mgr = native_manager()
        grad_pt = grad_grp.to(ctx.wdtype).unsqueeze(2).expand(B, G, tph, d_out)
        if ctx.drop_mask is not None:                          # scale each table's grad by its keep-mask
            grad_pt = grad_pt * ctx.drop_mask.unsqueeze(-1).to(grad_pt.dtype)
        grad_pt = grad_pt.reshape(B, nt, d_out).contiguous()
        main_w = (1.0 - wt).contiguous()                       # [B,nt]   -> grad into W[c]
        alt_w = wt.reshape(B, nt, 1).contiguous()              # [B,nt,1] -> grad into W[c']
        wgrad, gc_main, gc_alt = mgr.lprojection_backward_na1_smooth(
            grad_pt, W, li, lai, tif, tif, main_w, alt_w, _THREADS)
        dLdw = gc_alt.reshape(B, nt) - gc_main.reshape(B, nt)   # dValue/dw = W[c']-W[c]; dL/dw = <grad, .>
        grad_z, dts, dtl = _softsign_grads_from_dLdw(
            dLdw, delta, t_soft, t_select, a_g, b_g, ctx.single, B, G, d_in, nt)
        return (wgrad.reshape(G, tph, K, d_out).to(ctx.wdtype), grad_z.to(ctx.wdtype),
                None, None, None, None, None, None, dts, dtl, None, None)
