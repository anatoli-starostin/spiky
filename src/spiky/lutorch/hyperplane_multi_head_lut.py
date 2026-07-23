"""HyperplaneMultiHeadLUT — learned-hyperplane generalization of FastMultiHeadLut.

This is a clone of `FastMultiHeadLut` (`fast_multi_head_lut.py`) in which the
index-computation front-end is generalized from **fixed anchor-pair sign
tests** to **fully learned affine hyperplanes**.

In `FastMultiHeadLut` each of the `NAP` index bits comes from a fixed anchor
pair `(p1, p2)`::

    bit_i = 1[ x[p1] - x[p2] > 0 ]

i.e. a comparator against the special hyperplane with normal `(e_p1 - e_p2)`
through the origin. Here we replace every anchor-pair test with a general
learned hyperplane::

    bit_i = 1[ <w_i, x> + b_i > 0 ]

where `w_i` is a trainable weight vector over all `input_dim` components and
`b_i` is a trainable bias / threshold. Weight scope is **one set of `NAP`
hyperplanes per table**:

    w : [n_tables, NAP, input_dim]      (trainable)
    b : [n_tables, NAP]                 (trainable)

with `n_tables = n_heads * tables_per_head`.

Everything downstream of the sign-pack is structurally unchanged from
`FastMultiHeadLut`:
  - MSB-first bit packing into the `2^NAP`-row table index,
  - `F.embedding_bag(mode='sum')` reduce over the `tables_per_head` axis,
  - both `hard` and `hybrid_smooth` forward modes,
  - the always-soft, full-K-softmax surrogate backward.

Only the index-computation (the affine sign-pack front-end) and the gradients
it produces change. The soft surrogate now differentiates the affine
pre-activation `a_i = <w_i, x> + b_i` in place of the anchor-pair difference
`d_i = x[p1] - x[p2]`, so the input gradient becomes a **dense** contraction
through `w` (rather than a two-coordinate scatter) and additionally produces
gradients for `w` and `b`.

Anchor-pair-equivalent init (`hyperplane_init="anchor_pairs"`: `w_i = e_p1 -
e_p2`, `b_i = 0`) reproduces `FastMultiHeadLut` bit-for-bit, so the module is a
strict generalization and can be A/B'd against it.

See doc/lutorch/lutgpt_research_report.pdf (Section 2) and claude/thesis.md for
the soft-backward math.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs


# =============================================================================
# Index packing + bit-matrix helpers (identical convention to FastMultiHeadLut)
# =============================================================================

def _soft_bit_matrix_msb(nap: int, device, dtype=torch.float32) -> torch.Tensor:
    """[NAP, K] +/-1 bit-pattern matrix, MSB-first:
    bit_matrix[i, k] = +1 if (k >> (NAP-1-i)) & 1 else -1.
    Used by the soft backward to score every row b in {0,1}^NAP via
    ts(b) = sum_i p_i * chi_i(b)."""
    n = 1 << nap
    bits = ((torch.arange(n, device=device).unsqueeze(0)
             >> torch.arange(nap - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


def _msb_powers(nap: int, device) -> torch.Tensor:
    """powers[i] = 2^(NAP-1-i), MSB-first packing. The sign-pack index
        index = sum_i (a_i > 0) * powers[i]
    picks the row k that maximises sum_i sign(a_i) * bit_matrix[i, k]."""
    return (1 << torch.arange(nap - 1, -1, -1, device=device, dtype=torch.int64))


# =============================================================================
# Per-table hyperplane projection (the new dominant FLOP cost)
# =============================================================================

def _hyperplane_project(x: torch.Tensor, w: torch.Tensor,
                        b: torch.Tensor) -> torch.Tensor:
    """Affine pre-activation a_i = <w_i, x> + b_i for every table/hyperplane.

    x: [B, input_dim]; w: [n_tables, NAP, input_dim]; b: [n_tables, NAP].
    Returns a: [B, n_tables, NAP].

    Implemented as one fused GEMM: flatten w to [n_tables*NAP, input_dim] and
    matmul against x^T, then reshape and add the bias. A single matmul (rather
    than an einsum with a batched-per-table contraction) keeps this on one
    cuBLAS call, is autocast-eligible (runs in bf16 under autocast), and is the
    new dominant FLOP of the module — replacing the cheap x[a]-x[b] gather of
    the anchor-pair front-end.
    """
    n_tables, nap, input_dim = w.shape
    w2 = w.reshape(n_tables * nap, input_dim)          # [T*NAP, D]
    a = torch.matmul(x, w2.t())                        # [B, T*NAP]
    a = a.view(x.shape[0], n_tables, nap)              # [B, T, NAP]
    return a + b.unsqueeze(0)


# =============================================================================
# Hard forward body (matmul sign-pack + embedding_bag reduce)
# =============================================================================

def _hyperplane_lut_fwd_body(x, weights, w, b, powers, n_heads, tph, table_dim):
    """Hard forward.

    Computes the affine sign-pack index of the argmax row per (sample, table),
    then fuses gather + sum-reduce across the tables_per_head axis via
    F.embedding_bag(mode='sum'). Structurally identical to
    FastMultiHeadLut._soft_lut_fwd_body except the pairwise-difference gather is
    replaced by the hyperplane projection.
    """
    B, _ = x.shape
    n_tables = w.shape[0]
    n_outputs = weights.shape[2]
    a = _hyperplane_project(x, w, b)                    # [B, n_tables, NAP]
    bits = (a > 0).to(torch.int64)
    index = (bits * powers.view(1, 1, -1)).sum(dim=-1)  # [B, n_tables]
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tph
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs), index


_hyperplane_lut_fwd_body_c = torch.compile(_hyperplane_lut_fwd_body, dynamic=True)


# =============================================================================
# Shared soft backward (used by hard, hybrid_smooth, and full-soft forwards)
# =============================================================================

def _hyperplane_soft_bwd_body(grad_pt, x, weights, w, b,
                              bit_matrix, index, T_soft, T_sel,
                              accum_dtype: torch.dtype,
                              compute_weight_grad: bool = True,
                              wgrad_via_bmm: bool = False):
    """Soft backward pinned to the actually-chosen index.

    Differentiates the full-K softmax surrogate
        y = sum_k softmax(ts/T_sel)_k * W[t, k, :]
    where ts(k) = sum_i p_i * chi_i(k), p_i = sign(a_i) * |a_i| / (T_soft +
    |a_i|), and a_i = <w_i, x> + b_i is the affine pre-activation. `p_signs` is
    reconstructed from `index` (== sign(a) since index bit i == 1[a_i > 0]) so
    the surrogate's argmax matches the row the forward actually picked.

    Returns full-K softmax gradients for x, w, b, and the two log-temperatures,
    plus the LUT-table weight grad (1-row scatter at the chosen row, backend
    auto-picked exactly as in FastMultiHeadLut).

    The input-side path (x, w, b) is the porting core of this issue. In the
    anchor-pair version the pre-activation was d_i = x[p1] - x[p2], so the input
    gradient was a +/-d scatter at two coordinates. Here a_i is affine over all
    input_dim, so the gradient flows densely through w into x, and additionally
    into w and b.

    `compute_weight_grad=False` skips the 1-row LUT-weight scatter — used by the
    hybrid_smooth backward, which supplies its own 2-row weight grad.

    `wgrad_via_bmm=True` switches the LUT-weight scatter to a sparse-S + bmm
    pattern (bf16-storage path), matching FastMultiHeadLut.
    """
    B, n_tables_, n_outputs = grad_pt.shape
    n_tables, NAP, input_dim = w.shape
    K = bit_matrix.shape[1]
    w_dtype = weights.dtype

    a        = _hyperplane_project(x, w, b)             # [B, n_tables, NAP]
    denom    = T_soft + a.abs()

    # Bits actually used in forward (MSB-first packing): bit at position i is
    # bit (NAP-1-i) of the integer index. p_signs has the same +/-1 pattern
    # forward picked, so the surrogate's argmax matches the saved row.
    shifts   = torch.arange(NAP - 1, -1, -1, device=index.device, dtype=index.dtype)
    bits     = ((index.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 1).to(a.dtype)
    p_signs  = bits * 2.0 - 1.0
    p        = p_signs * a.abs() / denom

    ts       = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    z        = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)

    d_sel_soft = torch.einsum("bto,tko->btk", grad_pt.to(w_dtype), weights)

    # Softmax backward, idiomatic PyTorch (compile fuses).
    sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
    d_z      = sel_soft * (d_sel_soft - sum_term)
    d_ts     = d_z / T_sel
    grad_log_T_sel = -(d_z * z).sum()

    # dL/dp via GEMM; dL/da via the rational soft-sign Jacobian.
    # p = p_signs * |a|/denom -> dp/d|a| = p_signs * T_soft/denom^2;
    # d|a|/da = sign(a). Hence dp/da = p_signs * sign(a) * T_soft/denom^2.
    d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))
    d_a = d_p * p_signs * a.sign() * (T_soft / (denom * denom))   # [B, n_tables, NAP]
    grad_log_T_soft = -(d_a * a).sum()

    # --- LUT table-weight grad (unchanged from FastMultiHeadLut) --------------
    if compute_weight_grad:
        if wgrad_via_bmm:
            g_dtype = grad_pt.dtype
            S = torch.zeros(B, n_tables, K, dtype=g_dtype, device=weights.device)
            S.scatter_(2, index.unsqueeze(-1), 1.0)
            grad_weights = torch.einsum("btk,bto->tko", S, grad_pt).to(accum_dtype)
        else:
            flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * K
            flat_idx    = (index + flat_offset[None, :]).reshape(-1)
            grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=accum_dtype, device=weights.device)
            grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(accum_dtype))
            grad_weights = grad_w_flat.view(n_tables, K, n_outputs)
    else:
        grad_weights = None

    # --- Hyperplane-parameter grads (the new deliverable) --------------------
    # a = x @ w2^T + b, with w2 = w.reshape(T*NAP, D). So (with G = d_a):
    #   dL/dx      = G_flat @ w2                         [B, D]
    #   dL/dw2     = G_flat^T @ x  -> reshape [T, NAP, D]
    #   dL/db      = G.sum(dim=0)                        [T, NAP]
    w2       = w.reshape(n_tables * NAP, input_dim)
    d_a_flat = d_a.reshape(B, n_tables * NAP)
    grad_x   = torch.matmul(d_a_flat, w2.to(d_a_flat.dtype)).to(x.dtype)
    grad_w   = torch.matmul(d_a_flat.t(), x.to(d_a_flat.dtype)).view(n_tables, NAP, input_dim).to(w.dtype)
    grad_b   = d_a.sum(dim=0).to(b.dtype)

    return grad_x, grad_weights, grad_w, grad_b, grad_log_T_soft, grad_log_T_sel


_hyperplane_soft_bwd_body_c = torch.compile(_hyperplane_soft_bwd_body, dynamic=True)


def _pick_bwd_body(is_cuda: bool):
    return _hyperplane_soft_bwd_body_c if is_cuda else _hyperplane_soft_bwd_body


def _pick_fwd_body(is_cuda: bool):
    return _hyperplane_lut_fwd_body_c if is_cuda else _hyperplane_lut_fwd_body


# =============================================================================
# forward_mode="hard": hard forward + soft backward
# =============================================================================

class _HyperplaneMHLutSoft(torch.autograd.Function):
    """Hard forward (affine sign-pack + embedding_bag), soft backward."""

    @staticmethod
    def forward(ctx, x, weights, w, b, log_T_soft, log_T_sel,
                bit_matrix, powers, n_heads, tph, table_dim, use_bf16):
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        # Cast LUT weights to bf16 for compute when use_bf16 and storage is
        # fp32 — F.embedding_bag is not autocast-eligible. Storage stays fp32.
        compute_in_bf16 = use_bf16 and x.is_cuda and weights.dtype == torch.float32
        weights_compute = weights.to(torch.bfloat16) if compute_in_bf16 else weights
        fwd_body = _pick_fwd_body(x.is_cuda)
        with autocast_ctx:
            out, index = fwd_body(
                x, weights_compute, w, b, powers, n_heads, tph, table_dim,
            )
        if compute_in_bf16:
            out = out.to(weights.dtype)
        ctx.save_for_backward(x, weights, w, b, bit_matrix, index,
                              log_T_soft, log_T_sel, powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, w, b, bit_matrix, index,
         log_T_soft, log_T_sel, powers) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = w.shape[0]
        n_outputs = weights.shape[2]
        grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        wgrad_via_bmm = weights.dtype != torch.float32
        bwd_body = _pick_bwd_body(x.is_cuda)
        with autocast_ctx:
            grad_x, grad_w_lut, grad_w, grad_b, grad_log_Ts, grad_log_Tx = bwd_body(
                grad_pt, x, weights, w, b, bit_matrix,
                index, T_soft, T_sel, weights.dtype,
                wgrad_via_bmm=wgrad_via_bmm,
            )
        # 12 forward inputs -> 12 grad returns.
        return (grad_x, grad_w_lut, grad_w, grad_b, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None)


# =============================================================================
# forward_mode="hybrid_smooth": top-2 smooth forward + soft backward
# =============================================================================

def _hyperplane_smooth_weight_grad(grad_pt, main_index, alt_index, u,
                                   n_tables, K, n_outputs):
    """2-row LUT-weight gradient for hybrid_smooth backward.

    Identical to FastMultiHeadLut._hybrid_smooth_weight_grad: two fp32
    index_add scatters at main_index (weight 1-u) and alt_index (weight u).
    """
    B = grad_pt.shape[0]
    g32           = grad_pt.float()
    one_minus_u32 = (1.0 - u).float()
    u32           = u.float()
    offset = torch.arange(n_tables, device=grad_pt.device, dtype=main_index.dtype) * K
    main_flat = (main_index + offset).reshape(-1)
    alt_flat  = (alt_index  + offset).reshape(-1)
    grad_w_flat = torch.zeros(
        n_tables * K, n_outputs, dtype=torch.float32, device=grad_pt.device,
    )
    grad_w_flat.index_add_(
        0, main_flat, (one_minus_u32.unsqueeze(-1) * g32).reshape(-1, n_outputs)
    )
    grad_w_flat.index_add_(
        0, alt_flat,  (u32.unsqueeze(-1)         * g32).reshape(-1, n_outputs)
    )
    return grad_w_flat.view(n_tables, K, n_outputs)


def _hyperplane_smooth_fwd_gather(x, weights, w, b, powers,
                                  T_soft, T_sel, n_heads, tph, table_dim):
    """Smooth top-2 forward via two F.embedding gathers + blend.

      main = affine sign-pack of (a_i > 0).
      alt  = main with the bit at argmin |a| flipped.
      u    = sigmoid(-Delta/T_sel), Delta = 2*a_min / (T_soft + a_min).
      out  = sum_t [(1-u) * W[main] + u * W[alt]].
    """
    B, _ = x.shape
    n_tables = w.shape[0]
    n_outputs = weights.shape[2]

    a = _hyperplane_project(x, w, b)                              # [B, n_tables, NAP]
    bits = (a > 0).to(torch.int64)
    powers_view = powers.view(1, 1, -1)
    main_index = (bits * powers_view).sum(dim=-1)                 # [B, n_tables]

    abs_a = a.abs()
    p_star = abs_a.argmin(dim=-1)                                 # least-confident hyperplane
    flip_mask = powers.to(main_index.dtype)[p_star]
    alt_index = main_index ^ flip_mask

    a_min = abs_a.gather(-1, p_star.unsqueeze(-1)).squeeze(-1)
    delta_ts = 2.0 * a_min / (T_soft + a_min)
    u = torch.sigmoid(-delta_ts / T_sel)                          # in (0, 0.5]
    main_w = 1.0 - u

    table_offset = torch.arange(n_tables, device=weights.device,
                                dtype=main_index.dtype) * table_dim
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    main_flat_idx = (main_index + table_offset.view(1, -1)).reshape(-1)
    alt_flat_idx  = (alt_index  + table_offset.view(1, -1)).reshape(-1)
    main_rows = F.embedding(main_flat_idx, weights_flat).view(B, n_tables, n_outputs)
    alt_rows  = F.embedding(alt_flat_idx,  weights_flat).view(B, n_tables, n_outputs)
    blended = main_rows * main_w.unsqueeze(-1) + alt_rows * u.unsqueeze(-1)
    out = blended.view(B, n_heads, tph, n_outputs).sum(dim=2).to(weights.dtype)
    return out, main_index, alt_index, u


def _hyperplane_smooth_fwd_bmm(x, weights, w, b, powers,
                               T_soft, T_sel, n_heads, tph, table_dim, s_dtype):
    """Smooth top-2 forward via sparse-S + one bmm per head.

    Build a sparse selection mass S[B, n_tables, K] with two nonzeros per
    (b, t) — (1-u) at main_index and u at alt_index — then contract via
    bmm(S, W). Matches FastMultiHeadLut._hybrid_smooth_fwd_bmm; wins at
    per-head n_outputs >= 128.
    """
    B, _ = x.shape
    n_tables = w.shape[0]
    n_outputs = weights.shape[2]
    K = table_dim

    a = _hyperplane_project(x, w, b)                              # [B, n_tables, NAP]
    bits = (a > 0).to(torch.int64)
    main_index = (bits * powers.view(1, 1, -1)).sum(dim=-1)       # [B, n_tables]

    abs_a = a.abs()
    p_star = abs_a.argmin(dim=-1)
    flip_mask = powers.to(main_index.dtype)[p_star]
    alt_index = main_index ^ flip_mask

    a_min = abs_a.gather(-1, p_star.unsqueeze(-1)).squeeze(-1)
    delta_ts = 2.0 * a_min / (T_soft + a_min)
    u = torch.sigmoid(-delta_ts / T_sel)                          # in (0, 0.5]
    main_w = 1.0 - u

    S = torch.zeros(B, n_tables, K, dtype=s_dtype, device=x.device)
    S.scatter_(2, main_index.unsqueeze(-1), main_w.unsqueeze(-1).to(s_dtype))
    S.scatter_(2, alt_index.unsqueeze(-1),  u.unsqueeze(-1).to(s_dtype))

    tph_K = tph * K
    S_h = S.view(B, n_heads, tph_K).transpose(0, 1).contiguous()  # [n_heads, B, tph*K]
    W_h = weights.view(n_heads, tph_K, n_outputs)                 # [n_heads, tph*K, n_out]
    out_h = torch.bmm(S_h, W_h)                                   # [n_heads, B, n_out]
    out = out_h.transpose(0, 1).contiguous().to(weights.dtype)
    return out, main_index, alt_index, u


_hyperplane_smooth_fwd_gather_c = torch.compile(_hyperplane_smooth_fwd_gather, dynamic=True)
_hyperplane_smooth_fwd_bmm_c = torch.compile(_hyperplane_smooth_fwd_bmm, dynamic=True)
_hyperplane_smooth_weight_grad_c = torch.compile(_hyperplane_smooth_weight_grad, dynamic=True)


class _HyperplaneMHLutHybridSmooth(torch.autograd.Function):
    """Smooth top-2 forward + soft input/hyperplane grad + 2-row weight grad.

    Forward dispatches inline on per-head n_outputs, matching FastMultiHeadLut:
      n_outputs >= 128 -> sparse-S + tensor-core matmul
      n_outputs <  128 -> two embedding gathers + blend
    """

    @staticmethod
    def forward(ctx, x, weights, w, b, log_T_soft, log_T_sel,
                bit_matrix, powers, n_heads, tph, table_dim, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        on_cuda = x.is_cuda
        gather_body = _hyperplane_smooth_fwd_gather_c if on_cuda else _hyperplane_smooth_fwd_gather
        bmm_body    = _hyperplane_smooth_fwd_bmm_c    if on_cuda else _hyperplane_smooth_fwd_bmm
        with autocast_ctx:
            if weights.shape[2] >= 128:
                s_dtype = (torch.bfloat16
                           if use_bf16 and on_cuda
                           else weights.dtype)
                out, main_index, alt_index, u = bmm_body(
                    x, weights, w, b, powers,
                    T_soft, T_sel, n_heads, tph, table_dim, s_dtype,
                )
            else:
                out, main_index, alt_index, u = gather_body(
                    x, weights, w, b, powers,
                    T_soft, T_sel, n_heads, tph, table_dim,
                )
        ctx.save_for_backward(x, weights, w, b, bit_matrix,
                              main_index, alt_index, u,
                              log_T_soft, log_T_sel, powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, w, b, bit_matrix,
         main_index, alt_index, u,
         log_T_soft, log_T_sel, powers) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = w.shape[0]
        n_outputs = weights.shape[2]
        K = bit_matrix.shape[1]

        grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        bwd_body = _pick_bwd_body(x.is_cuda)
        with autocast_ctx:
            # Soft backward: grad_x/grad_w/grad_b + temp grads (input-side grads
            # are always soft). Its 1-row LUT-weight grad is discarded — we use
            # the 2-row hybrid weight grad below.
            grad_x, _unused, grad_w, grad_b, grad_log_Ts, grad_log_Tx = bwd_body(
                grad_pt, x, weights, w, b, bit_matrix,
                main_index, T_soft, T_sel, weights.dtype,
                compute_weight_grad=False,
            )

        wgrad_body = _hyperplane_smooth_weight_grad_c if x.is_cuda else _hyperplane_smooth_weight_grad
        grad_w_lut = wgrad_body(
            grad_pt, main_index, alt_index, u, n_tables, K, n_outputs,
        ).to(weights.dtype)

        # 12 forward inputs -> 12 grad returns.
        return (grad_x, grad_w_lut, grad_w, grad_b, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None)


# =============================================================================
# Full-soft surrogate forward + soft backward (gradcheck / reference target)
# =============================================================================

def _hyperplane_full_soft_fwd_body(x, weights, w, b, powers, bit_matrix,
                                   T_soft, T_sel, n_heads, tph):
    """Full-K softmax forward the surrogate backward is the exact gradient of.

    y = sum_k softmax(ts/T_sel)_k * W[t, k, :], reduced over tph. Uses the
    smooth pre-activation p = a / (T_soft + |a|) directly (== sign(a)*|a|/denom,
    which is C1 in a), so the forward is differentiable in x/w/b and matches the
    analytic soft backward for those inputs. Returns (out, index) with index the
    hard argmax row (saved so backward reconstructs the same p_signs).

    Not shipped as a forward mode (K-times the cost of hard/hybrid); used as the
    reference target for gradcheck and finite-difference tests.
    """
    B, _ = x.shape
    n_tables, NAP, _ = w.shape
    K = weights.shape[1]
    n_outputs = weights.shape[2]

    a = _hyperplane_project(x, w, b)                              # [B, n_tables, NAP]
    denom = T_soft + a.abs()
    p = a / denom                                                # smooth soft-sign
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    sel_soft = F.softmax(ts / T_sel, dim=-1)                     # [B, n_tables, K]
    blended = torch.einsum("btk,tko->bto", sel_soft, weights)   # [B, n_tables, n_out]
    out = blended.view(B, n_heads, tph, n_outputs).sum(dim=2)

    bits = (a > 0).to(torch.int64)
    index = (bits * powers.view(1, 1, -1)).sum(dim=-1)           # [B, n_tables]
    return out, index


class _HyperplaneMHLutFullSoft(torch.autograd.Function):
    """Full-soft forward + analytic soft backward.

    Exposed for gradcheck: the analytic backward for x, w, b is the exact
    gradient of this forward. (The LUT-weight grad returned by the shared
    backward is the 1-row surrogate scatter, NOT the dense full-soft weight
    grad, so gradcheck should mark only x/w/b as requiring grad.)
    """

    @staticmethod
    def forward(ctx, x, weights, w, b, log_T_soft, log_T_sel,
                bit_matrix, powers, n_heads, tph, table_dim, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        out, index = _hyperplane_full_soft_fwd_body(
            x, weights, w, b, powers, bit_matrix, T_soft, T_sel, n_heads, tph,
        )
        ctx.save_for_backward(x, weights, w, b, bit_matrix, index,
                              log_T_soft, log_T_sel, powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, w, b, bit_matrix, index,
         log_T_soft, log_T_sel, powers) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = w.shape[0]
        n_outputs = weights.shape[2]
        grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
        grad_x, grad_w_lut, grad_w, grad_b, grad_log_Ts, grad_log_Tx = _hyperplane_soft_bwd_body(
            grad_pt, x, weights, w, b, bit_matrix,
            index, T_soft, T_sel, weights.dtype,
        )
        return (grad_x, grad_w_lut, grad_w, grad_b, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None)


# =============================================================================
# Public module
# =============================================================================

_FORWARD_MODES = ("hard", "hybrid_smooth")
_HYPERPLANE_INITS = ("anchor_pairs", "random")


class HyperplaneMultiHeadLUT(nn.Module):
    """Learned-hyperplane generalization of FastMultiHeadLut.

    Each of the `n_heads * tables_per_head` lookup tables selects one of its
    `2^NAP` rows via `NAP` learned affine sign tests `1[<w_i, x> + b_i > 0]`,
    packs the bits MSB-first into a row index, gathers, and sum-reduces across
    the `tables_per_head` axis. Trains end-to-end via a full-K softmax surrogate
    backward that now produces gradients for the hyperplane weights `w` and
    biases `b` in addition to `x`, the LUT weights, and the two temperatures.

    Args mirror FastMultiHeadLut, plus:
        hyperplane_init: "anchor_pairs" (default) initializes each hyperplane to
            reproduce a fixed anchor pair (`w_i = e_p1 - e_p2`, `b_i = 0`), so
            the module is a strict, bit-for-bit generalization of
            FastMultiHeadLut and can be A/B'd against it. "random" uses
            small-norm Gaussian weight rows and zero bias for learning from
            scratch.
        hyperplane_init_scale: per-element std of the Gaussian rows when
            hyperplane_init="random".

    Forward signature:
        x: float [B, input_dim]
        returns: [B, n_heads, n_outputs] in weight_dtype.

    Exposed parameters: `hyperplane_weight` (w, [n_tables, NAP, input_dim]),
    `hyperplane_bias` (b, [n_tables, NAP]), `weights` (LUT tables), and the two
    log-temperatures (Parameters iff learnable_temps).
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        *,
        forward_mode: str = "hard",
        weight_dtype: torch.dtype = torch.float32,
        use_bf16: bool = True,
        hyperplane_init: str = "anchor_pairs",
        hyperplane_init_scale: float = 0.02,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        soft_score_temp: float = 0.5,
        select_temp: float = 0.5,
        learnable_temps: bool = False,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if forward_mode not in _FORWARD_MODES:
            raise ValueError(
                f"forward_mode must be one of {_FORWARD_MODES}, got {forward_mode!r}"
            )
        if hyperplane_init not in _HYPERPLANE_INITS:
            raise ValueError(
                f"hyperplane_init must be one of {_HYPERPLANE_INITS}, "
                f"got {hyperplane_init!r}"
            )
        if not (1 <= n_anchor_pairs <= 15):
            raise ValueError(
                f"n_anchor_pairs must be in [1, 15] (K = 2^NAP rows per table), "
                f"got {n_anchor_pairs}"
            )

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.tables_per_head = tables_per_head
        self.table_dim = 1 << n_anchor_pairs
        self.weight_dtype = weight_dtype
        self.forward_mode = forward_mode
        self.hyperplane_init = hyperplane_init
        self.use_bf16 = bool(use_bf16)

        n_lookup_tables = n_heads * tables_per_head
        self.n_lookup_tables = n_lookup_tables

        policy = (
            anchor_sampling_policy
            if anchor_sampling_policy is not None
            else AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE
        )
        if policy not in (
            AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
            AnchorSamplingPolicy.CANONICAL_DISTINCT,
        ):
            raise ValueError(
                f"anchor_sampling_policy must be CANONICAL_FULL_COVERAGE or "
                f"CANONICAL_DISTINCT, got {policy}"
            )
        self.anchor_sampling_policy = policy

        dev = device or torch.device("cpu")

        # Anchor pairs (used to seed the anchor-pairs-equivalent init, and kept
        # as buffers so the A/B init is reproducible / inspectable).
        anchor_a_long, anchor_b_long = get_balanced_anchor_pairs(
            n_tables=n_lookup_tables,
            n_anchor_pairs=n_anchor_pairs,
            input_dim=input_dim,
            device=dev,
            random_seed=random_seed,
            policy=policy,
            n_heads=n_heads,
        )
        self.register_buffer("soft_anchor_a_long", anchor_a_long.contiguous())
        self.register_buffer("soft_anchor_b_long", anchor_b_long.contiguous())

        # --- Hyperplane parameters -------------------------------------------
        w_init = torch.zeros(
            n_lookup_tables, n_anchor_pairs, input_dim,
            dtype=torch.float32, device=dev,
        )
        b_init = torch.zeros(
            n_lookup_tables, n_anchor_pairs, dtype=torch.float32, device=dev,
        )
        if hyperplane_init == "anchor_pairs":
            # w_i = e_{a} - e_{b}, b_i = 0  ->  <w_i, x> + b_i = x[a] - x[b].
            t_idx = torch.arange(n_lookup_tables, device=dev).view(-1, 1).expand(
                n_lookup_tables, n_anchor_pairs)
            n_idx = torch.arange(n_anchor_pairs, device=dev).view(1, -1).expand(
                n_lookup_tables, n_anchor_pairs)
            w_init[t_idx, n_idx, anchor_a_long] += 1.0
            w_init[t_idx, n_idx, anchor_b_long] -= 1.0
        else:  # "random"
            gen = None
            if random_seed is not None:
                gen = torch.Generator(device=dev).manual_seed(random_seed + 2)
            w_init.normal_(mean=0.0, std=hyperplane_init_scale, generator=gen)
        self.hyperplane_weight = nn.Parameter(w_init.to(weight_dtype))
        self.hyperplane_bias = nn.Parameter(b_init.to(weight_dtype))

        # --- LUT table weights (identical init to FastMultiHeadLut) ----------
        rng_kwargs: dict = {"device": dev}
        if random_seed is not None:
            rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed + 1)
        weights_init = (
            (torch.rand(n_lookup_tables, self.table_dim, n_outputs, **rng_kwargs) - 0.5)
            * (2.0 * initial_weights_noise)
        ).to(weight_dtype)
        self.weights = nn.Parameter(weights_init)

        # bit_matrix and MSB powers for the soft backward surrogate.
        self.register_buffer(
            "soft_bit_matrix",
            _soft_bit_matrix_msb(n_anchor_pairs, dev, dtype=torch.float32),
        )
        self.register_buffer("soft_powers", _msb_powers(n_anchor_pairs, dev))
        self.register_buffer(
            "_table_offset",
            torch.arange(n_lookup_tables, device=dev, dtype=torch.int64) * self.table_dim,
        )

        # log-parametrise the temperatures so unconstrained optimisation
        # keeps T positive.
        self.learnable_temps = bool(learnable_temps)
        log_Ts_init = math.log(float(soft_score_temp))
        log_Tx_init = math.log(float(select_temp))
        if self.learnable_temps:
            self.log_soft_score_temp = nn.Parameter(
                torch.tensor(log_Ts_init, dtype=torch.float32, device=dev)
            )
            self.log_select_temp = nn.Parameter(
                torch.tensor(log_Tx_init, dtype=torch.float32, device=dev)
            )
        else:
            self.register_buffer(
                "log_soft_score_temp",
                torch.tensor(log_Ts_init, dtype=torch.float32, device=dev),
            )
            self.register_buffer(
                "log_select_temp",
                torch.tensor(log_Tx_init, dtype=torch.float32, device=dev),
            )

    def _hard_eval(self, x: torch.Tensor) -> torch.Tensor:
        """no_grad hard-mode eval: affine sign-pack + embedding_bag reduce.

        No anchor-specific CUDA fast path here (the lutorch_cuda bit-pack kernel
        is hard-wired to pairwise differences); the compiled matmul body is the
        eval path. The bf16 LUT-weight cast mirrors the train forward.
        """
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.use_bf16 and x.is_cuda
            else torch.amp.autocast("cpu", enabled=False)
        )
        compute_in_bf16 = (
            self.use_bf16 and x.is_cuda and self.weights.dtype == torch.float32
        )
        weights_compute = (
            self.weights.to(torch.bfloat16) if compute_in_bf16 else self.weights
        )
        fwd_body = _pick_fwd_body(x.is_cuda)
        with autocast_ctx:
            out, _ = fwd_body(
                x, weights_compute, self.hyperplane_weight, self.hyperplane_bias,
                self.soft_powers, self.n_heads, self.tables_per_head, self.table_dim,
            )
        if compute_in_bf16:
            out = out.to(self.weights.dtype)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.forward_mode == "hybrid_smooth":
            return _HyperplaneMHLutHybridSmooth.apply(
                x, self.weights, self.hyperplane_weight, self.hyperplane_bias,
                self.log_soft_score_temp, self.log_select_temp,
                self.soft_bit_matrix, self.soft_powers,
                self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
            )
        # forward_mode == "hard"
        if not torch.is_grad_enabled():
            return self._hard_eval(x)
        return _HyperplaneMHLutSoft.apply(
            x, self.weights, self.hyperplane_weight, self.hyperplane_bias,
            self.log_soft_score_temp, self.log_select_temp,
            self.soft_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
        )

    def forward_full_soft(self, x: torch.Tensor) -> torch.Tensor:
        """Full-K softmax surrogate forward (reference / gradcheck target).

        The analytic soft backward for x/w/b is the exact gradient of this
        output. K-times the cost of the shipped forwards, so not a forward_mode.
        """
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        return _HyperplaneMHLutFullSoft.apply(
            x, self.weights, self.hyperplane_weight, self.hyperplane_bias,
            self.log_soft_score_temp, self.log_select_temp,
            self.soft_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
        )
