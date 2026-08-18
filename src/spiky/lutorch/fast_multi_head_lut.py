"""FastMultiHeadLut — multi-head LUT primitive for LUTGPT.

Each forward call gathers from `n_heads * tables_per_head` independent
2^n_anchor_pairs x n_outputs lookup tables, picks one row per table from a
sign-bit packing of pairwise differences x_a - x_b, and reduces across the
tables_per_head axis. Trains end-to-end via a soft surrogate backward.

Two forward modes:
  - "hard"          : hard sign-pack lookup, one row per table.
  - "hybrid_smooth" : top-2 soft blend of the main row and its Hamming-1 alt
                      at the least-confident anchor pair.

Backward (both modes; "always soft"):
  - Input and temperature gradients come from the full K-row softmax
    surrogate pinned to the chosen main row (gradient flows through every
    row score in the surrogate, even rows that did not appear in forward).
  - Weight gradient reflects the *actual* forward: a 1-row scatter at the
    chosen row in "hard" mode; a 2-row scatter at main + alt in
    "hybrid_smooth" mode.

See doc/lutorch/lutgpt_research_report.pdf for the math (Section~2).
"""
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs


# Hard-eval fast path: hand-written CUDA kernel from lutorch_cuda that fuses the
# pairwise differences and the sign-bit packing into one pass, returning the int64
# row indices directly. Replaces the compiled fp32 bit-pack in the hard-mode eval
# shortcut when available. Set SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS=1 to disable.
_USE_LUTORCH_CUSTOM_CUDA_KERNELS = (
    os.environ.get("SPIKY_LUTORCH_NO_CUSTOM_CUDA_KERNELS", "0") != "1"
)
_LUTORCH_CUDA_THREADS_PER_BLOCK = int(
    os.environ.get("SPIKY_LUTORCH_CUDA_THREADS_PER_BLOCK", "256")
)


def _get_native_lutorch_manager():
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]
        return get_lutorch_manager()
    except Exception:
        return None


@torch.compile(dynamic=True)
def _native_eval_bag_reduce(index, weights_flat, table_offset,
                             B, n_heads, tph, n_outputs):
    """Build flat_indices from MSB-first row indices and bag-sum.

    Compiled together so the add+reshape and the offsets arange live in one
    graph and can be fused with the embedding_bag launch.
    """
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights_flat.device, dtype=torch.long) * tph
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs)


# =============================================================================
# Index packing + bit-matrix helpers
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
        index = sum_i (d_i > 0) * powers[i]
    picks the row k that maximises sum_i sign(d_i) * bit_matrix[i, k]."""
    return (1 << torch.arange(nap - 1, -1, -1, device=device, dtype=torch.int64))


# =============================================================================
# Hard-forward body and eval shortcut
# =============================================================================

@torch.compile(dynamic=True)
def _soft_lut_fwd_body(x, weights, anchor_a_long, anchor_b_long, powers,
                       n_heads, tph, table_dim):
    """Compiled hard forward.

    Computes the sign-pack index of the argmax row per (sample, table) at fp32,
    then fuses gather + sum-reduce across the tables_per_head axis via
    F.embedding_bag(mode='sum').
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    bits = (d > 0).to(torch.int64)
    index = (bits * powers.view(1, 1, -1)).sum(dim=-1)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tph
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs), index


# =============================================================================
# Shared soft backward (used by both hard and hybrid_smooth forward modes)
# =============================================================================

@torch.compile(dynamic=True)
def _soft_lut_bwd_body(grad_pt, x, weights, anchor_a_long, anchor_b_long,
                        bit_matrix, index, T_soft, T_sel,
                        accum_dtype: torch.dtype,
                        compute_weight_grad: bool = True,
                        wgrad_via_bmm: bool = False):
    """Soft backward pinned to the actually-chosen index.

    Reconstructs p_signs from `index` so the surrogate softmax's argmax matches
    the row picked in forward. Returns full-K softmax gradients for x and the
    two log-temperatures, plus a 1-row weight grad at the chosen row.

    `compute_weight_grad=False` skips the 1-row weight scatter — used by
    hybrid_smooth backward, which supplies its own 2-row weight grad via
    `_hybrid_smooth_weight_grad`.

    `wgrad_via_bmm=True` switches the weight scatter to a sparse-S + bmm
    pattern: build a one-hot S[B, n_tables, K] in `grad_pt.dtype` (bf16 under
    autocast), then contract S against grad_pt via einsum. Used when weights
    are bf16 because bf16 atomic index_add is emulated/slow (e.g. +114% on
    L40S big modules vs fp32+index_add). With fp32 weights, index_add wins
    or ties on all measured shapes, so the caller picks index_add there.
    """
    B, n_tables_, n_outputs = grad_pt.shape
    n_tables, NAP = anchor_a_long.shape
    K = bit_matrix.shape[1]
    input_dim = x.shape[1]
    w_dtype = weights.dtype

    d        = x[:, anchor_a_long] - x[:, anchor_b_long]
    denom    = T_soft + d.abs()

    # Bits actually used in forward (MSB-first packing): bit at position i is
    # bit (NAP-1-i) of the integer index. p_signs has the same +/-1 pattern
    # forward picked, so the surrogate's argmax matches the saved row.
    shifts   = torch.arange(NAP - 1, -1, -1, device=index.device, dtype=index.dtype)
    bits     = ((index.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 1).to(d.dtype)
    p_signs  = bits * 2.0 - 1.0
    p        = p_signs * d.abs() / denom

    ts       = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    z        = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)

    d_sel_soft = torch.einsum("bto,tko->btk", grad_pt.to(w_dtype), weights)

    # Softmax backward, idiomatic PyTorch (compile fuses).
    sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
    d_z      = sel_soft * (d_sel_soft - sum_term)
    d_ts     = d_z / T_sel
    grad_log_T_sel = -(d_z * z).sum()

    # dL/dp via cuBLAS GEMM; dL/dd via the rational soft-sign Jacobian.
    # p = p_signs * |d|/denom -> dp/d|d| = p_signs * T_soft/denom^2;
    # d|d|/dd = sign(d). Hence dp/dd = p_signs * sign(d) * T_soft/denom^2.
    d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))
    d_d = d_p * p_signs * d.sign() * (T_soft / (denom * denom))
    grad_log_T_soft = -(d_d * d).sum()

    if compute_weight_grad:
        if wgrad_via_bmm:
            # Sparse-S + bmm: one-hot at chosen index in bf16, contracted against
            # bf16 grad_pt. cuBLAS bf16 tensor cores use fp32 accumulator
            # internally, then write bf16 output (the source of the ~0.25 ULP
            # precision drift vs index_add).
            g_dtype = grad_pt.dtype
            S = torch.zeros(B, n_tables, K, dtype=g_dtype, device=weights.device)
            S.scatter_(2, index.unsqueeze(-1), 1.0)
            grad_weights = torch.einsum("btk,bto->tko", S, grad_pt).to(accum_dtype)
        else:
            flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * K
            flat_idx    = (index + flat_offset[None, :]).reshape(-1)
            # Accumulate in accum_dtype (= bf16 under autocast) regardless of
            # weights.dtype; caller casts back to weights.dtype at the autograd
            # boundary. Keeps the K-row index_add bandwidth-light when weights
            # are fp32 master copies.
            grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=accum_dtype, device=weights.device)
            grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(accum_dtype))
            grad_weights = grad_w_flat.view(n_tables, K, n_outputs)
    else:
        grad_weights = None

    # dL/dx via scatter-add at anchor positions.
    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a_flat = anchor_a_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b_flat = anchor_b_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat     = d_d.reshape(B, -1).to(x.dtype)
    grad_x.scatter_add_(1, idx_a_flat,  d_flat)
    grad_x.scatter_add_(1, idx_b_flat, -d_flat)

    return grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel


# =============================================================================
# forward_mode="hard": hard forward + soft backward
# =============================================================================

class _FastMHLutSoft(torch.autograd.Function):
    """Hard forward (sign-pack + embedding_bag), soft backward."""

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix, powers,
                n_heads, tph, table_dim, use_bf16):
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        # Cast weights to bf16 for compute when use_bf16=True and storage is
        # fp32. F.embedding_bag (the gather op inside _soft_lut_fwd_body) is
        # not autocast-eligible, so without this explicit cast the gather and
        # the downstream einsum would run at fp32 even inside autocast(bf16).
        # Storage stays fp32; backward's accum_dtype is still weights.dtype.
        compute_in_bf16 = use_bf16 and x.is_cuda and weights.dtype == torch.float32
        weights_compute = weights.to(torch.bfloat16) if compute_in_bf16 else weights
        with autocast_ctx:
            out, index = _soft_lut_fwd_body(
                x, weights_compute, anchor_a_long, anchor_b_long, powers,
                n_heads, tph, table_dim,
            )
        # Preserve the contract that output dtype == weights storage dtype.
        # When the body computed in bf16 on fp32-stored weights, cast back so
        # downstream LayerNorms etc. still see fp32.
        if compute_in_bf16:
            out = out.to(weights.dtype)
        ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, index, log_T_soft, log_T_sel, powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, bit_matrix, index,
         log_T_soft, log_T_sel, powers) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = anchor_a_long.shape[0]
        n_outputs = weights.shape[2]
        grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        # Weight-grad backend is determined by storage dtype: bf16 weights must
        # use the sparse-S + bmm path (atomic index_add on bf16 is emulated and
        # ~2x slower than fp32 baseline), fp32 weights use index_add (wins or
        # ties on all measured shapes — see scratch benches in this repo).
        wgrad_via_bmm = weights.dtype != torch.float32
        with autocast_ctx:
            grad_x, grad_w, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                index, T_soft, T_sel, weights.dtype,
                wgrad_via_bmm=wgrad_via_bmm,
            )
        # 12 forward inputs -> 12 grad returns.
        return (grad_x, grad_w, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None)


# =============================================================================
# forward_mode="hybrid_smooth": top-2 smooth forward + soft backward
# =============================================================================

@torch.compile(dynamic=True)
def _hybrid_smooth_weight_grad(grad_pt, main_index, alt_index, u,
                               n_tables, K, n_outputs):
    """2-row weight gradient for hybrid_smooth backward.

    Two index_add scatters into a flat [n_tables*K, n_outputs] fp32
    accumulator, weighted by (1-u) at main_index and u at alt_index.
    Caller casts the returned fp32 grad to weights.dtype at the autograd
    boundary.

    Internal accumulator is fp32 because bf16 atomic accumulation loses
    precision badly: each LUT row collects O(B/K) ~ thousands of
    contributions, and a bf16 running sum at magnitude O(sqrt(B/K)) drifts
    far beyond the per-add rounding bound.

    The naive alternative — building a [B, n_tables, K] selection mass S
    with scatter_add and contracting via a bmm — materialises 6.3 GB of
    mostly-zero data at the publish recipe (B=32K, K=64); the two
    index_adds here only touch [B, n_tables, n_outputs] sources (~1.5 GB
    each), and the atomic-add contention on the small [n_tables*K, n_out]
    destination is well below HBM-bound territory. ~3.9x faster at
    LUTGPT shapes (~50 ms -> ~13 ms).
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


@torch.compile(dynamic=True)
def _hybrid_smooth_fwd_gather(x, weights, anchor_a_long, anchor_b_long, powers,
                               T_soft, T_sel, n_heads, tph, table_dim):
    """Smooth top-2 forward via two F.embedding gathers + blend.

      main = sign-pack of (x_a > x_b).
      alt  = main with the bit at argmin |d| flipped.
      u    = sigmoid(-Delta/T_sel), Delta = 2*d_min / (T_soft + d_min).
      out  = sum_t [(1-u) * W[main] + u * W[alt]].

    Wins at modules where per-head n_outputs is small (< 128, e.g. qk/v at
    n_heads=6 with d_qk=32/16): the gathers read only ~n_tables * n_outputs
    bf16, and compile fuses gather + multiply + sum into a streaming pattern
    with no materialised [B, n_tables, n_outputs] intermediates. The bmm
    fastpath loses here because its per-head matmul N dim is too narrow for
    tensor cores to amortise tile overhead.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                 # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    powers_view = powers.view(1, 1, -1)
    main_index = (bits * powers_view).sum(dim=-1)                  # [B, n_tables]

    abs_d = d.abs()
    p_star = abs_d.argmin(dim=-1)                                  # least-confident anchor
    flip_mask = powers.to(main_index.dtype)[p_star]
    alt_index = main_index ^ flip_mask

    # Exact top-2 softmax over {main, alt}: see paper note for derivation.
    d_min = abs_d.gather(-1, p_star.unsqueeze(-1)).squeeze(-1)
    delta_ts = 2.0 * d_min / (T_soft + d_min)
    u = torch.sigmoid(-delta_ts / T_sel)                           # in (0, 0.5]
    main_w = 1.0 - u

    table_offset = torch.arange(n_tables, device=weights.device,
                                dtype=main_index.dtype) * table_dim
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    main_flat_idx = (main_index + table_offset.view(1, -1)).reshape(-1)
    alt_flat_idx  = (alt_index  + table_offset.view(1, -1)).reshape(-1)
    main_rows = F.embedding(main_flat_idx, weights_flat).view(B, n_tables, n_outputs)
    alt_rows  = F.embedding(alt_flat_idx,  weights_flat).view(B, n_tables, n_outputs)
    blended = main_rows * main_w.unsqueeze(-1) + alt_rows * u.unsqueeze(-1)
    # Match the hard path's contract that out.dtype == weights.dtype. Under
    # bf16 autocast .sum() is promoted to fp32 for stability, so an explicit
    # final cast is needed.
    out = blended.view(B, n_heads, tph, n_outputs).sum(dim=2).to(weights.dtype)
    return out, main_index, alt_index, u


@torch.compile(dynamic=True)
def _hybrid_smooth_fwd_bmm(x, weights, anchor_a_long, anchor_b_long, powers,
                            T_soft, T_sel, n_heads, tph, table_dim, s_dtype):
    """Smooth top-2 forward via sparse-S + one bmm per head.

    Build a sparse selection mass S[B, n_tables, K] with two nonzeros per
    (b, t) — (1-u) at main_index and u at alt_index — then contract via
    bmm(S, W). At n_heads=1 the bmm collapses to one fat tensor-core matmul;
    at larger n_heads it splits into n_heads parallel matmuls.

    Wins at LUTGPT modules where per-head n_outputs >= 128 (residual_lut,
    emb_resid_lut, qk_lut at d_qk=64): on L40S the wins range from -30%
    (qk) to -48% (residual_lut, emb_resid_lut) vs the gather path because
    the matmul streams S + W instead of doing random-access gathers. S is
    98.5%-sparse at K=128 — fine because the matmul is HBM-bandwidth-bound,
    so unused FLOPs are free.

    `s_dtype` is the build dtype for S — pass bf16 under autocast to skip
    an fp32 -> bf16 cast at matmul time.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    K = table_dim

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                 # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    main_index = (bits * powers.view(1, 1, -1)).sum(dim=-1)        # [B, n_tables]

    abs_d = d.abs()
    p_star = abs_d.argmin(dim=-1)
    flip_mask = powers.to(main_index.dtype)[p_star]
    alt_index = main_index ^ flip_mask

    d_min = abs_d.gather(-1, p_star.unsqueeze(-1)).squeeze(-1)
    delta_ts = 2.0 * d_min / (T_soft + d_min)
    u = torch.sigmoid(-delta_ts / T_sel)                           # in (0, 0.5]
    main_w = 1.0 - u

    S = torch.zeros(B, n_tables, K, dtype=s_dtype, device=x.device)
    S.scatter_(2, main_index.unsqueeze(-1), main_w.unsqueeze(-1).to(s_dtype))
    S.scatter_(2, alt_index.unsqueeze(-1),  u.unsqueeze(-1).to(s_dtype))

    # Per-head contraction over (tph, K). n_tables = n_heads * tph, laid out
    # as [head0_t0..t(tph-1), head1_t0..., ...].
    tph_K = tph * K
    S_h = S.view(B, n_heads, tph_K).transpose(0, 1).contiguous()   # [n_heads, B, tph*K]
    W_h = weights.view(n_heads, tph_K, n_outputs)                  # [n_heads, tph*K, n_out]
    out_h = torch.bmm(S_h, W_h)                                    # [n_heads, B, n_out]
    out = out_h.transpose(0, 1).contiguous().to(weights.dtype)
    return out, main_index, alt_index, u


class _FastMHLutHybridSmooth(torch.autograd.Function):
    """Smooth top-2 forward + soft input grad + 2-row weight grad."""

    # Forward dispatches inline on per-head n_outputs:
    #   n_outputs >= 128 -> _hybrid_smooth_fwd_bmm (sparse-S + tensor-core matmul)
    #   n_outputs <  128 -> _hybrid_smooth_fwd_gather (two embedding gathers + blend)
    # Crossover measured on L40S at LUTGPT shapes (B=4096, NAP=4-6, tph=256-512):
    #     module        n_out  gather (ms)  bmm (ms)   pick
    #     emb_resid_lut   384         1.12      0.58   bmm
    #     residual_lut    384         1.09      0.57   bmm
    #     qk_lut          128         5.56      3.84   bmm
    #     v_lut            16         1.16     12.97   gather
    #     out_proj         96         2.68      3.64   gather
    # The 128 threshold matches the H100 sweep and the L40S numbers.

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix, powers,
                n_heads, tph, table_dim, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            if weights.shape[2] >= 128:
                s_dtype = (torch.bfloat16
                           if use_bf16 and x.is_cuda
                           else weights.dtype)
                out, main_index, alt_index, u = _hybrid_smooth_fwd_bmm(
                    x, weights, anchor_a_long, anchor_b_long, powers,
                    T_soft, T_sel, n_heads, tph, table_dim, s_dtype,
                )
            else:
                out, main_index, alt_index, u = _hybrid_smooth_fwd_gather(
                    x, weights, anchor_a_long, anchor_b_long, powers,
                    T_soft, T_sel, n_heads, tph, table_dim,
                )
        ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, main_index, alt_index, u,
                              log_T_soft, log_T_sel, powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, bit_matrix,
         main_index, alt_index, u,
         log_T_soft, log_T_sel, powers) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = anchor_a_long.shape[0]
        n_outputs = weights.shape[2]
        K = bit_matrix.shape[1]

        grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            # Soft backward gives us grad_x and the temperature grads; we
            # discard its 1-row weight grad (its compute_weight_grad=False
            # makes the accum_dtype passed here a no-op).
            grad_x, _grad_w_unused, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                main_index, T_soft, T_sel, weights.dtype,
                compute_weight_grad=False,
            )

        # _hybrid_smooth_weight_grad accumulates in fp32 internally and is
        # numerically lossless w.r.t. the inputs (bf16 grad_pt limits final
        # precision either way). Cast to weights.dtype at the autograd boundary.
        grad_weights = _hybrid_smooth_weight_grad(
            grad_pt, main_index, alt_index, u, n_tables, K, n_outputs,
        ).to(weights.dtype)

        # 12 forward inputs -> 12 grad returns.
        return (grad_x, grad_weights, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None)


# =============================================================================
# Public module
# =============================================================================

_FORWARD_MODES = ("hard", "hybrid_smooth")


class FastMultiHeadLut(nn.Module):
    """Multi-head LUT primitive used by LUTGPT.

    Args:
        input_dim: dimension of x.
        n_heads: number of output heads.
        n_outputs: per-head output dimension.
        n_anchor_pairs: per-table anchor pairs (NAP), in [1, 15]. Each table
            has K = 2^NAP rows.
        tables_per_head: number of LUT tables summed per head.
        forward_mode: "hard" (default) or "hybrid_smooth". Selects the
            forward path; backward is "soft" in both cases. May be flipped
            at runtime (e.g. soft -> hard finetune) by setting
            `module.forward_mode = "hard"`.
        weight_dtype: storage dtype for the LUT weights. Default
            torch.float32 (training-friendly: keeps an fp32 master copy
            and an fp32 .grad for the optimiser). Pass torch.bfloat16 for
            inference / smaller checkpoints.
        use_bf16: wrap forward and backward in bf16 autocast on CUDA when
            True. Independent of weight_dtype: with the default fp32
            weights + use_bf16=True, forward and weight-grad accumulation
            run in bf16 and only the final .grad is cast back to fp32.
            The hard-mode weight-grad backend is auto-picked from
            weights.dtype (bf16 storage -> sparse-S + bmm; fp32 storage
            -> index_add scatter) because the bf16 atomic index_add is
            emulated and slow.
        anchor_sampling_policy: how anchor pairs are drawn. Default
            AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE.
        soft_score_temp: T_soft (per-anchor sign sharpness).
        select_temp: T_sel (row-selection sharpness in hybrid_smooth, and
            the gradient sharpness in the soft surrogate).
        learnable_temps: if True, log T_soft and log T_sel are trainable
            Parameters; otherwise they are buffers.
        random_seed: seed for anchor sampling and weight init.
        initial_weights_noise: weights ~ Uniform[-sigma, +sigma], cast to
            weight_dtype.
        device: torch.device or None (-> CPU).

    Forward signature:
        x: float [B, input_dim]
        returns: [B, n_heads, n_outputs] in weight_dtype.
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
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        soft_score_temp: float = 0.5,
        select_temp: float = 0.5,
        learnable_temps: bool = True,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if forward_mode not in _FORWARD_MODES:
            raise ValueError(
                f"forward_mode must be one of {_FORWARD_MODES}, got {forward_mode!r}"
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
        anchor_a_long, anchor_b_long = get_balanced_anchor_pairs(
            n_tables=n_lookup_tables,
            n_anchor_pairs=n_anchor_pairs,
            input_dim=input_dim,
            device=dev,
            random_seed=random_seed,
            policy=policy,
            n_heads=n_heads,
        )
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
        # Anchor pairs as int64; reused by forward and backward.
        self.register_buffer("soft_anchor_a_long", anchor_a_long.contiguous())
        self.register_buffer("soft_anchor_b_long", anchor_b_long.contiguous())
        # Per-table row-block offsets used by the bag-reduce in hard eval:
        # entries 0..K-1 live in table 0, K..2K-1 in table 1, etc. Buffer so
        # we don't rebuild the arange every call.
        self.register_buffer(
            "_table_offset",
            torch.arange(n_lookup_tables, device=dev, dtype=torch.int64) * self.table_dim,
        )
        # Cache a bound reference to the MSB-first native eval kernel so the
        # forward path doesn't pay the manager getter or attribute lookup on
        # every call. None when lutorch_cuda is unavailable -- the compiled
        # forward body is then used as fallback.
        native_manager = (
            _get_native_lutorch_manager() if _USE_LUTORCH_CUSTOM_CUDA_KERNELS else None
        )
        self._native_eval_msb = (
            getattr(native_manager, "anchor_pairs_lookup_eval_forward_msb", None)
            if native_manager is not None else None
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

    def _hard_eval_native(self, x: torch.Tensor,
                          weights_compute: torch.Tensor) -> torch.Tensor:
        """Hard-mode eval via the MSB-first lutorch_cuda bit-pack kernel.

        Replaces the compiled fp32 bit-pack of _soft_lut_fwd_body with a single
        CUDA pass that emits int64 row indices in our MSB-first convention,
        feeding straight into a compiled embedding_bag reduce.
        """
        B = x.shape[0]
        n_tables = self.soft_anchor_a_long.shape[0]
        n_outputs = weights_compute.shape[2]
        index = self._native_eval_msb(
            x, self.soft_anchor_a_long, self.soft_anchor_b_long,
            0.0, _LUTORCH_CUDA_THREADS_PER_BLOCK,
        )  # [B, n_tables] int64
        weights_flat = weights_compute.view(n_tables * self.table_dim, n_outputs)
        return _native_eval_bag_reduce(
            index, weights_flat, self._table_offset,
            B, self.n_heads, self.tables_per_head, n_outputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.forward_mode == "hybrid_smooth":
            return _FastMHLutHybridSmooth.apply(
                x, self.weights, self.log_soft_score_temp, self.log_select_temp,
                self.soft_anchor_a_long, self.soft_anchor_b_long,
                self.soft_bit_matrix, self.soft_powers,
                self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
            )
        # forward_mode == "hard"
        if not torch.is_grad_enabled():
            # Eval: prefer the native CUDA bit-pack kernel when available; fall
            # back to the compiled forward body otherwise. The bf16 weight cast
            # mirrors _FastMHLutSoft.forward --- F.embedding_bag isn't
            # autocast-eligible, so we have to cast weights explicitly.
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
            use_native = (
                self._native_eval_msb is not None
                and x.is_cuda
                and x.dtype in (torch.float32, torch.float64)
            )
            with autocast_ctx:
                if use_native:
                    out = self._hard_eval_native(x, weights_compute)
                else:
                    out, _ = _soft_lut_fwd_body(
                        x, weights_compute,
                        self.soft_anchor_a_long, self.soft_anchor_b_long,
                        self.soft_powers,
                        self.n_heads, self.tables_per_head, self.table_dim,
                    )
            if compute_in_bf16:
                out = out.to(self.weights.dtype)
            return out
        return _FastMHLutSoft.apply(
            x, self.weights, self.log_soft_score_temp, self.log_select_temp,
            self.soft_anchor_a_long, self.soft_anchor_b_long,
            self.soft_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
        )
