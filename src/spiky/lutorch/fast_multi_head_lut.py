"""FastMultiHeadLUT — multi-head LUT primitive for LUTGPT.

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

See paper/tinymhl_hybrid_smooth.tex for the math.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs


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


def _ball_bit_matrix(nap: int, device, dtype=torch.float32) -> torch.Tensor:
    """[NAP+1, NAP] Hamming-1 ball bit-pattern matrix.
    Row 0 = main (all +1). Row 1+k flips anchor k (-1 at column k, +1 elsewhere).
    Used by the ball backward to score only the (NAP+1) row neighbours of the
    chosen main row via ts_r = sum_i ball_bit_matrix[r, i] * abs_p_i."""
    top = torch.ones((1, nap), device=device, dtype=dtype)
    bottom = 1.0 - 2.0 * torch.eye(nap, device=device, dtype=dtype)
    return torch.cat([top, bottom], dim=0)


# =============================================================================
# Hard-forward body and eval shortcut
# =============================================================================

@torch.compile
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

@torch.compile
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
    autocast), then contract S against grad_pt via einsum to get the per-row
    weight gradient. Wins at LUTGPT shapes when per-head n_outputs >= 128
    (out_proj, residual_lut, qk_lut: -16 to -37% bwd time vs index_add). At
    smaller n_outputs (v_lut at 64) the cost of materialising 1.5-2 GB of
    mostly-zero S cancels the atomic-scatter savings, so caller dispatches on
    n_outputs >= 128. Trades fp32 grad accumulation for one bf16 output
    truncation (~0.25 bf16 ULP, rel_rms ~2e-3 vs the fp32-accumulated
    baseline) — small enough for Lion (sign-based) optimisers and the size of
    the bf16 input grad floor at LUT-LM scale.
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
# Ball backward: gradient through only (NAP+1) Hamming-1 neighbour rows
# =============================================================================

@torch.compile
def _ball_lut_bwd_body(grad_pt, x, weights, anchor_a_long, anchor_b_long,
                       ball_bit_matrix, powers, index, T_soft, T_sel,
                       accum_dtype: torch.dtype,
                       compute_weight_grad: bool = True,
                       wgrad_via_bmm: bool = False):
    """Hamming-1 ball backward pinned to the chosen index.

    Surrogate softmax is over R = NAP+1 rows: the main row (saved `index`)
    plus the NAP rows reached by flipping a single anchor bit. ts_r is the
    dot product of the ball bit-pattern row r with abs_p; abs_p contains
    only the unsigned soft-sign magnitudes |d_i|/(T_soft+|d_i|).

    Memory profile vs full K-row backward: the K-column Z matrix is still
    materialised through one cuBLAS einsum ("tro,bto->btr"), but every
    other intermediate (ts, z, sel_soft, d_z, d_ts) shrinks to [B, t, R].
    At NAP=7 (R=8 vs K=128) the softmax/derivation arithmetic is the small
    savings; at large NAP the win compounds.

    `compute_weight_grad=False` skips the 1-row weight scatter — used by
    hybrid_smooth backward.
    """
    B, n_tables_, n_outputs = grad_pt.shape
    n_tables, NAP = anchor_a_long.shape
    R = ball_bit_matrix.shape[0]      # NAP + 1
    K = weights.shape[1]
    input_dim = x.shape[1]
    w_dtype = weights.dtype

    d        = x[:, anchor_a_long] - x[:, anchor_b_long]
    abs_d    = d.abs()
    denom    = T_soft + abs_d
    abs_p    = abs_d / denom                                          # [B, t, NAP]

    # ts_r = sum_i ball_bit_matrix[r, i] * abs_p[..., i]
    ts       = torch.einsum("btp,rp->btr", abs_p, ball_bit_matrix.to(abs_p.dtype))
    z        = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)

    # Full-K Z matrix (single GEMM) and gather (NAP+1) ball columns.
    Z_full   = torch.einsum("tro,bto->btr",
                            weights.to(grad_pt.dtype), grad_pt)        # [B, t, K]
    ball_idx = torch.cat([
        index.unsqueeze(-1),
        index.unsqueeze(-1) ^ powers.view(1, 1, NAP),
    ], dim=-1)                                                         # [B, t, R]
    Z        = Z_full.gather(-1, ball_idx)                             # [B, t, R]

    sum_term = (Z * sel_soft).sum(dim=-1, keepdim=True)
    d_z      = sel_soft * (Z - sum_term)
    d_ts     = d_z / T_sel
    grad_log_T_sel = -(d_z * z).sum()

    # d_abs_p[i] = sum_r d_ts[r] * ball_bit_matrix[r, i]
    d_abs_p  = torch.einsum("btr,rp->btp", d_ts, ball_bit_matrix.to(d_ts.dtype))
    # abs_p = abs_d / (T_soft + abs_d) -> d(abs_p)/d(abs_d) = T_soft / denom^2.
    # d(abs_p)/d(T_soft) = -abs_d / denom^2.
    d_abs_d  = d_abs_p * T_soft / (denom * denom)
    d_d      = d_abs_d * d.sign()
    grad_log_T_soft = -(d_abs_p * abs_d / (denom * denom)).sum() * T_soft

    if compute_weight_grad:
        if wgrad_via_bmm:
            # Single-row scatter via sparse-S + bmm. See `_soft_lut_bwd_body`
            # for the precision/speed tradeoff; same dispatch threshold applies.
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

    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a_flat = anchor_a_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b_flat = anchor_b_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat     = d_d.reshape(B, -1).to(x.dtype)
    grad_x.scatter_add_(1, idx_a_flat,  d_flat)
    grad_x.scatter_add_(1, idx_b_flat, -d_flat)

    return grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel


# =============================================================================
# Ball-gather backward: build Z via F.embedding-gather of only (NAP+1) rows
# per (b, t), B-chunked to bound peak memory. Avoids the K-row Z_full GEMM.
# =============================================================================

@torch.compile
def _ball_gather_Z_chunk(weights_flat, table_offset_R, ball_idx_chunk,
                          grad_pt_chunk, n_outputs):
    """One B-chunk of Z = sum_o W[t, ball_idx[b, t, r], o] * grad_pt[b, t, o].

    weights_flat: [n_tables * K, n_outputs]   (fp32 master, cast to grad_pt dtype)
    table_offset_R: [1, n_tables, 1] precomputed offsets
    ball_idx_chunk: [bb, n_tables, R] ball indices
    grad_pt_chunk: [bb, n_tables, n_outputs]  (bf16 under autocast)
    returns: [bb, n_tables, R]                (bf16)
    """
    bb, n_tables, R = ball_idx_chunk.shape
    flat_idx = (ball_idx_chunk + table_offset_R).reshape(bb * n_tables * R)
    # Gather in grad_pt's dtype so the einsum stays bf16 — gather payload
    # halves, and the dense einsum matches _ball_lut_bwd_body's bf16 path.
    W_flat_cast = weights_flat.to(grad_pt_chunk.dtype)
    W_ball = F.embedding(flat_idx, W_flat_cast).view(bb, n_tables, R, n_outputs)
    return torch.einsum("btro,bto->btr", W_ball, grad_pt_chunk)


def _ball_gather_lut_bwd_body(grad_pt, x, weights, anchor_a_long, anchor_b_long,
                                ball_bit_matrix, powers, index, T_soft, T_sel,
                                accum_dtype: torch.dtype,
                                compute_weight_grad: bool = True,
                                b_chunk: int = 1024,
                                wgrad_via_bmm: bool = False):
    """Same surrogate as `_ball_lut_bwd_body`, but skips the full K-row Z
    matrix. Computes Z[b, t, r] for r in the (NAP+1) Hamming-1 ball directly
    via F.embedding on those rows, chunked over B to bound peak memory.

    FLOPs for the Z computation scale as (NAP+1) instead of K=2^NAP — at
    NAP=10 that's 1024/11 ≈ 100× FLOPs savings; at NAP=7 about 16×. The
    chunk size trades peak memory (∝ b_chunk * n_tables * R * n_outputs)
    against kernel-launch overhead. Default b_chunk=1024 keeps per-chunk
    W_ball under ~3 GB at the LUTGPT shape (n_tables=512, R=8, n_out=384).
    """
    B, n_tables_, n_outputs = grad_pt.shape
    n_tables, NAP = anchor_a_long.shape
    R = ball_bit_matrix.shape[0]
    K = weights.shape[1]
    input_dim = x.shape[1]

    d        = x[:, anchor_a_long] - x[:, anchor_b_long]
    abs_d    = d.abs()
    denom    = T_soft + abs_d
    abs_p    = abs_d / denom

    ts       = torch.einsum("btp,rp->btr", abs_p, ball_bit_matrix.to(abs_p.dtype))
    z        = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)

    # Ball indices for the (NAP+1) rows; same as ball backward.
    ball_idx = torch.cat([
        index.unsqueeze(-1),
        index.unsqueeze(-1) ^ powers.view(1, 1, NAP),
    ], dim=-1)                                                         # [B, t, R]

    weights_flat   = weights.view(n_tables * K, n_outputs)
    table_offset_R = (torch.arange(n_tables, device=weights.device,
                                    dtype=index.dtype) * K).view(1, -1, 1)

    # Build Z chunk-by-chunk. Each chunk peaks at [b_chunk, t, R, n_out]
    # W_ball; never materialises a full [B, t, K] tensor.
    Z_chunks = []
    for start in range(0, B, b_chunk):
        end = min(start + b_chunk, B)
        Z_chunks.append(_ball_gather_Z_chunk(
            weights_flat, table_offset_R,
            ball_idx[start:end].contiguous(),
            grad_pt[start:end],
            n_outputs,
        ))
    Z = torch.cat(Z_chunks, dim=0)                                      # [B, t, R]

    sum_term = (Z * sel_soft).sum(dim=-1, keepdim=True)
    d_z      = sel_soft * (Z - sum_term)
    d_ts     = d_z / T_sel
    grad_log_T_sel = -(d_z * z).sum()

    d_abs_p  = torch.einsum("btr,rp->btp", d_ts, ball_bit_matrix.to(d_ts.dtype))
    d_abs_d  = d_abs_p * T_soft / (denom * denom)
    d_d      = d_abs_d * d.sign()
    grad_log_T_soft = -(d_abs_p * abs_d / (denom * denom)).sum() * T_soft

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
                anchor_a_long, anchor_b_long, bit_matrix, ball_bit_matrix,
                powers, n_heads, tph, table_dim, use_bf16, backward_mode):
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
        # Preserve the historical convention: output dtype = weights storage
        # dtype. When the body computed in bf16 on fp32-stored weights, cast
        # the output back so downstream LayerNorms etc. still see fp32.
        if compute_in_bf16:
            out = out.to(weights.dtype)
        ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, ball_bit_matrix, index,
                              log_T_soft, log_T_sel, powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        ctx.backward_mode = backward_mode
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, bit_matrix, ball_bit_matrix,
         index, log_T_soft, log_T_sel, powers) = ctx.saved_tensors
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
        # Mirror the forward's compute-dtype cast: pass bf16 weights to the
        # body when use_bf16=True and storage is fp32, so the body's einsum
        # runs at bf16 just like the forward gather. accum_dtype is still
        # weights.dtype (= fp32 here), so grad_w is accumulated at fp32.
        compute_in_bf16 = ctx.use_bf16 and x.is_cuda and weights.dtype == torch.float32
        weights_compute = weights.to(torch.bfloat16) if compute_in_bf16 else weights
        # Weight-grad H phase: always use the sparse-S + bmm path. Wins at
        # out_proj/residual_lut/qk_lut (n_out>=128) by ~30-40% over atomic-add
        # scatter; ties at v_lut shape (n_out=64) under fp32 weights; strictly
        # faster (~28%) under bf16 weights where atomic-add scatter is slow.
        # Removing the dispatch simplifies the code path at no measured cost.
        use_bmm_wgrad = True
        with autocast_ctx:
            if ctx.backward_mode == "ball":
                grad_x, grad_w, grad_log_Ts, grad_log_Tx = _ball_lut_bwd_body(
                    grad_pt, x, weights_compute, anchor_a_long, anchor_b_long,
                    ball_bit_matrix, powers, index, T_soft, T_sel, weights.dtype,
                    wgrad_via_bmm=use_bmm_wgrad,
                )
            elif ctx.backward_mode == "ball_gather":
                grad_x, grad_w, grad_log_Ts, grad_log_Tx = _ball_gather_lut_bwd_body(
                    grad_pt, x, weights_compute, anchor_a_long, anchor_b_long,
                    ball_bit_matrix, powers, index, T_soft, T_sel, weights.dtype,
                    wgrad_via_bmm=use_bmm_wgrad,
                )
            else:
                grad_x, grad_w, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                    grad_pt, x, weights_compute, anchor_a_long, anchor_b_long, bit_matrix,
                    index, T_soft, T_sel, weights.dtype,
                    wgrad_via_bmm=use_bmm_wgrad,
                )
        # 14 forward inputs -> 14 grad returns.
        return (grad_x, grad_w, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None, None, None)


# =============================================================================
# forward_mode="hybrid_smooth": top-2 smooth forward + soft backward
# =============================================================================

@torch.compile
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


@torch.compile
def _hybrid_smooth_fwd_bmm(x, weights, anchor_a_long, anchor_b_long, powers,
                            T_soft, T_sel, n_heads, tph, table_dim, s_dtype):
    """n_heads=1 forward: build sparse selection mass S[B, n_tables, K] with
    two nonzeros per (b, t), contract via one big tensor-core matmul.

    At n_heads=1 the bmm collapses to a single fat matmul whose tile sizes
    fully amortise tensor-core overhead. Replaces two random-access
    F.embedding gathers (~9.7 GB bf16 HBM at random offsets at NAP=7 out_proj
    shape) with one streaming bf16 matmul reading ~1.6 GB. S is built in
    `s_dtype` (bf16 under autocast, weights.dtype otherwise) to skip the
    fp32->bf16 cast inside the matmul. ~5 ms saved on G at LUTGPT out_proj.

    98.5%-sparse S at K=128 is fine — the matmul is HBM-bandwidth-bound,
    so the unused FLOPs are free.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    K = table_dim

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                 # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    main_index = (bits * powers.view(1, 1, -1)).sum(dim=-1)        # [B, n_tables]

    abs_d = d.abs()
    p_star = abs_d.argmin(dim=-1)                                  # least-confident anchor
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


@torch.compile
def _hybrid_smooth_fwd_gather(x, weights, anchor_a_long, anchor_b_long, powers,
                               T_soft, T_sel, n_heads, tph, table_dim):
    """n_heads>1 forward: gather + blend + sum across tph.

    The bmm path loses at n_heads>=2 because the per-head matmul shape
    (N=n_outputs/n_heads, often 64) is too narrow for tensor cores to
    amortise their tile overhead. At qkv shape (n_heads=6, NAP=6, tph=256,
    n_out=64) the random gathers read only ~4.8 GB of bf16 (n_outputs is
    small), and that random-access cost beats the bmm's setup cost.
    Compile fuses gather + multiply + sum into a streaming pattern with no
    materialised [B, n_tables, n_out] intermediates.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                 # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    main_index = (bits * powers.view(1, 1, -1)).sum(dim=-1)

    abs_d = d.abs()
    p_star = abs_d.argmin(dim=-1)
    flip_mask = powers.to(main_index.dtype)[p_star]
    alt_index = main_index ^ flip_mask

    d_min = abs_d.gather(-1, p_star.unsqueeze(-1)).squeeze(-1)
    delta_ts = 2.0 * d_min / (T_soft + d_min)
    u = torch.sigmoid(-delta_ts / T_sel)
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


# Shape-dependent fastpath: `_FastMHLutHybridSmooth.forward` dispatches inline
# between _hybrid_smooth_fwd_bmm (n_outputs >= 128) and _hybrid_smooth_fwd_gather
# (n_outputs < 128). The bmm splits as n_heads independent matmuls of shape
# [B, tph*K] @ [tph*K, n_outputs], so the per-head N axis (= the n_outputs
# constructor arg, not the total output dim) is what controls tensor-core
# tile efficiency. Crossover sweep (B=6144, NAP=6, tph=256):
#     n_outputs:    32   64  128  192  384
#     gather (ms):  ~30  ~30 ~53 ~73 ~189   (n_heads=6)
#     bmm    (ms):  ~30  ~31 ~26 ~30  ~71
# n_heads matters indirectly: gather scales linearly with n_tables, while
# bmm batches heads via tensor cores, so the bmm win compounds at high
# n_heads + large n_outputs (e.g. n_heads=6, n_outputs=384: -118 ms).
#
# LUTGPT module shapes hit by this dispatch:
#   - bmm wins:  out_proj (n_out=384), residual_lut (n_out=384),
#                emb_resid_lut (n_out=384)
#   - gather wins: qkv_lut (n_out=64), v_lut (n_out=64)


class _FastMHLutHybridSmooth(torch.autograd.Function):
    """Smooth top-2 forward + soft input grad + 2-row weight grad."""

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix, ball_bit_matrix,
                powers, n_heads, tph, table_dim, use_bf16, backward_mode):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            # Dispatch on per-head N (= n_outputs). Threshold 128 is the
            # tensor-core efficiency crossover at H100 bf16 + LUTGPT shapes;
            # see the note above _hybrid_smooth_fwd_bmm.
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
                              bit_matrix, ball_bit_matrix,
                              main_index, alt_index, u,
                              log_T_soft, log_T_sel, powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        ctx.backward_mode = backward_mode
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, bit_matrix, ball_bit_matrix,
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
            if ctx.backward_mode == "ball":
                grad_x, _grad_w_unused, grad_log_Ts, grad_log_Tx = _ball_lut_bwd_body(
                    grad_pt, x, weights, anchor_a_long, anchor_b_long,
                    ball_bit_matrix, powers, main_index, T_soft, T_sel,
                    weights.dtype, compute_weight_grad=False,
                )
            elif ctx.backward_mode == "ball_gather":
                grad_x, _grad_w_unused, grad_log_Ts, grad_log_Tx = _ball_gather_lut_bwd_body(
                    grad_pt, x, weights, anchor_a_long, anchor_b_long,
                    ball_bit_matrix, powers, main_index, T_soft, T_sel,
                    weights.dtype, compute_weight_grad=False,
                )
            else:
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

        # 14 forward inputs -> 14 grad returns.
        return (grad_x, grad_weights, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None, None, None)


# =============================================================================
# Public module
# =============================================================================

_FORWARD_MODES = ("hard", "hybrid_smooth")
_BACKWARD_MODES = ("dense_K", "ball", "ball_gather")


class FastMultiHeadLUT(nn.Module):
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
        backward_mode: str = "dense_K",
        weight_dtype: torch.dtype = torch.float32,
        use_bf16: bool = True,
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
        if backward_mode not in _BACKWARD_MODES:
            raise ValueError(
                f"backward_mode must be one of {_BACKWARD_MODES}, got {backward_mode!r}"
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
        self.backward_mode = backward_mode
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
        # [NAP+1, NAP] ball bit matrix for the ball backward surrogate.
        self.register_buffer(
            "ball_bit_matrix",
            _ball_bit_matrix(n_anchor_pairs, dev, dtype=torch.float32),
        )
        self.register_buffer("soft_powers", _msb_powers(n_anchor_pairs, dev))
        # Anchor pairs as int64; reused by forward and backward.
        self.register_buffer("soft_anchor_a_long", anchor_a_long.contiguous())
        self.register_buffer("soft_anchor_b_long", anchor_b_long.contiguous())

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.forward_mode == "hybrid_smooth":
            return _FastMHLutHybridSmooth.apply(
                x, self.weights, self.log_soft_score_temp, self.log_select_temp,
                self.soft_anchor_a_long, self.soft_anchor_b_long,
                self.soft_bit_matrix, self.ball_bit_matrix, self.soft_powers,
                self.n_heads, self.tables_per_head, self.table_dim,
                self.use_bf16, self.backward_mode,
            )
        # forward_mode == "hard"
        if not torch.is_grad_enabled():
            # Eval: reuse the compiled forward body and drop the index.
            autocast_ctx = (
                torch.amp.autocast("cuda", dtype=torch.bfloat16)
                if self.use_bf16 and x.is_cuda
                else torch.amp.autocast("cpu", enabled=False)
            )
            with autocast_ctx:
                out, _ = _soft_lut_fwd_body(
                    x, self.weights,
                    self.soft_anchor_a_long, self.soft_anchor_b_long,
                    self.soft_powers,
                    self.n_heads, self.tables_per_head, self.table_dim,
                )
            return out
        return _FastMHLutSoft.apply(
            x, self.weights, self.log_soft_score_temp, self.log_select_temp,
            self.soft_anchor_a_long, self.soft_anchor_b_long,
            self.soft_bit_matrix, self.ball_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim,
            self.use_bf16, self.backward_mode,
        )
