"""TinyMultiHeadLut — multi-head LUT primitive for LUTGPT.

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

from spiky.lutorch.tiny_anchor_pairs_lookup import TinyAnchorPairsLookup
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


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


def _embedding_bag_forward(weights: torch.Tensor, lookup_indices: torch.Tensor,
                           n_heads: int, tph: int) -> torch.Tensor:
    """Eval shortcut for hard mode: same gather+sum as the autograd Function,
    but skips the index-recompute path so eval doesn't pay it."""
    B, n_lookup_tables = lookup_indices.shape
    table_dim = weights.shape[1]
    n_outputs = weights.shape[2]
    weights_flat = weights.view(n_lookup_tables * table_dim, n_outputs)
    table_offset = (
        torch.arange(n_lookup_tables, device=weights.device, dtype=lookup_indices.dtype)
        * table_dim
    )
    flat_indices = (lookup_indices + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tph
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs)


# =============================================================================
# Shared soft backward (used by both hard and hybrid_smooth forward modes)
# =============================================================================

@torch.compile
def _soft_lut_bwd_body(grad_pt, x, weights, anchor_a_long, anchor_b_long,
                        bit_matrix, index, T_soft, T_sel,
                        compute_weight_grad: bool = True):
    """Soft backward pinned to the actually-chosen index.

    Reconstructs p_signs from `index` so the surrogate softmax's argmax matches
    the row picked in forward. Returns full-K softmax gradients for x and the
    two log-temperatures, plus a 1-row weight grad at the chosen row.

    `compute_weight_grad=False` skips the 1-row weight scatter — used by
    hybrid_smooth backward, which supplies its own 2-row weight grad via
    `_hybrid_smooth_weight_grad`.
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
        flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * K
        flat_idx    = (index + flat_offset[None, :]).reshape(-1)
        grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=w_dtype, device=weights.device)
        grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(w_dtype))
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

class _TinyMHLutSoft(torch.autograd.Function):
    """Hard forward (sign-pack + embedding_bag), soft backward."""

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix, powers,
                n_heads, tph, table_dim, use_bf16):
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            out, index = _soft_lut_fwd_body(
                x, weights, anchor_a_long, anchor_b_long, powers,
                n_heads, tph, table_dim,
            )
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
        with autocast_ctx:
            grad_x, grad_w, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                index, T_soft, T_sel,
            )
        # 12 forward inputs -> 12 grad returns.
        return (grad_x, grad_w, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None)


# =============================================================================
# forward_mode="hybrid_smooth": top-2 smooth forward + soft backward
# =============================================================================

@torch.compile
def _hybrid_smooth_weight_grad(grad_pt, main_index, alt_index, u,
                               n_tables, K, n_outputs, w_dtype):
    """2-row weight gradient for hybrid_smooth backward.

    Builds a per-(b, t) "selection mass" S of shape [B, n_tables, K] by
    scatter-adding (1-u) at main_index and u at alt_index. Then
    dW[t, k, o] = sum_b S[b, t, k] * grad_pt[b, t, o]
    in a single B-reducing einsum (cuBLAS bmm). Avoids the atomicAdd
    contention of a flat global index_add on hot K-row destinations.
    """
    B = grad_pt.shape[0]
    all_idx  = torch.stack([main_index, alt_index], dim=-1)
    weights2 = torch.stack([(1.0 - u).to(w_dtype),
                             u.to(w_dtype)], dim=-1)
    S = torch.zeros(B, n_tables, K, dtype=w_dtype, device=grad_pt.device)
    S.scatter_add_(-1, all_idx, weights2)
    return torch.einsum('btk,bto->tko', S, grad_pt.to(w_dtype))


@torch.compile
def _hybrid_smooth_lut_fwd_body(x, weights, anchor_a_long, anchor_b_long, powers,
                                 T_soft, T_sel, n_heads, tph, table_dim):
    """Smooth top-2 forward: blend main row and Hamming-1 alt at the
    least-confident anchor pair.

      main = sign-pack of (x_a > x_b).
      alt  = main with the bit at argmin |d| flipped.
      u    = sigmoid(-Delta/T_sel), Delta = 2*d_min / (T_soft + d_min).
      out  = sum_t [(1-u) * W[main] + u * W[alt]].
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
    out = blended.view(B, n_heads, tph, n_outputs).sum(dim=2)
    return out, main_index, alt_index, u


class _TinyMHLutHybridSmooth(torch.autograd.Function):
    """Smooth top-2 forward + soft input grad + 2-row weight grad."""

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
            out, main_index, alt_index, u = _hybrid_smooth_lut_fwd_body(
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
        w_dtype = weights.dtype

        grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            # Soft backward gives us grad_x and the temperature grads; we
            # discard its 1-row weight grad and overwrite with the 2-row
            # hybrid scatter below.
            grad_x, _grad_w_unused, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                main_index, T_soft, T_sel,
                compute_weight_grad=False,
            )

        grad_weights = _hybrid_smooth_weight_grad(
            grad_pt, main_index, alt_index, u, n_tables, K, n_outputs, w_dtype,
        )

        # 12 forward inputs -> 12 grad returns.
        return (grad_x, grad_weights, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None)


# =============================================================================
# Public module
# =============================================================================

_FORWARD_MODES = ("hard", "hybrid_smooth")


class TinyMultiHeadLut(nn.Module):
    """Multi-head LUT primitive used by LUTGPT.

    Args:
        input_dim: dimension of x (must be <= 32767, int16 anchor range).
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
            torch.bfloat16 saves ~2x memory; pass torch.float32 for
            fp32 master weights.
        use_bf16: wrap forward and backward in bf16 autocast on CUDA when
            True. Independent of weight_dtype (you can keep fp32 master
            weights and still compute in bf16).
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
        weight_dtype: torch.dtype = torch.bfloat16,
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
        if not (1 <= n_anchor_pairs <= 15):
            raise ValueError(
                f"n_anchor_pairs must be in [1, 15] (int16 lookup range), "
                f"got {n_anchor_pairs}"
            )
        if input_dim > 32767:
            raise ValueError(
                f"input_dim must be <= 32767 (int16 anchor index range), "
                f"got {input_dim}"
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
        self.lookup = TinyAnchorPairsLookup(
            input_dim=input_dim,
            n_tables=n_lookup_tables,
            n_anchor_pairs=n_anchor_pairs,
            n_heads=n_heads,
            random_seed=random_seed,
            device=device,
            anchor_sampling_policy=policy,
        )

        dev = device or torch.device("cpu")
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
        # Anchor pairs as int64 — cast once, reused by forward and backward.
        self.register_buffer(
            "soft_anchor_a_long",
            self.lookup.anchor_pairs_a.long().contiguous(),
        )
        self.register_buffer(
            "soft_anchor_b_long",
            self.lookup.anchor_pairs_b.long().contiguous(),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.forward_mode == "hybrid_smooth":
            return _TinyMHLutHybridSmooth.apply(
                x, self.weights, self.log_soft_score_temp, self.log_select_temp,
                self.soft_anchor_a_long, self.soft_anchor_b_long,
                self.soft_bit_matrix, self.soft_powers,
                self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
            )
        # forward_mode == "hard"
        if not torch.is_grad_enabled():
            # Eval shortcut: skip the autograd Function and its index recompute.
            d = x[:, self.soft_anchor_a_long] - x[:, self.soft_anchor_b_long]
            bits = (d > 0).to(torch.int64)
            index = (bits * self.soft_powers.view(1, 1, -1)).sum(dim=-1)
            return _embedding_bag_forward(
                self.weights, index, self.n_heads, self.tables_per_head,
            )
        return _TinyMHLutSoft.apply(
            x, self.weights, self.log_soft_score_temp, self.log_select_temp,
            self.soft_anchor_a_long, self.soft_anchor_b_long,
            self.soft_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
        )
