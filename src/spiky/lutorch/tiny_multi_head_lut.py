"""TinyMultiHeadLut — minimal MultiHeadLut variant for fast training/inference.

Sibling of MultiHeadLut, mirroring TinyAnchorPairsLookup's "stripped-down for
the common case" style. Hardcoded simplifications:

  - smooth_mode = False (no per-alternative soft routing)
  - n_alternatives = 1 (single carrier, hard lookup)
  - n_anchor_pairs <= 15 (TinyAnchorPairsLookup int16 constraint)
  - input_dim <= 32767 (int16 anchor index limit)
  - cmp_eps = 0
  - anchor_sampling_policy ∈ {CANONICAL_FULL_COVERAGE, CANONICAL_DISTINCT}
  - shuffle_per_head = True
  - n_buckets = 1 (no bucket conditioning)

Design choices vs MultiHeadLut:
  - Weights stored in user-chosen dtype (default torch.bfloat16) for ~2× memory
    savings vs fp32 MultiHeadLut. Forward gather, backward scatter both work
    natively in bf16/fp16 on H100+ (Tensor Core paths).
  - No per-table-output materialisation: forward computes the [B, n_lookup_tables,
    n_outputs] gather, immediately reduces to [B, n_heads, n_outputs] by summing
    over tables_per_head. Backward through advanced indexing handled by autograd.
  - Custom Adam optimiser (TinyMultiHeadLutOptimizer) keeps m, v in the same
    weight dtype (bf16 by default) for matching memory savings.

Forward signature (matches MultiHeadLut's reduced output):
  x: float [B, input_dim]
  returns: [B, n_heads, n_outputs] in weights' dtype.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from spiky.lutorch.tiny_anchor_pairs_lookup import TinyAnchorPairsLookup
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def _soft_bit_matrix_msb(nap: int, device, dtype=torch.float32) -> torch.Tensor:
    """[NAP, K] bit pattern matrix with MSB-first convention:
    bit_matrix[i, k] = +1 if (k >> (NAP-1-i)) & 1 else -1.
    Used by the soft-mode pure-PyTorch backward."""
    n = 1 << nap
    bits = ((torch.arange(n, device=device).unsqueeze(0)
             >> torch.arange(nap - 1, -1, -1, device=device).unsqueeze(1)) & 1)
    return ((bits.float() - 0.5) * 2.0).to(dtype)


def _msb_powers(nap: int, device) -> torch.Tensor:
    """powers[i] = 2^(NAP-1-i) — MSB-first packing matching _soft_bit_matrix_msb.
    index = Σ_i (d_i > 0) * powers[i]  produces the row k that maximizes
    Σ_i sign(d_i) * bit_matrix[i, k] for the soft-mode argmax."""
    return (1 << torch.arange(nap - 1, -1, -1, device=device, dtype=torch.int64))

# Toggle to disable the native fused backward kernel (PyTorch fallback path).
_USE_TINY_MHLUT_NATIVE_BWD = True

_NATIVE_MHLUT = None
def _get_tiny_mhlut_native():
    """Lazily fetch the native LUTorchManager and check for the fused-bwd binding."""
    global _NATIVE_MHLUT
    if _NATIVE_MHLUT is not None:
        return _NATIVE_MHLUT
    try:
        import lutorch_cuda  # noqa: F401  (loaded for side effects)
        m = lutorch_cuda.get_lutorch_manager()
        if hasattr(m, 'tiny_mhlut_backward_na1'):
            _NATIVE_MHLUT = m
            return m
    except Exception:
        pass
    return None


def _embedding_bag_forward(weights: torch.Tensor, lookup_indices: torch.Tensor,
                           n_heads: int, tables_per_head: int) -> torch.Tensor:
    """Fused gather + reduce via F.embedding_bag (mode='sum'). Used by both
    the training autograd Function and the eval no-grad shortcut."""
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
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tables_per_head
    out_flat = torch.nn.functional.embedding_bag(
        flat_indices, weights_flat, offsets=offsets, mode='sum',
    )
    return out_flat.view(B, n_heads, n_outputs)


class _TinyMHLutGatherReduce(torch.autograd.Function):
    """Gather + reduce + STE-carrier-thread, with recompute-in-backward.

    Forward:
        weights: [n_lookup_tables, table_dim, n_outputs]
        lookup_indices: [B, n_lookup_tables] int64 (chosen anchor pair, "main")
        lookup_alt_indices: [B, n_lookup_tables] int64 (runner-up, "alt")
        lookup_indices_grad_c: [B, n_lookup_tables] zero-valued main carrier
            with autograd link to x (via TinyAnchorPairsLookup).
        lookup_alt_indices_grad_c: [B, n_lookup_tables] zero-valued alt carrier
            with the matching autograd link.
        n_heads, tables_per_head: shape ints (saved on ctx).

    Both carriers are required for x.grad correctness: TinyAnchorPairsLookup's
    backward computes the STE update as
        du = (grad_main - grad_alt) * uncertainty_derivative,
    matching MHLut's `lprojection_backward_na1_carriers_kernel`. Threading
    only the main carrier (i.e. setting grad_alt=0) silently breaks
    numerical equivalence with MultiHeadLut and causes a structural ~+0.03
    bpb gap in downstream training.

    Saves only `weights`, `lookup_indices`, `lookup_alt_indices` (the
    [B, n_lookup_tables, n_outputs] gather is NOT saved — recomputed in
    backward).
    """

    @staticmethod
    def forward(ctx, weights, lookup_indices, lookup_alt_indices,
                lookup_indices_grad_c, lookup_alt_indices_grad_c,
                n_heads: int, tables_per_head: int):
        out = _embedding_bag_forward(weights, lookup_indices, n_heads, tables_per_head)
        # Save weights (parameter — no extra mem) and both lookup index
        # tensors (small int64). The big gather is never materialised.
        ctx.save_for_backward(weights, lookup_indices, lookup_alt_indices)
        ctx.n_heads = n_heads
        ctx.tables_per_head = tables_per_head
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [B, n_heads, n_outputs] in weights' dtype (or upstream's).
        weights, lookup_indices, lookup_alt_indices = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tables_per_head
        B, n_lookup_tables = lookup_indices.shape
        n_outputs = weights.shape[2]
        table_dim = weights.shape[1]

        if grad_out.dtype != weights.dtype:
            grad_out = grad_out.to(weights.dtype)

        # Native fused path: weights + carriers (main+alt) kernels modeled on
        # MHLut's lprojection_backward_na1_*. Returns grad_main AND grad_alt
        # so TinyAnchorPairsLookup's bwd can compute du = (grad_main-grad_alt)
        # * uncertainty_derivative for the STE update on x.grad.
        native = _get_tiny_mhlut_native() if _USE_TINY_MHLUT_NATIVE_BWD else None
        if native is not None and weights.is_cuda:
            grad_weights, grad_main, grad_alt = native.tiny_mhlut_backward_na1(
                grad_out.contiguous(),
                weights,
                lookup_indices.contiguous(),
                lookup_alt_indices.contiguous(),
                tph,
            )
            return grad_weights, None, None, grad_main, grad_alt, None, None

        # PyTorch fallback path. Mirror native: compute grad_main and grad_alt
        # by gathering at main and alt indices respectively.
        table_ix = torch.arange(n_lookup_tables, device=weights.device).view(1, -1).expand(B, -1)
        out_main = weights[table_ix, lookup_indices]      # [B, n_lookup_tables, n_outputs]
        out_alt  = weights[table_ix, lookup_alt_indices]
        grad_view = grad_out.unsqueeze(2)                 # [B, n_heads, 1, n_outputs]
        grad_main = (out_main.view(B, n_heads, tph, n_outputs) * grad_view).sum(-1) \
                    .view(B, n_lookup_tables).contiguous()
        grad_alt  = (out_alt.view(B, n_heads, tph, n_outputs)  * grad_view).sum(-1) \
                    .view(B, n_lookup_tables).contiguous()

        flat_lookup = lookup_indices.reshape(-1)
        table_offset = (
            torch.arange(n_lookup_tables, device=weights.device, dtype=lookup_indices.dtype) * table_dim
        ).unsqueeze(0).expand(B, -1).reshape(-1)
        fully_flat_idx = table_offset + flat_lookup
        grad_per_lookup = (
            grad_out.unsqueeze(2)
                    .expand(B, n_heads, tph, n_outputs)
                    .reshape(B * n_lookup_tables, n_outputs)
                    .contiguous()
        )
        grad_weights_flat = torch.zeros(
            n_lookup_tables * table_dim, n_outputs,
            dtype=weights.dtype, device=weights.device,
        )
        grad_weights_flat.index_add_(0, fully_flat_idx, grad_per_lookup)
        grad_weights = grad_weights_flat.view(n_lookup_tables, table_dim, n_outputs)

        return grad_weights, None, None, grad_main, grad_alt, None, None


def _per_table_gather_forward(weights: torch.Tensor, lookup_indices: torch.Tensor,
                              n_heads: int, tables_per_head: int) -> torch.Tensor:
    """Per-table gather (no reduce). Returns [B, n_heads, tph, n_outputs]."""
    B, n_lookup_tables = lookup_indices.shape
    n_outputs = weights.shape[2]
    table_ix = torch.arange(n_lookup_tables, device=weights.device).view(1, -1).expand(B, -1)
    out = weights[table_ix, lookup_indices]                         # [B, n_lookup_tables, n_outputs]
    return out.view(B, n_heads, tables_per_head, n_outputs)


def _balanced_coverage_indices(n_tables: int, n_per_row: int, n_slots: int,
                               generator: torch.Generator,
                               max_retries: int = 32) -> torch.Tensor:
    """Build a [n_tables, n_per_row] long tensor with two guarantees:

    1. Every row contains `n_per_row` distinct values in [0, n_slots).
    2. Each value j ∈ [0, n_slots) appears either
       ⌊n_tables·n_per_row/n_slots⌋ or ⌈n_tables·n_per_row/n_slots⌉ times
       across the whole tensor (balanced coverage).

    For n_per_row == 1 the construction is trivial (shuffle of a balanced
    bag). For n_per_row > 1 we use a greedy capacity-weighted sampler:
    each table is filled by sampling `n_per_row` distinct slots without
    replacement, weighted by their remaining capacity. Higher-capacity
    slots are preferred, which spreads load away from exhaustion. If the
    greedy gets stuck (a row would need more distinct slots than have
    capacity left), the table order is reshuffled and the run restarts.
    """
    T, S, N = n_tables, n_per_row, n_slots
    if S > N:
        raise ValueError(f"n_per_row={S} cannot exceed n_slots={N}")

    base = (T * S) // N
    extra = (T * S) - base * N
    target = torch.full((N,), base, dtype=torch.long)
    if extra > 0:
        bonus = torch.randperm(N, generator=generator)[:extra]
        target[bonus] += 1
    # invariant: target.sum() == T*S

    if S == 1:
        bag = torch.repeat_interleave(torch.arange(N, dtype=torch.long), target)
        perm = torch.randperm(T, generator=generator)
        return bag[perm].view(T, 1)

    for _ in range(max_retries):
        result = torch.empty(T, S, dtype=torch.long)
        remaining = target.clone()
        order = torch.randperm(T, generator=generator).tolist()
        ok = True
        for t in order:
            n_avail = int((remaining > 0).sum().item())
            if n_avail < S:
                ok = False
                break
            picks = torch.multinomial(
                remaining.float(), S, replacement=False, generator=generator,
            )
            result[t] = picks
            remaining[picks] -= 1
        if ok:
            return result

    # Fallback: when greedy can't pack distinct picks within balanced budget
    # (typically when S is a substantial fraction of N), revert to i.i.d.
    # randperm sampling per table. Each row stays distinct; per-slot counts
    # follow a multinomial close to the balanced target on average but no
    # longer hit the strict floor/ceil guarantee.
    import warnings
    warnings.warn(
        f"_balanced_coverage_indices: greedy failed after {max_retries} "
        f"retries for (n_tables={T}, n_per_row={S}, n_slots={N}); falling "
        f"back to i.i.d. randperm sampling. Per-slot counts will be "
        f"approximately balanced but not exact.",
        RuntimeWarning, stacklevel=2,
    )
    result = torch.empty(T, S, dtype=torch.long)
    for t in range(T):
        result[t] = torch.randperm(N, generator=generator)[:S]
    return result


class _TinyMHLutGather(torch.autograd.Function):
    """Per-table gather with carrier-grad threading. Returns
    [B, n_heads, tph, n_outputs] without reducing across tables.

    Mirrors the carrier-grad logic of `_TinyMHLutGatherReduce` so that
    `TinyAnchorPairsLookup.backward` receives the same (grad_main, grad_alt)
    needed to compute `du = (grad_main - grad_alt) * uncertainty_derivative`
    for the STE update on x.grad.
    """

    @staticmethod
    def forward(ctx, weights, lookup_indices, lookup_alt_indices,
                lookup_indices_grad_c, lookup_alt_indices_grad_c,
                n_heads: int, tables_per_head: int):
        out = _per_table_gather_forward(weights, lookup_indices, n_heads, tables_per_head)
        ctx.save_for_backward(weights, lookup_indices, lookup_alt_indices)
        ctx.n_heads = n_heads
        ctx.tables_per_head = tables_per_head
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # grad_out: [B, n_heads, tph, n_outputs]
        weights, lookup_indices, lookup_alt_indices = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tables_per_head
        B, n_lookup_tables = lookup_indices.shape
        n_outputs = weights.shape[2]
        table_dim = weights.shape[1]

        if grad_out.dtype != weights.dtype:
            grad_out = grad_out.to(weights.dtype)

        # Recompute gathers for STE carrier grads (mirrors
        # _TinyMHLutGatherReduce; native fast path is reduce-only).
        table_ix = torch.arange(n_lookup_tables, device=weights.device).view(1, -1).expand(B, -1)
        out_main = weights[table_ix, lookup_indices]      # [B, n_lookup_tables, n_outputs]
        out_alt  = weights[table_ix, lookup_alt_indices]
        grad_main = (out_main.view(B, n_heads, tph, n_outputs) * grad_out).sum(-1) \
                    .view(B, n_lookup_tables).contiguous()
        grad_alt  = (out_alt.view(B, n_heads, tph, n_outputs)  * grad_out).sum(-1) \
                    .view(B, n_lookup_tables).contiguous()

        # grad_weights: scatter grad_out onto (table, lookup_index) entries.
        flat_lookup = lookup_indices.reshape(-1)
        table_offset = (
            torch.arange(n_lookup_tables, device=weights.device, dtype=lookup_indices.dtype) * table_dim
        ).unsqueeze(0).expand(B, -1).reshape(-1)
        fully_flat_idx = table_offset + flat_lookup
        grad_per_lookup = grad_out.reshape(B * n_lookup_tables, n_outputs).contiguous()
        grad_weights_flat = torch.zeros(
            n_lookup_tables * table_dim, n_outputs,
            dtype=weights.dtype, device=weights.device,
        )
        grad_weights_flat.index_add_(0, fully_flat_idx, grad_per_lookup)
        grad_weights = grad_weights_flat.view(n_lookup_tables, table_dim, n_outputs)

        return grad_weights, None, None, grad_main, grad_alt, None, None


# =====================================================================
# Soft-mode pieces: TinyMHLut-fast forward + pure-PyTorch soft backward,
# both wrapped in @torch.compile so inductor fuses everything.
#
# Forward saves only small inputs (x, weights, anchor_*, bit_matrix, index,
# flat_offset, log_T_*). Backward recomputes p, ts, sel_soft inside one
# compiled function that produces all four gradients. No HBM materialisation
# of the [B, n_tables, K] tensors as saved activations.
#
# Beats torch.compile'd SoftMHLut(hard=True) on every shape we tested
# (~30% faster total, ~35% lower peak memory) with identical gradients.
# =====================================================================

@torch.compile
def _soft_lut_fwd_body(x, weights, anchor_a_long, anchor_b_long, powers,
                        bit_matrix, T_soft, n_heads, tph, table_dim,
                        bf16_argmax, noise_eps):
    """Compiled forward.

    bf16_argmax=False — fastest. fp32 sign(x_a - x_b) bit-pack for index.
    bf16_argmax=True  — match SoftMHLut(use_bf16=True): compute bf16
                        ts = einsum(p, bm) under bf16 autocast, argmax it.

    noise_eps > 0: when in fp32 sign-bit-pack path, RANDOMLY flip the bit at
                   positions where |d[i]| < noise_eps. This injects structured
                   noise on low-confidence comparisons — testing the
                   hypothesis that bf16's argmax regularization is doing this
                   implicitly. Ignored when bf16_argmax=True.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    if bf16_argmax:
        p = d / (T_soft + d.abs())
        ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
        index = ts.argmax(dim=-1)
    else:
        bits = (d > 0).to(torch.int64)
        if noise_eps > 0.0:
            # At low-confidence bits (|d| < noise_eps), flip with 50% probability.
            rand_bits = torch.empty_like(d).bernoulli_(0.5).to(torch.int64)
            low_conf = (d.abs() < noise_eps)
            bits = torch.where(low_conf, rand_bits, bits)
        index = (bits * powers.view(1, 1, -1)).sum(dim=-1)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tph
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs), index


@torch.compile
def _soft_lut_bwd_body(grad_out, x, weights, anchor_a_long, anchor_b_long,
                        bit_matrix, index, T_soft, T_sel, n_heads, tph):
    """Compiled backward — Gumbel-STE consistent.

    Reconstructs `p` so that `argmax(sel_soft) ≡ saved index` (including any
    noise flips applied in forward). Extract bits actually used in forward
    from `index`, build `p_signs = ±1` matching those bits, then
    `p = p_signs * |d| / (T_soft + |d|)`. In the no-noise case this is
    bit-identical to `p = d/(T+|d|)`; under noise it makes the soft pipeline's
    argmax match the chosen index, so softmax-backward gradients are
    self-consistent with the picked row.
    """
    B, _, n_outputs = grad_out.shape
    n_tables, NAP = anchor_a_long.shape
    K = bit_matrix.shape[1]
    input_dim = x.shape[1]
    w_dtype = weights.dtype

    d        = x[:, anchor_a_long] - x[:, anchor_b_long]
    denom    = T_soft + d.abs()

    # Bits actually used in forward (MSB-first packing): bit at position i is
    # bit (NAP-1-i) of the integer index.
    shifts   = torch.arange(NAP - 1, -1, -1, device=index.device, dtype=index.dtype)
    bits     = ((index.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 1).to(d.dtype)
    p_signs  = bits * 2.0 - 1.0      # ±1, matches the bits used in forward
    p        = p_signs * d.abs() / denom

    ts       = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    z        = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)

    # Broadcast grad_out → [B, n_tables, n_outputs] for per-table scatter/GEMM.
    grad_pt    = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
    d_sel_soft = torch.einsum("bto,tko->btk", grad_pt.to(w_dtype), weights)

    # Softmax backward expressed as idiomatic PyTorch (compile fuses).
    sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
    d_z      = sel_soft * (d_sel_soft - sum_term)
    d_ts     = d_z / T_sel
    grad_log_T_sel = -(d_z * z).sum()

    # dL/dp via cuBLAS GEMM; dL/dd via rational-sign Jacobian.
    # p = p_signs * |d| / denom, so dp/d|d| = p_signs * T_soft/denom^2 and
    # d|d|/dd = sign(d). Hence dp/dd = p_signs * sign(d) * T_soft/denom^2.
    d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))
    d_d = d_p * p_signs * d.sign() * (T_soft / (denom * denom))
    # grad_log_T_soft = T_soft * dL/dT_soft = -sum(d_d * d) (same algebra as
    # the simple p=d/denom case; the p_signs * sign(d) factors cancel out).
    grad_log_T_soft = -(d_d * d).sum()

    # dL/dweights via scatter at saved (table, index).
    flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * K
    flat_idx    = (index + flat_offset[None, :]).reshape(-1)
    grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=w_dtype, device=weights.device)
    grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(w_dtype))
    grad_weights = grad_w_flat.view(n_tables, K, n_outputs)

    # dL/dx via scatter-add at anchor positions.
    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a_flat = anchor_a_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b_flat = anchor_b_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat     = d_d.reshape(B, -1).to(x.dtype)
    grad_x.scatter_add_(1, idx_a_flat,  d_flat)
    grad_x.scatter_add_(1, idx_b_flat, -d_flat)

    return grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel


class _TinyMHLutSoft(torch.autograd.Function):
    """Forward: TinyMHLut-fast (sign-pack + embedding_bag).
    Backward: pure-PyTorch soft path, gradients matching SoftMHLut(hard=True).
    """

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix, powers,
                n_heads, tph, table_dim, use_bf16, argmax_noise_eps,
                bf16_argmax):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            out, index = _soft_lut_fwd_body(
                x, weights, anchor_a_long, anchor_b_long, powers,
                bit_matrix, T_soft, n_heads, tph, table_dim, bool(bf16_argmax),
                float(argmax_noise_eps),
            )
        ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, index, log_T_soft, log_T_sel)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, bit_matrix, index,
         log_T_soft, log_T_sel) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            grad_x, grad_w, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                grad_out, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                index, T_soft, T_sel, ctx.n_heads, ctx.tph,
            )
        return (grad_x, grad_w, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None, None, None)


class TinyMultiHeadLut(nn.Module):
    """Multi-head LUT with TinyAnchorPairsLookup + bf16 (default) weights.

    Args:
        input_dim: Dimension of input tensor (must be <= 32767).
        n_heads: Number of heads.
        n_outputs: Number of output dimensions per head.
        n_anchor_pairs: Number of anchor pairs per table (1..15).
        tables_per_head: Number of lookup tables per head.
        weight_dtype: Storage dtype for weights (default torch.bfloat16).
        anchor_sampling_policy: CANONICAL_FULL_COVERAGE (default) or
            CANONICAL_DISTINCT.
        partition_sets: Optional list-of-lists restricting CANONICAL_DISTINCT
            sampling to within-partition pairs.
        random_seed: Seed for anchor sampling and weight init.
        initial_weights_noise: Uniform [-σ, +σ] init for weights (default 0.001).
        device: torch.device or None.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        *,
        weight_dtype: torch.dtype = torch.bfloat16,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        partition_sets: Optional[list] = None,
        partition_pair_weights: Optional[list] = None,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        device: Optional[torch.device] = None,
        sparse_scatter_n_outputs: Optional[int] = None,
        sparse_scatter_seed: Optional[int] = None,
        sparse_scatter_balanced: bool = True,
        max_anchor_distance: Optional[int] = None,
        local_window_starts: str = "linspace",
        aligned_local_scatter: bool = False,
        # Soft-mode (backward only): rational-soft-sign + softmax gradient
        # path that gives the same gradients as SoftMHLut(hard=True). Forward
        # is unchanged (sign-pack + embedding_bag, native fast path).
        backward_mode: str = "ste",
        soft_score_temp: float = 0.5,
        select_temp: float = 0.5,
        learnable_temps: bool = False,
        use_bf16: bool = True,
        argmax_noise_eps: float = 0.0,
        bf16_argmax: bool = False,
    ):
        super().__init__()
        if not (1 <= n_anchor_pairs <= 15):
            raise ValueError(
                f"TinyMultiHeadLut requires 1 <= n_anchor_pairs <= 15 "
                f"(int16 lookup-index range), got {n_anchor_pairs}"
            )
        if input_dim > 32767:
            raise ValueError(
                f"TinyMultiHeadLut requires input_dim <= 32767 (int16 anchor "
                f"index range), got {input_dim}"
            )

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.tables_per_head = tables_per_head
        self.table_dim = 1 << n_anchor_pairs  # 2 ** n_anchor_pairs
        self.weight_dtype = weight_dtype

        # Sparse-scatter mode: each table's `n_outputs` values are scattered
        # into a fixed random subset of `sparse_scatter_n_outputs` slots.
        # Decouples per-table weights from the final output dim.
        self.sparse_scatter_n_outputs = sparse_scatter_n_outputs
        self.sparse_scatter_balanced = sparse_scatter_balanced
        self.aligned_local_scatter = aligned_local_scatter
        if sparse_scatter_n_outputs is not None:
            if sparse_scatter_n_outputs < n_outputs:
                raise ValueError(
                    f"sparse_scatter_n_outputs ({sparse_scatter_n_outputs}) "
                    f"must be >= n_outputs ({n_outputs}); each table contributes "
                    f"n_outputs values to a {n_outputs}-subset of "
                    f"sparse_scatter_n_outputs"
                )
            gen = torch.Generator()
            if sparse_scatter_seed is not None:
                gen.manual_seed(int(sparse_scatter_seed))
            elif random_seed is not None:
                gen.manual_seed(int(random_seed) + 1234567)
            if aligned_local_scatter:
                # Each (head, table)'s scatter destinations are sampled from
                # the SAME contiguous window [s_t, s_t + max_anchor_distance]
                # used by the anchor lookup. Requires input_dim ==
                # sparse_scatter_n_outputs (so input and output windows
                # share index space).
                if max_anchor_distance is None:
                    raise ValueError(
                        "aligned_local_scatter requires max_anchor_distance"
                    )
                if input_dim != sparse_scatter_n_outputs:
                    raise ValueError(
                        f"aligned_local_scatter requires input_dim "
                        f"({input_dim}) == sparse_scatter_n_outputs "
                        f"({sparse_scatter_n_outputs})"
                    )
                if local_window_starts != "linspace":
                    raise ValueError(
                        "aligned_local_scatter currently only supports "
                        "local_window_starts='linspace' (deterministic starts)"
                    )
                if n_outputs > max_anchor_distance + 1:
                    raise ValueError(
                        f"aligned_local_scatter: n_outputs ({n_outputs}) > "
                        f"max_anchor_distance+1 ({max_anchor_distance + 1}) — "
                        f"can't sample n_outputs distinct slots from a "
                        f"width-{max_anchor_distance + 1} window"
                    )
                from spiky.lutorch.lut_helpers import local_window_starts as _starts_fn
                K = max_anchor_distance
                starts = _starts_fn(
                    n_tables=n_heads * tables_per_head,
                    input_dim=input_dim, max_distance=K, n_heads=n_heads,
                    starts_mode="linspace",
                ).view(n_heads, tables_per_head)
                scatter_idx = torch.empty(
                    n_heads, tables_per_head, n_outputs, dtype=torch.long,
                )
                for h in range(n_heads):
                    for t in range(tables_per_head):
                        perm = torch.randperm(K + 1, generator=gen)[:n_outputs]
                        scatter_idx[h, t] = perm + starts[h, t]
            elif sparse_scatter_balanced:
                # Balanced coverage: every output slot of dim
                # sparse_scatter_n_outputs is the destination of either
                # ⌊tables_per_head*n_outputs/sparse_scatter_n_outputs⌋ or
                # ⌈...⌉ table-output positions per head.
                scatter_idx = torch.stack([
                    _balanced_coverage_indices(
                        n_tables=tables_per_head, n_per_row=n_outputs,
                        n_slots=sparse_scatter_n_outputs, generator=gen,
                    )
                    for _ in range(n_heads)
                ], dim=0)                                              # [H, tph, n_outputs]
            else:
                # i.i.d. uniform sampling without replacement per (head, table).
                scatter_idx = torch.empty(n_heads, tables_per_head, n_outputs, dtype=torch.long)
                for h in range(n_heads):
                    for t in range(tables_per_head):
                        perm = torch.randperm(sparse_scatter_n_outputs, generator=gen)
                        scatter_idx[h, t] = perm[:n_outputs]
            if device is not None:
                scatter_idx = scatter_idx.to(device)
            self.register_buffer('scatter_indices', scatter_idx)

        n_lookup_tables = n_heads * tables_per_head
        self.n_lookup_tables = n_lookup_tables

        # Anchor lookup (int16 path).
        self.lookup = TinyAnchorPairsLookup(
            input_dim=input_dim,
            n_tables=n_lookup_tables,
            n_anchor_pairs=n_anchor_pairs,
            n_heads=n_heads,
            random_seed=random_seed,
            device=device,
            partition_sets=partition_sets,
            partition_pair_weights=partition_pair_weights,
            anchor_sampling_policy=anchor_sampling_policy,
            max_anchor_distance=max_anchor_distance,
            local_window_starts=local_window_starts,
        )

        # Weights: [n_lookup_tables, table_dim, n_outputs] in weight_dtype.
        # Init: uniform[-σ, +σ] cast to weight_dtype.
        dev = device or torch.device("cpu")
        rng_kwargs: dict = {"device": dev}
        if random_seed is not None:
            rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed + 1)
        weights_init = (
            (torch.rand(n_lookup_tables, self.table_dim, n_outputs, **rng_kwargs) - 0.5)
            * (2.0 * initial_weights_noise)
        ).to(weight_dtype)
        self.weights = nn.Parameter(weights_init)

        # ----- soft-backward mode setup -----
        if backward_mode not in ("ste", "soft"):
            raise ValueError(f"backward_mode must be 'ste' or 'soft', got {backward_mode!r}")
        self.backward_mode = backward_mode
        self.use_bf16 = bool(use_bf16)
        self.argmax_noise_eps = float(argmax_noise_eps)
        self.bf16_argmax = bool(bf16_argmax)
        if backward_mode == "soft":
            if sparse_scatter_n_outputs is not None:
                raise NotImplementedError(
                    "soft backward_mode does not yet support sparse_scatter_n_outputs"
                )
            if partition_sets is not None:
                raise NotImplementedError(
                    "soft backward_mode does not yet support partition_sets"
                )
            # bit_matrix and powers (MSB-first, matching `_soft_lut_bwd_body`).
            self.register_buffer(
                "soft_bit_matrix",
                _soft_bit_matrix_msb(n_anchor_pairs, dev, dtype=torch.float32),
            )
            self.register_buffer("soft_powers", _msb_powers(n_anchor_pairs, dev))
            # Anchor pairs as int64 — cast once, reused by forward & backward.
            self.register_buffer(
                "soft_anchor_a_long",
                self.lookup.anchor_pairs_a.long().contiguous(),
            )
            self.register_buffer(
                "soft_anchor_b_long",
                self.lookup.anchor_pairs_b.long().contiguous(),
            )
            self.learnable_temps = bool(learnable_temps)
            if self.learnable_temps:
                # Log-parametrize so unconstrained optimization keeps T positive.
                self.log_soft_score_temp = nn.Parameter(
                    torch.tensor(math.log(float(soft_score_temp)), dtype=torch.float32, device=dev)
                )
                self.log_select_temp = nn.Parameter(
                    torch.tensor(math.log(float(select_temp)), dtype=torch.float32, device=dev)
                )
            else:
                self.register_buffer(
                    "log_soft_score_temp",
                    torch.tensor(math.log(float(soft_score_temp)), dtype=torch.float32, device=dev),
                )
                self.register_buffer(
                    "log_select_temp",
                    torch.tensor(math.log(float(select_temp)), dtype=torch.float32, device=dev),
                )

    def _soft_forward(self, x: torch.Tensor) -> torch.Tensor:
        """soft backward_mode forward path. Forward output is identical to
        SoftMHLut(hard=True) on the same weights and anchor pairs (sign-pack
        argmax = soft argmax). Backward gives soft gradients to x and the
        temperatures, sparse one-hot scatter to weights."""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        return _TinyMHLutSoft.apply(
            x, self.weights, self.log_soft_score_temp, self.log_select_temp,
            self.soft_anchor_a_long, self.soft_anchor_b_long,
            self.soft_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
            self.argmax_noise_eps, self.bf16_argmax,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: float [B, input_dim]
        Returns:
            [B, n_heads, n_outputs] in weight_dtype, or
            [B, n_heads, sparse_scatter_n_outputs] if sparse_scatter is active.
        """
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )

        # Soft backward_mode: bypass TAPL entirely; use the pure-PyTorch
        # `_TinyMHLutSoft` Function (TinyMHLut-fast forward + soft backward).
        if self.backward_mode == "soft":
            return self._soft_forward(x)

        # TinyAnchorPairsLookup returns BOTH the chosen ("main") and runner-up
        # ("alt") int16 lookup indices, plus their zero-valued carriers
        # (lookup_indices_grad_c, lookup_alt_indices_grad_c) whose gradients
        # back-flow through the anchor STE kernel into x.grad. We must thread
        # both carriers through our autograd Function — dropping the alt one
        # silently breaks numerical equivalence with MultiHeadLut.
        (lookup_indices, lookup_alt_indices, _alt_deltas,
         lookup_indices_grad_c, lookup_alt_indices_grad_c) = self.lookup(x)
        # TAPL returns lookup_alt_indices with a trailing n_alt=1 dim
        # (multi-alt API parity); squeeze for our na=1 path.
        lookup_indices = lookup_indices.to(torch.int64)
        lookup_alt_indices = lookup_alt_indices.squeeze(-1).to(torch.int64)

        # ===== argmax_noise_eps in STE mode =====
        # Match the regularization mechanism we found in soft-mode: at low-
        # confidence bit positions (|d_i| < eps), randomly flip that bit in
        # lookup_indices with 50% probability. TAPL uses LSB-first bit pack
        # (bit i ↔ powers[i] = 1 << i), so flip = XOR with `(1 << i)`.
        if self.argmax_noise_eps > 0.0 and lookup_indices_grad_c is not None:
            idx_a = self.lookup.anchor_pairs_a.long()
            idx_b = self.lookup.anchor_pairs_b.long()
            d = x[:, idx_a] - x[:, idx_b]                          # [B, T, NAP]
            low_conf = (d.abs() < self.argmax_noise_eps)
            rand_bits = torch.empty_like(d).bernoulli_(0.5).bool()
            flip_mask_per_bit = (low_conf & rand_bits).to(torch.int64)
            powers_lsb = (1 << torch.arange(d.shape[-1], device=x.device, dtype=torch.int64))
            flip_mask = (flip_mask_per_bit * powers_lsb).sum(dim=-1)  # [B, T]
            lookup_indices = lookup_indices ^ flip_mask

        sparse = self.sparse_scatter_n_outputs is not None

        # Eval / no_grad path: TAPL returns None carriers; nothing to
        # backprop, so skip the autograd Function.
        if lookup_indices_grad_c is None:
            if sparse:
                per_table = _per_table_gather_forward(
                    self.weights, lookup_indices, self.n_heads, self.tables_per_head,
                )                                                  # [B, H, tph, n_outputs]
                return self._scatter(per_table)                    # [B, H, sparse_scatter_n_outputs]
            return _embedding_bag_forward(
                self.weights, lookup_indices, self.n_heads, self.tables_per_head,
            )

        # Training path: thread BOTH carriers through the autograd Function
        # so its backward returns grad_main AND grad_alt. Dropping the alt
        # carrier silently breaks numerical equivalence with MultiHeadLut.
        if sparse:
            per_table = _TinyMHLutGather.apply(
                self.weights, lookup_indices, lookup_alt_indices,
                lookup_indices_grad_c, lookup_alt_indices_grad_c.squeeze(-1),
                self.n_heads, self.tables_per_head,
            )                                                      # [B, H, tph, n_outputs]
            return self._scatter(per_table)                        # [B, H, sparse_scatter_n_outputs]
        return _TinyMHLutGatherReduce.apply(
            self.weights, lookup_indices, lookup_alt_indices,
            lookup_indices_grad_c, lookup_alt_indices_grad_c.squeeze(-1),
            self.n_heads, self.tables_per_head,
        )

    def _scatter(self, per_table: torch.Tensor) -> torch.Tensor:
        """Scatter-add per-table outputs into the wider sparse_scatter_n_outputs
        dense vector via the fixed random per-(head, table) index subsets.
        per_table: [B, n_heads, tables_per_head, n_outputs]
        returns:   [B, n_heads, sparse_scatter_n_outputs]
        """
        B, H, T, S = per_table.shape
        out = per_table.new_zeros(B, H, self.sparse_scatter_n_outputs)
        idx = self.scatter_indices.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, H, T * S)
        out.scatter_add_(2, idx, per_table.reshape(B, H, T * S))
        return out
