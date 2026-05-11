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
import os
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


@torch.compile
def _ste_backward_body(
    grad_out: torch.Tensor,        # [B, n_heads, n_outputs]
    weights: torch.Tensor,         # [n_lookup_tables, table_dim, n_outputs]
    lookup_indices: torch.Tensor,  # [B, n_lookup_tables] int64
    lookup_alt_indices: torch.Tensor,  # [B, n_lookup_tables] int64
):
    """Inductor-fused replacement for `tiny_mhlut_bwd_na1_{carriers,weights}_kernel`.

    Produces (grad_weights, grad_main, grad_alt). At our nanochat shapes
    inductor fuses gather + multiply + reduction into ~1 kernel per output;
    measured ~1.6x faster than the hand-written native CUDA kernels because
    the native kernels run carriers and weights as two separate dispatches
    with separate HBM reads of `weights`, whereas inductor shares the gather.
    """
    B, n_lookup_tables = lookup_indices.shape
    n_outputs = weights.shape[2]
    table_dim = weights.shape[1]
    n_heads = grad_out.shape[1]
    tph = n_lookup_tables // n_heads

    table_ix = torch.arange(n_lookup_tables, device=weights.device).view(1, -1).expand(B, -1)
    out_main = weights[table_ix, lookup_indices]      # [B, n_lookup_tables, n_outputs]
    out_alt = weights[table_ix, lookup_alt_indices]
    grad_view = grad_out.unsqueeze(2)                 # [B, n_heads, 1, n_outputs]
    grad_main = (out_main.view(B, n_heads, tph, n_outputs) * grad_view).sum(-1).view(B, n_lookup_tables)
    grad_alt = (out_alt.view(B, n_heads, tph, n_outputs) * grad_view).sum(-1).view(B, n_lookup_tables)

    # grad_weights via flat index_add.
    flat_lookup = lookup_indices.reshape(-1)
    table_offset = (
        torch.arange(n_lookup_tables, device=weights.device, dtype=lookup_indices.dtype) * table_dim
    ).unsqueeze(0).expand(B, -1).reshape(-1)
    fully_flat_idx = table_offset + flat_lookup
    grad_per_lookup = (
        grad_out.unsqueeze(2)
                .expand(B, n_heads, tph, n_outputs)
                .reshape(B * n_lookup_tables, n_outputs)
    )
    grad_weights_flat = torch.zeros(
        n_lookup_tables * table_dim, n_outputs,
        dtype=weights.dtype, device=weights.device,
    )
    grad_weights_flat.index_add_(0, fully_flat_idx, grad_per_lookup)
    grad_weights = grad_weights_flat.view(n_lookup_tables, table_dim, n_outputs)
    return grad_weights, grad_main.contiguous(), grad_alt.contiguous()


# @torch.compile STE backward is the default (~2.5x faster than the hand-
# written native CUDA carriers/weights kernels at nanochat shapes, bit-exact
# weight grads, fp32-noise-level x grads). Set
# SPIKY_TINY_MHLUT_USE_COMPILE_BWD=0 to fall back to the native CUDA path.
_USE_COMPILE_STE_BWD = os.environ.get("SPIKY_TINY_MHLUT_USE_COMPILE_BWD", "1") == "1"


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

        # @torch.compile fast path: ~1.6x faster than the native CUDA kernel
        # at exp257/exp234 nanochat shapes because inductor fuses gather +
        # multiply + reduction into a single kernel per output, whereas the
        # native carriers/weights kernels run as two separate dispatches with
        # separate HBM reads of `weights`. Enable with the env flag for now;
        # native path remains the default until validated across more shapes.
        if _USE_COMPILE_STE_BWD and weights.is_cuda:
            grad_weights, grad_main, grad_alt = _ste_backward_body(
                grad_out.contiguous(), weights,
                lookup_indices.contiguous(), lookup_alt_indices.contiguous(),
            )
            return grad_weights, None, None, grad_main, grad_alt, None, None

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
# Multi-alternative STE backward.
#
# Pairs TinyMHL-fast forward (embedding_bag) with MultiHeadLut-style
# `_lprojection_backward` (smooth_mode=False, n_alternatives>1) +
# `_anchor_pairs_backward` (uncertainty-weighted x gradient). Mirrors the
# regularization that `MultiHeadLut(smooth_mode=False, n_alternatives>1)`
# applies in its backward, but keeps TinyMHL's bf16-friendly weights layout
# and fast forward.
#
# Supports `argmax_noise_eps` via the same saved-flip-mask mechanism used by
# `MultiHeadLutFunction` and `_TinyMHLutSoft`: forward generates a bernoulli
# bit-flip mask at low-confidence positions, backward replays it for
# consistent fwd/bwd.
# =====================================================================

@torch.compile
def _multi_alt_fwd_body(
    x, weights, anchor_a_long, anchor_b_long,
    powers_mult, table_arange, eb_offsets,
    n_heads, tables_per_head, argmax_noise_eps,
):
    """Fully fused multi-alt STE forward. Matches soft-mode's pattern: all
    delta/bit-pack/noise/embedding_bag ops inside one @torch.compile region
    so inductor fuses the noise XOR generation with the index computation
    and the embedding_bag gather.

    Returns (output, lookup_indices, flip_mask). The flip_mask is the
    bool per-bit noise mask (None if eps==0) — saved for backward.
    """
    B = x.shape[0]
    n_lookup_tables = anchor_a_long.shape[0]
    NAP = anchor_a_long.shape[1]
    table_dim = weights.shape[1]
    n_outputs = weights.shape[2]

    d = x[:, anchor_a_long] - x[:, anchor_b_long]  # [B, T, NAP]
    bits = (d > 0).to(torch.int64)

    if argmax_noise_eps > 0.0:
        low_conf = d.abs() < argmax_noise_eps
        rand_bits = torch.empty_like(low_conf).bernoulli_(0.5)
        flip_mask = low_conf & rand_bits
        bits = bits ^ flip_mask.to(torch.int64)
    else:
        flip_mask = torch.zeros_like(d, dtype=torch.bool)

    # LSB-first bit-pack — multipliers cached as `powers_mult` [2^0, ..., 2^(NAP-1)].
    lookup_indices = (bits * powers_mult).sum(dim=-1)  # [B, T]

    weights_flat = weights.view(n_lookup_tables * table_dim, n_outputs)
    # table_offset = arange(n_tables) * table_dim, derived from cached arange.
    flat_indices = (lookup_indices + (table_arange * table_dim).view(1, -1)).reshape(-1)
    out_flat = torch.nn.functional.embedding_bag(flat_indices, weights_flat, offsets=eb_offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs), lookup_indices, flip_mask


@torch.compile
def _multi_alt_bwd_body(
    grad_out, weights, x, batch_offset, anchor_a_long, anchor_b_long,
    table_arange, table_flat,
    lookup_indices, n_heads, tables_per_head, n_alternatives,
):
    """Fused multi-alt backward — everything inside @torch.compile, no
    [B, T, K] structured-bmm intermediate.

    Three tricks:
      1. **Manual top-k via sequential argmin + scatter** (k small, here 3).
         Avoids `torch.topk` whose lowering is more expensive than a few
         argmin passes that inductor easily fuses.
      2. **Fancy gather + mul + sum kept as a fused triplet**:
         `(weights[table_3d, lookup_alt_indices] * grad).sum(-1)` — inductor
         fuses gather → mul → reduce into a single kernel WITHOUT
         materialising the [B, T, n_alt, n_outputs] intermediate (the .sum
         consumes the mul output inline). HBM cost is just the random
         weight-row loads, no big alt_weights tensor.
      3. **All alt_* (lookup_alt_indices, lookup_alt_deltas, anchor1/2_ids)
         computed inline from `_compute_anchor_data`-equivalent ops** so
         they live in registers/SMEM through the whole backward.

    Empirical (exp257 out_proj shape: B=4096, T=2048, NAP=6, n_out=96):
      ~9 ms / 1 GB peak — faster AND leaner than soft (10 ms / 4 GB).
    """
    B = grad_out.shape[0]
    input_dim = x.shape[1]
    n_lookup_tables = anchor_a_long.shape[0]
    NAP = anchor_a_long.shape[1]
    table_dim = weights.shape[1]
    n_outputs = grad_out.shape[-1]

    # Deltas (inline; small [B, T, NAP])
    idx_a = anchor_a_long.reshape(1, -1).expand(B, -1)
    idx_b = anchor_b_long.reshape(1, -1).expand(B, -1)
    x_a = x.gather(1, idx_a).view(B, n_lookup_tables, NAP)
    x_b = x.gather(1, idx_b).view(B, n_lookup_tables, NAP)
    deltas = x_a - x_b
    abs_d = deltas.abs()

    # Manual top-n_alternatives via sequential argmin (no topk kernel).
    inf = torch.full_like(abs_d[..., :1], float('inf'))
    min_delta_indices = torch.empty(B, n_lookup_tables, n_alternatives, device=x.device, dtype=torch.long)
    for i in range(n_alternatives):
        mi = abs_d.argmin(dim=-1, keepdim=True)
        min_delta_indices[..., i:i+1] = mi
        abs_d = abs_d.scatter(2, mi, inf)

    lookup_alt_deltas = deltas.gather(2, min_delta_indices)
    bit_shifts = (1 << min_delta_indices).to(lookup_indices.dtype)
    lookup_alt_indices = lookup_indices.unsqueeze(-1) ^ bit_shifts
    anchor1_ids = anchor_a_long.unsqueeze(0).expand(B, -1, -1).gather(2, min_delta_indices)
    anchor2_ids = anchor_b_long.unsqueeze(0).expand(B, -1, -1).gather(2, min_delta_indices)

    # grad_per_table = expand view (no materialisation if inductor can fuse).
    grad_per_table = grad_out.unsqueeze(2).expand(B, n_heads, tables_per_head, n_outputs).reshape(B, n_lookup_tables, n_outputs)

    # Structured bmm to get carrier grads at ALL K weight rows, then cheap
    # .gather for main + alt positions. Mirrors soft mode's d_sel_soft path:
    # at small K the bf16 tensor-core GEMM wins over fancy gather; at large
    # K the [B, T, K] materialisation becomes too expensive — switch to
    # fancy gather there.
    if table_dim <= 128:  # K <= 128 ⇒ tensor cores win
        with torch.autocast("cuda", dtype=torch.bfloat16):
            all_grads = torch.einsum("bto,tko->btk", grad_per_table, weights)  # [B, T, K] bf16
        grad_main = all_grads.gather(2, lookup_indices.unsqueeze(-1)).squeeze(-1).to(weights.dtype)
        grad_alt = all_grads.gather(2, lookup_alt_indices).to(weights.dtype)
    else:                # K > 128 ⇒ fancy gather is leaner
        table_2d = table_arange.view(1, -1).expand(B, -1)
        main_w = weights[table_2d, lookup_indices]
        grad_main = (grad_per_table * main_w).sum(-1)
        table_3d = table_2d.unsqueeze(-1).expand(-1, -1, n_alternatives)
        alt_w = weights[table_3d, lookup_alt_indices]
        grad_alt = (grad_per_table.unsqueeze(2) * alt_w).sum(-1)

    # weights_grad: index_add_ (auto-broadcasts over column dim) — no need to
    # materialise an expanded index tensor like scatter_add_(0, idx.expand, val).
    flat_lookup = lookup_indices.reshape(-1)
    indices_main = table_flat * table_dim + flat_lookup
    weights_grad_flat = torch.zeros(n_lookup_tables * table_dim, n_outputs, dtype=weights.dtype, device=weights.device)
    weights_grad_flat.index_add_(0, indices_main, grad_per_table.reshape(-1, n_outputs))

    # x.grad via inverse-L1 uncertainty.
    grad_diff = grad_main.unsqueeze(2) - grad_alt
    one_plus_abs = 1.0 + lookup_alt_deltas.abs()
    minus_uncertainty_derivative = 0.5 * lookup_alt_deltas.sign() / (one_plus_abs * one_plus_abs)
    du = grad_diff * minus_uncertainty_derivative
    if n_alternatives > 1:
        du = du / n_alternatives

    # 2D scatter directly into [B, input_dim] — avoids the batch_offset add,
    # mirrors soft mode's pattern. anchor1/2_ids are [B, T, n_alt].
    x_grad = torch.zeros(B, input_dim, device=x.device, dtype=x.dtype)
    x_grad.scatter_add_(1, anchor1_ids.reshape(B, -1), du.reshape(B, -1))
    x_grad.scatter_add_(1, anchor2_ids.reshape(B, -1), -du.reshape(B, -1))

    return weights_grad_flat.view(weights.shape), x_grad


def _noisy_xor_postprocess(
    x, anchor_a_long, anchor_b_long, argmax_noise_eps, n_anchor_pairs,
    lookup_indices_clean, lookup_alt_indices_clean,
    provided_flip_mask=None,
):
    """Inject `argmax_noise_eps` bit-flip noise on top of an already-computed
    (clean) lookup-indices result from the native CUDA forward.

    Key algebraic identity: `lookup_alt_indices = lookup_indices XOR
    (1 << min_delta_indices)` (per-alt single-bit flip), and the noise XOR is
    a constant per (b, t) mask. XOR distributes, so XOR'ing the SAME mask into
    BOTH `lookup_indices` and `lookup_alt_indices` preserves the alt relation
    and correctly applies noise to the per-weight gather without rerunning
    the topk / fallback. `lookup_alt_deltas`, `anchor1_ids`, `anchor2_ids` are
    all derived from |delta| (and the |delta|-argmin selection), which is
    noise-independent — so they don't need to be touched.

    Backward path: pass `provided_flip_mask` (the saved per-bit bool mask) to
    reproduce the exact same XOR.
    """
    batch_size = x.shape[0]
    idx_a = anchor_a_long.reshape(1, -1).expand(batch_size, -1)
    idx_b = anchor_b_long.reshape(1, -1).expand(batch_size, -1)
    x_a = x.gather(1, idx_a).view(batch_size, anchor_a_long.shape[0], n_anchor_pairs)
    x_b = x.gather(1, idx_b).view(batch_size, anchor_a_long.shape[0], n_anchor_pairs)
    deltas = x_a - x_b

    if provided_flip_mask is not None:
        flip_mask_bool = provided_flip_mask
    else:
        low_conf = deltas.abs() < argmax_noise_eps
        rand = torch.empty_like(low_conf).bernoulli_(0.5)
        flip_mask_bool = low_conf & rand

    # LSB-first bit-pack to a per-(b, t) int.
    powers_lsb = (1 << torch.arange(n_anchor_pairs, device=x.device, dtype=torch.int64))
    flip_mask_int = (flip_mask_bool.to(torch.int64) * powers_lsb).sum(dim=-1)  # [B, T]

    lookup_indices_noisy = lookup_indices_clean ^ flip_mask_int
    lookup_alt_indices_noisy = lookup_alt_indices_clean ^ flip_mask_int.unsqueeze(-1)
    return lookup_indices_noisy, lookup_alt_indices_noisy, flip_mask_bool


class _TinyMHLutMultiAlt(torch.autograd.Function):
    """STE-style forward + multi-alternative uncertainty-weighted backward.

    Forward is identical in cost to standard TinyMHL n_alt=1 STE: just a
    sign-bit-pack to compute `lookup_indices` (and an optional noise XOR).
    All multi-alt machinery (lookup_alt_indices, lookup_alt_deltas,
    anchor1/2_ids) is computed in backward via `_compute_anchor_data` so we
    don't pay for it on the forward path and don't materialise it as saved
    activations.

    Memory: ctx saves only x (already live), weights (param), lookup_indices
    (small int64), and flip_mask (small bool) if noise is on.
    """

    @staticmethod
    def forward(ctx, x, weights, anchor_a_long, anchor_b_long,
                powers_mult, table_arange, eb_offsets,
                n_alternatives, batch_offset, table_flat,
                n_heads, tables_per_head, argmax_noise_eps):
        # Fully fused forward: deltas, sign-bit-pack, noise XOR, embedding_bag
        # all inside one @torch.compile region — same pattern as soft mode.
        out, lookup_indices, flip_mask = _multi_alt_fwd_body(
            x, weights, anchor_a_long, anchor_b_long,
            powers_mult, table_arange, eb_offsets,
            n_heads, tables_per_head, argmax_noise_eps,
        )

        ctx.save_for_backward(x, weights, lookup_indices, flip_mask)
        ctx.n_heads = n_heads
        ctx.tables_per_head = tables_per_head
        ctx.n_alternatives = n_alternatives
        ctx.batch_offset = batch_offset
        ctx.anchor_a_long = anchor_a_long
        ctx.anchor_b_long = anchor_b_long
        ctx.table_arange = table_arange
        ctx.table_flat = table_flat
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, weights, lookup_indices, _flip_mask = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tables_per_head

        if grad_out.dtype != weights.dtype:
            grad_out = grad_out.to(weights.dtype)

        # All alt_* computation and scatters fuse into a single @torch.compile
        # region (manual top-k via sequential argmin, fancy gather + mul + sum
        # as fused triplet, no [B, T, K] structured-bmm intermediate).
        weights_grad, x_grad = _multi_alt_bwd_body(
            grad_out, weights, x, ctx.batch_offset,
            ctx.anchor_a_long, ctx.anchor_b_long,
            ctx.table_arange, ctx.table_flat,
            lookup_indices, n_heads, tph, ctx.n_alternatives,
        )

        # 13 forward inputs -> 13 grad returns.
        return (x_grad, weights_grad,
                None, None, None, None, None, None, None, None, None, None, None)


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
        n_alternatives: int = 1,
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
        self.n_alternatives = int(n_alternatives)
        if self.n_alternatives < 1:
            raise ValueError(f"n_alternatives must be >= 1, got {self.n_alternatives}")
        if self.n_alternatives > 1:
            if backward_mode != "ste":
                raise ValueError(
                    "n_alternatives > 1 requires backward_mode='ste' "
                    f"(got backward_mode={backward_mode!r}). For soft mode the "
                    "rational-soft-sign pipeline already smooths all NAP positions."
                )
            if sparse_scatter_n_outputs is not None:
                raise NotImplementedError(
                    "n_alternatives > 1 does not yet support sparse_scatter_n_outputs"
                )
            if partition_sets is not None:
                raise NotImplementedError(
                    "n_alternatives > 1 does not yet support partition_sets"
                )
            # Cache int64 anchor pairs + LSB-first powers + batch offset for
            # `_compute_anchor_data` / `_anchor_pairs_lookup_forward_fallback*`
            # (these helpers require long-typed indices for `gather`).
            self.register_buffer(
                "_multialt_anchor_a_long",
                self.lookup.anchor_pairs_a.long().contiguous(),
            )
            self.register_buffer(
                "_multialt_anchor_b_long",
                self.lookup.anchor_pairs_b.long().contiguous(),
            )
            # LSB-first SHIFT AMOUNTS (matches `_anchor_pairs_lookup_forward_fallback`'s
            # `(bits << powers).sum`, where powers = [0, 1, ..., NAP-1] is broadcast
            # over the bit dim, NOT the MSB-first multipliers used by soft mode).
            self.register_buffer(
                "_multialt_powers_long",
                torch.arange(n_anchor_pairs, device=dev, dtype=torch.int64).view(1, 1, -1).contiguous(),
            )
            # LSB-first bit-pack MULTIPLIERS [2^0, 2^1, ..., 2^(NAP-1)] for the
            # fused fwd/bwd compile bodies — avoids recomputing arange/shift each call.
            self.register_buffer(
                "_multialt_powers_mult",
                (1 << torch.arange(n_anchor_pairs, device=dev, dtype=torch.int64)).contiguous(),
            )
            # arange(n_lookup_tables) — used as table_2d, table_3d, table_flat in
            # the bwd body. Cached 1D; the body broadcasts/views as needed.
            self.register_buffer(
                "_multialt_table_arange",
                torch.arange(n_lookup_tables, device=dev, dtype=torch.int64).contiguous(),
            )
            self._multialt_batch_offset = None       # lazily built on first forward
            self._multialt_eb_offsets = None         # embedding_bag offsets cache
            self._multialt_table_flat = None         # arange(n_lookup_tables).repeat(B)
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

        # Multi-alternative STE backward. Uses self.n_alternatives top-|delta|
        # anchor positions selected via manual argmin (no topk kernel; cheap
        # for small k). Fully fuseable inside the @torch.compile body.
        if self.n_alternatives > 1:
            B = x.shape[0]
            n_lookup_tables = self.n_heads * self.tables_per_head
            expected_bo_len = B * n_lookup_tables * self.n_alternatives
            if (
                self._multialt_batch_offset is None
                or self._multialt_batch_offset.numel() != expected_bo_len
                or self._multialt_batch_offset.device != x.device
            ):
                self._multialt_batch_offset = (
                    torch.arange(B, device=x.device, dtype=torch.long)
                    .repeat_interleave(n_lookup_tables * self.n_alternatives)
                    * self.input_dim
                ).contiguous()
            # embedding_bag offsets cache: depends only on B (and fixed n_heads, tph)
            expected_eb_len = B * self.n_heads
            if (
                self._multialt_eb_offsets is None
                or self._multialt_eb_offsets.numel() != expected_eb_len
                or self._multialt_eb_offsets.device != x.device
            ):
                self._multialt_eb_offsets = (
                    torch.arange(expected_eb_len, device=x.device, dtype=torch.long) * self.tables_per_head
                ).contiguous()
            # table_flat = arange(n_lookup_tables).repeat(B) — for weights_grad scatter
            expected_tf_len = B * n_lookup_tables
            if (
                self._multialt_table_flat is None
                or self._multialt_table_flat.numel() != expected_tf_len
                or self._multialt_table_flat.device != x.device
            ):
                self._multialt_table_flat = self._multialt_table_arange.repeat(B).contiguous()
            return _TinyMHLutMultiAlt.apply(
                x, self.weights,
                self._multialt_anchor_a_long, self._multialt_anchor_b_long,
                self._multialt_powers_mult, self._multialt_table_arange,
                self._multialt_eb_offsets,
                self.n_alternatives, self._multialt_batch_offset,
                self._multialt_table_flat,
                self.n_heads, self.tables_per_head, self.argmax_noise_eps,
            )

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
