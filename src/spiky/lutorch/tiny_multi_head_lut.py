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
# Toggle to disable the native sparse_scatter forward kernel.
_USE_TINY_MHLUT_NATIVE_SPARSE_FWD = True
# When True, soft-mode backward uses sel_soft to softly attribute weight gradient
# across all K rows per (batch, table), instead of the hard index_add at the
# chosen row only. Higher per-row gradient density at cost of einsum work and a
# small bias toward sel_soft's shape. Disables the native fused bwd kernel.
_GLOBAL_SOFT_WEIGHT_GRAD = False

def enable_soft_weight_grad(flag: bool = True) -> None:
    """Set the global soft-weight-gradient toggle (see _GLOBAL_SOFT_WEIGHT_GRAD)."""
    global _GLOBAL_SOFT_WEIGHT_GRAD
    _GLOBAL_SOFT_WEIGHT_GRAD = bool(flag)

# When True, soft-mode backward divides each (table, row) weight gradient by the
# number of batch elements that landed there, converting the per-row scatter-add
# from SUM to MEAN. Equalizes effective update magnitude across rows regardless
# of visit-count imbalance (helps cold rows; reduces hot-row momentum).
_GLOBAL_PER_ROW_GRAD_NORM = False

def set_per_row_grad_norm(flag: bool = True) -> None:
    """Set the global per-row visit-count gradient normalization toggle."""
    global _GLOBAL_PER_ROW_GRAD_NORM
    _GLOBAL_PER_ROW_GRAD_NORM = bool(flag)

# When True, soft-mode backward gates the per-row weight gradient by a per-coord
# "dominance" rule using the row's centroid as the partner: for each output
# coord a, the gradient is kept iff (W[k*][a] - mean(W[k*])) * (grad_pt[a] -
# mean(grad_pt)) >= 0 — i.e., the gradient is fighting an incorrect dominance
# with the centroid. When the dominance is already correct (product < 0),
# strengthening it further does nothing useful for downstream LUTs (which only
# see pairwise dominances) so the gradient is zeroed. Also projects out the
# mean direction of grad_pt (gauge for downstream LUT and unembedder consumers).
_GLOBAL_DOMINANCE_GATE = False

def set_dominance_gate(flag: bool = True) -> None:
    """Set the global dominance-gate toggle (see _GLOBAL_DOMINANCE_GATE)."""
    global _GLOBAL_DOMINANCE_GATE
    _GLOBAL_DOMINANCE_GATE = bool(flag)

# When True, the soft-mode backward builds the softmax over HARD-sign (±1) match
# scores — `ts = einsum(p_signs, bit_matrix)` integer-valued — instead of the
# soft-magnitude `p = p_signs·|d|/denom`. The argmax FORWARD is sign-only so it's
# unchanged; this isolates the effect of hard vs soft signs on the backward
# gradient alone (the TinyMHLut analog of MatmulMHL's hard_sign_ste / exp500).
# The input-gradient Jacobian (T_soft/denom²) is kept as the STE surrogate —
# without it hard signs give zero input gradient.
_GLOBAL_HARD_SIGN_BWD = False

def set_hard_sign_bwd(flag: bool = True) -> None:
    """Set the global hard-sign-backward toggle (see _GLOBAL_HARD_SIGN_BWD)."""
    global _GLOBAL_HARD_SIGN_BWD
    _GLOBAL_HARD_SIGN_BWD = bool(flag)

# When True, the soft-mode backward computes the per-(table,row) visit count for
# this backward (how many batch elements selected each row) and stashes it in
# _GLOBAL_VISIT_COUNT_REGISTRY keyed by weights.data_ptr(). The gradient itself
# is UNCHANGED (unlike _GLOBAL_PER_ROW_GRAD_NORM, which folds 1/count into grad).
# This exposes the exact diagonal Hessian (= visit count) to a Gauss-Newton-style
# optimizer that wants `m / (count + lambda)` per row. Default off.
_GLOBAL_STASH_VISIT_COUNT = False
_GLOBAL_VISIT_COUNT_REGISTRY = {}

def enable_visit_count_stash(flag: bool = True) -> None:
    """Set the global visit-count stash toggle (see _GLOBAL_STASH_VISIT_COUNT)."""
    global _GLOBAL_STASH_VISIT_COUNT
    _GLOBAL_STASH_VISIT_COUNT = bool(flag)

def get_visit_count(weights) -> "torch.Tensor":
    """Return the last-backward per-row visit count [n_tables*table_dim] for a
    LUT weight param, or None if not stashed yet."""
    return _GLOBAL_VISIT_COUNT_REGISTRY.get(weights.data_ptr())

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


def _get_tiny_mhlut_sparse_native():
    """Native LUTorchManager iff it has the sparse_scatter forward binding."""
    m = _get_tiny_mhlut_native()
    if m is not None and hasattr(m, 'tiny_mhlut_sparse_scatter_forward'):
        return m
    return None


def _build_sparse_scatter_inverse_map(scatter_indices: torch.Tensor,
                                       sparse_n_outputs: int):
    """Precompute the slot-major inverse of `scatter_indices` for the
    native sparse_scatter forward kernel.

    scatter_indices: [n_heads, tables_per_head, n_outputs] long.
    Returns (slot_offsets, contrib_table, contrib_local_i):
        slot_offsets    [n_heads, sparse_n_outputs + 1] long — prefix sum
                        of contributors per output slot per head.
        contrib_table   [n_heads, tph * n_outputs]      long — GLOBAL table
                        index of each contributor (sorted by destination
                        slot within each head).
        contrib_local_i [n_heads, tph * n_outputs]      long — per-table
                        local output index of each contributor.
    """
    H, T, N = scatter_indices.shape
    flat_dest = scatter_indices.reshape(H, T * N)
    sorted_dest, perm = flat_dest.sort(dim=1, stable=True)
    flat_src = torch.arange(T * N, device=scatter_indices.device,
                             dtype=torch.long).unsqueeze(0).expand(H, -1)
    sorted_src = flat_src.gather(1, perm)
    contrib_local_t = sorted_src // N
    contrib_local_i = sorted_src % N
    head_offset = (torch.arange(H, device=scatter_indices.device,
                                 dtype=torch.long) * T).unsqueeze(1)
    contrib_global_t = head_offset + contrib_local_t            # GLOBAL table idx
    counts = torch.zeros(H, sparse_n_outputs, device=scatter_indices.device,
                          dtype=torch.long)
    counts.scatter_add_(1, sorted_dest, torch.ones_like(sorted_dest))
    slot_offsets = torch.zeros(H, sparse_n_outputs + 1, device=scatter_indices.device,
                                dtype=torch.long)
    slot_offsets[:, 1:] = counts.cumsum(dim=1)
    return slot_offsets.contiguous(), contrib_global_t.contiguous(), contrib_local_i.contiguous()


@torch.compile
def _soft_index_signpack(x, anchor_a_long, anchor_b_long, powers, noise_eps):
    """Index-only helper for sparse forward: sign-bit-pack with optional
    Bernoulli noise on low-confidence bits. Returns `index: [B, n_tables]`."""
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    bits = (d > 0).to(torch.int64)
    if noise_eps > 0.0:
        rand_bits = torch.empty_like(d).bernoulli_(0.5).to(torch.int64)
        low_conf = (d.abs() < noise_eps)
        bits = torch.where(low_conf, rand_bits, bits)
    return (bits * powers.view(1, 1, -1)).sum(dim=-1)


@torch.compile
def _soft_index_einsum_bf16(x, anchor_a_long, anchor_b_long, bit_matrix, T_soft):
    """Index-only helper for sparse forward: bf16 einsum + argmax (no scatter)."""
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    p = d / (T_soft + d.abs())
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    return ts.argmax(dim=-1).to(torch.int64)


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
                        bit_matrix, T_soft, n_heads, tph, table_dim, noise_eps):
    """Compiled forward.

    fp32 sign(x_a - x_b) bit-pack for the index (provably equal to
    argmax(soft_ts) at fp32 — see test_softmhlut_argmax_equals_signbit_pack_at_fp32).

    noise_eps > 0: RANDOMLY flip the bit at positions where |d[i]| < noise_eps.
    This injects structured noise on low-confidence comparisons — the explicit
    replacement for bf16's implicit argmax regularisation.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
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
def _soft_lut_fwd_body_per_table(x, weights, anchor_a_long, anchor_b_long, powers,
                                  bit_matrix, T_soft, n_heads, tph, table_dim, noise_eps):
    """Compiled forward — per-table variant (no embedding_bag reduce).

    Identical to `_soft_lut_fwd_body` for the index computation; differs only
    in returning per-table gather `[B, n_heads, tph, n_outputs]` instead of
    the [B, n_heads, n_outputs] sum. Used by the sparse_scatter path: callers
    apply scatter_add into a wider output dim after this returns.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    bits = (d > 0).to(torch.int64)
    if noise_eps > 0.0:
        rand_bits = torch.empty_like(d).bernoulli_(0.5).to(torch.int64)
        low_conf = (d.abs() < noise_eps)
        bits = torch.where(low_conf, rand_bits, bits)
    index = (bits * powers.view(1, 1, -1)).sum(dim=-1)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    out_per_table = F.embedding(flat_indices, weights_flat)        # [B*n_tables, n_outputs]
    return out_per_table.view(B, n_heads, tph, n_outputs), index


@torch.compile
def _soft_lut_fwd_body_scatter(x, weights, anchor_a_long, anchor_b_long, powers,
                                bit_matrix, T_soft, n_heads, tph, table_dim, noise_eps,
                                scatter_indices, sparse_n_outputs):
    """Compiled forward — fused per-table gather + scatter_add into sparse output.

    Returns `[B, n_heads, sparse_n_outputs]` directly. The per-table intermediate
    `[B, n_heads, tph, n_outputs]` (which would be ~2 GB at exp318 shapes) is
    constructed inside the compiled body so Inductor can fuse it with the
    scatter_add and avoid a full HBM round-trip.

    scatter_indices: [n_heads, tph, n_outputs] long. Maps each (head, table_local,
    local_output) tuple to a destination slot in [0, sparse_n_outputs).
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    bits = (d > 0).to(torch.int64)
    if noise_eps > 0.0:
        rand_bits = torch.empty_like(d).bernoulli_(0.5).to(torch.int64)
        low_conf = (d.abs() < noise_eps)
        bits = torch.where(low_conf, rand_bits, bits)
    index = (bits * powers.view(1, 1, -1)).sum(dim=-1)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    out_per_table = F.embedding(flat_indices, weights_flat)        # [B*n_tables, n_outputs]
    out_per_table = out_per_table.view(B, n_heads, tph, n_outputs)
    out = out_per_table.new_zeros(B, n_heads, sparse_n_outputs)
    idx = scatter_indices.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, n_heads, tph * n_outputs)
    out.scatter_add_(2, idx, out_per_table.reshape(B, n_heads, -1))
    return out, index


@torch.compile
def _soft_lut_fwd_body_einsum(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, T_soft, n_heads, tph, table_dim):
    """Compiled forward — bf16 einsum path (SoftMHLut(hard=True) parity).

    Replaces the fp32 sign-bit-pack with the same `argmax(softmax(ts / T_sel))`
    computation that SoftMHLut(hard=True) does, where `ts = einsum(p, bit_matrix)`
    runs as a bf16 matmul under autocast. The bf16 rounding inside the einsum
    is the *original* implicit regularizer that the explicit Bernoulli-flip
    path proxies — and it's per-row tie-break (between rows whose `ts` scores
    are within ~ULP of each other), not per-bit sign flip.

    Memory: only `index` ([B, n_tables]) is materialized as saved activation
    — torch.compile fuses einsum → argmax so the [B, n_tables, 2^NAP] tensor
    never reaches HBM. Compute cost is higher than sign-pack (a bf16 matmul
    of shape [B*n_tables, NAP] x [NAP, 2^NAP]) but well-served by Tensor Cores.

    No noise injection here; the bf16 rounding inside the einsum is the noise.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    p = d / (T_soft + d.abs())
    # bf16 autocast (set by caller) downcasts the einsum operands → bf16 matmul.
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    index = ts.argmax(dim=-1).to(torch.int64)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tph
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs), index


@torch.compile
def _soft_lut_fwd_body_einsum_per_table(x, weights, anchor_a_long, anchor_b_long,
                                         bit_matrix, T_soft, n_heads, tph, table_dim):
    """Compiled forward — bf16 einsum, per-table variant (no embedding_bag reduce).

    See `_soft_lut_fwd_body_einsum` for the index computation. Returns
    per-table gather `[B, n_heads, tph, n_outputs]` for sparse_scatter callers.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    p = d / (T_soft + d.abs())
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    index = ts.argmax(dim=-1).to(torch.int64)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    out_per_table = F.embedding(flat_indices, weights_flat)        # [B*n_tables, n_outputs]
    return out_per_table.view(B, n_heads, tph, n_outputs), index


def _soft_lut_fwd_body_prob(x, weights, anchor_a_long, anchor_b_long,
                            bit_matrix, T_soft, T_sel, n_heads, tph, table_dim):
    """Probabilistic forward: sample one row per (batch, table) from softmax(ts/T_sel).

    Like the einsum argmax path but replaces argmax with torch.multinomial — each
    row whose softmax weight is non-negligible has a chance of being selected every
    step. At small batch sizes this gives more uniform row coverage than argmax,
    where the top row monopolises gradient signal and cold rows starve.

    Cannot be @torch.compile-d (multinomial is a dynamic op); runs in eager mode.
    Backward is identical to _soft_lut_bwd_body (same STE formula with sampled idx).
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    p = d / (T_soft + d.abs())
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    probs = F.softmax(ts.float() / T_sel, dim=-1)          # [B, n_tables, K]
    flat_probs = probs.reshape(B * n_tables, -1).contiguous()
    idx_flat = torch.multinomial(flat_probs, num_samples=1, replacement=True).squeeze(-1)
    index = idx_flat.reshape(B, n_tables)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    n_bags = B * n_heads
    offsets = torch.arange(n_bags, device=weights.device, dtype=torch.long) * tph
    out_flat = F.embedding_bag(flat_indices, weights_flat, offsets=offsets, mode='sum')
    return out_flat.view(B, n_heads, n_outputs), index


@torch.compile
def _soft_lut_fwd_body_winner(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, T_soft, T_sel, n_heads, tph, table_dim):
    """Scaled-hard forward: out = softmax(ts/T_sel)[winner] * W[winner].

    Picks the argmax row (winner) like hard forward, but multiplies its weights
    by the winner's softmax coefficient (= max of the selection softmax, a scalar
    in (1/K, 1] per table). Confident selection (one row dominates) → coeff≈1 ≈
    plain argmax; uncertain selection → output attenuated. Single-row lookup at
    inference (same bandwidth as ste/soft), but the coeff is a smooth, fully
    differentiable confidence gate giving an x-gradient path through the scores.
    Deterministic: train and eval forward are identical.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    p = d / (T_soft + d.abs())
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    sel = F.softmax(ts / T_sel, dim=-1)
    index = ts.argmax(dim=-1)                                      # [B, n_tables]
    coeff = sel.gather(-1, index.unsqueeze(-1)).squeeze(-1)        # [B, n_tables]
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    W_winner = F.embedding(flat_indices, weights_flat).view(B, n_tables, n_outputs)
    out_pt = coeff.unsqueeze(-1).to(W_winner.dtype) * W_winner
    out = out_pt.view(B, n_heads, tph, n_outputs).sum(dim=2)
    return out, index


@torch.compile
def _soft_lut_bwd_body_winner(grad_pt, x, weights, anchor_a_long, anchor_b_long,
                               bit_matrix, index, T_soft, T_sel, n_heads, tph):
    """Backward for scaled-hard (soft_winner) forward.

    out = coeff * W[winner], coeff = softmax(ts/T_sel)[winner] (W-independent).
      - weight grad: coeff * grad_pt, index_add at winner row.
      - x / temp grad: through coeff only (winner index fixed, STE-style).
        dcoeff/dz_j = coeff * (1{j=winner} - sel_j).
    """
    B, n_tables_, n_outputs = grad_pt.shape
    n_tables, NAP = anchor_a_long.shape
    K = bit_matrix.shape[1]
    input_dim = x.shape[1]
    w_dtype = weights.dtype

    d     = x[:, anchor_a_long] - x[:, anchor_b_long]
    denom = T_soft + d.abs()
    p     = d / denom
    ts    = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    z     = ts / T_sel
    sel   = F.softmax(z, dim=-1)
    coeff = sel.gather(-1, index.unsqueeze(-1)).squeeze(-1)        # [B, n_tables]

    flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * K
    flat_idx    = (index + flat_offset[None, :]).reshape(-1)
    W_winner = F.embedding(flat_idx, weights.view(n_tables * K, n_outputs)).view(B, n_tables, n_outputs)

    # weight gradient: coeff-scaled, hard index_add at winner row.
    grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=w_dtype, device=weights.device)
    grad_w_contrib = (coeff.unsqueeze(-1).to(w_dtype) * grad_pt.to(w_dtype)).reshape(-1, n_outputs)
    grad_w_flat.index_add_(0, flat_idx, grad_w_contrib)
    grad_weights = grad_w_flat.view(n_tables, K, n_outputs)

    # coeff gradient → ts → p → d → x.
    dL_dcoeff = (grad_pt.to(w_dtype) * W_winner).sum(dim=-1)       # [B, n_tables]
    onehot = F.one_hot(index, num_classes=K).to(sel.dtype)        # [B, n_tables, K]
    dL_dz = dL_dcoeff.unsqueeze(-1).to(sel.dtype) * coeff.unsqueeze(-1) * (onehot - sel)
    dL_dts = dL_dz / T_sel
    grad_log_T_sel = -(dL_dz * z).sum()

    dL_dp = torch.einsum("btk,pk->btp", dL_dts, bit_matrix.to(dL_dts.dtype))
    d_d = dL_dp * (T_soft / (denom * denom))
    grad_log_T_soft = -(d_d * d).sum()

    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a_flat = anchor_a_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b_flat = anchor_b_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat     = d_d.reshape(B, -1).to(x.dtype)
    grad_x.scatter_add_(1, idx_a_flat,  d_flat)
    grad_x.scatter_add_(1, idx_b_flat, -d_flat)

    return grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel


@torch.compile
def _soft_lut_fwd_body_einsum_scatter(x, weights, anchor_a_long, anchor_b_long,
                                       bit_matrix, T_soft, n_heads, tph, table_dim,
                                       scatter_indices, sparse_n_outputs):
    """Compiled forward — bf16 einsum + fused gather + scatter_add. See
    `_soft_lut_fwd_body_scatter` for fusion rationale; this is the einsum variant.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    p = d / (T_soft + d.abs())
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    index = ts.argmax(dim=-1).to(torch.int64)
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    table_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * table_dim
    flat_indices = (index + table_offset.view(1, -1)).reshape(-1)
    out_per_table = F.embedding(flat_indices, weights_flat)        # [B*n_tables, n_outputs]
    out_per_table = out_per_table.view(B, n_heads, tph, n_outputs)
    out = out_per_table.new_zeros(B, n_heads, sparse_n_outputs)
    idx = scatter_indices.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, n_heads, tph * n_outputs)
    out.scatter_add_(2, idx, out_per_table.reshape(B, n_heads, -1))
    return out, index


@torch.compile
def _soft_lut_bwd_body(grad_pt, x, weights, anchor_a_long, anchor_b_long,
                        bit_matrix, index, T_soft, T_sel, n_heads, tph,
                        visit_norm: bool = False,
                        dominance_gate: bool = False,
                        hard_sign_ste: bool = False,
                        compute_weight_grad: bool = True):
    """Compiled backward — Gumbel-STE consistent.

    Reconstructs `p` so that `argmax(sel_soft) ≡ saved index` (including any
    noise flips applied in forward). Extract bits actually used in forward
    from `index`, build `p_signs = ±1` matching those bits, then
    `p = p_signs * |d| / (T_soft + |d|)`. In the no-noise case this is
    bit-identical to `p = d/(T+|d|)`; under noise it makes the soft pipeline's
    argmax match the chosen index, so softmax-backward gradients are
    self-consistent with the picked row.

    grad_pt: [B, n_tables, n_outputs] — per-table upstream gradient. Callers
    construct this from grad_out: in dense mode by expanding [B, H, n_outputs]
    along the tph axis; in sparse_scatter mode it's a direct reshape of
    [B, H, tph, n_outputs] (autograd of the external scatter_add).
    """
    B, n_tables_, n_outputs = grad_pt.shape
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
    if hard_sign_ste:
        # Hard signs: softmax over INTEGER hamming scores (exact exp-hamming
        # kernel). Forward (argmax) is sign-only -> unchanged; only the bwd
        # gradient shaping changes. dp/dd surrogate (T_soft/denom²) kept below.
        p    = p_signs
    else:
        p    = p_signs * d.abs() / denom

    ts       = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    z        = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)

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

    # dL/dweights via scatter at saved (table, index). Skipped when the caller
    # only needs grad_x / grad_log_Ts / grad_log_Tx (e.g. hybrid_smooth modes
    # supply their own 2-row weight grad via _hybrid_smooth_weight_grad).
    if compute_weight_grad:
        flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * K
        flat_idx    = (index + flat_offset[None, :]).reshape(-1)
        grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=w_dtype, device=weights.device)
        if dominance_gate:
            # Per-coord centroid-dominance gate: keep grad_centered[a] iff its sign
            # agrees with W_centered[a] (= the gradient is fighting current dominance).
            # Zero gradient where dominance is already correct.
            w_flat_all = weights.view(n_tables * K, n_outputs)
            w_at = F.embedding(flat_idx, w_flat_all)              # [B*n_tables, n_outputs]
            grad_w_in = grad_pt.reshape(-1, n_outputs).to(w_dtype)
            w_centered = w_at - w_at.mean(dim=-1, keepdim=True)
            g_centered = grad_w_in - grad_w_in.mean(dim=-1, keepdim=True)
            keep = (w_centered * g_centered) >= 0
            grad_to_write = torch.where(keep, g_centered, torch.zeros_like(g_centered))
            grad_w_flat.index_add_(0, flat_idx, grad_to_write)
        else:
            grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(w_dtype))
        if visit_norm:
            # Convert per-row SUM to per-row MEAN by dividing by visit count.
            counts = torch.bincount(flat_idx, minlength=n_tables * K).to(w_dtype).clamp_(min=1)
            grad_w_flat = grad_w_flat / counts.unsqueeze(-1)
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


@torch.compile
def _soft_lut_bwd_body_prob(grad_pt, x, weights, anchor_a_long, anchor_b_long,
                             bit_matrix, index, T_soft, T_sel, n_heads, tph):
    """Backward for probabilistic-forward mode.

    Weight gradient: hard index_add_ at sampled `index` (same as STE).
    Input gradient: derived from the ACTUAL softmax distribution p → ts → sel_soft,
    NOT from index-bit reconstruction. Since the forward samples from
    softmax(ts/T_sel), the gradient of E[output] w.r.t. input integrates over
    the full distribution — it does not depend on which specific row was sampled.
    """
    B, n_tables_, n_outputs = grad_pt.shape
    n_tables, NAP = anchor_a_long.shape
    K = bit_matrix.shape[1]
    input_dim = x.shape[1]
    w_dtype = weights.dtype

    d     = x[:, anchor_a_long] - x[:, anchor_b_long]
    denom = T_soft + d.abs()
    p     = d / denom                           # actual p — no index reconstruction

    ts       = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    z        = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)

    d_sel_soft = torch.einsum("bto,tko->btk", grad_pt.to(w_dtype), weights)

    sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
    d_z      = sel_soft * (d_sel_soft - sum_term)
    d_ts     = d_z / T_sel
    grad_log_T_sel = -(d_z * z).sum()

    # dp/dd = T_soft / denom^2 (straightforward, p = d/denom).
    d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))
    d_d = d_p * (T_soft / (denom * denom))
    grad_log_T_soft = -(d_d * d).sum()

    # Weight gradient: hard index_add at sampled index (STE-style).
    flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * K
    flat_idx    = (index + flat_offset[None, :]).reshape(-1)
    grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=w_dtype, device=weights.device)
    grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(w_dtype))
    grad_weights = grad_w_flat.view(n_tables, K, n_outputs)

    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a_flat = anchor_a_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b_flat = anchor_b_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat     = d_d.reshape(B, -1).to(x.dtype)
    grad_x.scatter_add_(1, idx_a_flat,  d_flat)
    grad_x.scatter_add_(1, idx_b_flat, -d_flat)

    return grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel


@torch.compile
def _soft_lut_bwd_body_soft_w(grad_pt, x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, index, T_soft, T_sel, n_heads, tph):
    """Soft weight-gradient variant of `_soft_lut_bwd_body`.

    Identical to the standard body except the per-row weight gradient is
    computed via sel_soft-weighted attribution across all K rows, instead of
    the hard `index_add` at the chosen row only. Each (batch, table) thus
    contributes to all K rows densely, increasing per-row update frequency.
    """
    B, n_tables_, n_outputs = grad_pt.shape
    n_tables, NAP = anchor_a_long.shape
    K = bit_matrix.shape[1]
    input_dim = x.shape[1]
    w_dtype = weights.dtype

    d        = x[:, anchor_a_long] - x[:, anchor_b_long]
    denom    = T_soft + d.abs()

    shifts   = torch.arange(NAP - 1, -1, -1, device=index.device, dtype=index.dtype)
    bits     = ((index.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 1).to(d.dtype)
    p_signs  = bits * 2.0 - 1.0
    p        = p_signs * d.abs() / denom

    ts       = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))
    z        = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)

    d_sel_soft = torch.einsum("bto,tko->btk", grad_pt.to(w_dtype), weights)

    sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
    d_z      = sel_soft * (d_sel_soft - sum_term)
    d_ts     = d_z / T_sel
    grad_log_T_sel = -(d_z * z).sum()

    d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))
    d_d = d_p * p_signs * d.sign() * (T_soft / (denom * denom))
    grad_log_T_soft = -(d_d * d).sum()

    # Soft weight gradient: distribute grad_pt across all K rows via sel_soft.
    # sel_soft sums to 1 over K, so total per-token L1 mass equals the hard path.
    grad_weights = torch.einsum("btk,bto->tko", sel_soft.to(w_dtype), grad_pt.to(w_dtype))

    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a_flat = anchor_a_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b_flat = anchor_b_long.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat     = d_d.reshape(B, -1).to(x.dtype)
    grad_x.scatter_add_(1, idx_a_flat,  d_flat)
    grad_x.scatter_add_(1, idx_b_flat, -d_flat)

    return grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel


@torch.compile
def _soft_lut_bwd_body_topk(grad_pt, x, weights, anchor_a_long, anchor_b_long,
                             bit_matrix, powers_msb, index, T_soft, T_sel,
                             n_heads, tph, topk_n_alt):
    """Soft-math backward with softmax masked to {chosen + top-K 1-bit-flip
    neighbors} (Approach B: mask -inf on ts BEFORE softmax, keep dense
    tensor-core path).

    Math equivalent to the alternative gather-based implementation, but keeps
    the cheap full-K einsums for ts, d_sel_soft, and d_p. softmax(ts_masked)
    naturally renormalizes over the kept rows (since masked rows have
    ts=-inf → sel_soft=0). The full-soft chain rule then propagates gradient
    only through the kept rows automatically.

    Cost: ~= full soft (same einsums, plus a cheap mask construction).
    Memory: same as full soft (no [B, T, K_top] savings).
    Quality: softmax over (1 + topk_n_alt) rows out of K_full = 2^NAP rows.

    Selection of kept rows:
      - chosen index (saved from forward).
      - The topk_n_alt 1-bit-flip neighbors at the smallest-|d| anchor
        positions (= rows with highest sel_soft contribution after chosen).
      - When topk_n_alt >= NAP, all 1-bit-flip neighbors are kept.
    """
    B, n_tables_, n_outputs = grad_pt.shape
    n_tables, NAP = anchor_a_long.shape
    K_full = bit_matrix.shape[1]
    input_dim = x.shape[1]
    w_dtype = weights.dtype

    d        = x[:, anchor_a_long] - x[:, anchor_b_long]
    denom    = T_soft + d.abs()

    # Reconstruct p so argmax(sel_soft) == saved index (matches `_soft_lut_bwd_body`).
    shifts   = torch.arange(NAP - 1, -1, -1, device=index.device, dtype=index.dtype)
    bits     = ((index.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 1).to(d.dtype)
    p_signs  = bits * 2.0 - 1.0
    p        = p_signs * d.abs() / denom

    ts       = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))

    # Pick the top-K bit positions to flip (smallest |d| = highest neighbor
    # sel_soft). When topk_n_alt == NAP, this is just all positions.
    if topk_n_alt >= NAP:
        # All NAP positions: avoid the topk() call.
        top_pos = (
            torch.arange(NAP, device=index.device, dtype=index.dtype)
            .view(1, 1, -1)
            .expand(B, n_tables, -1)
        )
    else:
        abs_d = d.abs()
        _, top_pos = abs_d.topk(topk_n_alt, dim=-1, largest=False)

    # Build the row indices to keep: chosen + 1-bit-flip neighbors.
    selected_powers = powers_msb.view(1, 1, -1).expand(B, n_tables, -1).gather(-1, top_pos)
    alt_indices = index.unsqueeze(-1) ^ selected_powers              # [B, T, topk_n_alt]
    kept_indices = torch.cat([index.unsqueeze(-1), alt_indices], dim=-1)  # [B, T, 1+topk_n_alt]

    # Approach (B): mask ts to -inf at non-kept rows so the dense softmax
    # naturally renormalizes over the kept subset. Keeps the cheap full-K
    # tensor-core path for d_sel_soft and d_p; adds a bool mask + one extra
    # [B, T, K] tensor (ts_masked).
    mask = torch.zeros(B, n_tables, K_full, dtype=torch.bool, device=index.device)
    mask.scatter_(2, kept_indices, True)
    ts_masked = ts.masked_fill(~mask, float('-inf'))

    z         = ts_masked / T_sel
    sel_soft  = F.softmax(z, dim=-1)                                # zero at non-kept rows

    d_sel_soft = torch.einsum("bto,tko->btk", grad_pt.to(w_dtype), weights)

    sum_term = (d_sel_soft * sel_soft).sum(dim=-1, keepdim=True)
    d_z      = sel_soft * (d_sel_soft - sum_term)
    d_ts     = d_z / T_sel
    # grad_log_T_sel: clamp z to 0 at -inf rows (d_z is already 0 there since
    # sel_soft is 0). Avoid 0 * -inf = NaN.
    z_safe = torch.where(mask, z, torch.zeros_like(z))
    grad_log_T_sel = -(d_z * z_safe).sum()

    d_p = torch.einsum("btk,pk->btp", d_ts, bit_matrix.to(d_ts.dtype))
    d_d = d_p * p_signs * d.sign() * (T_soft / (denom * denom))
    grad_log_T_soft = -(d_d * d).sum()

    # Weight gradient: hard index_add at chosen row (identical to full soft).
    flat_offset = torch.arange(n_tables, device=weights.device, dtype=index.dtype) * K_full
    flat_idx    = (index + flat_offset[None, :]).reshape(-1)
    grad_w_flat = torch.zeros(n_tables * K_full, n_outputs, dtype=w_dtype, device=weights.device)
    grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs).to(w_dtype))
    grad_weights = grad_w_flat.view(n_tables, K_full, n_outputs)

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

    sparse modes (controlled by `scatter_indices`):
      - dense (scatter_indices is None): forward returns `[B, H, n_outputs]`,
        backward broadcasts grad_out across the tph axis.
      - sparse_scatter (scatter_indices: [H, tph, n_outputs]): forward fuses
        per-table gather + scatter_add inside the compiled body and returns
        `[B, H, sparse_n_outputs]` directly. Backward gathers grad_out at
        scatter_indices to reconstruct grad_pt — this avoids exposing the
        2 GB-class per-table tensor to PyTorch's autograd graph.
    """

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix, powers,
                n_heads, tph, table_dim, use_bf16, argmax_noise_eps,
                einsum_bf16_forward, scatter_indices, sparse_n_outputs,
                sparse_slot_offsets=None, sparse_contrib_table=None,
                sparse_contrib_local_i=None,
                topk_n_alt=0):
        sparse = scatter_indices is not None
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        # Native sparse_scatter forward path: ~2.7x faster than the
        # @torch.compile gather+scatter_add body at exp318 shapes (eliminates
        # scatter_add's atomic-contention cost). Requires all three
        # precomputed inverse-map buffers and the CUDA binding.
        native_sparse = (sparse
                         and _USE_TINY_MHLUT_NATIVE_SPARSE_FWD
                         and x.is_cuda
                         and sparse_slot_offsets is not None
                         and sparse_contrib_table is not None
                         and sparse_contrib_local_i is not None
                         and _get_tiny_mhlut_sparse_native() is not None)
        with autocast_ctx:
            if einsum_bf16_forward:
                if native_sparse:
                    index = _soft_index_einsum_bf16(
                        x, anchor_a_long, anchor_b_long, bit_matrix, T_soft,
                    )
                    out = _get_tiny_mhlut_sparse_native().tiny_mhlut_sparse_scatter_forward(
                        weights, index.contiguous(),
                        sparse_slot_offsets, sparse_contrib_table, sparse_contrib_local_i,
                        n_heads, tph, sparse_n_outputs,
                    )
                elif sparse:
                    out, index = _soft_lut_fwd_body_einsum_scatter(
                        x, weights, anchor_a_long, anchor_b_long,
                        bit_matrix, T_soft, n_heads, tph, table_dim,
                        scatter_indices, sparse_n_outputs,
                    )
                else:
                    out, index = _soft_lut_fwd_body_einsum(
                        x, weights, anchor_a_long, anchor_b_long,
                        bit_matrix, T_soft, n_heads, tph, table_dim,
                    )
            else:
                if native_sparse:
                    index = _soft_index_signpack(
                        x, anchor_a_long, anchor_b_long, powers,
                        float(argmax_noise_eps),
                    )
                    out = _get_tiny_mhlut_sparse_native().tiny_mhlut_sparse_scatter_forward(
                        weights, index.contiguous(),
                        sparse_slot_offsets, sparse_contrib_table, sparse_contrib_local_i,
                        n_heads, tph, sparse_n_outputs,
                    )
                elif sparse:
                    out, index = _soft_lut_fwd_body_scatter(
                        x, weights, anchor_a_long, anchor_b_long, powers,
                        bit_matrix, T_soft, n_heads, tph, table_dim,
                        float(argmax_noise_eps),
                        scatter_indices, sparse_n_outputs,
                    )
                else:
                    out, index = _soft_lut_fwd_body(
                        x, weights, anchor_a_long, anchor_b_long, powers,
                        bit_matrix, T_soft, n_heads, tph, table_dim,
                        float(argmax_noise_eps),
                    )
        if sparse:
            ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                                  bit_matrix, index, log_T_soft, log_T_sel,
                                  scatter_indices, powers)
        else:
            ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                                  bit_matrix, index, log_T_soft, log_T_sel,
                                  powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        ctx.sparse = sparse
        ctx.topk_n_alt = int(topk_n_alt)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        if ctx.sparse:
            (x, weights, anchor_a_long, anchor_b_long, bit_matrix, index,
             log_T_soft, log_T_sel, scatter_indices, powers) = ctx.saved_tensors
        else:
            (x, weights, anchor_a_long, anchor_b_long, bit_matrix, index,
             log_T_soft, log_T_sel, powers) = ctx.saved_tensors
            scatter_indices = None
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = anchor_a_long.shape[0]
        n_outputs = weights.shape[2]
        # Per-table grad_pt: in sparse mode gather grad_out at scatter_indices
        # (inverse of scatter_add); in dense mode broadcast across the tph axis.
        if ctx.sparse:
            idx = scatter_indices.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, n_heads, tph * n_outputs)
            grad_pt = grad_out.gather(2, idx).reshape(B, n_tables, n_outputs)
        else:
            grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            if ctx.topk_n_alt > 0:
                grad_x, grad_w, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body_topk(
                    grad_pt, x, weights, anchor_a_long, anchor_b_long,
                    bit_matrix, powers,
                    index, T_soft, T_sel, ctx.n_heads, ctx.tph, ctx.topk_n_alt,
                )
            else:
                if _GLOBAL_SOFT_WEIGHT_GRAD:
                    grad_x, grad_w, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body_soft_w(
                        grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                        index, T_soft, T_sel, ctx.n_heads, ctx.tph,
                    )
                else:
                    grad_x, grad_w, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                        grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                        index, T_soft, T_sel, ctx.n_heads, ctx.tph,
                        visit_norm=_GLOBAL_PER_ROW_GRAD_NORM,
                        dominance_gate=_GLOBAL_DOMINANCE_GATE,
                        hard_sign_ste=_GLOBAL_HARD_SIGN_BWD,
                    )
        # Stash per-(table,row) visit count for a Gauss-Newton optimizer. The
        # gradient grad_w is the SUM over tokens hitting each row; count is how
        # many tokens that was — i.e. the exact diagonal Hessian of this linear
        # (for fixed selection) layer. Keyed by storage so the optimizer can find
        # it from its own param handle.
        if _GLOBAL_STASH_VISIT_COUNT:
            _K = bit_matrix.shape[1]
            _flat_off = torch.arange(n_tables, device=index.device, dtype=index.dtype) * _K
            _flat_idx = (index + _flat_off[None, :]).reshape(-1)
            _GLOBAL_VISIT_COUNT_REGISTRY[weights.data_ptr()] = torch.bincount(
                _flat_idx, minlength=n_tables * _K).detach()
        # 20 forward inputs (x, weights, log_T_soft, log_T_sel, anchor_a_long,
        # anchor_b_long, bit_matrix, powers, n_heads, tph, table_dim, use_bf16,
        # argmax_noise_eps, einsum_bf16_forward, scatter_indices, sparse_n_outputs,
        # sparse_slot_offsets, sparse_contrib_table, sparse_contrib_local_i, topk_n_alt)
        # → 20 grad returns.
        return (grad_x, grad_w, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None)


@torch.compile
def _hybrid_smooth_weight_grad(grad_pt, main_index, alt_index, u, n_tables, K, n_outputs, w_dtype):
    """2-row weight gradient for hybrid_smooth backward via S+einsum.

    Build a per-(b, t) "selection mass" S of shape [B, n_tables, K] by scatter-
    adding the two row weights (1-u at main_index, u at alt_index) into K
    slots — only 2 writes per (b, t), no inter-batch collisions. Then dW
    follows from a single B-reducing einsum:

        dW[t, k, o] = sum_b S[b, t, k] * grad_pt[b, t, o]

    Same gradient as the original 2-row global-atomic scatter (mathematically
    equivalent; matches within bf16 noise on real inputs), but replaces a
    collision-heavy `index_add_` over a large flat destination tensor with a
    cheap local scatter + a cuBLAS GEMM. On L40S/exp666 this drops the
    backward from ~1130 ms → ~520 ms per step (~2.2× backward speedup, ~2.1×
    total step speedup) by removing the atomicAdd serialization on K=16/K=64
    destinations (256/64 collisions per row in the old layout).

    Memory: S is [B, n_tables, K] of `w_dtype` — for exp666 LUTGPT at B*T=4096:
    ~200 MB for the K=16 LUT, ~800 MB transiently for K=64 LUTs (allocated
    and freed per call). Both fit comfortably in HBM at production batch
    sizes; for very large B*T*n_tables*K shapes the same body can be chunked
    along the B axis without changing the math.
    """
    B = grad_pt.shape[0]
    # Stack the two destination indices and the two row weights for one
    # batched scatter into S (main writes (1-u), alt writes u, distinct cols).
    all_idx  = torch.stack([main_index, alt_index], dim=-1)              # [B, n_tables, 2]
    weights2 = torch.stack([(1.0 - u).to(w_dtype),
                             u.to(w_dtype)], dim=-1)                       # [B, n_tables, 2]
    S = torch.zeros(B, n_tables, K, dtype=w_dtype, device=grad_pt.device)
    S.scatter_add_(-1, all_idx, weights2)                                 # [B, n_tables, K]
    # Per-table B-reducing GEMM. Inductor lowers to cuBLAS bmm.
    return torch.einsum('btk,bto->tko', S, grad_pt.to(w_dtype))


@torch.compile
def _hybrid_smooth_lut_fwd_body(x, weights, anchor_a_long, anchor_b_long, powers,
                                 T_soft, T_sel, n_heads, tph, table_dim):
    """Smooth forward: blend main row and Hamming-1 alternative at least-confident
    anchor pair. Mirrors standard MultiHeadLut(smooth_mode=True, n_alternatives=1).

    Returns:
        out: [B, n_heads, n_outputs] — (1 - u) * W[main] + u * W[alt]
        main_index: [B, n_tables] — argmax row index (sign-packed)
        alt_index: [B, n_tables] — Hamming-1 neighbor at argmin |d| anchor pair
        u: [B, n_tables] — uncertainty in (0, 0.5]; alt_weight = u, main_weight = 1 - u
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    NAP = anchor_a_long.shape[1]

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                 # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)                                # [B, n_tables, NAP]
    powers_view = powers.view(1, 1, -1)                            # [1, 1, NAP]
    main_index = (bits * powers_view).sum(dim=-1)                  # [B, n_tables]

    # Least-confident anchor pair: argmin |d| along NAP dim.
    abs_d = d.abs()                                                # [B, n_tables, NAP]
    p_star = abs_d.argmin(dim=-1)                                  # [B, n_tables]
    # Flip the bit at position p_star: XOR with powers[p_star].
    flip_mask = powers.to(main_index.dtype)[p_star]                # [B, n_tables]
    alt_index = main_index ^ flip_mask                             # [B, n_tables]

    # Exact top-2 softmax over (main, alt). Soft-mode per-anchor score is
    # p[i] = sign(d[i]) * |d[i]| / (T_soft + |d[i]|); the row score for `main`
    # is sum_i |p[i]|, and `alt` differs only in bit p_star, so:
    #   Δts = ts[main] - ts[alt] = 2 * |d_min| / (T_soft + |d_min|)
    # The top-2 softmax weight on alt is then sigmoid(-Δts / T_sel). Both
    # T_soft and T_sel enter the formula, matching the underlying soft-mode
    # forward this approximates. u ∈ (0, 0.5].
    d_min = abs_d.gather(-1, p_star.unsqueeze(-1)).squeeze(-1)     # [B, n_tables]
    delta_ts = 2.0 * d_min / (T_soft + d_min)                       # [B, n_tables]
    u = torch.sigmoid(-delta_ts / T_sel)                            # [B, n_tables]
    main_w = 1.0 - u                                                # [B, n_tables]

    # Explicit gather + scale + tph-sum. Avoids F.embedding_bag's slow
    # per_sample_weights path. Under bf16 autocast the [B, n_tables, n_outputs]
    # intermediate is half-size and Inductor can fuse multiply+sum.
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


@torch.compile
def _hybrid_smooth_lut_fwd_body_scatter(x, weights, anchor_a_long, anchor_b_long, powers,
                                         T_soft, T_sel, n_heads, tph, table_dim,
                                         scatter_indices, sparse_n_outputs):
    """Hybrid-smooth (n_alt=1) forward fused with scatter_add into a sparse output.

    Same 2-row blend as `_hybrid_smooth_lut_fwd_body`, but instead of summing across
    the tph axis the per-table contribution is scatter_added into a [B, n_heads,
    sparse_n_outputs] tensor using `scatter_indices: [n_heads, tph, n_outputs]`.

    Returns:
        out: [B, n_heads, sparse_n_outputs]
        main_index, alt_index, u: same as `_hybrid_smooth_lut_fwd_body`.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]

    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    bits = (d > 0).to(torch.int64)
    powers_view = powers.view(1, 1, -1)
    main_index = (bits * powers_view).sum(dim=-1)

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
    blended_pt = blended.view(B, n_heads, tph, n_outputs)

    out = blended_pt.new_zeros(B, n_heads, sparse_n_outputs)
    idx = scatter_indices.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, n_heads, tph * n_outputs)
    out.scatter_add_(2, idx, blended_pt.reshape(B, n_heads, -1))
    return out, main_index, alt_index, u


@torch.compile
def _hybrid_smooth_lut_fwd_body_scatter_segred(
    x, weights, anchor_a_long, anchor_b_long, powers,
    T_soft, T_sel, n_heads, tph, table_dim,
    slot_offsets, contrib_global_t, contrib_local_i, sparse_n_outputs,
):
    """Same as `_hybrid_smooth_lut_fwd_body_scatter` but replaces the atomic
    `scatter_add_` with a deterministic gather-via-inverse-map + segment_reduce.
    Avoids heavy atomic contention when many tables write to the same output slot
    (e.g. tph=512, n_per=192, sparse_n=384 → 256 writes/slot)."""
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]

    d = x[:, anchor_a_long] - x[:, anchor_b_long]
    bits = (d > 0).to(torch.int64)
    powers_view = powers.view(1, 1, -1)
    main_index = (bits * powers_view).sum(dim=-1)

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

    # Gather via inverse map (sorted by destination slot per head), then segment-reduce.
    flat_idx = (contrib_global_t * n_outputs + contrib_local_i).reshape(-1)
    blended_flat = blended.view(B, n_tables * n_outputs)
    gathered = blended_flat.index_select(1, flat_idx).view(B, n_heads, tph * n_outputs)
    offsets_b = slot_offsets.unsqueeze(0).expand(B, -1, -1).contiguous()
    out = torch.segment_reduce(gathered, 'sum', offsets=offsets_b, axis=2)
    return out, main_index, alt_index, u


class _TinyMHLutHybridSmooth(torch.autograd.Function):
    """Hybrid smooth forward + soft input grad + 2-row weight grad.

    Forward (smooth, like standard MultiHeadLut with n_alternatives=1, smooth=True):
      - Pick main row via sign-bit packing of (x_a > x_b).
      - Pick alt row by flipping the bit at the least-confident anchor pair
        (smallest |x_a - x_b|).
      - Uncertainty u = 0.5 / (1 + |d_min|/T_soft) in (0, 0.5].
      - Output row = (1 - u) * W[main] + u * W[alt], summed across tables.

    Backward:
      - Input/temperature gradients: SOFT, via _soft_lut_bwd_body — i.e., full
        softmax over all K rows, gradient flows back through every row score.
      - Weight gradient: scatter (1 - u) * grad_pt at W[main],
        u * grad_pt at W[alt]; all other rows zero. This matches the forward's
        2-row blend (no soft attribution to non-chosen rows).
    """

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

        # Dense: broadcast grad_out across tph axis.
        grad_pt = grad_out.unsqueeze(2).expand(B, n_heads, tph, n_outputs).reshape(B, n_tables, n_outputs)

        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            # Reuse soft backward for grad_x, grad_log_Ts, grad_log_Tx.
            # _soft_lut_bwd_body returns (grad_x, grad_w_full, grad_log_Ts, grad_log_Tx)
            # where grad_w_full uses single-row scatter at `index` — we discard it
            # and rewrite grad_w with the 2-row hybrid scatter below.
            grad_x, _grad_w_unused, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                main_index, T_soft, T_sel, ctx.n_heads, ctx.tph,
                compute_weight_grad=False,
            )

        # Hybrid weight gradient: fused 2-row scatter at main/alt rows, scaled
        # by (1 - u) and u. @torch.compile fuses multiply + reshape + index_add.
        grad_weights = _hybrid_smooth_weight_grad(
            grad_pt, main_index, alt_index, u, n_tables, K, n_outputs, w_dtype,
        )

        # 12 forward inputs → 12 grad returns.
        return (grad_x, grad_weights, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None)


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
        # Soft-mode forward variant: if True, replace the fp32 sign-bit-pack
        # with the SoftMHLut(hard=True)-style bf16 einsum + argmax. The bf16
        # rounding inside the einsum is the implicit per-row tie-break
        # regularizer that motivated the explicit `argmax_noise_eps` proxy.
        # Backward is unaffected (it already runs the einsum under bf16
        # autocast). Requires `backward_mode='soft'` and `use_bf16=True`.
        # Mutually exclusive in spirit with `argmax_noise_eps>0`: when this
        # path is on, the noise flag is ignored.
        einsum_bf16_forward: bool = False,
        # backward_mode='hybrid_smooth' only: number of alternatives in the
        # forward smooth blend. 1 = top-2 (main + Hamming-1 neighbor at argmin |d|),
        # NAP = full Hamming-1 ball (NAP+1 rows). Other values not supported.
        hybrid_smooth_n_alt: int = 1,
        # backward_mode='hybrid_smooth' + n_alt=NAP only: if True, use plain
        # autograd backward (chain rule through softmax + abs_p directly,
        # gives "self-consistent" input gradient instead of soft K-row surrogate).
        # Inductor can fuse forward+backward. False = use manual backward with
        # soft K-row surrogate for input grad (matches exp611 style).
        hybrid_smooth_autograd: bool = False,
        # backward_mode='hybrid_smooth' + autograd=True only: if True, use a
        # custom autograd.Function that streams the (n_alt+1) row gathers
        # (forward) and re-gathers in backward, instead of letting plain
        # autograd retain all rows across the forward pass. Slower per call
        # but cuts per-call saved memory ~10×, enabling exp611-class deployment
        # where the plain autograd path OOMs from cross-layer activation
        # accumulation. No-op when autograd=False.
        hybrid_smooth_save_memory: bool = False,
        # backward_mode='hybrid_smooth' + autograd=True + save_memory=True only:
        # if True, use the exp611-style K-row soft surrogate for input gradient.
        # Forward and weight gradient are unchanged (still (n_alt+1)-row softmax
        # forward, (n_alt+1)-row weight scatter); only the input grad swaps from
        # self-consistent gather of (n_alt+1) anchor positions to denser
        # attribution across all K=2^NAP rows back to all NAP anchors. Tests
        # whether dense input grad helps at exp611-class scale.
        hybrid_smooth_dense_input_grad: bool = False,
        # backward_mode='hybrid_smooth' only (mutually exclusive with the
        # standard `save_memory` path). Forward computes full K-row softmax
        # sel_soft = softmax(ts/T_sel, dim=K), extracts the top-(n_alt+1) mass
        # values WITHOUT renormalising, and uses those as the row weights
        # (so probs no longer sum to 1 — they sum to mass(top-(n_alt+1))). This
        # makes forward and dense backward use the SAME K-row softmax: gradient
        # consistency by construction, no surrogate. Output magnitude attenuates
        # under uncertainty (kept mass < 1), giving built-in confidence gating.
        hybrid_smooth_unrenorm_forward: bool = False,
        n_alternatives: int = 1,
        # Multi-alt STE: u(d) = β / (T + |d|). T is a learnable temperature
        # (when `learnable_temps=True`) controlling the gradient breakpoint
        # — shapes the relative weighting across alts, which Adam can't
        # reproduce. β is hardcoded to 0.5 (matches the legacy multi-alt
        # default): it's a uniform multiplicative scale on x.grad and Adam's
        # per-parameter second-moment normalisation absorbs its effect.
        uncertainty_T_init: float = 1.0,
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
        if backward_mode not in ("ste", "soft", "soft_topk", "prob", "soft_winner", "hybrid_smooth"):
            raise ValueError(f"backward_mode must be 'ste', 'soft', 'soft_topk', 'prob', 'soft_winner', or 'hybrid_smooth', got {backward_mode!r}")
        self.backward_mode = backward_mode
        # Sentinel: hybrid_smooth_n_alt=-1 means "use n_anchor_pairs" (full ball).
        self.hybrid_smooth_n_alt = (
            int(n_anchor_pairs) if int(hybrid_smooth_n_alt) == -1
            else int(hybrid_smooth_n_alt)
        )
        self.hybrid_smooth_autograd = bool(hybrid_smooth_autograd)
        self.hybrid_smooth_save_memory = bool(hybrid_smooth_save_memory)
        self.hybrid_smooth_dense_input_grad = bool(hybrid_smooth_dense_input_grad)
        self.hybrid_smooth_unrenorm_forward = bool(hybrid_smooth_unrenorm_forward)
        if backward_mode == "hybrid_smooth":
            if self.hybrid_smooth_autograd:
                # Generic autograd path supports any n_alt in [1, NAP].
                if not (1 <= self.hybrid_smooth_n_alt <= n_anchor_pairs):
                    raise ValueError(
                        f"hybrid_smooth_n_alt must be in [1, NAP={n_anchor_pairs}]; "
                        f"got {self.hybrid_smooth_n_alt}"
                    )
            else:
                # Manual autograd.Function path: supports any n_alt in [1, NAP].
                # n_alt=1 uses the legacy `_TinyMHLutHybridSmooth` (sigmoid form);
                # n_alt=NAP uses `_TinyMHLutHybridSmoothNap` (full Hamming-1 ball);
                # 1 < n_alt < NAP uses the generalised `_TinyMHLutHybridSmoothKalt`.
                if not (1 <= self.hybrid_smooth_n_alt <= n_anchor_pairs):
                    raise ValueError(
                        f"hybrid_smooth_n_alt must be in [1, NAP={n_anchor_pairs}]; "
                        f"got {self.hybrid_smooth_n_alt}"
                    )
        self.use_bf16 = bool(use_bf16)
        self.argmax_noise_eps = float(argmax_noise_eps)
        self.einsum_bf16_forward = bool(einsum_bf16_forward)
        if self.einsum_bf16_forward:
            if backward_mode not in ("soft", "soft_topk", "prob", "soft_winner"):
                raise ValueError(
                    "einsum_bf16_forward=True requires backward_mode='soft' or 'soft_topk'; "
                    f"got backward_mode={backward_mode!r}"
                )
            if not self.use_bf16:
                raise ValueError(
                    "einsum_bf16_forward=True requires use_bf16=True (the "
                    "noise comes from bf16 rounding inside the einsum)."
                )
        self.n_alternatives = int(n_alternatives)
        if self.n_alternatives < 1:
            raise ValueError(f"n_alternatives must be >= 1, got {self.n_alternatives}")
        if backward_mode == "soft_topk":
            if self.n_alternatives < 1:
                raise ValueError("soft_topk requires n_alternatives >= 1")
            if self.n_alternatives > n_anchor_pairs:
                raise ValueError(
                    f"soft_topk: n_alternatives ({self.n_alternatives}) cannot exceed "
                    f"n_anchor_pairs ({n_anchor_pairs}); only 1-bit-flip neighbors are considered"
                )
        elif self.n_alternatives > 1:
            if backward_mode != "ste":
                raise ValueError(
                    "n_alternatives > 1 requires backward_mode='ste' or 'soft_topk' "
                    f"(got backward_mode={backward_mode!r}). For soft mode the "
                    "rational-soft-sign pipeline already smooths all NAP positions."
                )
            if sparse_scatter_n_outputs is not None:
                raise NotImplementedError(
                    "n_alternatives > 1 does not yet support sparse_scatter_n_outputs"
                )
            # partition_sets only affects init-time anchor-pair sampling (stored
            # as int16 in self.lookup.anchor_pairs_a/b); the multi-alt backward
            # gathers from these buffers without knowing or caring about
            # partition membership, so the combination is safe.
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

            # Uncertainty kernel: u(d) = β / (T + |d|), with β = 0.5 fixed
            # (hardcoded in `_multi_alt_bwd_body`) and T learnable when
            # `learnable_temps=True`. Adam's per-parameter second-moment
            # normalisation absorbs β's uniform-scale effect, so learning
            # it doesn't change effective dynamics — only T shapes the
            # gradient direction (relative weighting across alts).
            self.multialt_learnable_temps = bool(learnable_temps)
            log_T_init = math.log(float(uncertainty_T_init))
            if self.multialt_learnable_temps:
                self.log_uncertainty_T = nn.Parameter(
                    torch.tensor(log_T_init, dtype=torch.float32, device=dev)
                )
            else:
                self.register_buffer(
                    "log_uncertainty_T",
                    torch.tensor(log_T_init, dtype=torch.float32, device=dev),
                )
        if backward_mode in ("soft", "soft_topk", "prob", "soft_winner", "hybrid_smooth"):
            # sparse_scatter_n_outputs is supported: forward returns per-table
            # [B, H, tph, n_outputs] from `_TinyMHLutSoft.apply(..., sparse=True)`,
            # then `_scatter` reduces into the wider output dim. Backward
            # reshapes upstream `[B, H, tph, n_outputs]` (from scatter_add's
            # autograd) directly to `grad_pt` without the tph-broadcast.
            #
            # partition_sets only affects init-time anchor-pair sampling (stored
            # as int16 in self.lookup.anchor_pairs_a/b); the soft backward
            # gathers from these buffers without knowing or caring about
            # partition membership, so the combination is safe.
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
            # Precompute the slot-major inverse map for the native
            # sparse_scatter forward kernel (one-time at init). Eliminates
            # scatter_add_'s atomic-contention cost: ~2.7x forward speedup at
            # exp318 shapes. Buffers are tiny (a few MB total), persistent so
            # they're saved/loaded with checkpoints.
            if sparse_scatter_n_outputs is not None:
                so, ct, ci = _build_sparse_scatter_inverse_map(
                    self.scatter_indices, sparse_scatter_n_outputs,
                )
                self.register_buffer('_sparse_slot_offsets',    so)
                self.register_buffer('_sparse_contrib_table',   ct)
                self.register_buffer('_sparse_contrib_local_i', ci)

    def _soft_winner_forward(self, x: torch.Tensor) -> torch.Tensor:
        """backward_mode='soft_winner' forward: out = softmax_winner_coeff * W[winner]."""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}")
        return _TinyMHLutSoftWinner.apply(
            x, self.weights, self.log_soft_score_temp, self.log_select_temp,
            self.soft_anchor_a_long, self.soft_anchor_b_long,
            self.soft_bit_matrix,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
        )

    def _hybrid_smooth_forward(self, x: torch.Tensor) -> torch.Tensor:
        """backward_mode='hybrid_smooth':
          - hybrid_smooth_n_alt=1: top-2 softmax (main + Hamming-1 neighbor at argmin |d|).
          - hybrid_smooth_n_alt=NAP: full Hamming-1 ball, (NAP+1)-way softmax.
        Soft K-row input grad + (n_alt+1)-row weight scatter. No sparse_scatter / einsum_bf16."""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}")
        if self.sparse_scatter_n_outputs is not None:
            # sparse_scatter is currently supported only for the default manual
            # n_alt=1 path (the most common hybrid_smooth configuration).
            if self.hybrid_smooth_autograd or self.hybrid_smooth_n_alt != 1:
                raise NotImplementedError(
                    "hybrid_smooth sparse_scatter only supports the default "
                    "n_alt=1 manual path (hybrid_smooth_autograd=False)."
                )
            return _TinyMHLutHybridSmoothSparseScatter.apply(
                x, self.weights, self.log_soft_score_temp, self.log_select_temp,
                self.soft_anchor_a_long, self.soft_anchor_b_long,
                self.soft_bit_matrix, self.soft_powers,
                self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
                self.scatter_indices, self.sparse_scatter_n_outputs,
                getattr(self, '_sparse_slot_offsets', None),
                getattr(self, '_sparse_contrib_table', None),
                getattr(self, '_sparse_contrib_local_i', None),
            )
        # Autograd path: supports any n_alt in [1, NAP]. Self-consistent input
        # grad (chain rule through u and probs, no soft K-row surrogate).
        if self.hybrid_smooth_autograd:
            if self.hybrid_smooth_unrenorm_forward:
                # Pure-autograd full-K forward with top-(n_alt+1) mass extraction,
                # no renormalisation. Forward & backward share the same K-row
                # sel_soft → gradient consistency by construction. Output dtype
                # matches weights (fp32 by default) — consistent with memeff path.
                return _hybrid_smooth_unrenorm_forward_impl(
                    x, self.weights,
                    self.log_soft_score_temp, self.log_select_temp,
                    self.soft_anchor_a_long, self.soft_anchor_b_long,
                    self.soft_powers, self.soft_bit_matrix,
                    self.n_heads, self.tables_per_head, self.table_dim,
                    self.hybrid_smooth_n_alt,
                )
            if self.hybrid_smooth_save_memory:
                # Memory-efficient custom autograd.Function: stream forward,
                # re-gather in backward. Required at exp611-class scale where
                # plain autograd OOMs from cross-layer activation accumulation.
                return _TinyMHLutHybridSmoothMemEff.apply(
                    x, self.weights, self.log_soft_score_temp, self.log_select_temp,
                    self.soft_anchor_a_long, self.soft_anchor_b_long,
                    self.soft_powers, self.soft_bit_matrix,
                    self.n_heads, self.tables_per_head, self.table_dim,
                    self.hybrid_smooth_n_alt, self.use_bf16,
                    self.hybrid_smooth_dense_input_grad,
                )
            autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                            if self.use_bf16 and x.is_cuda
                            else torch.amp.autocast("cpu", enabled=False))
            with autocast_ctx:
                return _hybrid_smooth_kalt_fwd_autograd(
                    x, self.weights, self.log_soft_score_temp, self.log_select_temp,
                    self.soft_anchor_a_long, self.soft_anchor_b_long, self.soft_powers,
                    self.n_heads, self.tables_per_head, self.table_dim,
                    self.hybrid_smooth_n_alt,
                )
        # Manual autograd.Function paths.
        if self.hybrid_smooth_n_alt == 1:
            return _TinyMHLutHybridSmooth.apply(
                x, self.weights, self.log_soft_score_temp, self.log_select_temp,
                self.soft_anchor_a_long, self.soft_anchor_b_long,
                self.soft_bit_matrix, self.soft_powers,
                self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
            )
        # 1 < n_alt < NAP: use generalised manual path (K-row soft input grad,
        # (n_alt+1)-row smooth forward, (n_alt+1)-row weight scatter).
        if self.hybrid_smooth_n_alt < self.n_anchor_pairs:
            return _TinyMHLutHybridSmoothKalt.apply(
                x, self.weights, self.log_soft_score_temp, self.log_select_temp,
                self.soft_anchor_a_long, self.soft_anchor_b_long,
                self.soft_bit_matrix, self.soft_powers,
                self.n_heads, self.tables_per_head, self.table_dim,
                self.hybrid_smooth_n_alt, self.use_bf16,
            )
        # n_alt == NAP: full Hamming-1 ball, no topk needed.
        return _TinyMHLutHybridSmoothNap.apply(
            x, self.weights, self.log_soft_score_temp, self.log_select_temp,
            self.soft_anchor_a_long, self.soft_anchor_b_long,
            self.soft_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
        )

    def _prob_forward(self, x: torch.Tensor) -> torch.Tensor:
        """backward_mode='prob' forward: stochastic during train, argmax at eval."""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}")
        if not self.training:
            # Deterministic argmax — no autograd wrapper needed at eval.
            T_soft = self.log_soft_score_temp.exp()
            autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                            if self.use_bf16 and x.is_cuda
                            else torch.amp.autocast("cpu", enabled=False))
            with autocast_ctx:
                out, _ = _soft_lut_fwd_body_einsum(
                    x, self.weights, self.soft_anchor_a_long, self.soft_anchor_b_long,
                    self.soft_bit_matrix, T_soft,
                    self.n_heads, self.tables_per_head, self.table_dim,
                )
            return out
        return _TinyMHLutProb.apply(
            x, self.weights, self.log_soft_score_temp, self.log_select_temp,
            self.soft_anchor_a_long, self.soft_anchor_b_long,
            self.soft_bit_matrix,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
        )

    def _soft_forward(self, x: torch.Tensor) -> torch.Tensor:
        """soft backward_mode forward path. Forward output is identical to
        SoftMHLut(hard=True) on the same weights and anchor pairs (sign-pack
        argmax = soft argmax). Backward gives soft gradients to x and the
        temperatures, sparse one-hot scatter to weights.

        When `sparse_scatter_n_outputs` is configured, the autograd Function
        fuses the per-table gather + scatter_add into its compiled body and
        returns `[B, H, sparse_n_outputs]` directly — no external `_scatter`
        call, no exposed [B, H, tph, n_outputs] intermediate. Otherwise
        returns the embedding_bag-reduced `[B, H, n_outputs]`.
        """
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.sparse_scatter_n_outputs is not None:
            scatter_indices = self.scatter_indices
            sparse_n = self.sparse_scatter_n_outputs
            slot_offsets   = getattr(self, '_sparse_slot_offsets', None)
            contrib_table  = getattr(self, '_sparse_contrib_table', None)
            contrib_local_i = getattr(self, '_sparse_contrib_local_i', None)
        else:
            scatter_indices = None
            sparse_n = 0
            slot_offsets = contrib_table = contrib_local_i = None
        topk_n_alt = self.n_alternatives if self.backward_mode == "soft_topk" else 0
        return _TinyMHLutSoft.apply(
            x, self.weights, self.log_soft_score_temp, self.log_select_temp,
            self.soft_anchor_a_long, self.soft_anchor_b_long,
            self.soft_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
            self.argmax_noise_eps, self.einsum_bf16_forward,
            scatter_indices, sparse_n,
            slot_offsets, contrib_table, contrib_local_i,
            topk_n_alt,
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
        # `_soft_forward` internally fuses gather+scatter_add when
        # sparse_scatter_n_outputs is set; no external `_scatter` needed.
        # 'soft_topk' shares the same forward path; backward differs by
        # restricting the soft attribution to 1 + n_alternatives rows.
        if self.backward_mode in ("soft", "soft_topk"):
            return self._soft_forward(x)

        if self.backward_mode == "prob":
            return self._prob_forward(x)

        if self.backward_mode == "soft_winner":
            return self._soft_winner_forward(x)

        if self.backward_mode == "hybrid_smooth":
            return self._hybrid_smooth_forward(x)

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
                self.log_uncertainty_T,
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


