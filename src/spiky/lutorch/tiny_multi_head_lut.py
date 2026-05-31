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
    lookup_indices, log_uncertainty_T,
    n_heads, tables_per_head, n_alternatives,
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

    # x.grad via inverse-L1 uncertainty u(d) = 0.5 / (T + |d|).
    #   du/dd = -0.5 · sign(d) / (T + |d|)²
    # T = exp(log_uncertainty_T) is learnable (controls breakpoint, shapes
    # relative weighting across alts — Adam can't reproduce that).
    # β=0.5 is hardcoded: it's a uniform multiplicative scale on x.grad and
    # Adam's per-parameter second-moment normalisation absorbs its effect.
    T = log_uncertainty_T.exp()
    grad_diff = grad_main.unsqueeze(2) - grad_alt
    T_plus_abs = T + lookup_alt_deltas.abs()
    inv_denom_sq = 1.0 / (T_plus_abs * T_plus_abs)
    minus_uncertainty_derivative = 0.5 * lookup_alt_deltas.sign() * inv_denom_sq
    du = grad_diff * minus_uncertainty_derivative
    if n_alternatives > 1:
        du = du / n_alternatives

    # 2D scatter directly into [B, input_dim] — avoids the batch_offset add,
    # mirrors soft mode's pattern. anchor1/2_ids are [B, T, n_alt].
    x_grad = torch.zeros(B, input_dim, device=x.device, dtype=x.dtype)
    x_grad.scatter_add_(1, anchor1_ids.reshape(B, -1), du.reshape(B, -1))
    x_grad.scatter_add_(1, anchor2_ids.reshape(B, -1), -du.reshape(B, -1))

    # Gradient for log_uncertainty_T via the "imaginary smooth forward"
    # trick (same pattern soft mode uses for log_T_soft / log_T_sel):
    #   ∂L/∂T = (1/n_alt) · Σ (grad_alt - grad_main) · (-0.5)/(T+|d|)²
    #   grad_log_T = T · ∂L/∂T
    # In our code grad_diff = grad_main - grad_alt = -(grad_alt - grad_main),
    # giving the formula below.
    n_alt_div = float(n_alternatives) if n_alternatives > 1 else 1.0
    grad_log_uncertainty_T = (T * 0.5 / n_alt_div) * (grad_diff * inv_denom_sq).sum()

    return weights_grad_flat.view(weights.shape), x_grad, grad_log_uncertainty_T


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
    def forward(ctx, x, weights, log_uncertainty_T,
                anchor_a_long, anchor_b_long,
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

        ctx.save_for_backward(x, weights, lookup_indices, flip_mask,
                              log_uncertainty_T)
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
        (x, weights, lookup_indices, _flip_mask,
         log_uncertainty_T) = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tables_per_head

        if grad_out.dtype != weights.dtype:
            grad_out = grad_out.to(weights.dtype)

        # All alt_* computation and scatters fuse into a single @torch.compile
        # region (manual top-k via sequential argmin, fancy gather + mul + sum
        # as fused triplet, no [B, T, K] structured-bmm intermediate).
        weights_grad, x_grad, grad_log_T = _multi_alt_bwd_body(
            grad_out, weights, x, ctx.batch_offset,
            ctx.anchor_a_long, ctx.anchor_b_long,
            ctx.table_arange, ctx.table_flat,
            lookup_indices, log_uncertainty_T,
            n_heads, tph, ctx.n_alternatives,
        )

        # 14 forward inputs (x, weights, log_T, anchor_a, anchor_b,
        # powers_mult, table_arange, eb_offsets, n_alternatives, batch_offset,
        # table_flat, n_heads, tables_per_head, argmax_noise_eps) -> 14
        # grad returns. Only x, weights, log_T get real grads.
        return (x_grad, weights_grad, grad_log_T,
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
                        hard_sign_ste: bool = False):
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

    # dL/dweights via scatter at saved (table, index).
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
    """Fused 2-row weight gradient for hybrid_smooth backward.

    Inductor fuses the scale + reshape + index_add into a single kernel,
    avoiding the 3 large [B, n_tables, n_outputs] intermediates that the
    eager-mode version materialises.
    """
    flat_offset = torch.arange(n_tables, device=grad_pt.device, dtype=main_index.dtype) * K
    main_flat_idx = (main_index + flat_offset[None, :]).reshape(-1)
    alt_flat_idx  = (alt_index  + flat_offset[None, :]).reshape(-1)
    main_w_exp = (1.0 - u).unsqueeze(-1).to(w_dtype)
    u_exp = u.unsqueeze(-1).to(w_dtype)
    grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=w_dtype, device=grad_pt.device)
    grad_w_flat.index_add_(0, main_flat_idx, (grad_pt * main_w_exp).reshape(-1, n_outputs).to(w_dtype))
    grad_w_flat.index_add_(0, alt_flat_idx,  (grad_pt * u_exp).reshape(-1, n_outputs).to(w_dtype))
    return grad_w_flat.view(n_tables, K, n_outputs)


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
            )

        # Hybrid weight gradient: fused 2-row scatter at main/alt rows, scaled
        # by (1 - u) and u. @torch.compile fuses multiply + reshape + index_add.
        grad_weights = _hybrid_smooth_weight_grad(
            grad_pt, main_index, alt_index, u, n_tables, K, n_outputs, w_dtype,
        )

        # 12 forward inputs → 12 grad returns.
        return (grad_x, grad_weights, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None)


class _TinyMHLutHybridSmoothSparseScatter(torch.autograd.Function):
    """hybrid_smooth (n_alt=1) with sparse_scatter output.

    Forward: same 2-row smooth blend per table as `_TinyMHLutHybridSmooth`, but
    each table's contribution is scatter_added into a [B, n_heads, sparse_n_outputs]
    output via `scatter_indices` instead of summed across the tph axis. The
    per-table [B, n_heads, tph, n_outputs] intermediate stays inside the compiled
    body — Inductor fuses gather + blend + scatter_add.

    Backward:
      - grad_pt is reconstructed by GATHERING grad_out at scatter_indices
        (inverse of the forward scatter_add) → [B, n_tables, n_outputs].
      - Input / temperature gradients: dense K-row soft chain via
        `_soft_lut_bwd_body` (identical to non-sparse hybrid_smooth).
      - Weight gradient: 2-row scatter via `_hybrid_smooth_weight_grad`.
    """

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix, powers,
                n_heads, tph, table_dim, use_bf16,
                scatter_indices, sparse_n_outputs,
                slot_offsets=None, contrib_global_t=None, contrib_local_i=None):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        # Fast path: gather+segment_reduce via precomputed inverse map (no atomic
        # contention, deterministic). Falls back to atomic scatter_add when the
        # inverse-map buffers aren't supplied.
        use_segred = (slot_offsets is not None
                      and contrib_global_t is not None
                      and contrib_local_i is not None)
        with autocast_ctx:
            if use_segred:
                out, main_index, alt_index, u = _hybrid_smooth_lut_fwd_body_scatter_segred(
                    x, weights, anchor_a_long, anchor_b_long, powers,
                    T_soft, T_sel, n_heads, tph, table_dim,
                    slot_offsets, contrib_global_t, contrib_local_i, sparse_n_outputs,
                )
            else:
                out, main_index, alt_index, u = _hybrid_smooth_lut_fwd_body_scatter(
                    x, weights, anchor_a_long, anchor_b_long, powers,
                    T_soft, T_sel, n_heads, tph, table_dim,
                    scatter_indices, sparse_n_outputs,
                )
        ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, main_index, alt_index, u,
                              log_T_soft, log_T_sel, powers, scatter_indices)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, bit_matrix,
         main_index, alt_index, u,
         log_T_soft, log_T_sel, powers, scatter_indices) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = anchor_a_long.shape[0]
        n_outputs = weights.shape[2]
        K = bit_matrix.shape[1]
        w_dtype = weights.dtype

        # Gather grad_out at scatter_indices to reconstruct per-table grad_pt.
        idx = scatter_indices.unsqueeze(0).expand(B, -1, -1, -1).reshape(B, n_heads, tph * n_outputs)
        grad_pt = grad_out.gather(2, idx).reshape(B, n_tables, n_outputs)

        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            # Dense K-row soft chain for input + temperature grads (discard grad_w).
            grad_x, _grad_w_unused, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                main_index, T_soft, T_sel, ctx.n_heads, ctx.tph,
            )

        grad_weights = _hybrid_smooth_weight_grad(
            grad_pt, main_index, alt_index, u, n_tables, K, n_outputs, w_dtype,
        )

        # 17 forward inputs (x, weights, log_T_soft, log_T_sel, anchor_a_long,
        # anchor_b_long, bit_matrix, powers, n_heads, tph, table_dim, use_bf16,
        # scatter_indices, sparse_n_outputs, slot_offsets, contrib_global_t,
        # contrib_local_i) → 17 grad returns.
        return (grad_x, grad_weights, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None)


@torch.compile
def _hybrid_smooth_kalt_fwd_autograd(x, weights, log_T_soft, log_T_sel,
                                       anchor_a_long, anchor_b_long, powers,
                                       n_heads, tph, table_dim, n_alt):
    """Self-consistent forward for hybrid_smooth, supporting any n_alt ∈ [1, NAP].

    - n_alt = NAP: uses all NAP anchor positions (full Hamming-1 ball).
    - n_alt < NAP: picks the n_alt least-confident anchor positions via
      sequential argmin (one argmin + scatter-INF per k). Fuses much better
      under @torch.compile than torch.topk for small k, with no algorithmic
      change. n_alt=1 reduces to argmin |abs_p|.

    Backward via plain autograd: weight grad scatters at (n_alt+1) rows scaled by
    softmax probs; input grad propagates ONLY through smooth path (no soft K-row
    surrogate).
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    NAP = anchor_a_long.shape[1]
    T_soft = log_T_soft.exp()
    T_sel  = log_T_sel.exp()

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    powers_view = powers.view(1, 1, -1)
    main_index = (bits * powers_view).sum(dim=-1)                # [B, n_tables]

    abs_d = d.abs()
    abs_p = abs_d / (T_soft + abs_d)                              # [B, n_tables, NAP]

    if n_alt == NAP:
        delta_ts = 2.0 * abs_p                                    # [B, n_tables, NAP]
        flip_powers = powers.view(1, 1, -1).expand(B, n_tables, -1).to(main_index.dtype)
    else:
        # Sequential argmin: cheap for small n_alt, fuses under torch.compile.
        # `torch.topk` for k=3 of 6/8 is ~10x slower under Inductor than three
        # argmins because topk's quickselect kernel doesn't fuse with neighbours.
        INF = torch.finfo(abs_p.dtype).max
        abs_p_mask = abs_p
        pos_list = []
        for _k in range(n_alt):
            idx_k = abs_p_mask.argmin(dim=-1, keepdim=True)         # [B, n_tables, 1]
            pos_list.append(idx_k)
            abs_p_mask = abs_p_mask.scatter(-1, idx_k, INF)
        topk_pos = torch.cat(pos_list, dim=-1)                       # [B, n_tables, n_alt]
        delta_ts = 2.0 * abs_p.gather(-1, topk_pos)               # [B, n_tables, n_alt]
        flip_powers = powers.to(main_index.dtype)[topk_pos]       # [B, n_tables, n_alt]

    alt_indices = main_index.unsqueeze(-1) ^ flip_powers          # [B, n_tables, n_alt]

    logits_alts = -delta_ts / T_sel                               # [B, n_tables, n_alt]
    logits_main = torch.zeros_like(logits_alts[..., :1])          # [B, n_tables, 1]
    logits = torch.cat([logits_main, logits_alts], dim=-1)        # [B, n_tables, n_alt+1]
    probs = torch.softmax(logits, dim=-1)                          # [B, n_tables, n_alt+1]

    table_offset = torch.arange(n_tables, device=weights.device,
                                dtype=main_index.dtype) * table_dim
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    main_flat_idx = (main_index + table_offset.view(1, -1)).reshape(-1)
    main_rows = F.embedding(main_flat_idx, weights_flat).view(B, n_tables, n_outputs)
    out_per_table = main_rows * probs[..., 0:1]
    for k in range(n_alt):
        alt_flat_idx_k = (alt_indices[..., k] + table_offset.view(1, -1)).reshape(-1)
        alt_rows_k = F.embedding(alt_flat_idx_k, weights_flat).view(B, n_tables, n_outputs)
        out_per_table = out_per_table + alt_rows_k * probs[..., k + 1: k + 2]
    out = out_per_table.view(B, n_heads, tph, n_outputs).sum(dim=2)
    return out


@torch.compile
def _hybrid_smooth_nap_fwd_autograd(x, weights, log_T_soft, log_T_sel,
                                     anchor_a_long, anchor_b_long, powers,
                                     n_heads, tph, table_dim):
    """Self-consistent forward for hybrid_smooth with n_alt=NAP, using
    plain differentiable PyTorch ops. PyTorch autograd computes the full
    chain rule through softmax over (NAP+1) ball rows + per-anchor signed
    rational p, giving exact gradients of the actual forward to:
      - weights (scatter at main and alt rows, scaled by softmax probs)
      - x (chain rule through abs_p → delta_ts → probs)
      - log_T_soft, log_T_sel (through the temperatures in p and softmax)

    main_index / alt_indices are computed from `(d > 0)` (non-differentiable);
    autograd treats them as constants. Input gradient flows ONLY through the
    smooth (probs / abs_p) path — no soft K-row surrogate.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    NAP = anchor_a_long.shape[1]
    T_soft = log_T_soft.exp()
    T_sel = log_T_sel.exp()

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                # [B, n_tables, NAP]
    # Indices (non-diff)
    bits = (d > 0).to(torch.int64)
    powers_view = powers.view(1, 1, -1)
    main_index = (bits * powers_view).sum(dim=-1)                # [B, n_tables]
    alt_indices = main_index.unsqueeze(-1) ^ powers.view(1, 1, -1).to(main_index.dtype)

    # Soft probs (diff)
    abs_d = d.abs()
    abs_p = abs_d / (T_soft + abs_d)                              # [B, n_tables, NAP]
    delta_ts = 2.0 * abs_p                                        # [B, n_tables, NAP]
    logits_alts = -delta_ts / T_sel                               # [B, n_tables, NAP]
    logits_main = torch.zeros_like(logits_alts[..., :1])          # [B, n_tables, 1]
    logits = torch.cat([logits_main, logits_alts], dim=-1)        # [B, n_tables, NAP+1]
    probs = torch.softmax(logits, dim=-1)                          # [B, n_tables, NAP+1]

    # Gather + blend (iterative to bound memory)
    table_offset = torch.arange(n_tables, device=weights.device,
                                dtype=main_index.dtype) * table_dim
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    main_flat_idx = (main_index + table_offset.view(1, -1)).reshape(-1)
    main_rows = F.embedding(main_flat_idx, weights_flat).view(B, n_tables, n_outputs)
    out_per_table = main_rows * probs[..., 0:1]
    for k in range(NAP):
        alt_flat_idx_k = (alt_indices[..., k] + table_offset.view(1, -1)).reshape(-1)
        alt_rows_k = F.embedding(alt_flat_idx_k, weights_flat).view(B, n_tables, n_outputs)
        out_per_table = out_per_table + alt_rows_k * probs[..., k + 1: k + 2]
    out = out_per_table.view(B, n_heads, tph, n_outputs).sum(dim=2)
    return out


@torch.compile
def _hybrid_smooth_nap_weight_grad(grad_pt, main_index, alt_indices, probs,
                                    n_tables, K, n_outputs, w_dtype):
    """(NAP+1)-row weight scatter for hybrid_smooth with n_alternatives=NAP.

    Scatters grad_pt * probs[k] at main_index (k=0) and each alt_index_k (k=1..NAP).
    Inductor fuses the multiplications + index_add into a few kernels.
    """
    NAP = alt_indices.shape[-1]
    flat_offset = torch.arange(n_tables, device=grad_pt.device,
                               dtype=main_index.dtype) * K
    grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=w_dtype,
                               device=grad_pt.device)
    # Main: prob index 0
    main_flat_idx = (main_index + flat_offset[None, :]).reshape(-1)
    p_main = probs[..., 0:1].to(w_dtype)
    grad_w_flat.index_add_(0, main_flat_idx,
                            (grad_pt * p_main).reshape(-1, n_outputs).to(w_dtype))
    # NAP alts: prob indices 1..NAP
    for k in range(NAP):
        alt_flat_idx_k = (alt_indices[..., k] + flat_offset[None, :]).reshape(-1)
        p_alt_k = probs[..., k + 1: k + 2].to(w_dtype)
        grad_w_flat.index_add_(0, alt_flat_idx_k,
                                (grad_pt * p_alt_k).reshape(-1, n_outputs).to(w_dtype))
    return grad_w_flat.view(n_tables, K, n_outputs)


@torch.compile
def _hybrid_smooth_nap_lut_fwd_body(x, weights, anchor_a_long, anchor_b_long, powers,
                                      T_soft, T_sel, n_heads, tph, table_dim):
    """Smooth forward with n_alternatives=NAP: full Hamming-1 ball around main.

    For each of the NAP anchor positions, the corresponding alt row flips that one
    bit. Total (NAP+1) rows blended via exact (NAP+1)-way softmax over the row
    scores. Row scores are derived analytically from the per-anchor signed rational
    p[i] = sign(d[i]) * |d[i]|/(T_soft + |d[i]|); the score gap between main and
    alt_k is exactly 2*|p[k]|.

    Returns:
        out:          [B, n_heads, n_outputs]
        main_index:   [B, n_tables]
        alt_indices:  [B, n_tables, NAP] — main XOR powers[k]
        probs:        [B, n_tables, NAP+1] — softmax weights, probs[..., 0] = P(main)
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    NAP = anchor_a_long.shape[1]

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                  # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    powers_view = powers.view(1, 1, -1)
    main_index = (bits * powers_view).sum(dim=-1)                  # [B, n_tables]

    abs_d = d.abs()                                                # [B, n_tables, NAP]
    abs_p = abs_d / (T_soft + abs_d)                                # [B, n_tables, NAP]
    delta_ts = 2.0 * abs_p                                          # [B, n_tables, NAP]

    # (NAP+1)-way softmax: main logit = 0, alt_k logit = -delta_ts[k] / T_sel.
    logits_alts = -delta_ts / T_sel                                 # [B, n_tables, NAP]
    logits_main = torch.zeros_like(logits_alts[..., :1])            # [B, n_tables, 1]
    logits = torch.cat([logits_main, logits_alts], dim=-1)          # [B, n_tables, NAP+1]
    probs = torch.softmax(logits, dim=-1)                           # [B, n_tables, NAP+1]

    # alt_indices[b, t, k] = main_index[b, t] XOR powers[k].
    alt_indices = main_index.unsqueeze(-1) ^ powers.view(1, 1, -1).to(main_index.dtype)

    # Gather + blend, iteratively to keep memory bounded.
    table_offset = torch.arange(n_tables, device=weights.device,
                                dtype=main_index.dtype) * table_dim
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    main_flat_idx = (main_index + table_offset.view(1, -1)).reshape(-1)
    main_rows = F.embedding(main_flat_idx, weights_flat).view(B, n_tables, n_outputs)
    out_per_table = main_rows * probs[..., 0:1]                     # [B, n_tables, n_outputs]
    for k in range(NAP):
        alt_flat_idx_k = (alt_indices[..., k] + table_offset.view(1, -1)).reshape(-1)
        alt_rows_k = F.embedding(alt_flat_idx_k, weights_flat).view(B, n_tables, n_outputs)
        out_per_table = out_per_table + alt_rows_k * probs[..., k + 1: k + 2]
    out = out_per_table.view(B, n_heads, tph, n_outputs).sum(dim=2)
    return out, main_index, alt_indices, probs


class _TinyMHLutHybridSmoothNap(torch.autograd.Function):
    """Hybrid smooth with n_alternatives=NAP (full Hamming-1 ball).

    Forward: exact (NAP+1)-way softmax blend over main + NAP single-bit-flip
    alternatives. No topk needed — uses all NAP anchor positions.

    Backward:
      - Input/temperature gradients: soft K-row surrogate via _soft_lut_bwd_body
        (treats forward as if it picked one row from softmax over all K).
      - Weight gradient: (NAP+1)-row scatter, scaled by softmax probabilities,
        matching the forward's actual row participation.
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
            out, main_index, alt_indices, probs = _hybrid_smooth_nap_lut_fwd_body(
                x, weights, anchor_a_long, anchor_b_long, powers,
                T_soft, T_sel, n_heads, tph, table_dim,
            )
        ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, main_index, alt_indices, probs,
                              log_T_soft, log_T_sel, powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, bit_matrix,
         main_index, alt_indices, probs,
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
            grad_x, _grad_w_unused, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                main_index, T_soft, T_sel, ctx.n_heads, ctx.tph,
            )

        # (NAP+1)-row weight scatter, scaled by softmax probabilities.
        grad_weights = _hybrid_smooth_nap_weight_grad(
            grad_pt, main_index, alt_indices, probs, n_tables, K, n_outputs, w_dtype,
        )

        return (grad_x, grad_weights, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# Manual-path generalisation: hybrid_smooth with n_alt ∈ [1, NAP].
#
# Same structure as `_TinyMHLutHybridSmooth` (exp611's fast path) but generalised
# to (n_alt+1)-row softmax forward. Reuses the soft K-row input gradient via
# `_soft_lut_bwd_body` — that's where exp611's speed comes from — and adds an
# (n_alt+1)-row weight scatter for the weight gradient.
#
# Why this is fast vs the memeff path:
#   - One @torch.compile-fused forward body (Inductor sees everything)
#   - No B-chunking → no Python loop overhead, no kernel-launch multiplication
#   - The K-row soft input grad is a single einsum that Inductor fuses globally
#
# Memory cost at out_proj scale (B=8192, n_tables=1024, n_outputs=384, fp32):
#   - n_alt=1: 2 row tensors materialised at peak = 24 GB
#   - n_alt=3: 4 row tensors at peak = 48 GB  (fits H100 80GB)
#   - n_alt=NAP=6: 7 row tensors = 84 GB     (would NOT fit — use memeff instead)
#
# So this path is the right choice for n_alt ∈ [1, ~4] at exp611 scale; memeff
# is the right choice for larger n_alt.
# ---------------------------------------------------------------------------


@torch.compile
def _hybrid_smooth_lut_kalt_fwd_body(x, weights, anchor_a_long, anchor_b_long, powers,
                                       T_soft, T_sel, n_heads, tph, table_dim, n_alt):
    """Smooth forward with arbitrary n_alt: (n_alt+1)-row softmax blend.

    Returns: (out, main_index, alt_indices, probs).
      out:         [B, n_heads, n_outputs]
      main_index:  [B, n_tables]              (sign-packed)
      alt_indices: [B, n_tables, n_alt]        (main XOR flip_powers at top-n_alt positions)
      probs:       [B, n_tables, n_alt+1]      ((n_alt+1)-way softmax)
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    NAP = anchor_a_long.shape[1]

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    main_index = (bits * powers.view(1, 1, -1)).sum(dim=-1)      # [B, n_tables]

    abs_d = d.abs()
    abs_p = abs_d / (T_soft + abs_d)

    if n_alt == NAP:
        delta_ts = 2.0 * abs_p
        flip_powers = powers.view(1, 1, -1).expand(B, n_tables, -1).to(main_index.dtype)
    else:
        INF = torch.finfo(abs_p.dtype).max
        abs_p_mask = abs_p
        pos_list = []
        for _k in range(n_alt):
            idx_k = abs_p_mask.argmin(dim=-1, keepdim=True)
            pos_list.append(idx_k)
            abs_p_mask = abs_p_mask.scatter(-1, idx_k, INF)
        topk_pos = torch.cat(pos_list, dim=-1)
        delta_ts = 2.0 * abs_p.gather(-1, topk_pos)
        flip_powers = powers.to(main_index.dtype)[topk_pos]

    alt_indices = main_index.unsqueeze(-1) ^ flip_powers          # [B, n_tables, n_alt]

    logits_alts = -delta_ts / T_sel
    logits_main = torch.zeros_like(logits_alts[..., :1])
    logits = torch.cat([logits_main, logits_alts], dim=-1)
    probs = torch.softmax(logits, dim=-1)                          # [B, n_tables, n_alt+1]

    # Streaming FMA over all (n_alt+1) rows. In-place addcmul keeps peak live
    # rows == 2 (the temp `rows_k` and the accumulator). Inductor fuses the FMA.
    table_offset = torch.arange(n_tables, device=weights.device,
                                dtype=main_index.dtype) * table_dim
    weights_flat = weights.view(n_tables * table_dim, n_outputs)
    main_flat_idx = (main_index + table_offset.view(1, -1)).reshape(-1)
    main_rows = F.embedding(main_flat_idx, weights_flat).view(B, n_tables, n_outputs)
    blended = main_rows * probs[..., 0:1]
    for k in range(n_alt):
        alt_flat_idx_k = (alt_indices[..., k] + table_offset.view(1, -1)).reshape(-1)
        alt_rows_k = F.embedding(alt_flat_idx_k, weights_flat).view(B, n_tables, n_outputs)
        blended = blended + alt_rows_k * probs[..., k + 1: k + 2]
    out = blended.view(B, n_heads, tph, n_outputs).sum(dim=2)
    return out, main_index, alt_indices, probs


@torch.compile
def _hybrid_smooth_kalt_weight_grad(grad_pt, main_index, alt_indices, probs,
                                      n_tables, K, n_outputs, w_dtype):
    """(n_alt+1)-row weight scatter: dW[r, :] += sum_b probs[..., k] * grad_pt[b, t, :]
    for the chosen rows. Uses index_add per row; Inductor fuses multiply + scatter."""
    n_alt = alt_indices.shape[-1]
    flat_offset = torch.arange(n_tables, device=grad_pt.device,
                                dtype=main_index.dtype) * K
    grad_w_flat = torch.zeros(n_tables * K, n_outputs, dtype=w_dtype,
                                device=grad_pt.device)
    # Main row (prob index 0).
    main_flat_idx = (main_index + flat_offset[None, :]).reshape(-1)
    p_main = probs[..., 0:1].to(w_dtype)
    grad_w_flat.index_add_(0, main_flat_idx,
                             (grad_pt * p_main).reshape(-1, n_outputs).to(w_dtype))
    # Alt rows (prob indices 1..n_alt).
    for k in range(n_alt):
        alt_flat_idx_k = (alt_indices[..., k] + flat_offset[None, :]).reshape(-1)
        p_alt_k = probs[..., k + 1: k + 2].to(w_dtype)
        grad_w_flat.index_add_(0, alt_flat_idx_k,
                                 (grad_pt * p_alt_k).reshape(-1, n_outputs).to(w_dtype))
    return grad_w_flat.view(n_tables, K, n_outputs)


class _TinyMHLutHybridSmoothKalt(torch.autograd.Function):
    """Manual-path hybrid_smooth for arbitrary n_alt ∈ [1, NAP].

    Forward: (n_alt+1)-row softmax blend (smooth).
    Backward: K-row soft surrogate for input/T gradients (via `_soft_lut_bwd_body`,
    same as exp611). (n_alt+1)-row scatter for weight gradient.

    See module-level note above for memory/speed tradeoffs vs the memeff path.
    """

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix, powers,
                n_heads, tph, table_dim, n_alt, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            out, main_index, alt_indices, probs = _hybrid_smooth_lut_kalt_fwd_body(
                x, weights, anchor_a_long, anchor_b_long, powers,
                T_soft, T_sel, n_heads, tph, table_dim, n_alt,
            )
        ctx.save_for_backward(x, weights, anchor_a_long, anchor_b_long,
                              bit_matrix, main_index, alt_indices, probs,
                              log_T_soft, log_T_sel, powers)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.use_bf16 = use_bf16
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, anchor_a_long, anchor_b_long, bit_matrix,
         main_index, alt_indices, probs,
         log_T_soft, log_T_sel, powers) = ctx.saved_tensors
        T_soft = log_T_soft.exp()
        T_sel = log_T_sel.exp()
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = anchor_a_long.shape[0]
        n_outputs = weights.shape[2]
        K = bit_matrix.shape[1]
        w_dtype = weights.dtype

        grad_pt = (grad_out.unsqueeze(2)
                    .expand(B, n_heads, tph, n_outputs)
                    .reshape(B, n_tables, n_outputs))

        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            # K-row soft input/T grads (exp611's fast path).
            grad_x, _grad_w_unused, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                main_index, T_soft, T_sel, ctx.n_heads, ctx.tph,
            )

        # (n_alt+1)-row weight scatter, scaled by softmax probs.
        grad_weights = _hybrid_smooth_kalt_weight_grad(
            grad_pt, main_index, alt_indices, probs, n_tables, K, n_outputs, w_dtype,
        )

        # 13 forward inputs → 13 grad returns (None for non-tensor args).
        return (grad_x, grad_weights, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# Memory-efficient hybrid_smooth with n_alt < NAP.
#
# Same math as `_hybrid_smooth_kalt_fwd_autograd` (top-(n_alt+1) softmax over
# Hamming-1 ball at the n_alt least-confident anchor positions), but written
# as an autograd.Function that:
#   * Forward streams (gather → multiply by prob → accumulate → drop) per row,
#     so the peak live tensor count is ~2 row-shaped tensors instead of n_alt+1.
#   * Saves only compact tensors (indices, probs, d). The (n_alt+1) row gathers
#     are re-done in backward — trades a single gather pass for ~10× memory cut.
#   * Backward chunks over B so the materialised `grad_pt` (which is otherwise
#     B×n_tables×n_outputs ≈ 12 GB at exp611's out_proj) never lives in full.
#
# Targets exp611-class deployment where the simple autograd path saves ~48 GB
# per `out_proj` call × 24 calls per forward → OOM. Here per-call save drops to
# a few MB; cross-layer accumulation becomes tractable.
# ---------------------------------------------------------------------------

# Chunk size for forward and backward B-chunking. Tuned for exp611 out_proj scale:
#   B=8192, n_tables=1024, n_outputs=384, fp32 → 1.5 GB per row tensor per chunk.
# Smaller value → less memory, slightly more kernel overhead.
_HYBRID_SMOOTH_FWD_B_CHUNK = 1024
_HYBRID_SMOOTH_BWD_B_CHUNK = 1024


@torch.compile
def _hybrid_smooth_memeff_fwd_fused(
    x, weights, log_T_soft, log_T_sel,
    anchor_a_long, anchor_b_long, powers,
    n_heads, tph, table_dim, n_alt,
):
    """Fully fused forward: compute_probs + scatter + GEMM in ONE @torch.compile
    body. Inductor fuses the per-anchor rational, the sequential argmin loop,
    the softmax, the scatter, and the cuBLAS GEMM into a minimal kernel set —
    eliminates the Python chunking loop overhead from the per-chunk variant.

    Returns (out, d, main_index, alt_indices, probs, topk_pos) — d, main_index,
    alt_indices, probs are saved for backward; topk_pos is unused by dense path
    but returned for compatibility with the self-consistent path."""
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    NAP = anchor_a_long.shape[1]
    K = table_dim
    T_soft = log_T_soft.exp()
    T_sel = log_T_sel.exp()
    w_dtype = weights.dtype

    # ---- compute_probs ----
    d = x[:, anchor_a_long] - x[:, anchor_b_long]                # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    main_index = (bits * powers.view(1, 1, -1)).sum(dim=-1)      # [B, n_tables]

    abs_d = d.abs()
    abs_p = abs_d / (T_soft + abs_d)                              # [B, n_tables, NAP]

    if n_alt == NAP:
        delta_ts = 2.0 * abs_p
        flip_powers = powers.view(1, 1, -1).expand(B, n_tables, -1).to(main_index.dtype)
        topk_pos = main_index.unsqueeze(-1).expand(-1, -1, n_alt)  # placeholder (unused)
    else:
        INF = torch.finfo(abs_p.dtype).max
        abs_p_mask = abs_p
        pos_list = []
        for _k in range(n_alt):
            idx_k = abs_p_mask.argmin(dim=-1, keepdim=True)
            pos_list.append(idx_k)
            abs_p_mask = abs_p_mask.scatter(-1, idx_k, INF)
        topk_pos = torch.cat(pos_list, dim=-1)
        delta_ts = 2.0 * abs_p.gather(-1, topk_pos)
        flip_powers = powers.to(main_index.dtype)[topk_pos]

    alt_indices = main_index.unsqueeze(-1) ^ flip_powers          # [B, n_tables, n_alt]

    logits_alts = -delta_ts / T_sel
    logits_main = torch.zeros_like(logits_alts[..., :1])
    logits = torch.cat([logits_main, logits_alts], dim=-1)
    probs = torch.softmax(logits, dim=-1)                          # [B, n_tables, n_alt+1]

    # ---- scatter S + GEMM ----
    all_indices = torch.cat([main_index.unsqueeze(-1), alt_indices], dim=-1)
    S = torch.zeros(B, n_tables, K, dtype=w_dtype, device=weights.device)
    S.scatter_add_(-1, all_indices, probs.to(w_dtype))
    weights_view = weights.view(n_heads, tph, K, n_outputs)
    out = torch.einsum('bhtr,htro->bho',
                         S.view(B, n_heads, tph, K), weights_view)
    # Match chunked memeff's output dtype contract (= weights.dtype = fp32 in
    # production). Downstream layers handle fp32 input under autocast.
    out = out.to(w_dtype)

    return out, d, main_index, alt_indices, probs, topk_pos


@torch.compile
def _hybrid_smooth_memeff_fwd_chunk(weights_view, all_indices_chunk, probs_chunk,
                                      cb, n_heads, tph, table_dim, n_outputs, n_alt, w_dtype):
    """Per-chunk forward body via einsum.

    Builds S[b, t, r] = sum_k indicator(idx_k[b,t]==r) * probs[b,t,k] (selection
    mass on each row) via a collision-free scatter_add, then computes
    out[b, h, o] = sum_t sum_r S[b, t, r] * W[t, r, o] as a single batched GEMM.
    Same shape pattern as the dW einsum, so cuBLAS saturates the H100 — ~10×
    faster than the per-k gather + addcmul loop, with no per-row tensor allocated.
    """
    K = table_dim
    n_tables = n_heads * tph
    S = torch.zeros(cb, n_tables, K, dtype=w_dtype,
                    device=weights_view.device)
    S.scatter_add_(-1, all_indices_chunk, probs_chunk.to(w_dtype))
    return torch.einsum('bhtr,htro->bho',
                          S.view(cb, n_heads, tph, K), weights_view)


@torch.compile
def _hybrid_smooth_memeff_Z_chunk(weights_view, grad_out_chunk,
                                    cb, n_heads, tph, table_dim, n_outputs):
    """Per-chunk Z = W @ grad_out.

    Z[b, t, r] = sum_o W[t, r, o] * grad_out[b, h(t), o]. Single batched
    cuBLAS GEMM. Returned as [cb, n_tables, K] float32. Used by both the
    self-consistent input-grad path (gather n_alt+1 rows) and the K-row
    soft-surrogate path (chain through full K-row softmax)."""
    K = table_dim
    n_tables = n_heads * tph
    Z = torch.einsum('htro,bho->bhtr', weights_view,
                       grad_out_chunk.to(torch.float32))
    return Z.reshape(cb, n_tables, K)


@torch.compile
def _hybrid_smooth_memeff_dprobs_chunk(weights_view, all_indices_chunk,
                                         grad_out_chunk,
                                         cb, n_heads, tph, table_dim, n_outputs):
    """Self-consistent (n_alt+1)-row dprobs via Z + gather. See Z_chunk above."""
    Z = _hybrid_smooth_memeff_Z_chunk(
        weights_view, grad_out_chunk, cb, n_heads, tph, table_dim, n_outputs,
    )
    return Z.gather(-1, all_indices_chunk)


def _hybrid_smooth_memeff_drd_chunk_eager(weights_view, grad_out_chunk, p_signs_chunk,
                                            abs_p_chunk, T_sel,
                                            cb, n_heads, tph, table_dim, n_outputs,
                                            bit_matrix):
    """K-row soft-surrogate input gradient body (exp611 style), eager.

    Treats the forward as if it were the full K-row softmax pipeline:
      p[b,t,i] = p_signs[b,t,i] * abs_p[b,t,i]              (signed rational)
      ts[b,t,r] = sum_i p[b,t,i] * bit_matrix[i,r]            (row scores)
      sel_soft = softmax(ts / T_sel)
      out_implicit = sum_r sel_soft[r] * W[t, r, :]
    Backward via plain softmax algebra against Z = W @ grad_out:
      d_z   = sel_soft * (Z - (Z*sel_soft).sum(-1, keepdim=True))
      d_ts  = d_z / T_sel
      d_p   = einsum("btk,ik->bti", d_ts, bit_matrix)
    Returns (d_p_chunk, dlog_T_sel_contrib). dtype-agnostic; downstream callers
    handle bf16 autocast as needed.
    """
    K = table_dim
    n_tables = n_heads * tph
    # Native-dtype Z; don't force fp32 here so fp64 gradcheck works.
    Z = torch.einsum('htro,bho->bhtr', weights_view, grad_out_chunk).reshape(
        cb, n_tables, K)
    p = p_signs_chunk * abs_p_chunk                          # [cb, n_tables, NAP]
    ts = torch.einsum('bti,ir->btr', p, bit_matrix)          # [cb, n_tables, K]
    z = ts / T_sel
    sel_soft = torch.softmax(z, dim=-1)
    sum_term = (Z * sel_soft).sum(-1, keepdim=True)
    d_z = sel_soft * (Z - sum_term)                          # [cb, n_tables, K]
    d_ts = d_z / T_sel
    d_p = torch.einsum('btr,ir->bti', d_ts, bit_matrix)      # [cb, n_tables, NAP]
    dlog_T_sel_contrib = -(d_z * z).sum()
    return d_p, dlog_T_sel_contrib


# Compiled fast path: used for fp32 / bf16 production. Compile traps on mixed
# fp32/fp64 dtypes (the test path), so we keep the eager body as fallback.
_hybrid_smooth_memeff_drd_chunk_compiled = torch.compile(
    _hybrid_smooth_memeff_drd_chunk_eager
)


def _hybrid_smooth_memeff_drd_chunk(weights_view, grad_out_chunk, p_signs_chunk,
                                      abs_p_chunk, T_sel,
                                      cb, n_heads, tph, table_dim, n_outputs,
                                      bit_matrix):
    """Dispatcher: compiled path for fp32/bf16, eager for fp64 (gradcheck)."""
    if weights_view.dtype == torch.float64 or bit_matrix.dtype == torch.float64:
        return _hybrid_smooth_memeff_drd_chunk_eager(
            weights_view, grad_out_chunk, p_signs_chunk, abs_p_chunk, T_sel,
            cb, n_heads, tph, table_dim, n_outputs, bit_matrix,
        )
    return _hybrid_smooth_memeff_drd_chunk_compiled(
        weights_view, grad_out_chunk, p_signs_chunk, abs_p_chunk, T_sel,
        cb, n_heads, tph, table_dim, n_outputs, bit_matrix,
    )


@torch.compile
def _hybrid_smooth_memeff_dw_einsum(all_indices_chunk, probs_chunk, grad_out_chunk,
                                      cb, n_heads, tph, table_dim, n_outputs):
    """dW via einsum: rephrase weight gradient as a per-table batched GEMM.

    Key idea: introduce S[b, t, r] = sum_k indicator(idx_k[b,t]==r) * probs[b,t,k]
    — the "selection mass" each row receives from the (n_alt+1) softmax weights.
    Then dW[t, r, o] = sum_b S[b, t, r] * grad_out[b, h(t), o] is a batched GEMM,
    handled by cuBLAS in a single fused kernel.

    Avoids materialising grad_pt (n_tables × n_outputs broadcast) entirely; only
    S (B × n_tables × table_dim) plus the small dW output are allocated.
    Returns dW_chunk: [n_heads*tph, table_dim, n_outputs] in float32.
    """
    K = table_dim
    n_tables = n_heads * tph
    # S construction: (n_alt+1) writes per (b, t), each to a distinct k slot —
    # no collision so scatter_add is collision-free (cheap).
    S = torch.zeros(cb, n_tables, K, dtype=torch.float32,
                    device=all_indices_chunk.device)
    S.scatter_add_(-1, all_indices_chunk, probs_chunk.to(torch.float32))
    # Per-table batched GEMM: einsum saturates cuBLAS.
    S_view = S.view(cb, n_heads, tph, K)
    dW_chunk = torch.einsum('bhtr,bho->htro', S_view,
                              grad_out_chunk.to(torch.float32))
    return dW_chunk.reshape(n_tables, K, n_outputs)


def _hybrid_smooth_memeff_bwd_chunk(weights_view, all_indices_chunk, probs_chunk,
                                     grad_out_chunk,
                                     cb, n_tables, n_outputs, n_alt,
                                     n_heads, tph, table_dim):
    """Per-chunk backward body, both parts via einsum tricks:
      1. dprobs via Z = W @ grad_out + gather (~10 ms full B at exp611 scale)
      2. dW    via S (scatter) + S @ grad_out (~10 ms full B at exp611 scale)
    Neither materialises grad_pt or per-row gathered tensors.
    Returns (dprobs_chunk, dW_chunk).
    """
    dprobs_chunk = _hybrid_smooth_memeff_dprobs_chunk(
        weights_view, all_indices_chunk, grad_out_chunk,
        cb, n_heads, tph, table_dim, n_outputs,
    )
    dW_chunk = _hybrid_smooth_memeff_dw_einsum(
        all_indices_chunk, probs_chunk, grad_out_chunk,
        cb, n_heads, tph, table_dim, n_outputs,
    )
    return dprobs_chunk, dW_chunk


@torch.compile
def _hybrid_smooth_memeff_bwd_dense_fused(
    x, weights, log_T_soft, log_T_sel,
    anchor_a_long, anchor_b_long, bit_matrix,
    d, main_index, all_indices, probs, grad_out,
    n_heads, tph, table_dim,
):
    """Fully fused dense-input-grad backward for memeff.

    Computes (dx, dW, dlog_T_sel, dlog_T_soft) inside ONE @torch.compile body so
    Inductor can fuse the K-row softmax chain rule, Z GEMM, dW GEMM, and dx
    scatter into a minimal set of kernels (~1-2 instead of the ~5-10 you get
    from the chunked path's separate compiled bodies).

    No B-chunking — peak memory is ~5 GB at exp611 out_proj scale (Z + S
    tensors), well within the 80 GB H100 budget.

    Mathematically identical to the chunked dense path
    (`_hybrid_smooth_memeff_drd_chunk` + `_hybrid_smooth_memeff_dw_einsum`).
    """
    B = x.shape[0]
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    NAP = anchor_a_long.shape[1]
    K = table_dim
    T_soft = log_T_soft.exp()
    T_sel = log_T_sel.exp()
    weights_view = weights.view(n_heads, tph, K, n_outputs)
    grad_out_f = grad_out.to(torch.float32)

    # Z = einsum(W, grad_out): full K-row row-by-row dot with grad_out.
    Z = torch.einsum('htro,bho->bhtr', weights_view, grad_out_f).reshape(
        B, n_tables, K)

    # dW via S + einsum (same shape pattern; cuBLAS GEMM).
    S = torch.zeros(B, n_tables, K, dtype=torch.float32, device=weights.device)
    S.scatter_add_(-1, all_indices, probs.to(torch.float32))
    S_view = S.view(B, n_heads, tph, K)
    dW = torch.einsum('bhtr,bho->htro', S_view, grad_out_f).reshape(
        n_tables, K, n_outputs)

    # K-row soft surrogate chain to dx and dlog_T_sel.
    abs_d = d.abs()
    denom = T_soft + abs_d
    abs_p = abs_d / denom
    # p_signs from main_index bits (MSB-first).
    shifts = torch.arange(NAP - 1, -1, -1, device=x.device, dtype=main_index.dtype)
    bits_main = ((main_index.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 1).to(d.dtype)
    p_signs = bits_main * 2.0 - 1.0
    p = p_signs * abs_p                                          # [B, n_tables, NAP]
    ts = torch.einsum('bti,ir->btr', p, bit_matrix)               # [B, n_tables, K]
    z = ts / T_sel
    sel_soft = torch.softmax(z, dim=-1)
    sum_term = (Z * sel_soft).sum(-1, keepdim=True)
    d_z = sel_soft * (Z - sum_term)
    d_ts = d_z / T_sel
    d_p = torch.einsum('btr,ir->bti', d_ts, bit_matrix)            # [B, n_tables, NAP]
    dlog_T_sel = -(d_z * z).sum()

    dabs_p = d_p * p_signs
    dabs_d = dabs_p * (T_soft / (denom * denom))
    dd = dabs_d * torch.sign(d)
    # dx scatter via index_add at anchor positions.
    dx = torch.zeros_like(x)
    a_flat = anchor_a_long.reshape(-1)
    b_flat = anchor_b_long.reshape(-1)
    dd_flat = dd.reshape(B, -1)
    dx.index_add_(1, a_flat, dd_flat)
    dx.index_add_(1, b_flat, -dd_flat)
    dlog_T_soft = (-dabs_p * abs_d / (denom * denom)).sum() * T_soft

    return dx, dW.to(weights.dtype), dlog_T_soft, dlog_T_sel


def _hybrid_smooth_compute_probs(x, log_T_soft, log_T_sel,
                                  anchor_a_long, anchor_b_long, powers, n_alt):
    """Common front-half of forward: produce main_index, alt_indices, probs, d.
    Pure tensor ops; no row gathers. Used by both fwd (to drive the streamed
    accumulation) and bwd (to chain gradients through probs/d/temps)."""
    B, _ = x.shape
    NAP = anchor_a_long.shape[1]
    T_soft = log_T_soft.exp()
    T_sel = log_T_sel.exp()

    d = x[:, anchor_a_long] - x[:, anchor_b_long]                # [B, n_tables, NAP]
    bits = (d > 0).to(torch.int64)
    main_index = (bits * powers.view(1, 1, -1)).sum(dim=-1)      # [B, n_tables]

    abs_d = d.abs()
    abs_p = abs_d / (T_soft + abs_d)                              # [B, n_tables, NAP]

    if n_alt == NAP:
        delta_ts = 2.0 * abs_p
        flip_powers = powers.view(1, 1, -1).expand(B, anchor_a_long.shape[0], -1).to(main_index.dtype)
        topk_pos = None
    else:
        # Sequential argmin (matches `_hybrid_smooth_kalt_fwd_autograd`).
        INF = torch.finfo(abs_p.dtype).max
        abs_p_mask = abs_p
        pos_list = []
        for _k in range(n_alt):
            idx_k = abs_p_mask.argmin(dim=-1, keepdim=True)
            pos_list.append(idx_k)
            abs_p_mask = abs_p_mask.scatter(-1, idx_k, INF)
        topk_pos = torch.cat(pos_list, dim=-1)
        delta_ts = 2.0 * abs_p.gather(-1, topk_pos)
        flip_powers = powers.to(main_index.dtype)[topk_pos]

    alt_indices = main_index.unsqueeze(-1) ^ flip_powers          # [B, n_tables, n_alt]

    logits_alts = -delta_ts / T_sel
    logits_main = torch.zeros_like(logits_alts[..., :1])
    logits = torch.cat([logits_main, logits_alts], dim=-1)
    probs = torch.softmax(logits, dim=-1)                          # [B, n_tables, n_alt+1]
    return d, main_index, alt_indices, probs, topk_pos


class _TinyMHLutHybridSmoothMemEff(torch.autograd.Function):
    """Memory-efficient hybrid_smooth, n_alt ∈ [1, NAP]. See module-level note above."""

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, powers, bit_matrix,
                n_heads, tph, table_dim, n_alt, use_bf16, dense_input_grad):
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            # One fully fused forward body (no Python chunk loop).
            out, d, main_index, alt_indices, probs, topk_pos = (
                _hybrid_smooth_memeff_fwd_fused(
                    x, weights, log_T_soft, log_T_sel,
                    anchor_a_long, anchor_b_long, powers,
                    n_heads, tph, table_dim, n_alt,
                )
            )

        # Save only compact tensors. Row tensors are NOT saved — recomputed in backward.
        ctx.save_for_backward(x, weights, log_T_soft, log_T_sel,
                              anchor_a_long, anchor_b_long, powers, bit_matrix,
                              d, main_index, alt_indices, probs, topk_pos)
        ctx.n_heads = n_heads
        ctx.tph = tph
        ctx.table_dim = table_dim
        ctx.n_alt = n_alt
        ctx.use_bf16 = use_bf16
        ctx.dense_input_grad = bool(dense_input_grad)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weights, log_T_soft, log_T_sel,
         anchor_a_long, anchor_b_long, powers, bit_matrix,
         d, main_index, alt_indices, probs, topk_pos) = ctx.saved_tensors
        n_heads = ctx.n_heads
        tph = ctx.tph
        table_dim = ctx.table_dim
        n_alt = ctx.n_alt
        dense_input_grad = ctx.dense_input_grad

        B = x.shape[0]
        n_tables = anchor_a_long.shape[0]
        n_outputs = weights.shape[2]
        NAP = anchor_a_long.shape[1]
        T_soft = log_T_soft.exp()
        T_sel = log_T_sel.exp()

        weights_view = weights.view(n_heads, tph, table_dim, n_outputs)
        all_indices = torch.cat([main_index.unsqueeze(-1), alt_indices], dim=-1)

        # Fast path: dense_input_grad uses one fully-fused @torch.compile body
        # so Inductor fuses K-row softmax + Z GEMM + dW GEMM + dx scatter into
        # ~1-2 kernels. Roughly 2× faster than the chunked path at exp611 scale.
        if dense_input_grad:
            grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel = (
                _hybrid_smooth_memeff_bwd_dense_fused(
                    x, weights, log_T_soft, log_T_sel,
                    anchor_a_long, anchor_b_long,
                    bit_matrix.to(torch.float32),
                    d, main_index, all_indices, probs, grad_out,
                    n_heads, tph, table_dim,
                )
            )
            return (grad_x, grad_weights, grad_log_T_soft, grad_log_T_sel,
                    None, None, None, None, None, None, None, None, None, None)

        abs_d = d.abs()
        denom = T_soft + abs_d
        abs_p = abs_d / denom                                         # [B, n_tables, NAP]

        grad_weights_flat = None
        chunk = _HYBRID_SMOOTH_BWD_B_CHUNK

        if False:  # legacy dense path (kept for fallback / comparison)
            # K-row soft-surrogate input grad: gradient flows back through ALL
            # K rows weighted by sel_soft (full softmax over K), and via
            # bit_matrix back to ALL NAP anchor positions. Forward and weight
            # grad unchanged. p_signs is derived from main_index's bits
            # (consistent with the forward's chosen row).
            shifts = torch.arange(NAP - 1, -1, -1, device=x.device, dtype=main_index.dtype)
            bits_main = ((main_index.unsqueeze(-1) >> shifts.view(1, 1, -1)) & 1).to(d.dtype)
            p_signs = bits_main * 2.0 - 1.0                           # ±1 [B, n_tables, NAP]

            # Compute dtype for d_p: fp32 in standard production (bf16 autocast
            # tolerates fp32 grad), but match weights' precision in fp64 unit-tests.
            grad_dtype = torch.float32 if x.dtype == torch.float32 else x.dtype
            d_p = torch.zeros(B, n_tables, NAP, dtype=grad_dtype, device=x.device)
            dlog_T_sel = torch.zeros((), dtype=grad_dtype, device=x.device)

            bit_matrix_f = bit_matrix.to(grad_dtype)
            for cs in range(0, B, chunk):
                ce = min(B, cs + chunk)
                cb = ce - cs
                d_p_chunk, dlog_T_sel_contrib = _hybrid_smooth_memeff_drd_chunk(
                    weights_view, grad_out[cs:ce],
                    p_signs[cs:ce], abs_p[cs:ce], T_sel,
                    cb, n_heads, tph, table_dim, n_outputs, bit_matrix_f,
                )
                d_p[cs:ce] = d_p_chunk
                dlog_T_sel = dlog_T_sel + dlog_T_sel_contrib
                # dW unchanged (uses the forward's actual selection mass S).
                dW_chunk = _hybrid_smooth_memeff_dw_einsum(
                    all_indices[cs:ce], probs[cs:ce], grad_out[cs:ce],
                    cb, n_heads, tph, table_dim, n_outputs,
                )
                grad_weights_flat = (dW_chunk if grad_weights_flat is None
                                      else grad_weights_flat.add_(dW_chunk))

            # dabs_p = d_p * p_signs (because p = p_signs * abs_p, p_signs constant).
            dabs_p = d_p * p_signs                                    # [B, n_tables, NAP]
        else:
            dprobs = torch.empty(B, n_tables, n_alt + 1,
                                  dtype=torch.float32, device=x.device)
            for cs in range(0, B, chunk):
                ce = min(B, cs + chunk)
                cb = ce - cs
                dprobs[cs:ce], dW_chunk = _hybrid_smooth_memeff_bwd_chunk(
                    weights_view, all_indices[cs:ce], probs[cs:ce],
                    grad_out[cs:ce],
                    cb, n_tables, n_outputs, n_alt, n_heads, tph, table_dim,
                )
                grad_weights_flat = (dW_chunk if grad_weights_flat is None
                                      else grad_weights_flat.add_(dW_chunk))

            # Chain self-consistent path: dprobs → dlogits → ddelta_ts → dabs_p.
            sum_term = (dprobs * probs).sum(-1, keepdim=True)
            dlogits = probs * (dprobs - sum_term)
            ddelta_ts = -dlogits[..., 1:] / T_sel
            dabs_p_topk = 2.0 * ddelta_ts

            if n_alt == NAP:
                dabs_p = dabs_p_topk
                abs_p_used = abs_p
            else:
                dabs_p = torch.zeros_like(abs_p)
                dabs_p.scatter_(-1, topk_pos, dabs_p_topk)
                abs_p_used = abs_p.gather(-1, topk_pos)

            delta_ts_recomp = 2.0 * abs_p_used
            dlog_T_sel = (dlogits[..., 1:] * delta_ts_recomp / T_sel).sum()

        grad_weights = grad_weights_flat.view_as(weights).to(weights.dtype)

        # ---- Common chain: dabs_p → dabs_d → dd → dx; dlog_T_soft. ----
        dabs_d = dabs_p * (T_soft / (denom * denom))                  # [B, n_tables, NAP]
        dd = dabs_d * torch.sign(d)

        dx = torch.zeros_like(x)
        a_flat = anchor_a_long.reshape(-1)
        b_flat = anchor_b_long.reshape(-1)
        dd_flat = dd.reshape(B, -1)
        dx.index_add_(1, a_flat, dd_flat)
        dx.index_add_(1, b_flat, -dd_flat)

        dlog_T_soft = (-dabs_p * abs_d / (denom * denom)).sum() * T_soft

        return (dx, grad_weights, dlog_T_soft, dlog_T_sel,
                None, None, None, None, None, None, None, None, None, None)


@torch.compile
def _hybrid_smooth_unrenorm_forward_impl(x, weights, log_T_soft, log_T_sel,
                                           anchor_a_long, anchor_b_long, powers,
                                           bit_matrix, n_heads, tph, table_dim, n_alt):
    """Unrenormalised full-K forward for hybrid_smooth.

    Computes the same K-row softmax as the dense backward
    (`sel_soft = softmax(ts/T_sel, dim=K)`), extracts the top-(n_alt+1) mass
    values without renormalisation, and uses them as row weights. Forward and
    backward share `sel_soft` → autograd produces an exact, self-consistent
    backward through the K-row softmax. No custom autograd.Function needed.

    Probs sum to mass(top-(n_alt+1)) ≤ 1 — output magnitude attenuates under
    uncertainty (built-in confidence gating).

    Memory: forward saves `ts` and `sel_soft` ([B, n_tables, K] each, ~MB at
    tiny scale, ~GB at exp611 scale). Intended for tiny-scale experiments
    first; if it transfers, a memory-efficient custom autograd.Function variant
    can be added.
    """
    B, _ = x.shape
    n_tables = anchor_a_long.shape[0]
    n_outputs = weights.shape[2]
    NAP = anchor_a_long.shape[1]
    K = table_dim
    T_soft = log_T_soft.exp()
    T_sel = log_T_sel.exp()

    # Per-anchor signed soft score: p = sign(d) * |d|/(T_soft+|d|) = d/(T_soft+|d|).
    d = x[:, anchor_a_long] - x[:, anchor_b_long]                # [B, n_tables, NAP]
    abs_d = d.abs()
    p = d / (T_soft + abs_d)                                      # signed rational

    # Main row index from sign bits (MSB-first), used only as an integer index.
    bits_main = (d > 0).to(torch.int64)
    main_index = (bits_main * powers.view(1, 1, -1)).sum(dim=-1)  # [B, n_tables]

    # Full K-row scores and softmax (identical to dense backward's path).
    ts = torch.einsum('bti,ir->btr', p, bit_matrix.to(p.dtype))   # [B, n_tables, K]
    sel_soft = torch.softmax(ts / T_sel, dim=-1)                  # [B, n_tables, K]

    # Pick top-(n_alt) alt positions: least-confident anchor positions (smallest
    # |p|), same sequential argmin trick as elsewhere. n_alt=NAP shortcuts.
    abs_p = abs_d / (T_soft + abs_d)
    if n_alt == NAP:
        flip_powers = powers.view(1, 1, -1).expand(B, n_tables, -1).to(main_index.dtype)
    else:
        INF = torch.finfo(abs_p.dtype).max
        abs_p_mask = abs_p
        pos_list = []
        for _k in range(n_alt):
            idx_k = abs_p_mask.argmin(dim=-1, keepdim=True)
            pos_list.append(idx_k)
            abs_p_mask = abs_p_mask.scatter(-1, idx_k, INF)
        topk_pos = torch.cat(pos_list, dim=-1)
        flip_powers = powers.to(main_index.dtype)[topk_pos]
    alt_indices = main_index.unsqueeze(-1) ^ flip_powers          # [B, n_tables, n_alt]
    all_indices = torch.cat([main_index.unsqueeze(-1), alt_indices], dim=-1)

    # Gather top-(n_alt+1) masses from the FULL K-softmax — no renormalisation.
    mass_kept = sel_soft.gather(-1, all_indices)                   # [B, n_tables, n_alt+1]

    # Scatter into K-dim (other rows = 0), then per-table batched GEMM.
    S = torch.zeros(B, n_tables, K, dtype=weights.dtype,
                    device=weights.device)
    S.scatter_add_(-1, all_indices, mass_kept.to(weights.dtype))
    weights_view = weights.view(n_heads, tph, K, n_outputs)
    return torch.einsum('bhtr,htro->bho',
                          S.view(B, n_heads, tph, K), weights_view)


class _TinyMHLutProb(torch.autograd.Function):
    """Probabilistic forward + STE backward.

    Forward: sample one row from softmax(ts/T_sel) per (batch, table) —
    unlike _TinyMHLutSoft (argmax), every row with non-trivial probability
    gets selected occasionally, increasing row gradient coverage at small batch.
    Backward: identical to _TinyMHLutSoft — _soft_lut_bwd_body with sampled idx.
    """

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix,
                n_heads, tph, table_dim, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            out, index = _soft_lut_fwd_body_prob(
                x, weights, anchor_a_long, anchor_b_long,
                bit_matrix, T_soft, T_sel, n_heads, tph, table_dim,
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
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = anchor_a_long.shape[0]
        n_outputs = weights.shape[2]
        grad_pt = (grad_out.unsqueeze(2)
                           .expand(B, n_heads, tph, n_outputs)
                           .reshape(B, n_tables, n_outputs))
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            grad_x, grad_w, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body_prob(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                index, T_soft, T_sel, n_heads, tph,
            )
        # 11 inputs → 11 grads
        return (grad_x, grad_w, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None)


class _TinyMHLutSoftWinner(torch.autograd.Function):
    """Scaled-hard forward (out = coeff_winner * W[winner]) + matching backward.

    Deterministic forward (argmax + softmax-winner-coeff), so train==eval.
    Single-row inference; the coeff is a differentiable confidence gate.
    """

    @staticmethod
    def forward(ctx, x, weights, log_T_soft, log_T_sel,
                anchor_a_long, anchor_b_long, bit_matrix,
                n_heads, tph, table_dim, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            out, index = _soft_lut_fwd_body_winner(
                x, weights, anchor_a_long, anchor_b_long,
                bit_matrix, T_soft, T_sel, n_heads, tph, table_dim,
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
        B = x.shape[0]
        n_heads = ctx.n_heads
        tph = ctx.tph
        n_tables = anchor_a_long.shape[0]
        n_outputs = weights.shape[2]
        grad_pt = (grad_out.unsqueeze(2)
                           .expand(B, n_heads, tph, n_outputs)
                           .reshape(B, n_tables, n_outputs))
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if ctx.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            grad_x, grad_w, grad_log_Ts, grad_log_Tx = _soft_lut_bwd_body_winner(
                grad_pt, x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                index, T_soft, T_sel, n_heads, tph,
            )
        # 11 inputs → 11 grads
        return (grad_x, grad_w, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None, None)


# =====================================================================
# TinyOrderedMultiHeadLut: ordering-table + weight-table separation.
#
# Architecture motivation: when anchor pairs form connected groups
# (triplets, quadruplets), some bit-patterns are unreachable due to
# transitivity (e.g. a>b>c>a is a cycle). Standard TinyMultiHeadLut
# wastes those rows. This module separates:
#
#   ordering_table [n_tables, 2^n_pairs]: bit-pattern → weight-row index.
#     Unreachable patterns → -1 (→ -inf in soft routing → zero sel).
#   weight_table [n_tables, max_orders, n_outputs]: actual stored values.
#     max_orders = n_reachable_orderings_per_table ≤ 2^n_pairs.
#
# The ordering_table is built AUTOMATICALLY from the sampled anchor pairs
# at init time: for each table, all 2^NAP bit-patterns are checked for
# acyclicity; reachable ones are numbered 0,1,... and unreachable ones
# get -1. For disjoint anchor pairs (the common case with
# CANONICAL_FULL_COVERAGE and large input_dim) all patterns are
# reachable, max_orders = 2^NAP, and the weight_table equals TinyMHLut's
# weight layout exactly. Connected anchor groups (triplets, quadruplets)
# yield max_orders < 2^NAP.
#
# STE-style: hard argmax forward, hard index_add_ weight gradient, soft
# input gradient through all reachable patterns (matches TinyMultiHeadLut
# backward_mode='soft' but restricted to the reachable subset).
# =====================================================================


def _build_ordering_table(
    anchor_a: torch.Tensor,   # [n_tables, NAP] int64
    anchor_b: torch.Tensor,   # [n_tables, NAP] int64
) -> "tuple[torch.Tensor, int, torch.Tensor]":
    """Derive ordering_table, max_orders, and reachable_bit_matrix from anchor pairs.

    For each table, enumerates all 2^NAP bit-patterns (MSB-first, matching
    _msb_powers) and tests whether the corresponding directed comparison
    graph is acyclic (= the pattern is realizable by some real-valued
    assignment to the anchor dimensions).

    Shortcut: if the undirected version of the graph is a forest
    (no shared anchor indices → no transitivity constraints), all 2^NAP
    patterns are reachable and the identity mapping is returned for that
    table in O(NAP) time.

    Returns:
        ordering_table       [n_tables, 2^NAP] int64  — reachable → [0, n_reach),
                             unreachable → -1.
        max_orders           int — max n_reach across all tables.
        reachable_bit_matrix [NAP, max_orders] float32 — sign matrix (+1/-1, MSB-first)
                             for the reachable patterns of table 0, ordered by their
                             weight-row index j=0..max_orders-1.  When all tables
                             share the same reachable set (CONNECTED_TRIPLETS,
                             CANONICAL_FULL_COVERAGE), this covers every table.
    """
    from collections import defaultdict, deque

    n_tables, NAP = anchor_a.shape
    K = 1 << NAP
    a_cpu = anchor_a.cpu().tolist()
    b_cpu = anchor_b.cpu().tolist()

    def _forest(pairs):
        """True iff the undirected graph on `pairs` is a forest (no cycle / self-loop)."""
        parent = {}
        def find(x):
            parent.setdefault(x, x)
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for u, v in pairs:
            if u == v:
                return False
            ru, rv = find(u), find(v)
            if ru == rv:
                return False
            parent[ru] = rv
        return True

    def _acyclic(dedges):
        """True iff the directed graph is acyclic (Kahn toposort)."""
        nodes = set(x for e in dedges for x in e)
        indeg = {n: 0 for n in nodes}
        adj = defaultdict(list)
        for u, v in dedges:
            adj[u].append(v)
            indeg[v] += 1
        q = deque(n for n in nodes if indeg[n] == 0)
        seen = 0
        while q:
            u = q.popleft(); seen += 1
            for w in adj[u]:
                indeg[w] -= 1
                if indeg[w] == 0:
                    q.append(w)
        return seen == len(nodes)

    ordering_table = torch.empty(n_tables, K, dtype=torch.long)
    max_orders = 0

    for t in range(n_tables):
        pairs = list(zip(a_cpu[t], b_cpu[t]))
        if _forest(pairs):
            for k in range(K):
                ordering_table[t, k] = k
            max_orders = max(max_orders, K)
        else:
            order_idx = 0
            for k in range(K):
                dedges = [
                    (a_cpu[t][i], b_cpu[t][i]) if (k >> (NAP - 1 - i)) & 1
                    else (b_cpu[t][i], a_cpu[t][i])
                    for i in range(NAP)
                ]
                if _acyclic(dedges):
                    ordering_table[t, k] = order_idx
                    order_idx += 1
                else:
                    ordering_table[t, k] = -1
            max_orders = max(max_orders, order_idx)

    # Build reachable_bit_matrix [NAP, max_orders] from table 0's reachable patterns.
    # Column j = sign pattern (+1/-1, MSB-first) for the pattern mapped to weight row j.
    tbl0 = ordering_table[0]                    # [K]
    pattern_for_row = [-1] * max_orders
    for k in range(K):
        j = tbl0[k].item()
        if j >= 0:
            pattern_for_row[j] = k
    reachable_bit_matrix = torch.zeros(NAP, max_orders, dtype=torch.float32)
    for j, k in enumerate(pattern_for_row):
        if k >= 0:
            for i in range(NAP):
                bit = (k >> (NAP - 1 - i)) & 1
                reachable_bit_matrix[i, j] = 2.0 * bit - 1.0

    return ordering_table, max_orders, reachable_bit_matrix


# ---------------------------------------------------------------------------
# Soft-forward path: output = softmax(ts/T_sel) @ weight_table.
# Used by TinyOrderedMultiHeadLut(soft_forward=True) for temperature annealing.
# ---------------------------------------------------------------------------

@torch.compile
def _ordered_soft_fwd_body(x, weight_table, anchor_a, anchor_b,
                            reachable_bit_matrix, T_soft, T_sel, n_heads, tph):
    B = x.shape[0]
    n_outputs = weight_table.shape[2]
    d = x[:, anchor_a] - x[:, anchor_b]
    p = d / (T_soft + d.abs())
    ts = torch.einsum("btp,pm->btm", p, reachable_bit_matrix.to(p.dtype))
    sel = F.softmax(ts / T_sel, dim=-1)
    out_pt = torch.einsum("btm,tmo->bto", sel.to(weight_table.dtype), weight_table)
    return out_pt.view(B, n_heads, tph, n_outputs).sum(dim=2)


def _ordered_soft_bwd_body(grad_out, x, weight_table, anchor_a, anchor_b,
                            reachable_bit_matrix, log_T_soft, log_T_sel, n_heads, tph):
    T_soft = log_T_soft.exp()
    T_sel  = log_T_sel.exp()
    B = x.shape[0]
    input_dim = x.shape[1]
    n_tables, max_orders, n_outputs = weight_table.shape
    w_dtype = weight_table.dtype

    d = x[:, anchor_a] - x[:, anchor_b]
    p = d / (T_soft + d.abs())
    ts = torch.einsum("btp,pm->btm", p, reachable_bit_matrix.to(p.dtype))
    z  = ts / T_sel
    sel = F.softmax(z, dim=-1)

    grad_pt = (grad_out.unsqueeze(2)
               .expand(B, n_heads, tph, n_outputs)
               .reshape(B, n_tables, n_outputs)
               .to(w_dtype))

    d_weight_table = torch.einsum("btm,bto->tmo", sel.to(w_dtype), grad_pt)

    d_sel_soft = torch.einsum("bto,tmo->btm", grad_pt, weight_table)
    sum_term   = (d_sel_soft * sel).sum(-1, keepdim=True)
    d_z        = sel * (d_sel_soft - sum_term)
    d_ts       = d_z / T_sel
    grad_log_T_sel  = -(d_z * z).sum()

    d_p = torch.einsum("btm,pm->btp", d_ts.to(p.dtype), reachable_bit_matrix.to(p.dtype))
    d_d = d_p * T_soft / (T_soft + d.abs()) ** 2
    grad_log_T_soft = -(d_d * d).sum()

    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a  = anchor_a.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b  = anchor_b.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat = d_d.reshape(B, -1).to(x.dtype)
    grad_x.scatter_add_(1, idx_a,  d_flat)
    grad_x.scatter_add_(1, idx_b, -d_flat)

    return grad_x, d_weight_table, grad_log_T_soft, grad_log_T_sel


class _TinyOrderedMHLutSoft(torch.autograd.Function):
    """Soft-forward variant: output = softmax(ts/T_sel) @ weight_table."""

    @staticmethod
    def forward(ctx, x, weight_table, log_T_soft, log_T_sel,
                anchor_a, anchor_b, reachable_bit_matrix,
                n_heads, tph, use_bf16):
        T_soft = log_T_soft.exp()
        T_sel  = log_T_sel.exp()
        ac = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
              if use_bf16 and x.is_cuda
              else torch.amp.autocast("cpu", enabled=False))
        with ac:
            out = _ordered_soft_fwd_body(
                x, weight_table, anchor_a, anchor_b,
                reachable_bit_matrix, T_soft, T_sel, n_heads, tph,
            )
        ctx.save_for_backward(x, weight_table, anchor_a, anchor_b,
                               reachable_bit_matrix, log_T_soft, log_T_sel)
        ctx.n_heads = n_heads
        ctx.tph     = tph
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weight_table, anchor_a, anchor_b,
         reachable_bit_matrix, log_T_soft, log_T_sel) = ctx.saved_tensors
        grad_x, d_wt, gs, gx = _ordered_soft_bwd_body(
            grad_out, x, weight_table, anchor_a, anchor_b,
            reachable_bit_matrix, log_T_soft, log_T_sel, ctx.n_heads, ctx.tph,
        )
        return (grad_x, d_wt, gs, gx, None, None, None, None, None, None)


# ---------------------------------------------------------------------------
# Hard-forward (STE) path — default.
# ---------------------------------------------------------------------------

@torch.compile
def _ordered_fwd_body(x, weight_table, anchor_a, anchor_b,
                       reachable_bit_matrix, T_soft, n_heads, tph):
    """Ordered hard-LUT forward — argmax over reachable patterns, gather one row.

    reachable_bit_matrix: [NAP, max_orders] float32
    weight_table:         [n_tables, max_orders, n_outputs]
    Returns: [B, H, O] output, [B, n_tables] argmax indices.
    T_sel is NOT needed — argmax(ts) == argmax(ts/T_sel) for any T_sel > 0.
    """
    B = x.shape[0]
    n_tables, max_orders, n_outputs = weight_table.shape
    d = x[:, anchor_a] - x[:, anchor_b]                                         # [B, T, NAP]
    p = d / (T_soft + d.abs())
    ts = torch.einsum("btp,pm->btm", p, reachable_bit_matrix.to(p.dtype))       # [B, T, M]
    idx = ts.argmax(dim=-1)                                                      # [B, T]
    flat_offset = torch.arange(n_tables, device=idx.device, dtype=idx.dtype) * max_orders
    flat_idx = (idx + flat_offset.unsqueeze(0)).reshape(-1)                      # [B*T]
    out_pt = F.embedding(flat_idx, weight_table.reshape(-1, n_outputs).float()
                         ).to(weight_table.dtype).view(B, n_tables, n_outputs)
    return out_pt.view(B, n_heads, tph, n_outputs).sum(dim=2), idx              # [B, H, O], [B, T]


def _ordered_bwd_body(grad_out, x, weight_table, anchor_a, anchor_b,
                       reachable_bit_matrix, log_T_soft, log_T_sel, idx, n_heads, tph):
    """Ordered LUT backward — hard weight gradient + soft input gradient (STE-style).

    Weight gradient: index_add_ at the argmax row (idx) only.
    Input gradient: reconstruct p_signs from reachable_bit_matrix[:, idx], run
    full softmax backward through all max_orders rows (same as TinyMultiHeadLut
    backward_mode='soft' but with reachable_bit_matrix instead of bit_matrix).
    """
    T_soft = log_T_soft.exp()
    T_sel = log_T_sel.exp()
    B = x.shape[0]
    input_dim = x.shape[1]
    n_tables, max_orders, n_outputs = weight_table.shape
    w_dtype = weight_table.dtype

    # Per-table upstream gradient [B, n_tables, n_outputs]
    grad_pt = (grad_out.unsqueeze(2)
               .expand(B, n_heads, tph, n_outputs)
               .reshape(B, n_tables, n_outputs)
               .to(w_dtype))

    # --- Hard weight gradient: index_add_ at selected row only ---
    flat_offset = torch.arange(n_tables, device=idx.device, dtype=idx.dtype) * max_orders
    flat_idx = (idx + flat_offset.unsqueeze(0)).reshape(-1)                      # [B*T]
    grad_w_flat = torch.zeros(n_tables * max_orders, n_outputs, dtype=w_dtype, device=weight_table.device)
    grad_w_flat.index_add_(0, flat_idx, grad_pt.reshape(-1, n_outputs))
    d_weight_table = grad_w_flat.view(n_tables, max_orders, n_outputs)

    # --- Soft input gradient (STE): reconstruct p_signs from saved idx ---
    # reachable_bit_matrix.T: [max_orders, NAP]; idx: [B, n_tables]
    # p_signs[b,t,:] = reachable_bit_matrix[:, idx[b,t]] = rbm.T[idx[b,t]]
    p_signs = reachable_bit_matrix.T[idx].to(x.dtype)                           # [B, T, NAP]
    d = x[:, anchor_a] - x[:, anchor_b]
    denom = T_soft + d.abs()
    p = p_signs * d.abs() / denom

    ts = torch.einsum("btp,pm->btm", p, reachable_bit_matrix.to(p.dtype))       # [B, T, M]
    z = ts / T_sel
    sel_soft = F.softmax(z, dim=-1)                                              # [B, T, M]

    d_sel_soft = torch.einsum("bto,tmo->btm", grad_pt.to(p.dtype), weight_table.to(p.dtype))
    sum_term = (d_sel_soft * sel_soft).sum(-1, keepdim=True)
    d_z = sel_soft * (d_sel_soft - sum_term)
    d_ts = d_z / T_sel
    grad_log_T_sel = -(d_z * z).sum()

    # dp/dd = p_signs * sign(d) * T_soft / denom^2  (same algebra as TinyMHLut soft bwd)
    d_p = torch.einsum("btm,pm->btp", d_ts, reachable_bit_matrix.to(d_ts.dtype))
    d_d = d_p * p_signs * d.sign() * (T_soft / (denom * denom))
    grad_log_T_soft = -(d_d * d).sum()

    grad_x = torch.zeros(B, input_dim, dtype=x.dtype, device=x.device)
    idx_a = anchor_a.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    idx_b = anchor_b.unsqueeze(0).expand(B, -1, -1).reshape(B, -1)
    d_flat = d_d.reshape(B, -1)
    grad_x.scatter_add_(1, idx_a,  d_flat)
    grad_x.scatter_add_(1, idx_b, -d_flat)

    return grad_x, d_weight_table, grad_log_T_soft, grad_log_T_sel


class _TinyOrderedMHLut(torch.autograd.Function):
    """Autograd wrapper for the ordered hard-LUT forward / STE backward.

    Forward: hard argmax over reachable orderings → gather one weight row.
    Backward: hard index_add_ for weight gradient; soft STE for input gradient.

    Forward inputs (10):
      x, weight_table, log_T_soft, log_T_sel,
      anchor_a, anchor_b, reachable_bit_matrix,
      n_heads(int), tph(int), use_bf16(bool)
    """

    @staticmethod
    def forward(ctx, x, weight_table, log_T_soft, log_T_sel,
                anchor_a, anchor_b, reachable_bit_matrix,
                n_heads, tph, use_bf16):
        T_soft = log_T_soft.exp()
        out, idx = _ordered_fwd_body(
            x, weight_table, anchor_a, anchor_b,
            reachable_bit_matrix, T_soft, n_heads, tph,
        )
        ctx.save_for_backward(x, weight_table, anchor_a, anchor_b,
                               reachable_bit_matrix, log_T_soft, log_T_sel, idx)
        ctx.n_heads = n_heads
        ctx.tph = tph
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x, weight_table, anchor_a, anchor_b,
         reachable_bit_matrix, log_T_soft, log_T_sel, idx) = ctx.saved_tensors
        grad_x, d_wt, grad_log_Ts, grad_log_Tx = _ordered_bwd_body(
            grad_out, x, weight_table, anchor_a, anchor_b,
            reachable_bit_matrix, log_T_soft, log_T_sel, idx, ctx.n_heads, ctx.tph,
        )
        # 10 forward inputs → 10 backward outputs.
        return (grad_x, d_wt, grad_log_Ts, grad_log_Tx,
                None, None, None, None, None, None)


class TinyOrderedMultiHeadLut(nn.Module):
    """Multi-head LUT with ordering-table + weight-table two-level lookup.

    Drop-in replacement for TinyMultiHeadLut(backward_mode='soft') with
    automatic reachability detection. Forward selects one weight row via
    hard argmax over reachable orderings; backward uses STE: hard index_add_
    for weight gradients (only the selected row), soft input gradient through
    all max_orders reachable patterns via reachable_bit_matrix.

    Constructor args are identical to TinyMultiHeadLut except:
      - backward_mode is always STE-style (no parameter)
      - argmax_noise_eps is not supported (omitted)
      - einsum_bf16_forward / n_alternatives are not supported (omitted)

    After init, inspect:
      self.max_orders           — weight_table rows per table
      self.ordering_table       — [n_tables, 2^NAP] int, -1 = unreachable
      self.reachable_bit_matrix — [NAP, max_orders] float32 sign matrix
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
        soft_score_temp: float = 0.5,
        select_temp: float = 0.5,
        learnable_temps: bool = False,
        use_bf16: bool = True,
        soft_forward: bool = False,
        max_anchor_distance: Optional[int] = None,
        local_window_starts: str = "linspace",
    ):
        super().__init__()
        if not (1 <= n_anchor_pairs <= 15):
            raise ValueError(f"n_anchor_pairs must be 1..15, got {n_anchor_pairs}")
        if input_dim > 32767:
            raise ValueError(f"input_dim must be <= 32767, got {input_dim}")

        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.tables_per_head = tables_per_head
        self.table_dim = 1 << n_anchor_pairs
        self.weight_dtype = weight_dtype
        self.soft_forward = bool(soft_forward)
        n_lookup_tables = n_heads * tables_per_head
        self.n_lookup_tables = n_lookup_tables

        dev = device or torch.device("cpu")

        # 1. Sample anchor pairs (identical to TinyMultiHeadLut).
        self.lookup = TinyAnchorPairsLookup(
            input_dim=input_dim,
            n_tables=n_lookup_tables,
            n_anchor_pairs=n_anchor_pairs,
            n_heads=n_heads,
            random_seed=random_seed,
            device=dev,
            partition_sets=partition_sets,
            partition_pair_weights=partition_pair_weights,
            anchor_sampling_policy=anchor_sampling_policy,
            max_anchor_distance=max_anchor_distance,
            local_window_starts=local_window_starts,
        )
        anchor_a_long = self.lookup.anchor_pairs_a.long().contiguous()
        anchor_b_long = self.lookup.anchor_pairs_b.long().contiguous()
        self.register_buffer('soft_anchor_a_long', anchor_a_long)
        self.register_buffer('soft_anchor_b_long', anchor_b_long)
        # 2. Auto-detect reachable orderings from sampled pairs.
        _ord_tbl, _max_orders, _rbm = _build_ordering_table(anchor_a_long, anchor_b_long)
        self.max_orders = _max_orders
        self.register_buffer('ordering_table',       _ord_tbl.to(dev))
        self.register_buffer('reachable_bit_matrix', _rbm.to(dev))

        # 3. Weight table: [n_tables, max_orders, n_outputs].
        rng_kwargs: dict = {"device": dev}
        if random_seed is not None:
            rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed + 1)
        wt_init = (
            (torch.rand(n_lookup_tables, _max_orders, n_outputs, **rng_kwargs) - 0.5)
            * (2.0 * initial_weights_noise)
        ).to(weight_dtype)
        self.weight_table = nn.Parameter(wt_init)

        # 4. Temperature scalars.
        self.use_bf16 = bool(use_bf16)
        self.learnable_temps = bool(learnable_temps)
        if self.learnable_temps:
            self.log_soft_score_temp = nn.Parameter(
                torch.tensor(math.log(float(soft_score_temp)), dtype=torch.float32, device=dev)
            )
            self.log_select_temp = nn.Parameter(
                torch.tensor(math.log(float(select_temp)), dtype=torch.float32, device=dev)
            )
        else:
            self.register_buffer(
                'log_soft_score_temp',
                torch.tensor(math.log(float(soft_score_temp)), dtype=torch.float32, device=dev),
            )
            self.register_buffer(
                'log_select_temp',
                torch.tensor(math.log(float(select_temp)), dtype=torch.float32, device=dev),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, input_dim]
        returns: [B, n_heads, n_outputs]
        """
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}")
        fn = _TinyOrderedMHLutSoft if self.soft_forward else _TinyOrderedMHLut
        return fn.apply(
            x, self.weight_table,
            self.log_soft_score_temp, self.log_select_temp,
            self.soft_anchor_a_long, self.soft_anchor_b_long,
            self.reachable_bit_matrix,
            self.n_heads, self.tables_per_head,
            self.use_bf16,
        )


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


# =====================================================================
# MatmulMultiHeadLut: fully-differentiable dense-matmul LUT (no STE).
#
# Same front end as TinyMultiHeadLut (anchor pairs, rational soft-sign,
# multiply by the ±1 bit-matrix to get per-row match scores `ts`), but the
# routing is NOT argmax/softmax. Instead each of the 2^NAP rows gets an
# INDEPENDENT rational gate in [0,1]:
#     c[k] = 0.5 * (1 + ts[k] / (T_sel + |ts[k]|))
# (the [0,1] analogue of the [-1,1] soft sign — matching row -> ~1,
#  anti-matching -> ~0), followed by a dense matmul against the
# [2^NAP, n_outputs] weight table:
#     out = c @ W  (summed over tables_per_head).
#
# Pure PyTorch + @torch.compile, NO custom autograd / NO STE. Every weight
# receives a gradient from every token (DENSE gradients) — eliminates the
# per-row gradient-sparsity bottleneck of the hard-routing LUTs, at the cost
# of a K×n_outputs matmul per table at both train and inference (not
# matmul-free). Upper-bound / exploration variant.
# =====================================================================


@torch.compile
def _matmul_mhlut_fwd_body(x, weights, anchor_a_long, anchor_b_long, bit_matrix,
                            T_soft, T_sel, n_heads, tph, gate_kind,
                            ln_weight, ln_bias, ham_weight, relu_bias, bias,
                            hard_sign_ste=0, gate_power=2.0):
    """x: [B, input_dim]; weights: [n_tables, K=2^NAP, n_out]; bit_matrix: [NAP, K] ±1.
    gate_kind: 0 unit `0.5*(1+ts/(T_sel+|ts|))` in (0,1);
               1 signed `ts/(T_sel+|ts|)` in (-1,1);
               2 layernorm over the K match-scores (+affine ln_weight/ln_bias);
               3 hamming -- multiply the soft score by a learnable per-hamming-shell
                 weight: c[k] = ts[k] * ham_weight[h_k], h_k = hamming(row_k, sign(p))
                 = (NAP - ts_hard[k])/2. No gate/LN/temperature. ham_weight is [NAP+1].
    Optional output bias [n_heads, n_out]. Returns [B, n_heads, n_out]."""
    B = x.shape[0]
    n_tables, K, n_out = weights.shape
    NAP = bit_matrix.shape[0]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]               # [B, n_tables, NAP]
    p = d / (T_soft + d.abs())                                   # [B, n_tables, NAP] in (-1,1) soft sign
    if hard_sign_ste:
        # STE: hard ±1 sign on forward, soft-sign gradient on backward. Makes ts an
        # exact integer Hamming score (softmax -> exact exp-Hamming kernel) at fwd,
        # while gradients still flow through the (-1,1) soft sign p.
        p_hard = torch.where(d > 0, 1.0, -1.0).to(p.dtype)
        p = p + (p_hard - p).detach()                            # fwd=p_hard, bwd grad via p
    ts = torch.einsum("btp,pk->btk", p, bit_matrix.to(p.dtype))  # [B, n_tables, K] match scores
    if gate_kind == 2:
        g = F.layer_norm(ts, (K,), weight=ln_weight, bias=ln_bias)
    elif gate_kind == 3:
        hard = torch.where(d > 0, 1.0, -1.0).to(p.dtype)        # hard sign per bit [B,nt,NAP]
        ts_hard = torch.einsum("btp,pk->btk", hard, bit_matrix.to(p.dtype))  # integer-valued
        h = ((NAP - ts_hard) * 0.5).round().long().clamp(0, NAP)  # [B,nt,K] hamming shell index
        g = ts * ham_weight.to(ts.dtype)[h]                      # soft score * learnable shell weight
    elif gate_kind == 4:
        g = F.softmax(ts / T_sel, dim=-1)                       # normalized exp hamming kernel (the real thing)
    elif gate_kind == 5:
        g = F.relu(ts + relu_bias)                             # thresholded sparse gate (unnormalized; common-mode)
    elif gate_kind == 6:
        r = F.relu(ts + relu_bias)                            # sparsemax-style: sparse AND normalized
        g = r / (r.sum(dim=-1, keepdim=True) + 1e-6)         # Σ=1 fixes common-mode; ReLU keeps it sparse/hardenable
    elif gate_kind == 7:
        r = F.gelu(ts + relu_bias)                           # smooth GELU: nonzero gradient everywhere (no ReLU dead rows)
        g = r / (r.sum(dim=-1, keepdim=True) + 1e-6)         # normalized -> no common-mode; dense-gradient like softmax
    elif gate_kind == 8:
        g = F.gelu(ts + relu_bias)                           # UNNORMALIZED GELU; signed -> mid rows cancel common-mode
    elif gate_kind == 9:
        g0 = ts / (2.0 * NAP) + 0.5                          # map ts in [-NAP,NAP] -> [0,1] (fraction of matching bits)
        g = g0 ** gate_power                                 # power -> polynomial Hamming-similarity kernel (unnormalized)
    elif gate_kind == 10:
        g0 = ts / (2.0 * NAP) + 0.5                          # map ts in [-NAP,NAP] -> [0,1]
        r = g0 ** gate_power                                 # power k (k=2 square, larger k = sharper)
        g = r / (r.sum(dim=-1, keepdim=True) + 1e-6)        # NORMALIZED (Σ=1) polynomial Hamming kernel
    else:
        g = ts / (T_sel + ts.abs())                             # (-1,1) signed
        if gate_kind == 0:
            g = 0.5 * (1.0 + g)                                 # (0,1) unit
    out_pt = torch.einsum("btk,tko->bto", g.to(weights.dtype), weights)  # [B, n_tables, n_out]
    out = out_pt.view(B, n_heads, tph, n_out).sum(dim=2)        # [B, n_heads, n_out]
    if bias is not None:
        out = out + bias
    return out


class MatmulMultiHeadLut(nn.Module):
    """Dense-matmul, fully-differentiable LUT (no STE/argmax/softmax).

    Drop-in for TinyMultiHeadLut(backward_mode='soft'): identical constructor
    surface and anchor sampling (same random_seed -> identical anchor pairs), so
    forking exp475 swaps only the routing+aggregation. See module-level comment.
    """

    def __init__(self, input_dim, n_heads, n_outputs, n_anchor_pairs, tables_per_head,
                 random_seed=None, device=None, weight_dtype=torch.float32,
                 anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
                 initial_weights_noise=0.001, soft_score_temp=0.5, select_temp=0.5,
                 learnable_temps=False, use_bf16=True,
                 gate_mode="unit", use_bias=False, hard_sign_ste=False, gate_power=2.0,
                 partition_sets=None, partition_pair_weights=None,
                 max_anchor_distance=None, local_window_starts="linspace", **_ignored):
        super().__init__()
        if not (1 <= n_anchor_pairs <= 15):
            raise ValueError(f"n_anchor_pairs must be in [1,15], got {n_anchor_pairs}")
        if gate_mode not in ("unit", "signed", "layernorm", "hamming", "softmax", "relu", "relu_norm", "gelu_norm", "gelu", "square", "square_norm"):
            raise ValueError(f"gate_mode invalid: {gate_mode!r}")
        self.gate_mode = gate_mode
        self.gate_kind = {"unit": 0, "signed": 1, "layernorm": 2, "hamming": 3,
                          "softmax": 4, "relu": 5, "relu_norm": 6, "gelu_norm": 7, "gelu": 8,
                          "square": 9, "square_norm": 10}[gate_mode]
        self.use_bias = bool(use_bias)
        self.hard_sign_ste = bool(hard_sign_ste)
        self.gate_power = float(gate_power)
        self.input_dim = input_dim
        self.n_heads = n_heads
        self.n_outputs = n_outputs
        self.n_anchor_pairs = n_anchor_pairs
        self.tables_per_head = tables_per_head
        self.table_dim = 1 << n_anchor_pairs           # K = 2^NAP
        self.weight_dtype = weight_dtype
        n_lookup_tables = n_heads * tables_per_head
        self.n_lookup_tables = n_lookup_tables
        dev = device or torch.device("cpu")

        # Anchor pairs — identical machinery / seed-determinism as TinyMultiHeadLut.
        self.lookup = TinyAnchorPairsLookup(
            input_dim=input_dim, n_tables=n_lookup_tables, n_anchor_pairs=n_anchor_pairs,
            n_heads=n_heads, random_seed=random_seed, device=dev,
            partition_sets=partition_sets, partition_pair_weights=partition_pair_weights,
            anchor_sampling_policy=anchor_sampling_policy,
            max_anchor_distance=max_anchor_distance, local_window_starts=local_window_starts,
        )
        self.register_buffer('soft_anchor_a_long', self.lookup.anchor_pairs_a.long().contiguous())
        self.register_buffer('soft_anchor_b_long', self.lookup.anchor_pairs_b.long().contiguous())
        self.register_buffer('soft_bit_matrix',
                             _soft_bit_matrix_msb(n_anchor_pairs, dev, dtype=torch.float32))

        rng_kwargs: dict = {"device": dev}
        if random_seed is not None:
            rng_kwargs["generator"] = torch.Generator(device=dev).manual_seed(random_seed + 1)
        wt = ((torch.rand(n_lookup_tables, self.table_dim, n_outputs, **rng_kwargs) - 0.5)
              * (2.0 * initial_weights_noise)).to(weight_dtype)
        self.weights = nn.Parameter(wt)

        # Optional learnable per-(head, output) bias — absorbs the input-independent
        # (DC) component so the weight table encodes only routing-dependent content.
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(n_heads, n_outputs, dtype=weight_dtype, device=dev))
        else:
            self.bias = None

        # gate_mode='layernorm': affine over the K match-scores (identity init).
        if self.gate_kind == 2:
            self.gate_ln_weight = nn.Parameter(torch.ones(self.table_dim, dtype=torch.float32, device=dev))
            self.gate_ln_bias = nn.Parameter(torch.zeros(self.table_dim, dtype=torch.float32, device=dev))
        else:
            self.gate_ln_weight = None
            self.gate_ln_bias = None

        # gate_mode='hamming': learnable per-shell weight ham_weight[NAP+1], init
        # linear 1 (hamming 0) -> 0 (hamming NAP). Indexed by integer hamming dist.
        if self.gate_kind == 3:
            _h = torch.arange(n_anchor_pairs + 1, dtype=torch.float32, device=dev)
            self.ham_weight = nn.Parameter(1.0 - _h / n_anchor_pairs)
        else:
            self.ham_weight = None

        # gate_mode='relu': learnable per-(table, row) threshold bias for ReLU(ts + b),
        # shape [n_tables, K], init 0. Per-table so each table sets its own firing
        # thresholds (the ReLU threshold is a nonlinearity W cannot absorb).
        if self.gate_kind in (5, 6, 7, 8):
            self.gate_relu_bias = nn.Parameter(
                torch.zeros(n_lookup_tables, self.table_dim, dtype=torch.float32, device=dev))
        else:
            self.gate_relu_bias = None

        self.use_bf16 = bool(use_bf16)
        self.learnable_temps = bool(learnable_temps)
        if self.learnable_temps:
            self.log_soft_score_temp = nn.Parameter(
                torch.tensor(math.log(float(soft_score_temp)), dtype=torch.float32, device=dev))
            self.log_select_temp = nn.Parameter(
                torch.tensor(math.log(float(select_temp)), dtype=torch.float32, device=dev))
        else:
            self.register_buffer('log_soft_score_temp',
                                 torch.tensor(math.log(float(soft_score_temp)), dtype=torch.float32, device=dev))
            self.register_buffer('log_select_temp',
                                 torch.tensor(math.log(float(select_temp)), dtype=torch.float32, device=dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}")
        T_soft = self.log_soft_score_temp.exp()
        T_sel = self.log_select_temp.exp()
        autocast_ctx = (torch.amp.autocast("cuda", dtype=torch.bfloat16)
                        if self.use_bf16 and x.is_cuda
                        else torch.amp.autocast("cpu", enabled=False))
        with autocast_ctx:
            return _matmul_mhlut_fwd_body(
                x, self.weights, self.soft_anchor_a_long, self.soft_anchor_b_long,
                self.soft_bit_matrix, T_soft, T_sel, self.n_heads, self.tables_per_head,
                self.gate_kind, self.gate_ln_weight, self.gate_ln_bias,
                self.ham_weight, self.gate_relu_bias, self.bias,
                int(self.hard_sign_ste), self.gate_power,
            )
