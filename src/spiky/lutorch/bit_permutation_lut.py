"""BitPermutationLUT — 1-bit-weight PermutationalLut for bit-level inference.

Design:
  - Inner anchor lookup: TinyAnchorPairsLookup (int16 indices, CANONICAL_FULL_COVERAGE,
    n_alternatives=1, uncertainty=INVERSE_L1 with bias=0.5).
  - Bit weights: one ±1 bit per (table, entry, output_nap slot), packed as
    int32 blocks of 32 bits. `n_blocks = ceil(output_nap / 32)`.
  - Output: per-canonical-pair dominance from summing signed bit votes. Sum
    is kept as int32 inside the kernel — every term is ±1, so sum ∈ [-K, K].
  - CANONICAL_FULL_COVERAGE sampling of output pairs ⇒ no per-slot sign tensor
    is needed (all signs are +1).
  - Forward: custom CUDA kernel only (no PyTorch fallback). Backward is
    stubbed for this implementation.

Tests compare forward output to an equivalent PermutationalLut configured
with ±1 float weights matching the same bit pattern.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

_FP8 = getattr(torch, "float8_e4m3fn", None)
_FP8_AMAX = 448.0  # float8_e4m3fn representable maximum

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy, _repair_intra_table_duplicates
from spiky.lutorch.ranking_tools import _canonical_borda_m
from spiky.lutorch.tiny_anchor_pairs_lookup import (
    TinyAnchorPairsLookup,
    _get_tiny_apl_native,
)


def _get_bit_permlut_native():
    try:
        from lutorch_cuda import get_lutorch_manager  # type: ignore[import]
        return get_lutorch_manager()
    except Exception:
        return None


def _build_output_structures_from_pairs(
    idx_a: torch.Tensor,             # [H, tph, output_nap] long
    idx_b: torch.Tensor,
    n_heads: int,
    tph: int,
    output_nap: int,
    n_outputs: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Build (idx_a, idx_b canonicalized, inv_idx, K_max, output_idx_per_table)
    from given idx_a/idx_b. If any pair has a > b, swap to canonical a < b.
    Duplicate pairs per table are allowed -- K_max simply grows.
    """
    P = n_outputs * (n_outputs - 1) // 2
    # Canonicalize: min(a, b) < max(a, b).
    a_can = torch.minimum(idx_a, idx_b)
    b_can = torch.maximum(idx_a, idx_b)

    tri_i = torch.triu_indices(n_outputs, n_outputs, offset=1, device=device)[0]
    tri_j = torch.triu_indices(n_outputs, n_outputs, offset=1, device=device)[1]
    pair_map = torch.full((n_outputs, n_outputs), -1, dtype=torch.long, device=device)
    pair_range = torch.arange(P, device=device)
    pair_map[tri_i, tri_j] = pair_range
    pair_map[tri_j, tri_i] = pair_range
    output_idx_per_table = pair_map[a_can, b_can]                   # [H, tph, output_nap]

    pair_idx = output_idx_per_table.reshape(n_heads, tph * output_nap)  # [H, TP]
    counts = torch.stack(
        [torch.bincount(pair_idx[h], minlength=P) for h in range(n_heads)], dim=0
    )  # [H, P]
    K_max = int(counts.max().item())

    TP = tph * output_nap
    inv_idx = torch.full((n_heads, P, K_max), -1, dtype=torch.int32, device=device)
    sort_order = pair_idx.argsort(dim=1, stable=True)            # [H, TP]
    pair_sorted = pair_idx.gather(1, sort_order)
    starts = torch.cat(
        [torch.zeros(n_heads, 1, dtype=counts.dtype, device=device), counts.cumsum(dim=1)[:, :-1]],
        dim=1,
    )  # [H, P]
    pos = torch.arange(TP, device=device).unsqueeze(0).expand(n_heads, -1)
    within_group = pos - starts.gather(1, pair_sorted)
    h_idx = torch.arange(n_heads, device=device).unsqueeze(1).expand(-1, TP)
    inv_idx[h_idx, pair_sorted, within_group] = sort_order.to(torch.int32)

    return (
        a_can.contiguous(),
        b_can.contiguous(),
        inv_idx.contiguous(),
        K_max,
        output_idx_per_table.to(torch.int32).contiguous(),
    )


def _sample_canonical_distinct_output_pairs(
    n_heads: int, tph: int, output_nap: int, n_outputs: int,
    random_seed: Optional[int], device: torch.device,
    anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample output pairs. Default policy is CANONICAL_FULL_COVERAGE, which
    gives (idx_a, idx_b) each shaped [n_heads, tph, output_nap] long, with
    a < b, distinct per table, and guaranteed coverage of the canonical pool
    when tph * output_nap >= C(n_outputs, 2). Callers may pass
    CANONICAL_DISTINCT for the legacy per-table i.i.d. sampling."""
    from spiky.lutorch.lut_helpers import get_balanced_anchor_pairs

    policy = (
        anchor_sampling_policy
        if anchor_sampling_policy is not None
        else AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE
    )
    seed = (random_seed + 2_000_003) if random_seed is not None else None
    idx_a_flat, idx_b_flat = get_balanced_anchor_pairs(
        n_tables=n_heads * tph,
        n_anchor_pairs=output_nap,
        input_dim=n_outputs,
        device=device,
        random_seed=seed,
        policy=policy,
        n_heads=n_heads,
        shuffle_per_head=True,
    )
    return (
        idx_a_flat.view(n_heads, tph, output_nap).long(),
        idx_b_flat.view(n_heads, tph, output_nap).long(),
    )


def _build_inv_idx(
    n_heads: int,
    tph: int,
    output_nap: int,
    n_outputs: int,
    random_seed: Optional[int],
    device: torch.device,
    anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Sample output pairs (default CANONICAL_FULL_COVERAGE) and build full
    structures.

    Returns (idx_a, idx_b, inv_idx [int32], K_max, output_idx_per_table [int32])."""
    idx_a, idx_b = _sample_canonical_distinct_output_pairs(
        n_heads, tph, output_nap, n_outputs, random_seed, device,
        anchor_sampling_policy=anchor_sampling_policy,
    )
    a, b, inv_idx, K_max, output_idx_per_table = _build_output_structures_from_pairs(
        idx_a, idx_b, n_heads, tph, output_nap, n_outputs, device,
    )
    return a, b, inv_idx, K_max, output_idx_per_table


def _sample_canonical_output_indices(
    n_heads: int,
    tph: int,
    output_nap: int,
    n_outputs: int,
    random_seed: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """Balanced full-coverage sampling of flat output channel indices with
    per-table distinctness.

    Returns: int64 [n_heads, tph, output_nap], values in [0, n_outputs).

    Algorithm: per-head, concatenate `ceil(tph*output_nap / n_outputs)`
    random permutations of [0, n_outputs) and reshape into [tph, output_nap].
    When `n_outputs % output_nap == 0` (e.g. the frequent case
    `output_nap == n_outputs`, or any divisor), table boundaries align with
    permutation boundaries and every row is distinct by construction — no
    repair pass is needed. Otherwise a greedy swap pass
    (`_repair_intra_table_duplicates`) fixes the boundary tables where two
    independent randperms mix. Mirrors `_get_canonical_full_coverage_pairs`
    in lut_helpers but operates on flat output indices instead of pair
    indices.

    Requires `output_nap <= n_outputs`: a table with more vote slots than
    distinct outputs would burn latent capacity on duplicate vote channels
    (two slots in the same table contributing to the same output) — the
    duplicates would alias gradient and waste bit budget — so this mode is
    rejected at construction time.
    """
    if output_nap > n_outputs:
        raise ValueError(
            f"output_nap ({output_nap}) must be <= n_outputs ({n_outputs}); "
            "with output_nap > n_outputs the per-table distinctness invariant "
            "is infeasible (pigeonhole) and the LUT would burn latent bits on "
            "duplicate vote channels — use a larger n_outputs or smaller output_nap."
        )

    if output_nap == n_outputs:
        # Every table must be a full permutation of [0, n_outputs); since the
        # bit_weights latents are independent per (head, table, slot), any
        # constant per-head output permutation is equivalent up to relabeling
        # of output channels. Use the identity arange and skip the randperm
        # loop entirely.
        base = torch.arange(n_outputs, device=device, dtype=torch.long)
        return base.view(1, 1, n_outputs).expand(n_heads, tph, n_outputs).contiguous()

    seed = (random_seed + 2_000_003) if random_seed is not None else None
    gen = (
        torch.Generator(device=device).manual_seed(seed)
        if seed is not None else None
    )
    P = n_outputs
    slots_per_head = tph * output_nap
    repeats = (slots_per_head + P - 1) // P
    aligned = (P % output_nap == 0)
    out = torch.empty(n_heads, tph, output_nap, dtype=torch.long, device=device)
    for h in range(n_heads):
        perm_cat = torch.cat([
            torch.randperm(P, device=device, generator=gen)
            for _ in range(repeats)
        ])[:slots_per_head]
        per_head_table = perm_cat.view(tph, output_nap).contiguous()
        if not aligned:
            _repair_intra_table_duplicates(per_head_table)
        out[h] = per_head_table
    return out


def _build_inv_idx_flat(
    output_idx_per_table: torch.Tensor,
    n_outputs: int,
    n_heads: int,
    tph: int,
    output_nap: int,
    device: torch.device,
) -> Tuple[torch.Tensor, int]:
    """Build inv_idx from a pre-existing flat output-index assignment.

    output_idx_per_table : long [n_heads, tph, output_nap], values in [0, n_outputs).
    Returns (inv_idx [n_heads, n_outputs, K_max] int32, K_max).
    Same algorithm as `_build_output_structures_from_pairs`, generalized to
    any flat output-index map (no pair semantics).
    """
    P = n_outputs
    flat = output_idx_per_table.reshape(n_heads, tph * output_nap)
    counts = torch.stack(
        [torch.bincount(flat[h], minlength=P) for h in range(n_heads)], dim=0
    )
    K_max = int(counts.max().item())
    TP = tph * output_nap
    inv_idx = torch.full((n_heads, P, K_max), -1, dtype=torch.int32, device=device)
    sort_order = flat.argsort(dim=1, stable=True)
    sorted_idx = flat.gather(1, sort_order)
    starts = torch.cat(
        [torch.zeros(n_heads, 1, dtype=counts.dtype, device=device),
         counts.cumsum(dim=1)[:, :-1]],
        dim=1,
    )
    pos = torch.arange(TP, device=device).unsqueeze(0).expand(n_heads, -1)
    within_group = pos - starts.gather(1, sorted_idx)
    h_idx = torch.arange(n_heads, device=device).unsqueeze(1).expand(-1, TP)
    inv_idx[h_idx, sorted_idx, within_group] = sort_order.to(torch.int32)
    return inv_idx.contiguous(), K_max


class _VoteGatherFn(torch.autograd.Function):
    """Gather ±1 votes from packed `bit_weights` at looked-up entries.

    The primitive behind "Part 1" of a bit-LUT forward: given the winning
    table entry per head-table (``lookup_indices``), return the ±1 votes
    at those entries as a float tensor.

    Forward : PyTorch gather + bit-unpack from ``bit_weights``.
              Output: [B, N, output_nap] ±1 float, where N = n_heads·tph.
    Backward : STE + rational Jacobian gate. ``gate = T/(T+|latent|)²``.
              Gradient is scattered into `latent.grad` at the positions
              read in forward (``(n, lookup_indices[b,n], v)``).
              `bit_weights` itself carries no gradient.
    """

    @staticmethod
    def forward(
        ctx,
        latent: torch.Tensor,            # [N, D, output_nap] — for STE gate
        bit_weights: torch.Tensor,        # [N, D, n_blocks] int32 packed ±1
        lookup_indices: torch.Tensor,     # [B, N] int16
        ste_gate_temperature: float,
        output_nap: int,
    ) -> torch.Tensor:
        B, N = lookup_indices.shape
        dev = bit_weights.device
        li = lookup_indices.long()                                     # [B, N]
        n_ix = torch.arange(N, device=dev).view(1, N).expand(B, N)     # [B, N]
        rows = bit_weights[n_ix, li]                                   # [B, N, n_blocks]
        v_ix = torch.arange(output_nap, device=dev)
        block_per = v_ix >> 5
        bit_per = v_ix & 31
        words = rows[..., block_per]                                   # [B, N, output_nap]
        bits = (words >> bit_per) & 1
        out = bits.to(torch.float32) * 2.0 - 1.0

        ctx.save_for_backward(latent, lookup_indices)
        ctx.ste_gate_temperature = float(ste_gate_temperature)
        ctx.output_nap = int(output_nap)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        latent, lookup_indices = ctx.saved_tensors
        if not latent.requires_grad:
            return None, None, None, None, None
        T = ctx.ste_gate_temperature
        on = ctx.output_nap
        B, N = lookup_indices.shape
        dev = grad_out.device

        latent_f32 = latent if latent.dtype == torch.float32 else latent.to(torch.float32)
        li = lookup_indices.long()
        n_ix = torch.arange(N, device=dev).view(1, N, 1).expand(B, N, on)
        v_ix = torch.arange(on, device=dev).view(1, 1, on).expand(B, N, on)
        li_ix = li.unsqueeze(-1).expand(B, N, on)

        latent_at = latent_f32[n_ix, li_ix, v_ix]                      # [B, N, on]
        denom = T + latent_at.abs()
        gate = T / (denom * denom)
        gated = grad_out.to(torch.float32) * gate

        D = latent_f32.shape[1]
        latent_grad = torch.zeros_like(latent_f32)
        flat_idx = (n_ix * D * on + li_ix * on + v_ix).reshape(-1)
        latent_grad.view(-1).index_add_(0, flat_idx, gated.reshape(-1))

        if latent.dtype != torch.float32:
            latent_grad = latent_grad.to(latent.dtype)
        return latent_grad, None, None, None, None


class _BitPermLutDomFunction(torch.autograd.Function):
    """Custom autograd for the bit dominance gather.

    Forward : one thread per (b, h, p); sums ±1 bits through the inverse index.
    Backward: dispatches between two STE variants:
      - hard (default) : uses discrete ±1 from `bit_weights` (full magnitude).
      - soft (opt-in)  : uses continuous fp8 `latent_fp8` in [-1, 1]. Can
                         under-train early when latent magnitudes are small;
                         good for late-training refinement.
    """

    @staticmethod
    def forward(
        ctx,
        lookup_indices,          # int16 [B, n_heads*tph]       (no grad)
        lookup_alt_indices,      # int16 [B, n_heads*tph, 1]    (no grad; saved for bwd)
        carriers_main,           # float [B, n_heads*tph]       (zeros, autograd link to x)
        carriers_alt,            # float [B, n_heads*tph, 1]    (zeros, autograd link to x)
        bit_weights,             # int32 [n_heads*tph, table_dim, n_blocks] (forward)
        latent_fp8,              # fp8   [n_heads*tph, table_dim, output_nap] (soft backward)
        latent_scale,            # float32 [n_heads*tph, 1, 1] per-table scale
        inv_idx,                 # int32 [n_heads, P, K]
        output_idx_per_table,       # int32 [n_heads, tph, output_nap]
        n_heads,
        tph,
        output_nap,
        n_pairs,
        scale,                   # float (0.5 / sqrt(N_votes_per_pair))
        soft_backward,           # bool
    ):
        native = _get_bit_permlut_native()
        out_int = native.bit_perm_lut_dom_gather_forward(
            lookup_indices.contiguous(),
            bit_weights.contiguous(),
            inv_idx.contiguous(),
            int(n_heads),
            int(tph),
            int(output_nap),
            int(n_pairs),
        )
        out_float = out_int.to(carriers_main.dtype) * scale
        if soft_backward:
            # Soft backward needs latent + its per-table scale.
            ctx.save_for_backward(
                lookup_indices, lookup_alt_indices, latent_fp8, latent_scale, output_idx_per_table,
            )
        else:
            ctx.save_for_backward(
                lookup_indices, lookup_alt_indices, bit_weights, output_idx_per_table,
            )
        ctx.n_heads = int(n_heads)
        ctx.tph = int(tph)
        ctx.output_nap = int(output_nap)
        ctx.n_pairs = int(n_pairs)
        ctx.scale = float(scale)
        ctx.soft_backward = bool(soft_backward)
        return out_float

    @staticmethod
    def backward(ctx, grad_out):
        native = _get_bit_permlut_native()
        if ctx.soft_backward:
            lookup_indices, lookup_alt_indices, latent_fp8, latent_scale, output_idx_per_table = ctx.saved_tensors
            # Dispatch on latent dtype.
            if latent_fp8.dtype == torch.float32:
                grad_main, grad_alt = native.bit_perm_lut_dom_gather_backward_latent_f32(
                    grad_out.contiguous().to(torch.float32),
                    lookup_indices, lookup_alt_indices, latent_fp8, output_idx_per_table,
                    ctx.n_heads, ctx.tph, ctx.output_nap, ctx.n_pairs, ctx.scale,
                )
            elif latent_fp8.dtype == torch.bfloat16:
                grad_main, grad_alt = native.bit_perm_lut_dom_gather_backward_latent_bf16(
                    grad_out.contiguous().to(torch.float32),
                    lookup_indices, lookup_alt_indices, latent_fp8, output_idx_per_table,
                    ctx.n_heads, ctx.tph, ctx.output_nap, ctx.n_pairs, ctx.scale,
                )
            else:
                grad_main, grad_alt = native.bit_perm_lut_dom_gather_backward_latent(
                    grad_out.contiguous().to(torch.float32),
                    lookup_indices, lookup_alt_indices, latent_fp8, latent_scale, output_idx_per_table,
                    ctx.n_heads, ctx.tph, ctx.output_nap, ctx.n_pairs, ctx.scale,
                )
        else:
            lookup_indices, lookup_alt_indices, bit_weights, output_idx_per_table = ctx.saved_tensors
            grad_main, grad_alt = native.bit_perm_lut_dom_gather_backward(
                grad_out.contiguous().to(torch.float32),
                lookup_indices, lookup_alt_indices, bit_weights, output_idx_per_table,
                ctx.n_heads, ctx.tph, ctx.output_nap, ctx.n_pairs, ctx.scale,
            )
        # 15 forward inputs → 15 gradient returns. Only the two carriers receive gradient.
        return (
            None,         # lookup_indices
            None,         # lookup_alt_indices
            grad_main,    # carriers_main
            grad_alt,     # carriers_alt
            None,         # bit_weights
            None,         # latent_fp8
            None,         # latent_scale
            None,         # inv_idx
            None,         # output_idx_per_table
            None, None, None, None, None, None,  # scalars incl. soft_backward
        )


class BitPermutationLUTInput(nn.Module):
    """Shared input-side state for BitPermutationLUT and BitPermutationLUTEx.

    Owns the pieces that are identical across both flavours of bit-LUT:
      - `TinyAnchorPairsLookup` for input anchor lookup.
      - Latent storage (fp8 / bf16 / fp32).
      - Packed ±1 `bit_weights` buffer (signs of the latent).

    Common helpers:
      - `set_bit_weights_from_signs(signs)` packs a float sign tensor.
      - `refresh_bit_weights()` re-packs from the current latent.
      - `gather_votes(x)` runs anchor lookup + returns gathered ±1 votes
        (float [B, n_heads·tph, output_nap]) with STE-gate gradient to the
        latent. This is the "Part 1 primitive" of any bit-LUT forward.

    Subclasses supply the aggregation from gathered votes to the final
    output (Part 2). `BitPermutationLUT` uses a fused CUDA kernel that
    folds Part 1+2 into one pass (`_BitPermLutDomFunction`); newer variants
    like `BitPermutationLUTEx` call `gather_votes` and build their own
    aggregation on top.
    """

    def __init__(
        self,
        n_inputs: int,
        n_heads: int,
        input_nap: int,
        output_nap: int,
        tph: int,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        latent_dtype: str = 'fp8',
        ste_gate_temperature: float = 0.1,
        device: Optional[torch.device] = None,
        partition_sets: Optional[list] = None,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
    ):
        super().__init__()
        if latent_dtype not in ('fp8', 'bf16', 'fp32'):
            raise ValueError(
                f"latent_dtype must be one of 'fp8', 'bf16', 'fp32', got {latent_dtype!r}"
            )
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if not (1 <= input_nap <= 15):
            raise ValueError(
                f"BitPermutationLUTInput requires 1 <= input_nap <= 15 "
                f"(TinyAnchorPairsLookup int16 lookup index), got {input_nap}"
            )
        if output_nap <= 0:
            raise ValueError(f"output_nap must be positive, got {output_nap}")

        self.n_inputs = n_inputs
        self.n_heads = n_heads
        self.input_nap = input_nap
        self.output_nap = output_nap
        self.tph = tph
        self.table_dim = 1 << input_nap
        self.n_blocks = (output_nap + 31) // 32
        self.latent_dtype = latent_dtype
        self.ste_gate_temperature = float(ste_gate_temperature)

        # Anchor lookup (flat tables, n_heads×tph rows).
        self.anchor = TinyAnchorPairsLookup(
            input_dim=n_inputs,
            n_tables=n_heads * tph,
            n_anchor_pairs=input_nap,
            n_heads=n_heads,
            random_seed=random_seed,
            device=dev,
            partition_sets=partition_sets,
            anchor_sampling_policy=anchor_sampling_policy,
        )
        self._anchor_sampling_policy_opt = anchor_sampling_policy

        if _FP8 is None or dev.type != "cuda":
            raise RuntimeError("BitPermutationLUTInput requires CUDA + fp8 support")

        # Latent init: uniform in [-initial_weights_noise, +initial_weights_noise].
        if random_seed is not None:
            gen = torch.Generator(device=dev).manual_seed(random_seed + 4_000_003)
        else:
            gen = None
        shape = (n_heads * tph, self.table_dim, output_nap)
        latent_init_f32 = (
            torch.rand(shape, device=dev, generator=gen) - 0.5
        ) * (2.0 * float(initial_weights_noise))

        if latent_dtype == 'fp8':
            amax = latent_init_f32.abs().amax(dim=(1, 2), keepdim=True).clamp(min=1e-20)
            latent_scale = _FP8_AMAX / amax
            latent_fp8 = (latent_init_f32 * latent_scale).to(_FP8)
            self.register_buffer('latent_fp8', latent_fp8.contiguous())
            self.register_buffer('latent_scale', latent_scale.contiguous())
        elif latent_dtype == 'bf16':
            self.register_buffer(
                'latent_bf16',
                latent_init_f32.to(torch.bfloat16).contiguous(),
            )
        else:
            self.register_buffer('latent_fp32', latent_init_f32.contiguous())

        bit_weights = torch.zeros(
            n_heads * tph, self.table_dim, self.n_blocks,
            device=dev, dtype=torch.int32,
        )
        self.register_buffer('bit_weights', bit_weights.contiguous())
        _get_bit_permlut_native().bit_pack_signs(
            latent_init_f32.contiguous(), self.bit_weights, int(output_nap),
        )

    # --- helpers ---------------------------------------------------------

    def _latent_for_gate(self) -> torch.Tensor:
        """Return a float-typed view of the latent for the STE gate."""
        if self.latent_dtype == 'fp32':
            return self.latent_fp32
        if self.latent_dtype == 'bf16':
            return self.latent_bf16
        # fp8 — dequantize per-table for the gate's |·| magnitude.
        return self.latent_fp8.to(torch.float32) / self.latent_scale

    def _latent_for_pack(self) -> torch.Tensor:
        """Return a float-typed view of the latent for re-packing signs."""
        if self.latent_dtype == 'fp32':
            return self.latent_fp32
        if self.latent_dtype == 'bf16':
            return self.latent_bf16.to(torch.float32)
        return self.latent_fp8.to(torch.float32) / self.latent_scale

    def refresh_bit_weights(self) -> None:
        """Re-pack `bit_weights` from the current latent. Call after the
        latent changes (e.g. after an optimizer step when training with
        standard torch.optim.Adam on a Parameter-promoted latent)."""
        self.set_bit_weights_from_signs(self._latent_for_pack().contiguous())

    def gather_votes(self, x: torch.Tensor):
        """Part 1 primitive: anchor lookup + gathered ±1 votes.

        Returns
        -------
        lookup_indices : int16 [B, n_heads·tph]
        votes          : float32 [B, n_heads·tph, output_nap]  (±1)
        """
        lookup_tuple = self.anchor(x)
        lookup_indices = lookup_tuple[0]
        votes = _VoteGatherFn.apply(
            self._latent_for_gate(),
            self.bit_weights,
            lookup_indices,
            self.ste_gate_temperature,
            self.output_nap,
        )
        return lookup_indices, votes

    def set_bit_weights_from_signs(self, signs: torch.Tensor) -> None:
        """Update bit_weights from a ±1 (or >0 / <=0) float tensor.

        signs: [n_heads * tph, table_dim, output_nap]  (float; positive → bit 1)
        """
        if signs.shape != (self.n_heads * self.tph, self.table_dim, self.output_nap):
            raise ValueError(
                f"signs shape must be [n_heads*tph, table_dim, output_nap] = "
                f"({self.n_heads * self.tph}, {self.table_dim}, {self.output_nap}), got {tuple(signs.shape)}"
            )
        native = _get_bit_permlut_native()
        if signs.is_cuda and native is not None:
            native.bit_pack_signs(
                signs.to(torch.float32).contiguous(),
                self.bit_weights,
                int(self.output_nap),
            )
            return
        bits = (signs > 0).to(torch.int32)
        packed = torch.zeros(
            self.n_heads * self.tph, self.table_dim, self.n_blocks,
            device=signs.device, dtype=torch.int32,
        )
        for k in range(self.output_nap):
            block_idx = k // 32
            bit_pos = k % 32
            packed[:, :, block_idx] |= (bits[:, :, k] << bit_pos)
        self.bit_weights.copy_(packed.to(self.bit_weights.device))


class BitMultiHeadLUT(BitPermutationLUTInput):
    """Generic multi-head bit-LUT mapping input to output via sparse voting.

    Forward: x (float [B, n_inputs]) -> votes (float [B, n_heads, n_outputs]).
    Each output channel accumulates the sum of ±1 bit votes drawn from
    `tph * output_nap` table slots through `inv_idx`. The output is purely a
    flat voting space; downstream is free to interpret it as pair-dominance,
    logits, or any other vector.

    Args:
        n_inputs:               input dimension.
        n_outputs:              flat output dimension P (number of voting
                                channels).
        n_heads:                number of heads (logical — n_tables = n_heads * tph).
        input_nap:              anchor pairs per table for input lookup (<= 15).
        output_nap:             signed votes per table (packed as bits).
        tph:                    tables per head.
        random_seed:            seed for anchor and output-channel sampling.
        output_idx_per_table:   optional override for the
                                [n_heads, tph, output_nap] integer tensor
                                mapping (head, table, anchor-pair-slot) ->
                                output channel in [0, n_outputs). If None,
                                balanced canonical-coverage sampling is used.
                                Wrappers (e.g. BitPermutationLUT) pass a
                                pre-built map to layer pair semantics on top
                                of this generic core.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_heads: int,
        input_nap: int,
        output_nap: int,
        tph: int,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        soft_backward: bool = False,
        latent_dtype: str = 'fp8',
        device: Optional[torch.device] = None,
        partition_sets: Optional[list] = None,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        input_anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        output_idx_per_table: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ):
        input_policy = (
            input_anchor_sampling_policy
            if input_anchor_sampling_policy is not None
            else anchor_sampling_policy
        )
        super().__init__(
            n_inputs=n_inputs, n_heads=n_heads,
            input_nap=input_nap, output_nap=output_nap, tph=tph,
            random_seed=random_seed,
            initial_weights_noise=initial_weights_noise,
            latent_dtype=latent_dtype, device=device,
            partition_sets=partition_sets,
            anchor_sampling_policy=input_policy,
        )
        dev = self.bit_weights.device
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")
        self.n_outputs = int(n_outputs)

        if output_idx_per_table is None:
            output_idx_per_table = _sample_canonical_output_indices(
                n_heads=n_heads, tph=tph, output_nap=output_nap,
                n_outputs=n_outputs, random_seed=random_seed, device=dev,
            )
        else:
            if tuple(output_idx_per_table.shape) != (n_heads, tph, output_nap):
                raise ValueError(
                    "output_idx_per_table must have shape "
                    f"[n_heads={n_heads}, tph={tph}, output_nap={output_nap}], "
                    f"got {tuple(output_idx_per_table.shape)}"
                )
            output_idx_per_table = output_idx_per_table.to(dev, dtype=torch.long)

        inv_idx, K_max = _build_inv_idx_flat(
            output_idx_per_table, self.n_outputs, n_heads, tph, output_nap, dev,
        )
        self.register_buffer('inv_idx', inv_idx.contiguous())
        self.register_buffer(
            'output_idx_per_table',
            output_idx_per_table.to(torch.int32).contiguous(),
        )
        self.K_max = K_max

        # `self.scale` is applied to the int vote-sum kernel output (forward
        # and backward). Default 1.0 = raw sums in [-K, +K] with K ≤ K_max;
        # the caller is free to follow up with normalization / learnable
        # scale / LayerNorm. Wrappers with a fixed downstream interpretation
        # (e.g. BitPermutationLUT for pair-dominance) pass an explicit value.
        self.scale = float(scale)
        self.soft_backward = bool(soft_backward)

    def get_bit_weights_as_signs(self) -> torch.Tensor:
        """Decode bit_weights to ±1 float tensor [N, table_dim, output_nap]."""
        out = torch.empty(
            self.n_heads * self.tph, self.table_dim, self.output_nap,
            device=self.bit_weights.device, dtype=torch.float32,
        )
        for k in range(self.output_nap):
            block_idx = k // 32
            bit_pos = k % 32
            bit = (self.bit_weights[:, :, block_idx] >> bit_pos) & 1
            out[:, :, k] = (2 * bit - 1).to(torch.float32)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run anchor lookup, then bit gather.

        x: float [B, input_dim]
        returns: float [B, n_heads, n_outputs]
        """
        if not x.is_cuda:
            raise RuntimeError("BitMultiHeadLUT is CUDA-only")
        if _get_bit_permlut_native() is None:
            raise RuntimeError("lutorch_cuda native extension not available")

        lookup_indices, lookup_alt_indices, _, carriers_main, carriers_alt = self.anchor(x)
        if carriers_main is None:
            # Eval / no-grad path: bypass autograd Function.
            native = _get_bit_permlut_native()
            out_int = native.bit_perm_lut_dom_gather_forward(
                lookup_indices.contiguous(),
                self.bit_weights.contiguous(),
                self.inv_idx.contiguous(),
                int(self.n_heads), int(self.tph), int(self.output_nap), int(self.n_outputs),
            )
            return out_int.to(x.dtype) * self.scale

        # The autograd Function only reads the latent tensors in the soft
        # backward path. For fp32 mode we pass latent_fp32 into the
        # `latent_fp8` slot (dtype check happens at dispatch, downstream);
        # `latent_scale` is unused for fp32.
        if self.latent_dtype == 'fp8':
            latent_fp8 = self.latent_fp8
            latent_scale = self.latent_scale
        elif self.latent_dtype == 'fp32':
            latent_fp8 = self.latent_fp32
            latent_scale = self.bit_weights   # placeholder
        else:  # 'bf16'
            latent_fp8 = self.latent_bf16
            latent_scale = self.bit_weights   # placeholder
        return _BitPermLutDomFunction.apply(
            lookup_indices, lookup_alt_indices,
            carriers_main, carriers_alt,
            self.bit_weights, latent_fp8, latent_scale,
            self.inv_idx, self.output_idx_per_table,
            self.n_heads, self.tph, self.output_nap, self.n_outputs,
            self.scale, self.soft_backward,
        )


class BitPermutationLUT(BitMultiHeadLUT):
    """Pair-dominance specialization of BitMultiHeadLUT (legacy API).

    Treats the output as the C(d_head, 2) pair-dominance vector for d_head
    ranked items: each output channel corresponds to a canonical (a, b) pair
    with a < b. Constructor signature is identical to the original
    BitPermutationLUT — `n_outputs` here means the rank dim d_head, and the
    actual output dim is P = d_head*(d_head-1)/2 (exposed as `n_pairs`).

    For non-dominance use cases (logits, generic feature voting, etc.) use
    BitMultiHeadLUT directly with `n_outputs` set to the literal output dim.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_heads: int,
        input_nap: int,
        output_nap: int,
        tph: int,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        soft_backward: bool = False,
        latent_dtype: str = 'fp8',
        device: Optional[torch.device] = None,
        partition_sets: Optional[list] = None,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        input_anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        output_anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
    ):
        if n_outputs < 2:
            raise ValueError(f"n_outputs must be >= 2, got {n_outputs}")
        d_head = int(n_outputs)
        n_pairs = d_head * (d_head - 1) // 2

        input_policy = (
            input_anchor_sampling_policy
            if input_anchor_sampling_policy is not None
            else anchor_sampling_policy
        )
        output_policy = (
            output_anchor_sampling_policy
            if output_anchor_sampling_policy is not None
            else anchor_sampling_policy
        )

        # Sample (idx_a, idx_b) using the canonical pair sampler so seed
        # determinism / pair coverage match the pre-redesign behaviour, then
        # map (a, b) -> canonical triu pair index in [0, n_pairs) and pass
        # the resulting flat index map to the BitMultiHeadLUT core.
        dev = torch.device(device) if device is not None else torch.device("cpu")
        idx_a, idx_b = _sample_canonical_distinct_output_pairs(
            n_heads, tph, output_nap, d_head, random_seed, dev,
            anchor_sampling_policy=output_policy,
        )
        a_can = torch.minimum(idx_a, idx_b)
        b_can = torch.maximum(idx_a, idx_b)
        tri_i = torch.triu_indices(d_head, d_head, offset=1, device=dev)[0]
        tri_j = torch.triu_indices(d_head, d_head, offset=1, device=dev)[1]
        pair_map = torch.full((d_head, d_head), -1, dtype=torch.long, device=dev)
        pair_range = torch.arange(n_pairs, device=dev)
        pair_map[tri_i, tri_j] = pair_range
        pair_map[tri_j, tri_i] = pair_range
        output_idx_per_table = pair_map[a_can, b_can]

        # Pair-dominance interpretation: each output is a sum of N ±1 votes
        # where N ≈ tph*output_nap/n_pairs. CLT-normalize so the per-pair
        # dominance lives at unit-variance scale (matching the legacy
        # BitPermutationLUT contract that downstream Borda / DominanceToVector
        # consumers depend on).
        n_votes_per_pair = tph * output_nap / float(n_pairs)
        dom_scale = 0.5 / math.sqrt(n_votes_per_pair)
        super().__init__(
            n_inputs=n_inputs,
            n_outputs=n_pairs,
            n_heads=n_heads,
            input_nap=input_nap,
            output_nap=output_nap,
            tph=tph,
            random_seed=random_seed,
            initial_weights_noise=initial_weights_noise,
            soft_backward=soft_backward,
            latent_dtype=latent_dtype,
            device=device,
            partition_sets=partition_sets,
            input_anchor_sampling_policy=input_policy,
            output_idx_per_table=output_idx_per_table,
            scale=dom_scale,
        )

        # Pair-dominance specifics. `self.n_outputs` from the parent equals
        # n_pairs (the kernel output dim) and is required by the kernel call
        # in `forward`; we also expose `n_pairs` and `d_head` so callers can
        # query the rank dim explicitly.
        self.d_head = d_head
        self.n_pairs = n_pairs
        self.register_buffer('idx_a', a_can.contiguous())
        self.register_buffer('idx_b', b_can.contiguous())
        self.register_buffer('dom_borda_m', _canonical_borda_m(d_head, dev))

    def load_pairs(
        self,
        anchor_pairs_a: torch.Tensor,        # [n_tables, input_nap] integer
        anchor_pairs_b: torch.Tensor,
        idx_a: torch.Tensor,                  # [n_heads, tph*output_nap] OR [n_heads, tph, output_nap] integer
        idx_b: torch.Tensor,
    ) -> None:
        """Replace anchor_pairs and output-pair buffers with externally
        provided values and rebuild all derived structures.

        Non-canonical pairs (a > b) are silently swapped to canonical form
        (min, max). Duplicate pairs within a table are permitted (K_max just
        grows).

        Useful for reproducing experiments where a student must share anchor
        layout with a teacher PermutationalLut (same `anchor_pairs_a/b`,
        `idx_a/b` from dataset.pt).
        """
        N = self.n_heads * self.tph
        dev = self.bit_weights.device

        # --- anchor pairs ---
        if anchor_pairs_a.shape != (N, self.input_nap) or anchor_pairs_b.shape != (N, self.input_nap):
            raise ValueError(
                f"anchor_pairs_{{a,b}} must have shape [n_heads*tph, input_nap] = [{N}, {self.input_nap}]"
            )
        a_in = torch.minimum(anchor_pairs_a, anchor_pairs_b).to(dev, dtype=torch.int16)
        b_in = torch.maximum(anchor_pairs_a, anchor_pairs_b).to(dev, dtype=torch.int16)
        self.anchor.anchor_pairs_a.copy_(a_in)
        self.anchor.anchor_pairs_b.copy_(b_in)

        # --- output pairs + derived structures ---
        idx_a_r = idx_a.reshape(self.n_heads, self.tph, self.output_nap).to(dev, dtype=torch.long)
        idx_b_r = idx_b.reshape(self.n_heads, self.tph, self.output_nap).to(dev, dtype=torch.long)
        a, b, inv_idx, K_max, output_idx_per_table = _build_output_structures_from_pairs(
            idx_a_r, idx_b_r, self.n_heads, self.tph, self.output_nap, self.d_head, dev,
        )
        # Replace buffers. `inv_idx` shape may change (new K_max), so use setattr via register_buffer.
        self.idx_a = a
        self.idx_b = b
        self.register_buffer('inv_idx', inv_idx)
        self.register_buffer('output_idx_per_table', output_idx_per_table)
        self.K_max = K_max
