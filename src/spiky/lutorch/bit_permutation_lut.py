"""BitPermutationLUT — 1-bit-weight PermutationalLut for bit-level inference.

Design:
  - Inner anchor lookup: TinyAnchorPairsLookup (int16 indices, CANONICAL_DISTINCT,
    n_alternatives=1, uncertainty=INVERSE_L1 with bias=0.5).
  - Bit weights: one ±1 bit per (table, entry, output_nap slot), packed as
    int32 blocks of 32 bits. `n_blocks = ceil(output_nap / 32)`.
  - Output: per-canonical-pair dominance from summing signed bit votes. Sum
    is kept as int32 inside the kernel — every term is ±1, so sum ∈ [-K, K].
  - CANONICAL_DISTINCT sampling of output pairs ⇒ no per-slot sign tensor
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

from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
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
    """Build (idx_a, idx_b canonicalized, inv_idx, K_max, pair_idx_per_slot)
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
    pair_idx_per_slot = pair_map[a_can, b_can]                   # [H, tph, output_nap]

    pair_idx = pair_idx_per_slot.reshape(n_heads, tph * output_nap)  # [H, TP]
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
        pair_idx_per_slot.to(torch.int32).contiguous(),
    )


def _sample_canonical_distinct_output_pairs(
    n_heads: int, tph: int, output_nap: int, n_outputs: int,
    random_seed: Optional[int], device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample output pairs via CANONICAL_DISTINCT policy. Returns (idx_a, idx_b)
    each shaped [n_heads, tph, output_nap] long, with a < b and distinct per table."""
    from spiky.lutorch.lut_helpers import get_balanced_anchor_pairs

    seed = (random_seed + 2_000_003) if random_seed is not None else None
    idx_a_flat, idx_b_flat = get_balanced_anchor_pairs(
        n_tables=n_heads * tph,
        n_anchor_pairs=output_nap,
        input_dim=n_outputs,
        device=device,
        random_seed=seed,
        policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Sample output pairs via CANONICAL_DISTINCT and build full structures.

    Returns (idx_a, idx_b, inv_idx [int32], K_max, pair_idx_per_slot [int32])."""
    idx_a, idx_b = _sample_canonical_distinct_output_pairs(
        n_heads, tph, output_nap, n_outputs, random_seed, device,
    )
    a, b, inv_idx, K_max, pair_idx_per_slot = _build_output_structures_from_pairs(
        idx_a, idx_b, n_heads, tph, output_nap, n_outputs, device,
    )
    return a, b, inv_idx, K_max, pair_idx_per_slot


def _canonical_borda_m(n_outputs: int, device: torch.device) -> torch.Tensor:
    """Canonical Borda matrix, pre-scaled by 1/sqrt(N-1) (matches PermLut lib)."""
    P = n_outputs * (n_outputs - 1) // 2
    tri_i, tri_j = torch.triu_indices(n_outputs, n_outputs, offset=1)
    m = torch.zeros(n_outputs, P, device=device)
    for p in range(P):
        m[int(tri_i[p]), p] = 1.0
        m[int(tri_j[p]), p] = -1.0
    return m / math.sqrt(max(n_outputs - 1, 1))


class _BitPermLutDomFunction(torch.autograd.Function):
    """Custom autograd for the bit dominance gather.

    Forward : one thread per (b, h, p); sums ±1 bits through the inverse index.
    Backward: one thread per (b, n = h*tph+t); projects grad_out back through
              bit weights at `entry_main` and `entry_alt` to produce
              grad_main [B, N] and grad_alt [B, N, 1] for the lookup carriers.
              Bits themselves carry no gradient (discrete).
    """

    @staticmethod
    def forward(
        ctx,
        lookup_indices,          # int16 [B, n_heads*tph]       (no grad)
        lookup_alt_indices,      # int16 [B, n_heads*tph, 1]    (no grad; saved for bwd)
        carriers_main,           # float [B, n_heads*tph]       (zeros, autograd link to x)
        carriers_alt,            # float [B, n_heads*tph, 1]    (zeros, autograd link to x)
        bit_weights,             # int32 [n_heads*tph, table_dim, n_blocks]
        inv_idx,                 # int32 [n_heads, P, K]
        pair_idx_per_slot,       # int32 [n_heads, tph, output_nap]
        n_heads,
        tph,
        output_nap,
        n_pairs,
        scale,                   # float (0.5 / sqrt(N_votes_per_pair))
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
        ctx.save_for_backward(lookup_indices, lookup_alt_indices, bit_weights, pair_idx_per_slot)
        ctx.n_heads = int(n_heads)
        ctx.tph = int(tph)
        ctx.output_nap = int(output_nap)
        ctx.n_pairs = int(n_pairs)
        ctx.scale = float(scale)
        return out_float

    @staticmethod
    def backward(ctx, grad_out):
        lookup_indices, lookup_alt_indices, bit_weights, pair_idx_per_slot = ctx.saved_tensors
        native = _get_bit_permlut_native()
        grad_main, grad_alt = native.bit_perm_lut_dom_gather_backward(
            grad_out.contiguous().to(torch.float32),
            lookup_indices, lookup_alt_indices, bit_weights, pair_idx_per_slot,
            ctx.n_heads, ctx.tph, ctx.output_nap, ctx.n_pairs, ctx.scale,
        )
        # 12 forward inputs → 12 gradient returns. Only the two carriers receive gradient.
        return (
            None,         # lookup_indices
            None,         # lookup_alt_indices
            grad_main,    # carriers_main
            grad_alt,     # carriers_alt
            None,         # bit_weights (discrete)
            None,         # inv_idx
            None,         # pair_idx_per_slot
            None, None, None, None, None,  # scalars
        )


class BitPermutationLUT(nn.Module):
    """1-bit-weight PermutationalLut for bit-level inference.

    Forward: x (float) -> dominance (float [B, n_heads, P]).
    Backward is stubbed — `bit_weights` is a buffer (not a Parameter).

    Args:
        n_inputs:      input dimension
        n_outputs:     output dimension (embedding of the downstream block)
        n_heads:       number of heads (logical — n_tables = n_heads * tph)
        input_nap:     anchor pairs per table for input lookup (<= 16)
        output_nap:    signed votes per table (packed as bits)
        tph:           tables per head
        random_seed:   seed for anchor and output-pair sampling
        device:        target device
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
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        dev = device or torch.device("cpu")
        if not (1 <= input_nap <= 16):
            raise ValueError(f"BitPermutationLUT requires 1 <= input_nap <= 16, got {input_nap}")
        if output_nap <= 0:
            raise ValueError(f"output_nap must be positive, got {output_nap}")
        if n_outputs < 2:
            raise ValueError(f"n_outputs must be >= 2, got {n_outputs}")

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_heads = n_heads
        self.input_nap = input_nap
        self.output_nap = output_nap
        self.tph = tph
        self.table_dim = 1 << input_nap
        self.n_blocks = (output_nap + 31) // 32
        self.n_pairs = n_outputs * (n_outputs - 1) // 2

        # Anchor lookup (shared-style: flat tables, no head separation).
        self.anchor = TinyAnchorPairsLookup(
            input_dim=n_inputs,
            n_tables=n_heads * tph,
            n_anchor_pairs=input_nap,
            n_heads=n_heads,
            random_seed=random_seed,
            device=dev,
        )

        # Output pair sampling (CANONICAL_DISTINCT) + inverse index.
        idx_a, idx_b, inv_idx, K_max, pair_idx_per_slot = _build_inv_idx(
            n_heads=n_heads, tph=tph, output_nap=output_nap,
            n_outputs=n_outputs, random_seed=random_seed, device=dev,
        )
        self.register_buffer('idx_a', idx_a.contiguous())          # [H, tph, output_nap] long (reference)
        self.register_buffer('idx_b', idx_b.contiguous())
        self.register_buffer('inv_idx', inv_idx.contiguous())      # [H, P, K_max] int32
        # Reverse of inv_idx: used by backward kernel to map slot → canonical pair.
        self.register_buffer('pair_idx_per_slot', pair_idx_per_slot.contiguous())  # [H, tph, output_nap] int32
        self.K_max = K_max

        # Per-output scaling. Input to gather kernel is a signed-vote ±1 count
        # (n_alt=1 STE forward produces ±1 magnitude pre-0.5-scaling). Applying
        # 0.5 here matches the /output_nap-era PermLut convention; the final
        # /sqrt(N_votes_per_pair) normalizes for CLT consistency.
        n_votes_per_pair = tph * output_nap / float(self.n_pairs)
        self.scale = 0.5 / math.sqrt(n_votes_per_pair)

        # Borda matrix for optional dominance→rank projection (pre-scaled).
        self.register_buffer('dom_borda_m', _canonical_borda_m(n_outputs, dev))

        # Bit weights — stored as int32 blocks. Initialized uniformly at ±1.
        # Not a Parameter: autograd does not train them; updates happen via
        # latent fp optimizer (e.g. fp8 Adam) that calls `set_bit_weights`.
        if random_seed is not None:
            gen = torch.Generator(device=dev)
            gen.manual_seed(random_seed + 4_000_003)
        else:
            gen = None
        bit_weights = torch.randint(
            low=torch.iinfo(torch.int32).min,
            high=torch.iinfo(torch.int32).max,
            size=(n_heads * tph, self.table_dim, self.n_blocks),
            device=dev, dtype=torch.int32, generator=gen,
        )
        self.register_buffer('bit_weights', bit_weights.contiguous())

    def load_pairs(
        self,
        anchor_pairs_a: torch.Tensor,        # [n_tables, input_nap] integer
        anchor_pairs_b: torch.Tensor,
        idx_a: torch.Tensor,                  # [n_heads, tph*output_nap] OR [n_heads, tph, output_nap] integer
        idx_b: torch.Tensor,
    ) -> None:
        """Replace the module's anchor_pairs and output-pair buffers with
        externally-provided values and rebuild all derived structures.

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
        a, b, inv_idx, K_max, pair_idx_per_slot = _build_output_structures_from_pairs(
            idx_a_r, idx_b_r, self.n_heads, self.tph, self.output_nap, self.n_outputs, dev,
        )
        # Replace buffers. `inv_idx` shape may change (new K_max), so use setattr via register_buffer.
        self.idx_a = a
        self.idx_b = b
        self.register_buffer('inv_idx', inv_idx)
        self.register_buffer('pair_idx_per_slot', pair_idx_per_slot)
        self.K_max = K_max

    def set_bit_weights_from_signs(self, signs: torch.Tensor) -> None:
        """Update bit_weights from a ±1 (or >0 / <=0) float tensor.

        signs: [n_heads * tph, table_dim, output_nap]  (float; positive → bit 1)
        """
        if signs.shape != (self.n_heads * self.tph, self.table_dim, self.output_nap):
            raise ValueError(
                f"signs shape must be [n_heads*tph, table_dim, output_nap] = "
                f"({self.n_heads * self.tph}, {self.table_dim}, {self.output_nap}), got {tuple(signs.shape)}"
            )
        bits = (signs > 0).to(torch.int32)  # [N, table_dim, output_nap]
        packed = torch.zeros(
            self.n_heads * self.tph, self.table_dim, self.n_blocks,
            device=signs.device, dtype=torch.int32,
        )
        for k in range(self.output_nap):
            block_idx = k // 32
            bit_pos = k % 32
            packed[:, :, block_idx] |= (bits[:, :, k] << bit_pos)
        self.bit_weights.copy_(packed.to(self.bit_weights.device))

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
        returns: float [B, n_heads, P]  (dominance scores)

        The most recent batch's `lookup_indices` is exposed as
        `self.last_lookup_indices` (detached, int16) for use by downstream
        optimizers that need to project output gradients back to weight
        space without re-running the anchor lookup. Cleared on next call.
        """
        if not x.is_cuda:
            raise RuntimeError("BitPermutationLUT is CUDA-only")
        if _get_bit_permlut_native() is None:
            raise RuntimeError("lutorch_cuda native extension not available")

        lookup_indices, lookup_alt_indices, _, carriers_main, carriers_alt = self.anchor(x)
        self.last_lookup_indices = lookup_indices.detach()
        if carriers_main is None:
            # Eval / no-grad path: bypass autograd Function.
            native = _get_bit_permlut_native()
            out_int = native.bit_perm_lut_dom_gather_forward(
                lookup_indices.contiguous(),
                self.bit_weights.contiguous(),
                self.inv_idx.contiguous(),
                int(self.n_heads), int(self.tph), int(self.output_nap), int(self.n_pairs),
            )
            return out_int.to(x.dtype) * self.scale

        return _BitPermLutDomFunction.apply(
            lookup_indices, lookup_alt_indices,
            carriers_main, carriers_alt,
            self.bit_weights, self.inv_idx, self.pair_idx_per_slot,
            self.n_heads, self.tph, self.output_nap, self.n_pairs,
            self.scale,
        )
