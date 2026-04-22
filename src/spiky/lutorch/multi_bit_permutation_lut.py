"""MultiBitPermutationLUT — K-bit-weight fork of BitPermutationLUT.

Simplifications vs BitPermutationLUT:
  * Latent is always bf16 (no fp8 / fp32).
  * Soft backward is always on (STE through the K-bit quantisation).
  * Supports K ∈ {2, 4, 8} (powers of two that pack cleanly into int32).

Quantisation mapping (no per-table scale):
  latent in bf16  →  q_int = round(latent * 2^(K-1)) clamped to [-2^(K-1), 2^(K-1)-1]
  packed two's-complement K-bit fields, `slots_per_block = 32 / K` per int32.

The "effective" weight contributed to each vote is the signed int in
[-2^(K-1), 2^(K-1)-1]. Forward sums these as int32; the caller applies
`scale = 0.5 / (2^(K-1) * sqrt(n_votes_per_pair))` so the post-scale output
has comparable range to BitPermutationLUT's ±1 summed votes.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from spiky.lutorch.bit_permutation_lut import (
    _build_inv_idx,
    _get_bit_permlut_native,
)
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import _canonical_borda_m
from spiky.lutorch.tiny_anchor_pairs_lookup import TinyAnchorPairsLookup


_SUPPORTED_BIT_WIDTHS = (2, 4, 8)


def _compute_n_blocks_k(output_nap: int, bit_width: int) -> int:
    return (output_nap * bit_width + 31) // 32


class _MultiBitLutDomFunction(torch.autograd.Function):
    """Custom autograd for MultiBitPermutationLUT forward.

    Forward : CUDA kernel `multi_bit_dom_gather_forward` gathers signed K-bit
              votes through `inv_idx` and returns an int32 sum [B, n_heads, P].
              We cast to the requested float dtype and apply `scale`.
    Backward: soft STE through the bf16 latent. With `temperature > 0` uses
              `rational(latent, T)` as the effective weight for the input-side
              grad — matches PermLut+vote_quant semantics so training dynamics
              track the rational-then-quantize pipeline used there.
    """

    @staticmethod
    def forward(
        ctx,
        lookup_indices,          # int16 [B, n_heads*tph]
        lookup_alt_indices,      # int16 [B, n_heads*tph, 1]
        carriers_main,           # float [B, n_heads*tph]
        carriers_alt,            # float [B, n_heads*tph, 1]
        bit_weights,             # int32 [n_heads*tph, table_dim, n_blocks_k]
        latent_bf16,             # bf16  [n_heads*tph, table_dim, output_nap]
        pair_idx_per_slot,       # int32 [n_heads, tph, output_nap]
        inv_idx,                 # int32 [n_heads, P, K_inv]
        n_heads,
        tph,
        output_nap,
        n_pairs,
        bit_width,
        scale,
        temperature,
    ):
        native = _get_bit_permlut_native()
        out_int = native.multi_bit_dom_gather_forward(
            lookup_indices.contiguous(),
            bit_weights.contiguous(),
            inv_idx.contiguous(),
            int(n_heads), int(tph), int(output_nap), int(n_pairs),
            int(bit_width),
        )
        out_float = out_int.to(carriers_main.dtype) * scale

        ctx.save_for_backward(
            lookup_indices, lookup_alt_indices, latent_bf16, pair_idx_per_slot,
        )
        ctx.n_heads = int(n_heads)
        ctx.tph = int(tph)
        ctx.output_nap = int(output_nap)
        ctx.n_pairs = int(n_pairs)
        ctx.scale = float(scale)
        ctx.temperature = float(temperature)
        return out_float

    @staticmethod
    def backward(ctx, grad_out):
        native = _get_bit_permlut_native()
        lookup_indices, lookup_alt_indices, latent_bf16, pair_idx_per_slot = ctx.saved_tensors

        grad_main, grad_alt = native.multi_bit_dom_gather_backward_latent_bf16(
            grad_out.contiguous().to(torch.float32),
            lookup_indices, lookup_alt_indices,
            latent_bf16, pair_idx_per_slot,
            ctx.n_heads, ctx.tph, ctx.output_nap, ctx.n_pairs, ctx.scale,
            ctx.temperature,
        )

        return (
            None,             # lookup_indices
            None,             # lookup_alt_indices
            grad_main,        # carriers_main
            grad_alt,         # carriers_alt
            None,             # bit_weights
            None,             # latent_bf16
            None,             # pair_idx_per_slot
            None,             # inv_idx
            None, None, None, None, None, None, None,  # scalars (+ temperature)
        )


class MultiBitPermutationLUT(nn.Module):
    """K-bit-weight LUT. See module docstring for semantics.

    Args:
        n_inputs, n_outputs, n_heads, input_nap, output_nap, tph:
            same geometry as BitPermutationLUT.
        bit_width: int in {2, 4, 8}. Bits per signed weight.
        random_seed: controls anchor sampling + output pair sampling + latent init.
        initial_weights_noise: amplitude ``a`` of uniform ``U(-a, +a)`` latent init.
        ste_gate_temperature: kept for parity / optimizer use.
        partition_sets, anchor_sampling_policy, input_/output_anchor_sampling_policy:
            same semantics as BitPermutationLUT.
        device: target device.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_heads: int,
        input_nap: int,
        output_nap: int,
        tph: int,
        bit_width: int = 4,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        ste_gate_temperature: float = 0.1,
        pre_quant_temperature: float = 0.1,
        device: Optional[torch.device] = None,
        partition_sets: Optional[list] = None,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        input_anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        output_anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
    ):
        super().__init__()
        if bit_width not in _SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be in {_SUPPORTED_BIT_WIDTHS}, got {bit_width}")
        if not (1 <= input_nap <= 15):
            raise ValueError(
                f"input_nap must be in [1, 15] (TinyAnchorPairsLookup int16 limit), "
                f"got {input_nap}"
            )
        if output_nap <= 0:
            raise ValueError(f"output_nap must be positive, got {output_nap}")
        if n_outputs < 2:
            raise ValueError(f"n_outputs must be >= 2, got {n_outputs}")

        dev = torch.device(device) if device is not None else torch.device("cpu")
        if dev.type != "cuda":
            raise RuntimeError("MultiBitPermutationLUT requires CUDA")

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_heads = n_heads
        self.input_nap = input_nap
        self.output_nap = output_nap
        self.tph = tph
        self.bit_width = int(bit_width)
        self.table_dim = 1 << input_nap
        self.n_blocks_k = _compute_n_blocks_k(output_nap, bit_width)
        self.n_pairs = n_outputs * (n_outputs - 1) // 2
        self.ste_gate_temperature = float(ste_gate_temperature)
        self.pre_quant_temperature = float(pre_quant_temperature)
        self.half_range = 1 << (bit_width - 1)  # 2^(K-1) — for scale derivation

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

        # Anchor lookup.
        self.anchor = TinyAnchorPairsLookup(
            input_dim=n_inputs,
            n_tables=n_heads * tph,
            n_anchor_pairs=input_nap,
            n_heads=n_heads,
            random_seed=random_seed,
            device=dev,
            partition_sets=partition_sets,
            anchor_sampling_policy=input_policy,
        )

        # Latent init — uniform U(-a, +a), matching BitPermLUT shape.
        if random_seed is not None:
            gen = torch.Generator(device=dev).manual_seed(random_seed + 4_000_003)
        else:
            gen = None
        shape = (n_heads * tph, self.table_dim, output_nap)
        latent_init_f32 = (
            torch.rand(shape, device=dev, generator=gen) - 0.5
        ) * (2.0 * float(initial_weights_noise))
        self.register_buffer('latent_bf16', latent_init_f32.to(torch.bfloat16).contiguous())

        # Output-pair structures + bit_weights buffer.
        idx_a, idx_b, inv_idx, K_max, pair_idx_per_slot = _build_inv_idx(
            n_heads=n_heads, tph=tph, output_nap=output_nap,
            n_outputs=n_outputs, random_seed=random_seed, device=dev,
            anchor_sampling_policy=output_policy,
        )
        self.register_buffer('idx_a', idx_a.contiguous())
        self.register_buffer('idx_b', idx_b.contiguous())
        self.register_buffer('inv_idx', inv_idx.contiguous())
        self.register_buffer('pair_idx_per_slot', pair_idx_per_slot.contiguous())
        self.K_max = K_max

        bit_weights = torch.zeros(
            n_heads * tph, self.table_dim, self.n_blocks_k,
            device=dev, dtype=torch.int32,
        )
        self.register_buffer('bit_weights', bit_weights.contiguous())

        # Initial pack.
        self.refresh_bit_weights()

        # Forward float scale: int sum in [-half_range·nvpp, +half_range·nvpp-...].
        # Divide by half_range to normalise each vote to [-1, 1), then by
        # sqrt(n_votes_per_pair) * 2 for Borda-matched scale (same convention
        # as BitPermutationLUT which used 0.5 / sqrt(nvpp)).
        n_votes_per_pair = tph * output_nap / float(self.n_pairs)
        self.scale = 0.5 / (self.half_range * math.sqrt(max(n_votes_per_pair, 1.0)))

        # Midrise correction: each vote's effective value is (2q+1)/(2*half_range)
        # rather than q/half_range. The extra "+1/(2*half_range)" per vote
        # accumulates into a per-pair DC offset: count_per_pair * scale / 2
        # (once `scale` already includes the 1/half_range factor).
        # count_per_pair derived from inv_idx (non-(-1) entries per pair).
        count_per_pair = (inv_idx >= 0).sum(dim=-1).to(torch.float32)  # [n_heads, n_pairs]
        pair_bias = 0.5 * count_per_pair * self.scale                 # [n_heads, n_pairs]
        self.register_buffer('pair_bias', pair_bias.contiguous())

        self.register_buffer('dom_borda_m', _canonical_borda_m(n_outputs, dev))

    # -- helpers ----------------------------------------------------------

    def refresh_bit_weights(self) -> None:
        """Pack the current bf16 latent into bit_weights via multi_bit_pack.
        Applies rational(latent, T=pre_quant_temperature) before quantisation."""
        native = _get_bit_permlut_native()
        native.multi_bit_pack(
            self.latent_bf16.contiguous(),
            self.bit_weights,
            int(self.output_nap),
            int(self.bit_width),
            float(self.pre_quant_temperature),
        )

    def _latent_for_gate(self) -> torch.Tensor:
        """Return the latent as float for the STE gate in the optimizer."""
        return self.latent_bf16

    # -- forward ----------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_inputs] -> dominance [B, n_heads, P]."""
        (
            lookup_indices, lookup_alt_indices, lookup_alt_deltas,
            carriers_main, carriers_alt,
        ) = self.anchor(x)

        if carriers_main is None:
            # Eval / no-grad fast path: bypass autograd Function.
            native = _get_bit_permlut_native()
            out_int = native.multi_bit_dom_gather_forward(
                lookup_indices.contiguous(),
                self.bit_weights.contiguous(),
                self.inv_idx.contiguous(),
                int(self.n_heads), int(self.tph), int(self.output_nap),
                int(self.n_pairs), int(self.bit_width),
            )
            return out_int.to(x.dtype) * self.scale + self.pair_bias

        return _MultiBitLutDomFunction.apply(
            lookup_indices, lookup_alt_indices, carriers_main, carriers_alt,
            self.bit_weights, self.latent_bf16, self.pair_idx_per_slot, self.inv_idx,
            self.n_heads, self.tph, self.output_nap, self.n_pairs,
            self.bit_width, self.scale, self.pre_quant_temperature,
        ) + self.pair_bias

    def extra_repr(self) -> str:
        return (
            f"n_inputs={self.n_inputs}, n_outputs={self.n_outputs}, "
            f"n_heads={self.n_heads}, input_nap={self.input_nap}, "
            f"output_nap={self.output_nap}, tph={self.tph}, "
            f"bit_width={self.bit_width}"
        )
