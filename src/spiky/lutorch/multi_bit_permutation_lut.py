"""K-bit-weight fork of BitMultiHeadLUT / BitPermutationLUT.

Two classes:
  MultiBitMultiHeadLUT   — generic flat-output module, K-bit signed weights.
                            Forward: x [B, n_inputs] -> votes [B, n_heads, n_outputs].
  MultiBitPermutationLUT — pair-dominance specialization (legacy API).
                            `n_outputs` arg means d_head; output dim is
                            P = d_head*(d_head-1)/2.

Differences vs BitMultiHeadLUT / BitPermutationLUT:
  * Latent is always bf16 (no fp8 / fp32).
  * Soft backward is always on (STE through the K-bit quantisation).
  * Supports K ∈ {2, 4, 8} (powers of two that pack cleanly into int32).

Quantisation mapping (no per-table scale):
  latent in bf16  →  q_int = round(latent * 2^(K-1)) clamped to [-2^(K-1), 2^(K-1)-1]
  packed two's-complement K-bit fields, `slots_per_block = 32 / K` per int32.

The "effective" weight contributed to each vote is the signed int in
[-2^(K-1), 2^(K-1)-1]. Forward sums these as int32; the caller-supplied
`scale` rescales the float output. Midrise quantisation adds a per-output
constant offset (`output_bias`) that's applied as a separate buffer.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

from spiky.lutorch.bit_permutation_lut import (
    _build_inv_idx,
    _build_inv_idx_flat,
    _sample_canonical_output_indices,
    _sample_canonical_distinct_output_pairs,
    _get_bit_permlut_native,
)
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.ranking_tools import _canonical_borda_m
from spiky.lutorch.tiny_anchor_pairs_lookup import TinyAnchorPairsLookup


_SUPPORTED_BIT_WIDTHS = (2, 4, 8)


def _compute_n_blocks_k(output_nap: int, bit_width: int) -> int:
    return (output_nap * bit_width + 31) // 32


class _MultiBitLutDomFunction(torch.autograd.Function):
    """Custom autograd for MultiBitMultiHeadLUT forward.

    Forward : CUDA kernel `multi_bit_dom_gather_forward` gathers signed K-bit
              votes through `inv_idx` and returns an int32 sum
              [B, n_heads, n_outputs]. We cast to the requested float dtype
              and apply `scale`.
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
        output_idx_per_table,    # int32 [n_heads, tph, output_nap]
        inv_idx,                 # int32 [n_heads, n_outputs, K_inv]
        n_heads,
        tph,
        output_nap,
        n_outputs,
        bit_width,
        scale,
        temperature,
    ):
        native = _get_bit_permlut_native()
        out_int = native.multi_bit_dom_gather_forward(
            lookup_indices.contiguous(),
            bit_weights.contiguous(),
            inv_idx.contiguous(),
            int(n_heads), int(tph), int(output_nap), int(n_outputs),
            int(bit_width),
        )
        out_float = out_int.to(carriers_main.dtype) * scale

        ctx.save_for_backward(
            lookup_indices, lookup_alt_indices, latent_bf16, output_idx_per_table,
        )
        ctx.n_heads = int(n_heads)
        ctx.tph = int(tph)
        ctx.output_nap = int(output_nap)
        ctx.n_outputs = int(n_outputs)
        ctx.scale = float(scale)
        ctx.temperature = float(temperature)
        return out_float

    @staticmethod
    def backward(ctx, grad_out):
        native = _get_bit_permlut_native()
        lookup_indices, lookup_alt_indices, latent_bf16, output_idx_per_table = ctx.saved_tensors

        grad_main, grad_alt = native.multi_bit_dom_gather_backward_latent_bf16(
            grad_out.contiguous().to(torch.float32),
            lookup_indices, lookup_alt_indices,
            latent_bf16, output_idx_per_table,
            ctx.n_heads, ctx.tph, ctx.output_nap, ctx.n_outputs, ctx.scale,
            ctx.temperature,
        )

        return (
            None,             # lookup_indices
            None,             # lookup_alt_indices
            grad_main,        # carriers_main
            grad_alt,         # carriers_alt
            None,             # bit_weights
            None,             # latent_bf16
            None,             # output_idx_per_table
            None,             # inv_idx
            None, None, None, None, None, None, None,  # scalars (+ temperature)
        )


class MultiBitMultiHeadLUT(nn.Module):
    """Generic K-bit-weight multi-head LUT.

    Forward: x (float [B, n_inputs]) -> votes (float [B, n_heads, n_outputs]).
    Each output channel accumulates the sum of signed K-bit votes drawn from
    `tph * output_nap` table slots through `inv_idx`. The output is purely a
    flat voting space; downstream is free to interpret it as pair-dominance,
    logits, or any other vector.

    Args:
        n_inputs:               input dimension.
        n_outputs:              flat output dimension P (number of voting
                                channels).
        n_heads:                number of heads (logical — n_tables = n_heads * tph).
        input_nap:              anchor pairs per table for input lookup (1..15).
        output_nap:             signed votes per table (≤ n_outputs).
        tph:                    tables per head.
        bit_width:              bits per signed weight; one of {2, 4, 8}.
        random_seed:            seed for anchor + output-channel sampling +
                                latent init.
        output_idx_per_table:   optional override for the
                                [n_heads, tph, output_nap] integer tensor
                                mapping (head, table, slot) -> output channel
                                in [0, n_outputs). If None, balanced
                                canonical-coverage flat sampling is used.
        scale:                  multiplier applied to the int vote-sum kernel
                                output (forward and backward). Default 1.0
                                (raw signed sums); wrappers with a fixed
                                downstream interpretation (e.g.
                                MultiBitPermutationLUT for pair-dominance)
                                pass an explicit value. The midrise
                                quantisation correction `output_bias` is
                                computed using `scale`.
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
        output_idx_per_table: Optional[torch.Tensor] = None,
        scale: float = 1.0,
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
        if n_outputs < 1:
            raise ValueError(f"n_outputs must be >= 1, got {n_outputs}")

        dev = torch.device(device) if device is not None else torch.device("cpu")
        if dev.type != "cuda":
            raise RuntimeError("MultiBitMultiHeadLUT requires CUDA")

        self.n_inputs = n_inputs
        self.n_outputs = int(n_outputs)
        self.n_heads = n_heads
        self.input_nap = input_nap
        self.output_nap = output_nap
        self.tph = tph
        self.bit_width = int(bit_width)
        self.table_dim = 1 << input_nap
        self.n_blocks_k = _compute_n_blocks_k(output_nap, bit_width)
        self.ste_gate_temperature = float(ste_gate_temperature)
        self.pre_quant_temperature = float(pre_quant_temperature)
        self.half_range = 1 << (bit_width - 1)  # 2^(K-1) — caller's scale derivation

        input_policy = (
            input_anchor_sampling_policy
            if input_anchor_sampling_policy is not None
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

        # Latent init — uniform U(-a, +a), bf16.
        if random_seed is not None:
            gen = torch.Generator(device=dev).manual_seed(random_seed + 4_000_003)
        else:
            gen = None
        shape = (n_heads * tph, self.table_dim, output_nap)
        latent_init_f32 = (
            torch.rand(shape, device=dev, generator=gen) - 0.5
        ) * (2.0 * float(initial_weights_noise))
        self.register_buffer('latent_bf16', latent_init_f32.to(torch.bfloat16).contiguous())

        # Output channel index per slot.
        if output_idx_per_table is None:
            output_idx_per_table = _sample_canonical_output_indices(
                n_heads=n_heads, tph=tph, output_nap=output_nap,
                n_outputs=self.n_outputs, random_seed=random_seed, device=dev,
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

        # Packed bit weights (zero-init; refresh fills from latent).
        bit_weights = torch.zeros(
            n_heads * tph, self.table_dim, self.n_blocks_k,
            device=dev, dtype=torch.int32,
        )
        self.register_buffer('bit_weights', bit_weights.contiguous())
        self.refresh_bit_weights()

        self.scale = float(scale)

        # Midrise quantisation correction. Each vote's effective value is
        # (2q+1)/(2*half_range) rather than q/half_range; the extra
        # "+1/(2*half_range)" per vote accumulates into a per-output DC
        # offset of `count_per_output * scale / 2` (once `scale` already
        # folds in 1/half_range, which is the legacy convention).
        count_per_output = (inv_idx >= 0).sum(dim=-1).to(torch.float32)  # [n_heads, n_outputs]
        output_bias = 0.5 * count_per_output * self.scale
        self.register_buffer('output_bias', output_bias.contiguous())

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
        """x: [B, n_inputs] -> votes [B, n_heads, n_outputs]."""
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
                int(self.n_outputs), int(self.bit_width),
            )
            return out_int.to(x.dtype) * self.scale + self.output_bias

        return _MultiBitLutDomFunction.apply(
            lookup_indices, lookup_alt_indices, carriers_main, carriers_alt,
            self.bit_weights, self.latent_bf16, self.output_idx_per_table, self.inv_idx,
            self.n_heads, self.tph, self.output_nap, self.n_outputs,
            self.bit_width, self.scale, self.pre_quant_temperature,
        ) + self.output_bias

    def extra_repr(self) -> str:
        return (
            f"n_inputs={self.n_inputs}, n_outputs={self.n_outputs}, "
            f"n_heads={self.n_heads}, input_nap={self.input_nap}, "
            f"output_nap={self.output_nap}, tph={self.tph}, "
            f"bit_width={self.bit_width}"
        )


class MultiBitPermutationLUT(MultiBitMultiHeadLUT):
    """Pair-dominance specialization of MultiBitMultiHeadLUT (legacy API).

    Treats the output as the C(d_head, 2) pair-dominance vector for d_head
    ranked items: each output channel corresponds to a canonical (a, b) pair
    with a < b. Constructor signature is identical to the original
    MultiBitPermutationLUT — `n_outputs` here means the rank dim d_head, and
    the actual output dim is P = d_head*(d_head-1)/2 (exposed as `n_pairs`).

    For non-dominance use cases (logits, generic feature voting, etc.) use
    MultiBitMultiHeadLUT directly with `n_outputs` set to the literal
    output dim.
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
        if bit_width not in _SUPPORTED_BIT_WIDTHS:
            raise ValueError(f"bit_width must be in {_SUPPORTED_BIT_WIDTHS}, got {bit_width}")
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

        # Sample canonical pairs and map to flat triu indices so seeded runs
        # match the pre-redesign behaviour.
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

        # Pair-dominance interpretation: each output is a sum of N signed
        # K-bit votes where N ≈ tph*output_nap/n_pairs. Normalise so per-pair
        # dominance sits at the same scale as BitPermutationLUT (the legacy
        # `0.5 / (half_range * sqrt(nvpp))` convention).
        half_range = 1 << (bit_width - 1)
        n_votes_per_pair = tph * output_nap / float(n_pairs)
        dom_scale = 0.5 / (half_range * math.sqrt(max(n_votes_per_pair, 1.0)))

        super().__init__(
            n_inputs=n_inputs,
            n_outputs=n_pairs,
            n_heads=n_heads,
            input_nap=input_nap,
            output_nap=output_nap,
            tph=tph,
            bit_width=bit_width,
            random_seed=random_seed,
            initial_weights_noise=initial_weights_noise,
            ste_gate_temperature=ste_gate_temperature,
            pre_quant_temperature=pre_quant_temperature,
            device=device,
            partition_sets=partition_sets,
            input_anchor_sampling_policy=input_policy,
            output_idx_per_table=output_idx_per_table,
            scale=dom_scale,
        )

        # Pair-dominance specifics. `self.n_outputs` from the parent equals
        # n_pairs; we expose `n_pairs` and `d_head` for callers that want
        # the rank-dim semantics.
        self.d_head = d_head
        self.n_pairs = n_pairs
        self.register_buffer('idx_a', a_can.contiguous())
        self.register_buffer('idx_b', b_can.contiguous())
        self.register_buffer('dom_borda_m', _canonical_borda_m(d_head, dev))
