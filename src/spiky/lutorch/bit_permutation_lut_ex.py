"""BitPermutationLUTEx — BitPermutationLUT + output-table WTA layer.

Architecture (two sequential steps):

    Part 1  x [B, E]   -->  votes [B, H, V_ex]     (fused kernel, gather ±1 bits)
    Part 2  votes      -->  dominance [B, H, P]    (WTA per output table + scatter)

Part 1 is the same fused kernel that powers BitPermutationLUT. The only
differences are the voting-space size ``V_ex = output_tph · 2^output_nap``
and the routing (per-table, not per-canonical-pair), both expressed via
``output_idx_per_table_ex`` / ``inv_idx_ex`` buffers. This means:

  - the kernel is reused verbatim (no new CUDA code);
  - ``BitPermutationLUTOptimizer`` trains the latent unchanged — pass
    ``[lut.voting]`` to it; the optimizer's `grad_out` hook and
    `_project_grad_out_to_weight_grad` operate on whatever "pair" space
    ``output_idx_per_table`` defines.

Part 2:
  1. Reshape [B, H, V_ex] -> [B, H, output_tph, 2^output_nap].
  2. WTALookup per output table (with soft grad via ``n_alternatives``).
  3. Gather the winner entry's ±1 pattern over its `output_nap` canonical
     pairs, scatter those into the final [B, H, P] dominance.

Hyperparameters:
    input_nap, input_tph       : input anchor lookup.
    voting_nap                 : ±1 bits per input entry (parent's output_nap).
    output_nap, output_tph     : output-table shape.
"""
import math
from typing import Optional

import torch
import torch.nn as nn

from spiky.lutorch.bit_permutation_lut import (
    BitPermutationLUTInput,
    _BitPermLutDomFunction,
    _canonical_borda_m,
)
from spiky.lutorch.lut_helpers import UncertaintyMode
from spiky.lutorch.wta_lookup import WTALookupFunction


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _build_routing_and_inv_idx(
    n_heads: int,
    input_tph: int,
    voting_nap: int,
    V: int,
    random_seed: Optional[int],
    device: torch.device,
):
    """Per-table routing over V-dim virtual-pair space.

    Returns
    -------
    output_idx_per_table : int32 [H, input_tph, voting_nap]
        target[h, t, v] ∈ [0, V) — the virtual pair each (table, slot)
        contributes to.
    inv_idx : int32 [H, V, K_max]
        for each virtual pair, list of contributing slot_idx
        (= table · voting_nap + slot), padded with -1.
    K_max : int
    """
    gen = None
    if random_seed is not None:
        gen = torch.Generator(device=device).manual_seed(random_seed + 6_000_003)
    routing = torch.randint(
        0, V, (n_heads, input_tph, voting_nap),
        generator=gen, device=device, dtype=torch.long,
    )
    output_idx_per_table = routing.to(torch.int32).contiguous()

    TP = input_tph * voting_nap
    routing_flat = routing.reshape(n_heads, TP)
    counts = torch.stack(
        [torch.bincount(routing_flat[h], minlength=V) for h in range(n_heads)],
        dim=0,
    )
    K_max = int(counts.max().item())

    inv_idx = torch.full((n_heads, V, K_max), -1, dtype=torch.int32, device=device)
    sort_order = routing_flat.argsort(dim=1, stable=True)
    routing_sorted = routing_flat.gather(1, sort_order)
    starts = torch.cat(
        [torch.zeros(n_heads, 1, dtype=counts.dtype, device=device),
         counts.cumsum(dim=1)[:, :-1]],
        dim=1,
    )
    pos = torch.arange(TP, device=device).unsqueeze(0).expand(n_heads, -1)
    within_group = pos - starts.gather(1, routing_sorted)
    h_idx = torch.arange(n_heads, device=device).unsqueeze(1).expand(-1, TP)
    inv_idx[h_idx, routing_sorted, within_group] = sort_order.to(torch.int32)
    return output_idx_per_table, inv_idx.contiguous(), K_max


def _build_output_table_pairs(
    n_heads: int, output_tph: int, output_nap: int, n_outputs: int,
    random_seed: Optional[int], device: torch.device,
) -> torch.Tensor:
    """[H, output_tph, output_nap] long — distinct canonical pair indices per table."""
    P = n_outputs * (n_outputs - 1) // 2
    if output_nap > P:
        raise ValueError(f"output_nap ({output_nap}) > C(n_outputs, 2) = {P}")
    gen = None
    if random_seed is not None:
        gen = torch.Generator(device=device).manual_seed(random_seed + 7_000_003)
    out = torch.empty(n_heads, output_tph, output_nap, dtype=torch.long, device=device)
    for h in range(n_heads):
        for t in range(output_tph):
            out[h, t] = torch.randperm(P, generator=gen, device=device)[:output_nap]
    return out


def _build_entry_patterns(output_nap: int, device: torch.device) -> torch.Tensor:
    """[2^output_nap, output_nap] ±1 — binary expansion of entry index."""
    D = 1 << output_nap
    k = torch.arange(D, device=device)
    pos = torch.arange(output_nap, device=device)
    bits = ((k.unsqueeze(1) >> pos.unsqueeze(0)) & 1).float()
    return bits * 2.0 - 1.0


# ----------------------------------------------------------------------------
# Part 1 submodule (reuses the fused kernel in a virtual-pair space)
# ----------------------------------------------------------------------------

class BitPermutationLUTVoting(BitPermutationLUTInput):
    """Part 1: anchor lookup + bit gather + scatter into V-dim virtual pair space.

    Forward output: [B, n_heads, V] float (± 0.5/√votes_per_V scaled sum).

    This module is intentionally hookable by `BitPermutationLUTOptimizer`:
    pass `[ex.voting]` to the optimizer and it trains the latent normally.
    `output_idx_per_table` and `inv_idx` are supplied externally so the parent
    class (Ex) controls the routing.
    """

    def __init__(
        self,
        n_inputs: int,
        n_heads: int,
        input_nap: int,
        voting_nap: int,
        input_tph: int,
        V: int,
        output_idx_per_table: torch.Tensor,   # int32 [H, input_tph, voting_nap]
        inv_idx: torch.Tensor,              # int32 [H, V, K_max]
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        soft_backward: bool = False,
        latent_dtype: str = 'fp8',
        device: Optional[torch.device] = None,
        partition_sets: Optional[list] = None,
    ):
        super().__init__(
            n_inputs=n_inputs, n_heads=n_heads,
            input_nap=input_nap, output_nap=voting_nap, tph=input_tph,
            random_seed=random_seed,
            initial_weights_noise=initial_weights_noise,
            latent_dtype=latent_dtype, device=device,
            partition_sets=partition_sets,
        )
        self.V = int(V)
        self.soft_backward = bool(soft_backward)

        self.register_buffer('output_idx_per_table', output_idx_per_table.contiguous())
        self.register_buffer('inv_idx', inv_idx.contiguous())

        n_votes_per_V = input_tph * voting_nap / float(max(V, 1))
        self.scale = 0.5 / math.sqrt(max(n_votes_per_V, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from spiky.lutorch.bit_permutation_lut import _get_bit_permlut_native
        lookup_indices, lookup_alt_indices, _, carriers_main, carriers_alt = self.anchor(x)

        # Eval / no-grad path: bypass autograd Function (carriers are None).
        if carriers_main is None:
            native = _get_bit_permlut_native()
            out_int = native.bit_perm_lut_dom_gather_forward(
                lookup_indices.contiguous(),
                self.bit_weights.contiguous(),
                self.inv_idx.contiguous(),
                int(self.n_heads), int(self.tph), int(self.output_nap), int(self.V),
            )
            return out_int.to(x.dtype) * self.scale

        # Train / grad path — hand off to _BitPermLutDomFunction.
        # For fp32/bf16, `latent_scale` is unused; pass bit_weights as placeholder.
        if self.latent_dtype == 'fp8':
            latent_fp8 = self.latent_fp8
            latent_scale = self.latent_scale
        elif self.latent_dtype == 'bf16':
            latent_fp8 = self.latent_bf16
            latent_scale = self.bit_weights   # placeholder
        else:
            latent_fp8 = self.latent_fp32
            latent_scale = self.bit_weights   # placeholder

        return _BitPermLutDomFunction.apply(
            lookup_indices, lookup_alt_indices, carriers_main, carriers_alt,
            self.bit_weights, latent_fp8, latent_scale,
            self.inv_idx, self.output_idx_per_table,
            self.n_heads, self.tph, self.output_nap,
            self.V, self.scale, self.soft_backward,
        )


# ----------------------------------------------------------------------------
# Full Ex = voting (Part 1) + WTA + pattern scatter (Part 2)
# ----------------------------------------------------------------------------

class BitPermutationLUTEx(nn.Module):
    """BitPermutationLUT + output-table WTA layer.

    Use:
        lut = BitPermutationLUTEx(...)
        opt = BitPermutationLUTOptimizer([lut.voting], lr=...)   # trains latent

    Args:
        n_inputs, n_outputs, n_heads
        input_nap, input_tph       : input-side anchor lookup config.
        voting_nap                 : ±1 votes per input entry.
        output_nap, output_tph     : output-table config (output_nap
                                     canonical pairs per table, 2^output_nap
                                     entries).
        random_seed                : seed; distinct offsets used internally
                                     for anchor / routing / output-pair sampling.
        soft_backward              : passthrough to Part 1 kernel's STE path.
        latent_dtype               : 'fp8' / 'bf16' / 'fp32'.
        wta_n_alternatives         : 1/2/3 — WTA soft-grad runners-up.
        uncertainty_mode           : INVERSE_L1 or INVERSE_QUADRATIC.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_heads: int,
        input_nap: int,
        input_tph: int,
        voting_nap: int,
        output_nap: int,
        output_tph: int,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        soft_backward: bool = False,
        latent_dtype: str = 'fp8',
        wta_n_alternatives: int = 1,
        uncertainty_mode: UncertaintyMode = UncertaintyMode.INVERSE_L1,
        device: Optional[torch.device] = None,
        partition_sets: Optional[list] = None,
    ):
        super().__init__()
        if n_outputs < 2:
            raise ValueError(f"n_outputs must be >= 2, got {n_outputs}")
        if output_nap <= 0 or output_nap > 16:
            raise ValueError(f"output_nap must be in [1, 16], got {output_nap}")
        if wta_n_alternatives not in (1, 2, 3):
            raise ValueError(f"wta_n_alternatives must be 1, 2, or 3, got {wta_n_alternatives}")

        dev = torch.device(device) if device is not None else torch.device("cpu")

        self.n_outputs = n_outputs
        self.n_heads = n_heads
        self.output_nap_out = int(output_nap)
        self.output_tph = int(output_tph)
        self.output_table_dim = 1 << output_nap
        self.V_ex = self.output_tph * self.output_table_dim
        self.n_pairs = n_outputs * (n_outputs - 1) // 2
        self.wta_n_alternatives = int(wta_n_alternatives)
        self.uncertainty_mode = uncertainty_mode
        self._uncertainty_mode_int = 0 if uncertainty_mode == UncertaintyMode.INVERSE_L1 else 1

        # Per-table routing + inv_idx over the virtual-pair space V_ex.
        output_idx_per_table_ex, inv_idx_ex, K_max = _build_routing_and_inv_idx(
            n_heads=n_heads, input_tph=input_tph, voting_nap=voting_nap,
            V=self.V_ex, random_seed=random_seed, device=dev,
        )
        self.K_max = K_max

        # Part 1 submodule — hookable by BitPermutationLUTOptimizer.
        self.voting = BitPermutationLUTVoting(
            n_inputs=n_inputs, n_heads=n_heads,
            input_nap=input_nap, voting_nap=voting_nap, input_tph=input_tph,
            V=self.V_ex,
            output_idx_per_table=output_idx_per_table_ex,
            inv_idx=inv_idx_ex,
            random_seed=random_seed,
            initial_weights_noise=initial_weights_noise,
            soft_backward=soft_backward,
            latent_dtype=latent_dtype, device=device,
            partition_sets=partition_sets,
        )

        # Part 2: output-table canonical pairs + ±1 entry patterns + Borda.
        self.register_buffer(
            'output_pair_indices',
            _build_output_table_pairs(
                n_heads, output_tph, output_nap, n_outputs, random_seed, dev,
            ),
        )
        self.register_buffer('entry_patterns', _build_entry_patterns(output_nap, dev))
        self.register_buffer('dom_borda_m', _canonical_borda_m(n_outputs, dev))

        # WTA batch_offset cache.
        self._wta_batch_offset: Optional[torch.Tensor] = None

    def _get_wta_batch_offset(self, bht: int, device: torch.device) -> torch.Tensor:
        n_alt = self.wta_n_alternatives
        expected = bht * n_alt
        if (
            self._wta_batch_offset is None
            or self._wta_batch_offset.numel() != expected
            or self._wta_batch_offset.device != device
        ):
            self._wta_batch_offset = (
                torch.arange(bht, device=device, dtype=torch.long)
                .repeat_interleave(n_alt) * self.output_table_dim
            ).contiguous()
        return self._wta_batch_offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, n_inputs] -> dominance [B, n_heads, n_pairs]."""
        B = x.shape[0]
        H = self.n_heads
        T_out = self.output_tph
        D_out = self.output_table_dim
        on_out = self.output_nap_out
        P = self.n_pairs

        # Part 1: hookable submodule — BitPermutationLUTOptimizer trains this.
        votes = self.voting(x)                                           # [B, H, V_ex]

        # Part 2: WTA over each output table + pattern scatter to P.
        scores_wta = votes.view(B, H, T_out, D_out).reshape(B * H, T_out, D_out).contiguous()
        bht = B * H * T_out
        batch_offset = self._get_wta_batch_offset(bht, x.device)
        winner_idx, alt_idx, _alt_deltas, grad_c_winner, grad_c_alt = (
            WTALookupFunction.apply(
                scores_wta, self.wta_n_alternatives,
                self._uncertainty_mode_int, batch_offset,
            )
        )

        winner_patterns = self.entry_patterns[winner_idx]                 # [B*H, T_out, on_out]
        alt_patterns    = self.entry_patterns[alt_idx]                    # [B*H, T_out, n_alt, on_out]

        effective = winner_patterns + winner_patterns.detach() * grad_c_winner.unsqueeze(-1)
        effective = effective + (alt_patterns.detach() * grad_c_alt.unsqueeze(-1)).sum(dim=2)
        effective = effective.view(B, H, T_out, on_out)

        pair_idx_flat = (
            self.output_pair_indices.view(1, H, T_out * on_out).expand(B, -1, -1).contiguous()
        )
        dominance = torch.zeros(B, H, P, device=x.device, dtype=effective.dtype)
        dominance.scatter_add_(
            2, pair_idx_flat, effective.reshape(B, H, T_out * on_out),
        )
        return dominance


# ----------------------------------------------------------------------------
# Simpler Part-2 variant: learnable linear readout V → P (no WTA, no output tables).
# ----------------------------------------------------------------------------

class BitPermutationLUTLin(nn.Module):
    """BitPermutationLUT + learnable linear readout from voting space V to P.

    Same Part 1 (BitPermutationLUTVoting) as BitPermutationLUTEx — gathers ±1
    bits and scatters them into an arbitrary V-dim voting space. Part 2 is a
    per-head learnable matrix W ∈ ℝ^{n_heads × V × n_pairs} instead of the
    WTA + output-table scatter. Output: `dominance[b, h, p] = Σ_v votes[b,h,v] · W[h,v,p]`.

    Rationale: WTA-STE training can have a large generalization gap (soft
    gradients optimize a relaxation that eval's argmax does not match). A
    linear readout keeps training and eval on the same deterministic path,
    trading the combinatorial interpretation for a cleaner gradient flow.

    Args:
        n_inputs, n_outputs, n_heads
        input_nap, input_tph       : input-side anchor lookup config.
        voting_nap                 : ±1 votes per input entry.
        V                          : voting-space size (routing target range).
        random_seed, initial_weights_noise, soft_backward, latent_dtype, device
        readout_init_scale         : initial std of W; defaults to 1/sqrt(V).
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_heads: int,
        input_nap: int,
        input_tph: int,
        voting_nap: int,
        V: int,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        soft_backward: bool = False,
        latent_dtype: str = 'fp8',
        device: Optional[torch.device] = None,
        readout_init_scale: Optional[float] = None,
        partition_sets: Optional[list] = None,
    ):
        super().__init__()
        if n_outputs < 2:
            raise ValueError(f"n_outputs must be >= 2, got {n_outputs}")
        if V <= 0:
            raise ValueError(f"V must be positive, got {V}")

        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.n_outputs = n_outputs
        self.n_heads = n_heads
        self.n_pairs = n_outputs * (n_outputs - 1) // 2
        self.V = int(V)

        output_idx_per_table_ex, inv_idx_ex, K_max = _build_routing_and_inv_idx(
            n_heads=n_heads, input_tph=input_tph, voting_nap=voting_nap,
            V=self.V, random_seed=random_seed, device=dev,
        )
        self.K_max = K_max

        self.voting = BitPermutationLUTVoting(
            n_inputs=n_inputs, n_heads=n_heads,
            input_nap=input_nap, voting_nap=voting_nap, input_tph=input_tph,
            V=self.V,
            output_idx_per_table=output_idx_per_table_ex,
            inv_idx=inv_idx_ex,
            random_seed=random_seed,
            initial_weights_noise=initial_weights_noise,
            soft_backward=soft_backward,
            latent_dtype=latent_dtype, device=device,
            partition_sets=partition_sets,
        )

        if readout_init_scale is None:
            readout_init_scale = 1.0 / math.sqrt(float(self.V))
        gen = torch.Generator(device=dev)
        if random_seed is not None:
            gen.manual_seed(random_seed + 8_000_003)
        self.readout = nn.Parameter(
            torch.randn(n_heads, self.V, self.n_pairs, generator=gen, device=dev)
            * float(readout_init_scale)
        )

        # Keep the output's std ≈ 1 per (B, H) sample regardless of readout
        # drift — matches the CLT-unit-variance scaling that BitPermutationLUT's
        # `scale = 0.5/√votes_per_pair` provides for the direct-scatter path.
        # Affine=False so the norm itself adds no trainable parameters.
        self.output_norm = nn.LayerNorm(self.n_pairs, elementwise_affine=False)

        self.register_buffer('dom_borda_m', _canonical_borda_m(n_outputs, dev))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        votes = self.voting(x)                                           # [B, H, V]
        # Per-head matmul: [B, H, V] × [H, V, P] → [B, H, P], then per-sample
        # LN across P so the output std stays ~1 (matches Borda/dominance scale).
        dom = torch.einsum('bhv,hvp->bhp', votes, self.readout)
        return self.output_norm(dom)
