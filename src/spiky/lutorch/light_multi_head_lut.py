"""LightMultiHeadLUT: a minimal pure-autograd LUT control layer.

A scientific control for ablating FastMultiHeadLut's backward. It reuses OUR
anchor-pair routing geometry (margins ``d = x[anchor_a] - x[anchor_b]``) and the
SAME confidence-score forms we added to FastMultiHeadLut, but its gradient is
LookupFFN's: a single forward, the hard sign address FULLY DETACHED (integer,
non-differentiable, no straight-through estimator), and the ONLY gradient to x
flowing through the differentiable confidence score ``score(|d|)``. Plain
autograd throughout -- no custom ``autograd.Function``, no softmax/temperature
surrogate, no STE -- so there is ZERO directional routing gradient.

Contrast with FastMultiHeadLut:
  Fast  : hard forward + soft (temperature) backward surrogate. x receives BOTH a
          directional routing gradient AND (with forward_confidence) a score
          gradient.
  Light : hard forward + pure autograd. x receives ONLY the score gradient; the
          routing DIRECTION is not learned. This is exactly LookupFFN's learning
          signal expressed on our anchor-pair geometry.

Clarity is the priority (this is a control for experiments), so there is one
forward path only: no bmm/gather dispatch, no hybrid_smooth, no exp_outputs
log-sum-exp readout, no routed-V, no dual-stream, and a single shared input
(``multi_head_input`` is intentionally not supported -- use FastMultiHeadLut for
that). The layer is torch.compile-friendly: no data-dependent Python control
flow on tensor values.
"""
from typing import Optional

import torch
import torch.nn as nn

from .lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs
# Reuse the EXACT score definition FastMultiHeadLut uses, so the two layers are
# directly comparable in an ablation (same "bounded"/"margin" forms).
from .fast_multi_head_lut import _confidence_score


class LightMultiHeadLUT(nn.Module):
    """Ensemble of ``n_tables`` anchor-pair LUTs, summed, gated by the score.

    Forward (identical in train and eval)::

        d     = x[:, anchor_a] - x[:, anchor_b]        # [B, n_tables, NAP] margins
        index = pack(sign(d.detach()))                 # [B, n_tables], integer, NO grad
        row   = tables[t, index[:, t]]                 # [B, n_tables, output_dim]
        score = confidence(|d|)                        # [B, n_tables], differentiable
        out   = sum_t score[:, t] * row[:, t]          # [B, output_dim]

    Args:
        input_dim: dimension of x.
        n_tables: number of independent tables summed together (the ensemble /
            the "multiple heads"). The output is their sum.
        output_dim: width of each stored row and of the layer output.
        n_anchor_pairs: NAP anchor pairs per table -> ``2 ** NAP`` rows per table.
        confidence_form: "bounded" (default) uses ``prod_j sigmoid(2|d_j|)`` in
            (0, 1]; "margin" uses ``(sum_j |d_j|) * prod_j sigmoid(2|d_j|)``
            (== the exact LookupFFN score ``sum|d| / prod(1+e^{-2|d|})``).
        anchor_sampling_policy: defaults to CANONICAL_FULL_COVERAGE (as Fast).
        random_seed: seed for anchor sampling and table init.
        initial_weights_noise: tables ~ Uniform[-noise, +noise] (matches Fast's
            default init for comparability).
        device: torch.device or None (-> CPU).

    Forward signature:
        x: float [B, input_dim]  ->  [B, output_dim]
    """

    def __init__(
        self,
        input_dim: int,
        n_tables: int,
        output_dim: int,
        n_anchor_pairs: int,
        *,
        confidence_form: str = "bounded",
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if confidence_form not in ("bounded", "margin"):
            raise ValueError(
                f"confidence_form must be 'bounded' or 'margin', got {confidence_form!r}"
            )
        if not (1 <= n_anchor_pairs <= 15):
            raise ValueError(
                f"n_anchor_pairs must be in [1, 15] (2^NAP rows per table), got {n_anchor_pairs}"
            )

        self.input_dim = input_dim
        self.n_tables = n_tables
        self.output_dim = output_dim
        self.n_anchor_pairs = n_anchor_pairs
        self.table_size = 1 << n_anchor_pairs
        self.confidence_form = confidence_form

        dev = device or torch.device("cpu")
        policy = anchor_sampling_policy or AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE
        # Same anchor-pair geometry as FastMultiHeadLut with n_heads=1 (all tables
        # form one summed head), so the routing margins are drawn identically.
        anchor_a, anchor_b = get_balanced_anchor_pairs(
            n_tables=n_tables, n_anchor_pairs=n_anchor_pairs, input_dim=input_dim,
            device=dev, random_seed=random_seed, policy=policy, n_heads=1,
        )
        self.register_buffer("anchor_a", anchor_a.contiguous())   # [n_tables, NAP]
        self.register_buffer("anchor_b", anchor_b.contiguous())   # [n_tables, NAP]
        # MSB-first bit-pack powers, matching FastMultiHeadLut's index convention.
        self.register_buffer(
            "powers",
            (2 ** torch.arange(n_anchor_pairs - 1, -1, -1, device=dev)).to(torch.int64),
        )
        # Per-table row-block offsets for the flat gather.
        self.register_buffer(
            "table_offset",
            torch.arange(n_tables, device=dev, dtype=torch.int64) * self.table_size,
        )

        gen = None
        if random_seed is not None:
            gen = torch.Generator(device=dev).manual_seed(random_seed + 1)
        u = torch.rand(n_tables, self.table_size, output_dim, device=dev, generator=gen) - 0.5
        self.tables = nn.Parameter(u * (2.0 * initial_weights_noise))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must be [B, {self.input_dim}], got {tuple(x.shape)}")
        B = x.shape[0]

        # Routing margins -- differentiable in x (this is what score() reads).
        d = x[:, self.anchor_a] - x[:, self.anchor_b]                # [B, n_tables, NAP]

        # Hard sign address, FULLY DETACHED: the comparison is taken on d.detach()
        # and yields an integer index, so NO gradient (and NO straight-through
        # estimator) flows through the code/routing direction. Detaching here is
        # explicit intent; a bool/integer index would carry no grad regardless.
        index = ((d.detach() > 0).to(torch.int64) * self.powers.view(1, 1, -1)).sum(dim=-1)  # [B, n_tables]

        # Gather one row per table. Grad flows to `tables` at the selected rows only.
        flat = self.tables.reshape(self.n_tables * self.table_size, self.output_dim)
        flat_idx = (index + self.table_offset.view(1, -1)).reshape(-1)
        rows = flat[flat_idx].view(B, self.n_tables, self.output_dim)  # [B, n_tables, output_dim]

        # Differentiable confidence gate -- the ONLY path from x to the output grad.
        score = _confidence_score(d, self.confidence_form)           # [B, n_tables]

        # Sum over tables = the ensemble output.
        return (rows * score.unsqueeze(-1)).sum(dim=1)               # [B, output_dim]

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, n_tables={self.n_tables}, "
                f"output_dim={self.output_dim}, n_anchor_pairs={self.n_anchor_pairs}, "
                f"table_size={self.table_size}, confidence_form={self.confidence_form!r}")
