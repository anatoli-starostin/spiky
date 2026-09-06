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
import torch.nn.functional as F
import torch.nn as nn

from .lut_helpers import AnchorSamplingPolicy, get_balanced_anchor_pairs
# Reuse the EXACT score definition FastMultiHeadLut uses, so the two layers are
# directly comparable in an ablation (same "bounded"/"margin" forms).
from .fast_multi_head_lut import _confidence_score, _get_native_lutorch_manager


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
            (== the exact LookupFFN score ``sum|d| / prod(1+e^{-2|d|})``);
            "bounded_norm" uses the geometric mean of the same sigmoids,
            ``prod_j sigmoid(2|d_j|) ** (1/NAP)`` -- same ordering as "bounded"
            but without its NAP-dependent attenuation.
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
        confidence_gain: float = 1.0,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        device: Optional[torch.device] = None,
        n_heads: int = 1,
        multi_head_input: bool = False,
    ):
        super().__init__()
        if confidence_form not in ("bounded", "margin", "bounded_norm"):
            raise ValueError(
                "confidence_form must be 'bounded', 'margin' or 'bounded_norm', "
                f"got {confidence_form!r}"
            )
        if not (1 <= n_anchor_pairs <= 15):
            raise ValueError(
                f"n_anchor_pairs must be in [1, 15] (2^NAP rows per table), got {n_anchor_pairs}"
            )

        if multi_head_input and n_tables % n_heads != 0:
            raise ValueError(
                f"multi_head_input requires n_tables divisible by n_heads; got "
                f"n_tables={n_tables}, n_heads={n_heads}"
            )
        if multi_head_input and n_heads < 1:
            raise ValueError(f"n_heads must be >= 1, got {n_heads}")

        self.input_dim = input_dim
        self.n_tables = n_tables
        self.output_dim = output_dim
        self.n_anchor_pairs = n_anchor_pairs
        self.table_size = 1 << n_anchor_pairs
        self.confidence_form = confidence_form
        if not (confidence_gain > 0):
            raise ValueError(
                f"confidence_gain must be > 0, got {confidence_gain!r}")
        self.confidence_gain = float(confidence_gain)
        self.multi_head_input = bool(multi_head_input)
        self.n_heads = n_heads if multi_head_input else 1
        self.tables_per_head = n_tables // self.n_heads

        dev = device or torch.device("cpu")
        policy = anchor_sampling_policy or AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE
        if self.multi_head_input:
            # BLOCK-DIAGONAL routing: head h reads its OWN [input_dim] slice of the
            # compressed code, so anchors index within a head, not across heads. Each
            # head draws from a fresh generator seeded (random_seed + h) -- the SAME
            # convention FastMultiHeadLut uses for multi_head_input=True, so the two
            # layers are initialisation-comparable head for head.
            a_list, b_list = [], []
            for h in range(self.n_heads):
                seed_h = None if random_seed is None else random_seed + h
                a_h, b_h = get_balanced_anchor_pairs(
                    n_tables=self.tables_per_head, n_anchor_pairs=n_anchor_pairs,
                    input_dim=input_dim, device=dev, random_seed=seed_h,
                    policy=policy, n_heads=1,
                )
                a_list.append(a_h)
                b_list.append(b_h)
            anchor_a = torch.stack(a_list)      # [n_heads, tables_per_head, NAP]
            anchor_b = torch.stack(b_list)
        else:
            # Same anchor-pair geometry as FastMultiHeadLut with n_heads=1 (all tables
            # form one summed head), so the routing margins are drawn identically.
            anchor_a, anchor_b = get_balanced_anchor_pairs(
                n_tables=n_tables, n_anchor_pairs=n_anchor_pairs, input_dim=input_dim,
                device=dev, random_seed=random_seed, policy=policy, n_heads=1,
            )
        self.register_buffer("anchor_a", anchor_a.contiguous())   # [n_tables, NAP] or [H, T, NAP]
        self.register_buffer("anchor_b", anchor_b.contiguous())
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

        if self.multi_head_input:
            # Per-head table draw from Generator(random_seed + h + 1), matching the
            # convention FastMultiHeadLut uses for its block-diagonal path so the two
            # layers start from comparable draws head for head.
            blocks = []
            for h in range(self.n_heads):
                g_h = (None if random_seed is None
                       else torch.Generator(device=dev).manual_seed(random_seed + h + 1))
                blocks.append(torch.rand(self.tables_per_head, self.table_size, output_dim,
                                         device=dev, generator=g_h) - 0.5)
            u = torch.cat(blocks, dim=0)        # [n_tables, table_size, output_dim]
        else:
            gen = None
            if random_seed is not None:
                gen = torch.Generator(device=dev).manual_seed(random_seed + 1)
            u = torch.rand(n_tables, self.table_size, output_dim,
                           device=dev, generator=gen) - 0.5
        self.tables = nn.Parameter(u * (2.0 * initial_weights_noise))

        # --- native CUDA bit-pack for the ADDRESS (opt-in, exact, train and eval) ---
        # FastMultiHeadLut uses lutorch_cuda's MSB-first kernel only at eval, and only
        # when its confidence gate is off -- the kernel returns just the packed index and
        # throws the margins away, so a gated layer cannot get its score from it.
        #
        # Light is in the same position for the SCORE (it still gathers |d| in torch), but
        # not for the ADDRESS: Light's address is detached by construction, so replacing
        # its sign+pack with the kernel is exact and is legal in TRAINING as well as eval.
        # Measured at the anchor sizing: torch sign+pack 1.27 ms vs native 0.12 ms, taking
        # the fused forward from 3.79 ms to ~2.64 ms.
        #
        # Anchors are flattened once here (with per-head offsets for the block-diagonal
        # case) because the kernel takes a 2-D [n_tables, NAP] anchor table over a flat x.
        self._native_msb = None
        mgr = _get_native_lutorch_manager()
        if mgr is not None:
            self._native_msb = getattr(mgr, "anchor_pairs_lookup_eval_forward_msb", None)
        if self.multi_head_input:
            head_off = torch.arange(self.n_heads, device=dev).view(self.n_heads, 1, 1) \
                * input_dim
            a_flat = (anchor_a + head_off).reshape(n_tables, n_anchor_pairs)
            b_flat = (anchor_b + head_off).reshape(n_tables, n_anchor_pairs)
        else:
            a_flat, b_flat = anchor_a, anchor_b
        self.register_buffer("native_anchor_a", a_flat.contiguous().to(torch.int64))
        self.register_buffer("native_anchor_b", b_flat.contiguous().to(torch.int64))

    def _pack_index(self, x_flat, d):
        """Packed row index [B, n_tables], MSB-first. Never differentiable.

        Prefers the native CUDA kernel, which does gather+sign+pack in one pass; falls
        back to the torch expression everywhere else (CPU, float64, no extension). Both
        produce the identical integer address -- a test asserts equality -- so this is a
        speed choice, never a numerics one.
        """
        if (self._native_msb is not None and x_flat.is_cuda
                and x_flat.dtype in (torch.float32, torch.float64)):
            return self._native_msb(x_flat, self.native_anchor_a, self.native_anchor_b,
                                    0.0, 256)
        shape = (1, 1, 1, -1) if d.dim() == 4 else (1, 1, -1)
        return ((d.detach() > 0).to(torch.int64)
                * self.powers.view(*shape)).sum(dim=-1)

    def _forward_multi_head(self, x: torch.Tensor) -> torch.Tensor:
        """Block-diagonal variant: x [B, n_heads, input_dim] -> [B, n_heads, output_dim].

        Identical mechanism to the shared-input path -- detached sign address, one row
        per table, differentiable confidence gate, sum over that head's tables -- with
        every quantity carrying a head axis and no mixing between heads.
        """
        B, H, T, NAP = x.shape[0], self.n_heads, self.tables_per_head, self.n_anchor_pairs
        if x.dim() != 3 or x.shape[1] != H or x.shape[2] != self.input_dim:
            raise ValueError(
                f"multi_head_input expects x of shape [B, {H}, {self.input_dim}], "
                f"got {tuple(x.shape)}"
            )
        # gather the per-head anchor coordinates out of each head's own slice
        idx_a = self.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
        idx_b = self.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
        d = (torch.gather(x, 2, idx_a) - torch.gather(x, 2, idx_b)).view(B, H, T, NAP)

        index = self._pack_index(x.reshape(B, H * self.input_dim), d).view(B, H, T)

        flat = self.tables.reshape(H * T * self.table_size, self.output_dim)
        flat_idx = (index + self.table_offset.view(1, H, T)).reshape(-1)

        score = _confidence_score(d, self.confidence_form,
                                  self.confidence_gain)                # [B, H, T]
        # One bag per (sample, head), summing that head's T tables.
        return self._bagged_sum(flat, flat_idx, score, B * H, T).view(B, H, self.output_dim)

    def _bagged_sum(self, flat, flat_idx, score, n_bags: int, bag_size: int):
        """sum_t score[.., t] * flat[flat_idx[.., t]], fused via F.embedding_bag.

        Mathematically identical to gathering the rows and doing
        ``(rows * score.unsqueeze(-1)).sum(over tables)`` -- which is how this layer was
        first written -- but it never materialises the [.., n_tables, output_dim] rows.
        At the anchor sizing those rows are 6144 x 4 x 256 x 48 x 4B = 1.2 GiB of traffic
        per layer per call, and removing them makes the forward ~2.3x faster and the peak
        memory ~2.4x smaller. The naive form is kept as the reference implementation in
        test_light_embedding_bag_fusion.py, which asserts the two agree.

        The confidence score enters as embedding_bag's `per_sample_weights`, which is
        exactly what it is: one scalar multiplying one gathered row. That is also what
        preserves this layer's defining property -- `flat_idx` is an integer tensor built
        from `d.detach()`, so it carries no gradient and there is still no STE; autograd
        reaches x ONLY through `per_sample_weights` -> score -> |d| -> d.

        `per_sample_weights` must share the table dtype, so with reduced-precision tables
        the score is rounded to that dtype before it multiplies (the naive form would have
        accumulated in the wider of the two). That is a deliberate consequence of fusing:
        it is what makes the fused kernel single-pass, and it only bites when tables are
        stored below fp32.
        """
        w = flat.dtype
        offsets = torch.arange(n_bags, device=flat.device, dtype=torch.long) * bag_size
        return F.embedding_bag(
            flat_idx, flat, offsets=offsets, mode="sum",
            per_sample_weights=score.reshape(-1).to(w),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.multi_head_input:
            return self._forward_multi_head(x)
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must be [B, {self.input_dim}], got {tuple(x.shape)}")
        B = x.shape[0]

        # Routing margins -- differentiable in x (this is what score() reads).
        d = x[:, self.anchor_a] - x[:, self.anchor_b]                # [B, n_tables, NAP]

        # Hard sign address, FULLY DETACHED: the comparison is taken on d.detach()
        # and yields an integer index, so NO gradient (and NO straight-through
        # estimator) flows through the code/routing direction. Detaching here is
        # explicit intent; a bool/integer index would carry no grad regardless.
        index = self._pack_index(x, d)                               # [B, n_tables]

        # Gather one row per table. Grad flows to `tables` at the selected rows only.
        flat = self.tables.reshape(self.n_tables * self.table_size, self.output_dim)
        flat_idx = (index + self.table_offset.view(1, -1)).reshape(-1)

        # Differentiable confidence gate -- the ONLY path from x to the output grad.
        score = _confidence_score(d, self.confidence_form,
                                  self.confidence_gain)              # [B, n_tables]

        # One bag per sample, summing all n_tables = the ensemble output.
        return self._bagged_sum(flat, flat_idx, score, B, self.n_tables)

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, n_tables={self.n_tables}, "
                f"output_dim={self.output_dim}, n_anchor_pairs={self.n_anchor_pairs}, "
                f"table_size={self.table_size}, confidence_form={self.confidence_form!r}")
