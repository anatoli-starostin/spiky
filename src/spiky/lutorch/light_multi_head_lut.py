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

import os

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
        # DEFAULT CHANGED to "margin" (was "bounded"). "margin" is the exact LookupFFN
        # kernel and is worth -0.034641 bpb over "bounded_norm" on this layer at
        # nap8/tph128 -- 10.3x the 0.00335 seed spread, measured on the clean pair
        # exp_g_0189 -> exp_g_0193, which differ in nothing but this string. "bounded" was
        # already a documented hazard (#112): it is a product over NAP factors, lands at
        # ~0.054 at nap=8 and diverges. Safe to change: every committed config that uses a
        # score sets this key explicitly (verified across all 89 runs -- 28 with the gate
        # on, 21 on the light path, zero relying on this fallback), so no historical run
        # rebuilds differently.
        confidence_form: str = "margin",
        confidence_gain: float = 1.0,
        anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
        random_seed: Optional[int] = None,
        initial_weights_noise: float = 0.001,
        device: Optional[torch.device] = None,
        n_heads: int = 1,
        multi_head_input: bool = False,
        read_top_n: int = 1,
        read_tau: float = 0.1,
        read_tau_learnable: bool = False,
        anchor_mode: str = "pair",
        pool_size: Optional[int] = None,
    ):
        super().__init__()
        if anchor_mode not in ("pair", "single"):
            raise ValueError(
                f"anchor_mode must be 'pair' or 'single', got {anchor_mode!r}")
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

        # --- addressing mode ----------------------------------------------------------
        # "pair"   (default, unchanged): bit_i = 1[x[a_i] - x[b_i] > 0]  (hyperplane e_a-e_b)
        # "single":                      bit_i = 1[x[c_i] > 0]           (one coordinate vs 0)
        # In "single" mode each bit is ONE coordinate's sign; c_i indexes the per-head pool
        # of `pool_size` features. pool_size decouples the addressing pool from output_dim;
        # currently supported only when it equals input_dim (index the existing per-head
        # compressed features directly -- no extra projection). Everything downstream is
        # identical: d = x[c] plays the role d = x[a]-x[b] had, so sign->bit, |d|->margin,
        # score, blend, read-out are untouched. The pair-only native CUDA kernel is disabled
        # in single mode (torch addressing path is used).
        self.anchor_mode = anchor_mode
        self.pool_size = int(pool_size) if pool_size is not None else input_dim
        if anchor_mode == "single" and self.pool_size != input_dim:
            raise NotImplementedError(
                "single anchor_mode currently supports pool_size == input_dim only "
                f"(got pool_size={self.pool_size}, input_dim={input_dim}); a larger pool "
                "would need a dedicated input_dim->pool_size hyperplane projection.")

        # --- top-n blended read-out (TRAINING-CAPABLE; default 1 == today's layer) -------
        # At read_top_n=1 nothing below is reachable and the layer is byte-for-byte the
        # module it always was. At n>1 the single addressed row becomes a normalised
        # convex combination of the n nearest cells, and -- unlike probe_soft_readout.py,
        # which was eval-only and detached the weights -- the weights here are
        # DIFFERENTIABLE in the margins. That is the point: dw/dm is a term that compares
        # the table rows of alternative cells and pushes z toward the better one, i.e. a
        # DIRECTIONAL ROUTING GRADIENT, which plain Light does not have at all. It makes
        # this layer a sparse top-n cousin of FastMultiHeadLut's full-2^NAP softmax
        # surrogate, at n gathers instead of 2^NAP.
        if read_top_n < 1 or read_top_n > n_anchor_pairs + 1:
            raise ValueError(
                f"read_top_n must be in [1, n_anchor_pairs+1] (the argmax cell plus at "
                f"most one flip per anchor); got {read_top_n} with "
                f"n_anchor_pairs={n_anchor_pairs}")
        if not (read_tau > 0):
            raise ValueError(f"read_tau must be > 0, got {read_tau!r}")
        self.read_top_n = int(read_top_n)
        self.read_tau_learnable = bool(read_tau_learnable)

        # --- blend temperature, stored as log_tau -------------------------------------
        # PARAMETERISED AS log_tau, NOT tau: it keeps tau > 0 for free (no clamping, no
        # projection step, no way for an optimiser step to make the softmax undefined), and
        # it conditions the problem -- tau is a SCALE, so a multiplicative step is the
        # natural one, and gradient descent on log_tau is exactly that.
        #
        # ONE SCALAR PER LAYER, and the measurement supports that choice rather than merely
        # permitting it. On exp_g_0193 the per-head medians of m_(1) inside a layer span a
        # factor of only 1.01-1.22 and the across-table CV is 0.017-0.021 -- flat. ACROSS
        # layers it is not flat at all: 0.0331 (L0) -> 0.1079 (L5), a 3.3x spread. So a
        # per-LAYER scalar is right and a single per-MODEL scalar would be wrong; splitting
        # further (per head or per table) has nothing to buy. See diag_margin_gap.py.
        #
        # FROZEN CASE IS A BUFFER, NOT A PARAMETER. Deliberate: `total_params` in the
        # trainer is `sum(p.numel() for p in model.parameters())` and counts frozen
        # parameters too, so registering a frozen tau as a Parameter would add +1 per layer
        # (+6 per model) and shift every leaderboard comparison by that much for no reason.
        # As a buffer the count is IDENTICAL to today's, and it is trivially absent from the
        # optimiser. It still lands in the state_dict under the same key, so a frozen
        # checkpoint loads into a learnable model and vice versa.
        self._read_tau_init = float(read_tau)   # registered at the end of __init__

        dev = device or torch.device("cpu")
        policy = anchor_sampling_policy or AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE
        if self.anchor_mode == "single":
            # SINGLE-anchor addressing: draw NAP single indices per table into the per-head
            # pool [0, pool_size). Frozen buffer, seeded per head (random_seed + h) like the
            # pairs so the two modes are init-comparable head for head. Shape matches
            # anchor_a: [n_heads, tables_per_head, NAP] (multi-head) or [n_tables, NAP].
            def _draw_c(nt, seed):
                g = (torch.Generator(device=dev).manual_seed(seed)
                     if seed is not None else None)
                return torch.randint(0, self.pool_size, (nt, n_anchor_pairs),
                                     device=dev, generator=g, dtype=torch.int64)
            if self.multi_head_input:
                anchor_c = torch.stack([
                    _draw_c(self.tables_per_head,
                            None if random_seed is None else random_seed + h)
                    for h in range(self.n_heads)
                ])                                   # [H, T, NAP]
            else:
                anchor_c = _draw_c(n_tables, random_seed)   # [n_tables, NAP]
            self.register_buffer("anchor_c", anchor_c.contiguous())
        elif self.multi_head_input:
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
            self.register_buffer("anchor_a", anchor_a.contiguous())
            self.register_buffer("anchor_b", anchor_b.contiguous())
        else:
            # Same anchor-pair geometry as FastMultiHeadLut with n_heads=1 (all tables
            # form one summed head), so the routing margins are drawn identically.
            anchor_a, anchor_b = get_balanced_anchor_pairs(
                n_tables=n_tables, n_anchor_pairs=n_anchor_pairs, input_dim=input_dim,
                device=dev, random_seed=random_seed, policy=policy, n_heads=1,
            )
            self.register_buffer("anchor_a", anchor_a.contiguous())   # [n_tables, NAP]
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
        # The native kernel is PAIR-only; in "single" mode it stays disabled (None) so
        # _pack_index and _fused_eval take the torch path over the single-index margins.
        self._native_msb = None
        self._native_msb_scored = None
        self._score_form_id = {"bounded_norm": 0, "bounded": 1, "margin": 2}[confidence_form]
        if self.anchor_mode == "pair":
            mgr = _get_native_lutorch_manager()
            if mgr is not None:
                self._native_msb = getattr(mgr, "anchor_pairs_lookup_eval_forward_msb", None)
                self._native_msb_scored = getattr(
                    mgr, "anchor_pairs_lookup_eval_forward_msb_scored", None)
            if self.multi_head_input:
                head_off = torch.arange(self.n_heads, device=dev).view(self.n_heads, 1, 1) \
                    * input_dim
                a_flat = (anchor_a + head_off).reshape(n_tables, n_anchor_pairs)
                b_flat = (anchor_b + head_off).reshape(n_tables, n_anchor_pairs)
            else:
                a_flat, b_flat = anchor_a, anchor_b
            self.register_buffer("native_anchor_a", a_flat.contiguous().to(torch.int64))
            self.register_buffer("native_anchor_b", b_flat.contiguous().to(torch.int64))

        log_tau = torch.log(torch.tensor(self._read_tau_init, device=dev))
        if self.read_tau_learnable:
            self.log_tau = nn.Parameter(log_tau)
        else:
            self.register_buffer("log_tau", log_tau)

        # --- torch.compile the forward for the blend path (default-ON, no config flag) ----
        # The n>1 blended read-out fans out into ~8-10 tiny elementwise kernels
        # (abs/min/bits/powers/gather/idx-cat/softmax/psw); compiling the forward folds them
        # into 1-2 (one graph break remains at F.embedding_bag). Measured ~-27% fwd+bwd on the
        # blend at the anchor sizing, ON TOP of the topk->min change. Scoped to read_top_n>1 so
        # the n==1 path stays byte-identical (protects the established light line's
        # reproducibility). Guarded: a compile failure -- or LUT_DISABLE_COMPILE=1 -- falls
        # back to eager. NOT bit-exact: compile reorders the embedding_bag-backward fp
        # accumulation (grad_tables ~3e-5, negligible vs bf16 epsilon).
        # Lazy + runtime-checked (see forward): compiled ONLY when read_top_n>1 at call time,
        # so the n==1 path -- and a runtime read_top_n->1 flip -- stays eager and byte-identical;
        # compiled once on the first n>1 forward. LUT_DISABLE_COMPILE=1 forces eager.
        self._compiled_fwd = None
        self._compile_enabled = os.environ.get('LUT_DISABLE_COMPILE') != '1'

    @property
    def read_tau(self):
        """The blend temperature, a 0-dim tensor. Differentiable iff read_tau_learnable.

        Read through `log_tau.exp()` on every use rather than cached, so that an optimiser
        step on log_tau takes effect immediately and a frozen buffer stays exactly its init.
        """
        return self.log_tau.exp()

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
        if self.anchor_mode == "single":
            idx_c = self.anchor_c.reshape(1, H, T * NAP).expand(B, H, T * NAP)
            d = torch.gather(x, 2, idx_c).view(B, H, T, NAP)          # single-coordinate value
        else:
            idx_a = self.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
            idx_b = self.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
            d = (torch.gather(x, 2, idx_a) - torch.gather(x, 2, idx_b)).view(B, H, T, NAP)

        index = self._pack_index(x.reshape(B, H * self.input_dim), d).view(B, H, T)

        flat = self.tables.reshape(H * T * self.table_size, self.output_dim)
        flat_idx = (index + self.table_offset.view(1, H, T)).reshape(-1)

        score = _confidence_score(d, self.confidence_form,
                                  self.confidence_gain)                # [B, H, T]
        if self.read_top_n > 1:
            return self._blend_bag(d, index, self.table_offset.view(1, H, T), flat,
                                   score, B * H, T).view(B, H, self.output_dim)
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

    def _blend_bag(self, d, index, offset, flat, score, n_bags: int, bag: int):
        """Top-`read_top_n` blended read-out: sum_t score_t * sum_i w_i * flat[c_i].

        CANDIDATES. The cells reachable by flipping one address bit are the Hamming-1
        neighbours, and under the code-space softmax (see WEIGHTS) flipping bit j costs
        `2 m_j`, so the nearest neighbours are the SMALLEST-margin bits, ascending. That is
        the same enumeration probe_soft_readout.py used. For n=3 we therefore take the two
        smallest SINGLE flips rather than one double flip, and that is forced rather than
        preferred: the double flip of the two smallest bits costs `2(m_j1 + m_j2)`, which is
        strictly greater than the second single flip's `2 m_j2` for any m_j1 > 0. So the two
        smallest singles ARE the two nearest cells; a double flip can never be third-nearest
        while an unused single flip remains. The candidate set is exactly a Hamming ball of
        radius 1 around the argmax, truncated to n.

        WEIGHTS. A softmax over the candidates' code-space log-weights, at temperature tau:

            logits = [0, -2 m_j1 / tau, -2 m_j2 / tau, ...]        w = softmax(logits)

        The zero is the argmax cell. This comes from `w(b) ∝ exp(<z, b>)`, whose argmax
        `b* = sign(d)` scores `sum_j m_j` and whose j-flip scores `sum_j m_j - 2 m_j`, so
        only the difference `-2 m_j` matters and the normaliser cancels.

        WHY tau EXISTS, recorded because getting this wrong wasted a first attempt. The
        naive tau=1 weight `exp(-2 m_min)` looks principled and is useless here: `m_min` is
        the MINIMUM of `n_anchor_pairs` margins, an order statistic that stays around
        0.05-0.13 even when typical margins are 0.4-0.8, so the neighbour's normalised share
        came out at median 0.46 / mean 0.44, above 0.3 for 96% of (token, table) pairs
        (diag_blend_weight.py). That is a coin flip between two unrelated rows, not a blend,
        and it moved eval bpb by -0.0014 on one checkpoint and +0.0208 on another -- opposite
        signs, no coherent reading. tau in [0.05, 0.3] fixes it; tau=0.1 was the working
        value across five checkpoints.

        COMPOSITION WITH THE CONFIDENCE SCORE. The blend weights are normalised over the n
        candidates (`sum_i w_i = 1`) and sit INSIDE the score: `score_t * sum_i w_i row_i`.
        So `score` keeps its exact meaning -- a per-(token, table) gate on that table's whole
        contribution -- and `w` only redistributes *which row* that table reads. The two
        compose multiplicatively and never interact.

        THE n=1 LIMIT IS EXACT. With one candidate the softmax is `softmax([0]) = [1]`
        identically, so this reduces to `score_t * flat[c_0]`, which is `_bagged_sum`. It is
        exact, not approximate, and is asserted by test.

        GRADIENT. `per_sample_weights` is differentiable, so autograd flows to `flat` at all
        n gathered rows (as before, but now n of them) AND -- new -- through `w` into `m`
        into `d` into `x`. That second path is the directional routing gradient: dw_i/dm
        weights each alternative row by how much better it would have been. Plain autograd
        throughout; no STE, no custom autograd.Function.
        """
        n = self.read_top_n
        K = d.shape[-1]
        m = d.abs()                                        # differentiable, NOT detached
        # the (n-1) cheapest flips: smallest margins, ascending
        if n == 2:
            mv, mj = m.min(dim=-1, keepdim=True)           # fused argmin == topk(k=1,largest=False)
        else:
            mv, mj = torch.topk(m, k=n - 1, dim=-1, largest=False)      # [..., n-1]
        bits = (d.detach() > 0).to(torch.int64)
        pw = self.powers[mj]                                            # [..., n-1]
        bsel = torch.gather(bits, -1, mj)                               # [..., n-1]
        # bit==1 -> clear it (subtract 2^k); bit==0 -> set it (add 2^k)
        idx = torch.cat([index.unsqueeze(-1),
                         index.unsqueeze(-1) + pw * (1 - 2 * bsel)], dim=-1)  # [..., n]

        # The argmax cell's logit is 0. Built from `m`, not from `mv`: at n=1 `mv` has an
        # EMPTY last dim, so `mv[..., :1]` would stay empty and the softmax would collapse
        # to nothing. `m` always has n_anchor_pairs >= 1 columns, so this is always [..., 1]
        # and the n=1 limit really is softmax([0]) = [1].
        logits = torch.cat([torch.zeros_like(m[..., :1]),
                            -2.0 * mv / self.read_tau], dim=-1)         # [..., n]
        w = torch.softmax(logits, dim=-1)                               # [..., n]

        flat_idx = (idx + offset.unsqueeze(-1)).reshape(-1)             # [.. * n]
        psw = (score.unsqueeze(-1) * w).reshape(-1).to(flat.dtype)
        offsets = torch.arange(n_bags, device=flat.device,
                               dtype=torch.long) * (bag * n)
        return F.embedding_bag(flat_idx, flat, offsets=offsets, mode="sum",
                               per_sample_weights=psw)

    def _fused_eval(self, x_flat):
        """Eval-only fast path: address AND score from one native pass, then the bag.

        Returns None when unavailable, so the caller falls through to the autograd path.

        The margins never leave registers here: the kernel gathers each anchor pair,
        decides its sign bit, and accumulates the score in the same loop. That removes BOTH
        the torch margin gather (0.65 ms) and the torch score (1.02 ms) at the anchor
        sizing, for 0.006 ms of extra arithmetic inside the kernel -- the score really is
        nearly free once the margins are already in hand.

        Eval only, and that is a real restriction rather than an oversight: autograd needs
        the margins as a differentiable tensor to reach x through the score, and this path
        deliberately never materialises them. Training therefore keeps the torch path,
        where the forward is a small share of the step anyway (2.7 ms of a 9.1 ms
        fwd+bwd), so the leverage is here, at inference.
        """
        if (self._native_msb_scored is None or not x_flat.is_cuda
                or x_flat.dtype not in (torch.float32, torch.float64)):
            return None
        index, score = self._native_msb_scored(
            x_flat, self.native_anchor_a, self.native_anchor_b, 0.0,
            self._score_form_id, self.confidence_gain, 256,
        )
        flat = self.tables.reshape(self.n_tables * self.table_size, self.output_dim)
        flat_idx = (index + self.table_offset.view(1, -1)).reshape(-1)
        n_bags = x_flat.shape[0] * self.n_heads
        return self._bagged_sum(flat, flat_idx, score, n_bags, self.tables_per_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Default-ON torch.compile for the n>1 blend path (no config flag). Checked at CALL
        # time on read_top_n, so n==1 (or a runtime flip to 1) runs eager and byte-identical.
        # Guarded: a compile failure falls back to eager permanently.
        if self.read_top_n > 1 and self._compile_enabled:
            if self._compiled_fwd is None:
                try:
                    self._compiled_fwd = torch.compile(self._forward_impl, fullgraph=False)
                except Exception:
                    self._compile_enabled = False
                    return self._forward_impl(x)
            return self._compiled_fwd(x)
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # The fused eval kernel returns the packed index and the score only -- it never
        # materialises the margins, and the blend needs them to choose and weight the
        # neighbours. So it is unavailable at read_top_n > 1, by construction rather than
        # by oversight.
        if not torch.is_grad_enabled() and self.read_top_n == 1:
            B = x.shape[0]
            x_flat = x.reshape(B, -1)
            if x_flat.shape[1] == (self.n_heads * self.input_dim if self.multi_head_input
                                   else self.input_dim):
                out = self._fused_eval(x_flat.contiguous())
                if out is not None:
                    return (out.view(B, self.n_heads, self.output_dim)
                            if self.multi_head_input else out)
        if self.multi_head_input:
            return self._forward_multi_head(x)
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"x must be [B, {self.input_dim}], got {tuple(x.shape)}")
        B = x.shape[0]

        # Routing margins -- differentiable in x (this is what score() reads).
        if self.anchor_mode == "single":
            d = x[:, self.anchor_c]                                  # [B, n_tables, NAP]
        else:
            d = x[:, self.anchor_a] - x[:, self.anchor_b]            # [B, n_tables, NAP]

        # Hard sign address, FULLY DETACHED: the comparison is taken on d.detach()
        # and yields an integer index, so NO gradient (and NO straight-through
        # estimator) flows through the code/routing direction. Detaching here is
        # explicit intent; a bool/integer index would carry no grad regardless.
        index = self._pack_index(x, d)                               # [B, n_tables]

        # Gather one row per table. Grad flows to `tables` at the selected rows only.
        flat = self.tables.reshape(self.n_tables * self.table_size, self.output_dim)
        flat_idx = (index + self.table_offset.view(1, -1)).reshape(-1)

        # Differentiable confidence gate. At read_top_n=1 this is the ONLY path from x to
        # the output grad; at n>1 the blend weights add a second, directional one.
        score = _confidence_score(d, self.confidence_form,
                                  self.confidence_gain)              # [B, n_tables]

        if self.read_top_n > 1:
            return self._blend_bag(d, index, self.table_offset.view(1, -1), flat,
                                   score, B, self.n_tables)
        # One bag per sample, summing all n_tables = the ensemble output.
        return self._bagged_sum(flat, flat_idx, score, B, self.n_tables)

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, n_tables={self.n_tables}, "
                f"output_dim={self.output_dim}, n_anchor_pairs={self.n_anchor_pairs}, "
                f"table_size={self.table_size}, confidence_form={self.confidence_form!r}")
