"""Ternary straight-through hyperplanes for HyperplaneMultiHeadLUT.

`TernaryHyperplaneMultiHeadLUT` learns the routing exactly as
`HyperplaneMultiHeadLUT` does, but the hyperplane weights that reach the projection
are quantized to {-1, 0, +1}. The real-valued weights are shadow weights: they exist
to be trained and can be discarded afterwards, leaving a routing that needs no
multiplications at all.

    s_i = stanh(w_i, T) = tanh(w_i / (2T))          x0 = 0, fixed (not trainable)
    q_hard_i = +1 if s_i >  0.5
               -1 if s_i < -0.5
                0 otherwise
    q_i = s_i + (q_hard_i - s_i).detach()           straight-through estimator

so the forward VALUE is exactly ternary while the gradient flows through the smooth
`s = tanh(w / (2T))`, reaching both `w` and `log_T`.

Because `|tanh(u)| > 1/2  <=>  |u| > atanh(1/2)`, the dead zone has a closed form:

    q_i = 0   <=>   |w_i| <= 2T * atanh(1/2) = T * ln 3 ~= 1.0986 * T

T therefore *is* the zero-band width in weight units: T -> 0 drives every weight to
+-1 (no zeros), large T drives everything to 0. This is what makes the choice of
`ternary_temp_init` load-bearing rather than cosmetic -- see below.

This is a strict generalization of anchor pairs. A FastMultiHeadLut anchor pair is
the ternary vector with exactly one +1 and one -1; here a hyperplane may use any
number of +-1 entries, and may drop entries to 0.

PARAMETER COUNT IS UNCHANGED DURING TRAINING. The continuous `hyperplane_weight`
stays a trainable Parameter of the same shape as in HyperplaneMultiHeadLUT
(`[n_tables, nap, input_dim]`), plus one scalar `log_ternary_temp` per table. The
saving is a DEPLOYMENT-time property: after training, call `bake_ternary_weights()`,
store the {-1, 0, +1} result (2 bits per entry, or as two index lists), and the
float weights are never needed again.

Design decisions, all reported explicitly rather than assumed:

  * BIAS IS DROPPED ENTIRELY. There is no bias term and no flag for one: the routing
    test is purely `<q, x> > 0`, the pure anchor-pair "compare to zero" form.
    HyperplaneMultiHeadLUT's per-hyperplane bias is retained only as a frozen
    all-zero buffer, because the inherited autograd Functions take it as a positional
    argument -- it is not a Parameter, receives no gradient, and never moves off zero.
    So `TernaryHyperplaneMultiHeadLUT` has one FEWER learnable tensor than
    HyperplaneMultiHeadLUT, not one more.

  * T IS PER TABLE, stored as `log_ternary_temp` with shape `[n_tables, 1, 1]` so it
    broadcasts over the `(nap, input_dim)` axes of that table's hyperplane block.
    `n_tables = n_heads * tables_per_head`, the same flat table indexing
    HyperplaneMultiHeadLUT uses (head h owns rows `[h*tph, (h+1)*tph)`), so every
    hyperplane and every weight element of one table share one T. See
    `ternary_temp_per_head()` for the `[n_heads, tph]` view.

  * INFERENCE IS MULTIPLICATION-FREE once `w` is baked to `q`: the projection is
    `sum(x[i] for q_i=+1) - sum(x[i] for q_i=-1)` compared against `-b`, i.e. adds,
    subtracts and one threshold compare per hyperplane. The bias does not break this
    -- it is a threshold, not a coefficient. TRAINING-time forward still runs the
    same dense matmul HyperplaneMultiHeadLUT uses, just with `q` in place of `w`;
    that is fine and intended.

Everything downstream of the projection -- the sign/soft-score machinery, the
soft-backward surrogate, the selection temperatures, the table lookup and weights --
is INHERITED, not reimplemented. This class subclasses HyperplaneMultiHeadLUT and
overrides only the two places the hyperplane weights are consumed, substituting `q`
for `w`. Subclassing rather than copying keeps the two modules from drifting apart:
there is exactly one implementation of the routing math in the tree.

OPTIONAL SPARSITY PENALTY (off by default)
------------------------------------------
`nonzero_penalty_weight` (lambda, default 0.0) adds a soft L0-surrogate that
discourages DENSE routing -- fewer non-zero {-1,+1} components per hyperplane, i.e.
cheaper add/subtract kernels at deployment. It is a loss term only: the forward, the
routing and the parameter set are untouched, and at the default 0.0 the class is a
bit-for-bit no-op against the un-penalized version (verified in the sanity checks).

    surrogate = mean over ALL hyperplane weight elements of |tanh(w_i / (2T))|
    penalty   = lambda * surrogate

Each term is ~0 inside the dead zone and ~1 when saturated, so the surrogate is a
smooth stand-in for "fraction of components that are non-zero" -- the same `stanh`
the routing already uses, so nothing new is introduced. MEAN, not sum, so lambda has
the same meaning at any (n_tables, nap, input_dim) shape.

Wiring it into a training loop -- sum over the slots and add to the loss:

    pen = sum(m.sparsity_penalty() for m in model.modules()
              if isinstance(m, TernaryHyperplaneMultiHeadLUT))
    (loss + pen).backward()

The penalty gradient reaches BOTH `hyperplane_weight` (pulling weights toward the
dead zone) and `log_ternary_temp` (widening the band). Both routes genuinely sparsify,
so both are wired -- but note the failure mode: T -> infinity zeroes the routing
entirely and drives the penalty to 0, so lambda must stay small enough that the task
loss opposes it. Watch `ternary_stats()["frac_zero"]` and `T_max` during training; if
T inflation dominates, detaching T inside `sparsity_surrogate` is a one-line change.

Neither HyperplaneMultiHeadLUT nor CompressionMultiHeadLUT is modified by this file.
"""
import math
import warnings
from typing import Optional

import torch
import torch.nn as nn

from .hyperplane_multi_head_lut import (
    HyperplaneMultiHeadLUT,
    _HyperplaneMHLutSoft,
    _HyperplaneMHLutHybridSmooth,
    _HyperplaneMHLutFullSoft,
    _pick_fwd_body,
)

__all__ = ["TernaryHyperplaneMultiHeadLUT"]

# |w| <= T * ln 3  =>  q = 0.   (2 * atanh(1/2) == ln 3)
_ZERO_BAND_PER_T = math.log(3.0)

# T at which an anchor-pairs init (entries exactly +-1) survives quantization:
# it needs T * ln3 < 1, i.e. T < 1/ln3 ~= 0.9102.
_ANCHOR_SAFE_T = 1.0 / _ZERO_BAND_PER_T

# Extra init mode this subclass adds. The parent validates its own set and would
# reject an unknown name, so "balanced_ternary" is intercepted here and the parent is
# constructed with "random" before the weights are redrawn.
_BALANCED_TERNARY = "balanced_ternary"


def _balanced_sigma(temp: float, target_zero_frac: float = 1.0 / 3.0) -> float:
    """Weight std that makes a zero-mean Gaussian init quantize to equal thirds.

    For w ~ N(0, sigma^2) the component is 0 exactly when |w| <= band = T*ln3, so

        P(zero) = 2*Phi(band/sigma) - 1 = target
        =>  band/sigma = Phi^-1((1 + target)/2)
        =>  sigma      = band / Phi^-1((1 + target)/2)

    At the default target 1/3 that inverse-CDF factor is Phi^-1(2/3) ~= 0.43073, so
    sigma ~= 2.5504 * T. With T = 0.5 (band 0.5493) that gives sigma ~= 1.2754.

    P(+1) == P(-1) automatically, since the draw is symmetric about zero -- the only
    free knob is the scale relative to the band.
    """
    if not (0.0 < target_zero_frac < 1.0):
        raise ValueError(f"target_zero_frac must be in (0, 1), got {target_zero_frac}")
    band = temp * _ZERO_BAND_PER_T
    z = float(torch.special.ndtri(torch.tensor((1.0 + target_zero_frac) / 2.0,
                                               dtype=torch.float64)))
    return band / z


class TernaryHyperplaneMultiHeadLUT(HyperplaneMultiHeadLUT):
    """HyperplaneMultiHeadLUT with straight-through ternary {-1, 0, +1} routing.

    Drop-in for HyperplaneMultiHeadLUT: same constructor arguments, same forward
    signature (`x: [B, input_dim] -> [B, n_heads, n_outputs]`), same parameter shapes,
    plus `log_ternary_temp` of shape `[n_tables, 1, 1]`.

    Extra args:
        ternary_temp_init: initial T (NOT log T), one per table. Default 0.5.
            The default is chosen, not arbitrary: the dead zone is |w| <= T*ln3, so
            T=0.5 gives a band of 0.549. With the inherited default
            `hyperplane_init="anchor_pairs"` the weights are exactly {-1, 0, +1}
            already, so every +-1 survives (1 > 0.549) and every 0 stays 0 -- the
            module reduces EXACTLY to anchor-pair routing at step 0, the same
            bit-for-bit A/B property HyperplaneMultiHeadLUT has against
            FastMultiHeadLut. The "soft, T ~ 1.0" reflex is actively wrong here:
            T=1.0 puts the band at 1.0986 > 1, which quantizes an anchor-pairs init
            to ALL ZEROS and destroys the routing before training starts.

    There is no bias term and no flag for one: the routing test is `<q, x> > 0`.
    `hyperplane_bias` survives only as a frozen all-zero BUFFER, because the
    inherited autograd Functions take it as a positional argument.

    Init modes: the parent's `"anchor_pairs"` (default) and `"random"`, plus this
    subclass's `"balanced_ternary"` -- a zero-mean Gaussian draw whose std is chosen so
    the step-0 routing quantizes to approximately equal thirds of -1 / 0 / +1. Where
    `anchor_pairs` pins exactly 2 non-zeros per hyperplane and puts every component
    maximally far from the dead-zone boundary (zeros at w=0, +-1s at |w|=1, against a
    boundary at 0.5493), `balanced_ternary` spreads components across the boundary so a
    fraction of them are within reach of flipping. `sigma = T*ln3 / Phi^-1((1+f)/2)`,
    i.e. ~2.5504*T at the default third; `balanced_target_zero_frac` moves that target.
    The realized split is not assumed -- check it with `ternary_stats()`.

    Note on random init: `hyperplane_init="random"` draws rows with std
    `initial_weights_noise` (default 1e-3). Against a band of ~0.55 that quantizes to
    all zeros, so a random init needs `hyperplane_init_scale` on the order of T (or
    larger) to route at all. The constructor checks the realized init and warns if the
    ternary weights come out entirely zero, rather than letting a silently dead
    routing train for hours.
    """

    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        n_anchor_pairs: int,
        tables_per_head: int = 1,
        *,
        ternary_temp_init: float = 0.5,
        nonzero_penalty_weight: float = 0.0,
        balanced_target_zero_frac: float = 1.0 / 3.0,
        **kwargs,
    ):
        # "balanced_ternary" is this subclass's own init mode; the parent validates
        # against its own set and would reject the name, so build it as "random" and
        # redraw below. The parent's anchor_pairs / random paths are untouched.
        requested_init = kwargs.get("hyperplane_init", "anchor_pairs")
        balanced = requested_init == _BALANCED_TERNARY
        if balanced:
            kwargs["hyperplane_init"] = "random"
        super().__init__(
            input_dim, n_heads, n_outputs, n_anchor_pairs, tables_per_head, **kwargs
        )
        if ternary_temp_init <= 0:
            raise ValueError(
                f"ternary_temp_init must be > 0 (it is a temperature), "
                f"got {ternary_temp_init}"
            )
        dev = self.hyperplane_weight.device
        # One T per TABLE, broadcast over that table's (nap, input_dim) block.
        self.log_ternary_temp = nn.Parameter(
            torch.full((self.n_lookup_tables, 1, 1), math.log(float(ternary_temp_init)),
                       dtype=torch.float32, device=dev)
        )

        # NO BIAS. The routing test is <q, x> > 0. The inherited autograd Functions
        # take b as a positional argument, so it stays as a frozen all-zero BUFFER --
        # demoted from Parameter, so it receives no gradient, is not returned by
        # .parameters(), and can never move off zero.
        b = self.hyperplane_bias.detach().zero_()
        del self.hyperplane_bias
        self.register_buffer("hyperplane_bias", b)

        # Plain Python float, deliberately NOT a Parameter or buffer: it must not
        # appear in state_dict, so enabling or disabling the penalty never changes a
        # checkpoint's shape and never invalidates an existing run.
        if nonzero_penalty_weight < 0:
            raise ValueError(
                f"nonzero_penalty_weight must be >= 0, got {nonzero_penalty_weight}"
            )
        self.nonzero_penalty_weight = float(nonzero_penalty_weight)

        # Balanced-ternary init: redraw w so the step-0 routing is ~equal thirds of
        # -1 / 0 / +1, instead of anchor_pairs' 2 nonzeros per hyperplane with every
        # component sitting maximally far from the dead-zone boundary.
        self.balanced_sigma = None
        if balanced:
            self.balanced_sigma = _balanced_sigma(ternary_temp_init,
                                                  balanced_target_zero_frac)
            gen = None
            if kwargs.get("random_seed") is not None:
                gen = torch.Generator(device=self.hyperplane_weight.device)
                # +2: the parent already consumed random_seed for the anchors and
                # random_seed+1 for the LUT tables.
                gen.manual_seed(int(kwargs["random_seed"]) + 2)
            with torch.no_grad():
                self.hyperplane_weight.normal_(0.0, self.balanced_sigma, generator=gen)
            self.hyperplane_init = _BALANCED_TERNARY   # report what was ACTUALLY used

        nz = int((self.hard_ternary_weight() != 0).sum())
        if nz == 0:
            warnings.warn(
                f"TernaryHyperplaneMultiHeadLUT: the ternary routing is ENTIRELY ZERO "
                f"at init (dead zone |w| <= T*ln3 = {ternary_temp_init * _ZERO_BAND_PER_T:.4f} "
                f"swallows every weight). Every hyperplane test would be constant. "
                f"Lower ternary_temp_init (< {_ANCHOR_SAFE_T:.4f} for an anchor_pairs "
                f"init) or raise hyperplane_init_scale for a random init.",
                RuntimeWarning,
                stacklevel=2,
            )

    # ---- the ternary routing -------------------------------------------------

    @property
    def ternary_temp(self) -> torch.Tensor:
        """T, positive by construction, shape [n_tables, 1, 1]."""
        return self.log_ternary_temp.exp()

    def ternary_temp_per_head(self) -> torch.Tensor:
        """T viewed as [n_heads, tables_per_head] -- the natural table indexing."""
        return self.ternary_temp.view(self.n_heads, self.tables_per_head)

    def zero_band(self) -> torch.Tensor:
        """Half-width of the dead zone in weight units: |w| <= T*ln3 maps to 0."""
        return self.ternary_temp.view(-1) * _ZERO_BAND_PER_T

    def soft_ternary_weight(self) -> torch.Tensor:
        """s = tanh(w / (2T)) -- the smooth surrogate the STE differentiates."""
        return torch.tanh(self.hyperplane_weight / (2.0 * self.ternary_temp))

    def hard_ternary_weight(self) -> torch.Tensor:
        """q_hard in {-1, 0, +1}, no autograd. This is what you bake and ship."""
        with torch.no_grad():
            s = self.soft_ternary_weight()
            return torch.sign(s) * (s.abs() > 0.5).to(s.dtype)

    def ternary_weight(self) -> torch.Tensor:
        """q = s + (q_hard - s).detach().

        Forward value is exactly ternary; gradient flows through s to both
        `hyperplane_weight` and `log_ternary_temp`.
        """
        s = self.soft_ternary_weight()
        q_hard = torch.sign(s) * (s.abs() > 0.5).to(s.dtype)
        return s + (q_hard - s).detach()

    # ---- optional soft sparsity penalty (off by default) --------------------

    def sparsity_surrogate(self) -> torch.Tensor:
        """RAW, UNWEIGHTED smooth surrogate for the fraction of non-zero components.

        `mean(|tanh(w / (2T))|)` over every hyperplane weight element. Each term is
        ~0 inside the dead zone (component quantizes to 0) and ~1 when saturated
        (component quantizes to +-1), so the mean approximates the fraction of
        components the routing actually uses. MEAN rather than sum, so the value is
        dimension-invariant and lambda means the same thing at any shape.

        Always differentiable and always computed -- this is the quantity to LOG.
        `sparsity_penalty()` is what you add to the loss.
        """
        return self.soft_ternary_weight().abs().mean()

    def sparsity_penalty(self) -> torch.Tensor:
        """LAMBDA-WEIGHTED penalty term, ready to add straight to the loss.

        Returns `nonzero_penalty_weight * sparsity_surrogate()`.

        When `nonzero_penalty_weight == 0` (the default) it returns a detached zero
        scalar and computes NOTHING: no surrogate, no graph nodes, no gradient. Adding
        it to a loss is then a genuine no-op, which is what keeps this class
        bit-for-bit identical to the un-penalized version at default settings.

        Sum it over the slots and add it once:

            pen = sum(m.sparsity_penalty() for m in model.modules()
                      if isinstance(m, TernaryHyperplaneMultiHeadLUT))
            (loss + pen).backward()
        """
        if self.nonzero_penalty_weight == 0.0:
            return torch.zeros((), device=self.hyperplane_weight.device,
                               dtype=torch.float32)
        return self.nonzero_penalty_weight * self.sparsity_surrogate()

    def bake_ternary_weights(self) -> torch.Tensor:
        """Deployment helper: the {-1, 0, +1} routing, detached and on CPU.

        After this the float `hyperplane_weight` is no longer needed -- the forward
        depends on it only through q. Returned as int8, which is how you would store
        it (or convert to two index lists per hyperplane for a multiplication-free
        add/subtract kernel).
        """
        return self.hard_ternary_weight().to(torch.int8).cpu()

    def ternary_stats(self) -> dict:
        """Fractions of +1 / 0 / -1 in the realized routing, for monitoring."""
        q = self.hard_ternary_weight()
        n = q.numel()
        return {
            "frac_pos": float((q > 0).sum()) / n,
            "frac_zero": float((q == 0).sum()) / n,
            "frac_neg": float((q < 0).sum()) / n,
            "nonzero_per_hyperplane": float((q != 0).sum()) / (q.shape[0] * q.shape[1]),
            "T_min": float(self.ternary_temp.min().detach()),
            "T_max": float(self.ternary_temp.max().detach()),
        }

    # ---- forwards: identical to the parent, with q substituted for w ---------

    def _hard_eval(self, x: torch.Tensor) -> torch.Tensor:
        """no_grad hard eval on the TERNARY routing.

        Overridden so eval uses the same {-1, 0, +1} weights training commits to --
        evaluating with the continuous w would measure a model that will never be
        deployed.
        """
        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.bfloat16)
            if self.use_bf16 and x.is_cuda
            else torch.amp.autocast("cpu", enabled=False)
        )
        compute_in_bf16 = (
            self.use_bf16 and x.is_cuda and self.weights.dtype == torch.float32
        )
        weights_compute = (
            self.weights.to(torch.bfloat16) if compute_in_bf16 else self.weights
        )
        fwd_body = _pick_fwd_body(x.is_cuda)
        q = self.hard_ternary_weight()
        with autocast_ctx:
            out, _ = fwd_body(
                x, weights_compute, q, self.hyperplane_bias,
                self.soft_powers, self.n_heads, self.tables_per_head, self.table_dim,
            )
        if compute_in_bf16:
            out = out.to(self.weights.dtype)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.forward_mode == "hybrid_smooth":
            return _HyperplaneMHLutHybridSmooth.apply(
                x, self.weights, self.ternary_weight(), self.hyperplane_bias,
                self.log_soft_score_temp, self.log_select_temp,
                self.soft_bit_matrix, self.soft_powers,
                self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
            )
        # forward_mode == "hard"
        if not torch.is_grad_enabled():
            return self._hard_eval(x)
        return _HyperplaneMHLutSoft.apply(
            x, self.weights, self.ternary_weight(), self.hyperplane_bias,
            self.log_soft_score_temp, self.log_select_temp,
            self.soft_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
        )

    def forward_full_soft(self, x: torch.Tensor) -> torch.Tensor:
        """Full-K softmax surrogate forward (reference / gradcheck target)."""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [B, {self.input_dim}], got {tuple(x.shape)}"
            )
        return _HyperplaneMHLutFullSoft.apply(
            x, self.weights, self.ternary_weight(), self.hyperplane_bias,
            self.log_soft_score_temp, self.log_select_temp,
            self.soft_bit_matrix, self.soft_powers,
            self.n_heads, self.tables_per_head, self.table_dim, self.use_bf16,
        )


# =============================================================================
# Sanity checks. Run: python -m spiky.lutorch.ternary_hyperplane_multi_head_lut
# =============================================================================

def _sanity() -> None:
    torch.manual_seed(0)
    D, H, OUT, NAP, TPH, B = 16, 2, 8, 3, 4, 32
    kw = dict(input_dim=D, n_heads=H, n_outputs=OUT, n_anchor_pairs=NAP,
              tables_per_head=TPH, random_seed=1234)
    m = TernaryHyperplaneMultiHeadLUT(**kw)
    ref = HyperplaneMultiHeadLUT(**kw)
    x = torch.randn(B, D)
    ok = True

    print('=== (1) the projection weights are exactly ternary ===')
    q = m.ternary_weight()
    vals = torch.unique(q)
    print(f'  unique values in q: {vals.tolist()}')
    is_tern = bool(torch.isin(vals, torch.tensor([-1.0, 0.0, 1.0])).all())
    print(f'  q subset of {{-1, 0, +1}}: {is_tern}')
    print(f'  forward value == q_hard exactly: '
          f'{bool(torch.equal(q, m.hard_ternary_weight()))}')
    print(f'  q carries grad (STE attached): {q.requires_grad}')
    ok &= is_tern and bool(torch.equal(q, m.hard_ternary_weight()))

    print('\n=== (2) STE gradient reaches w and log_T ===')
    m.zero_grad(set_to_none=True)
    m(x).sum().backward()
    gw, gT = m.hyperplane_weight.grad, m.log_ternary_temp.grad
    print(f'  grad hyperplane_weight : finite={bool(torch.isfinite(gw).all())} '
          f'nonzero={bool((gw != 0).any())}  |g|_max={float(gw.abs().max()):.3e}')
    print(f'  grad log_ternary_temp  : finite={bool(torch.isfinite(gT).all())} '
          f'nonzero={bool((gT != 0).any())}  |g|_max={float(gT.abs().max()):.3e}')
    print(f'  grad LUT tables        : nonzero={bool((m.weights.grad != 0).any())}')
    ok &= bool((gw != 0).any()) and bool((gT != 0).any())

    print('\n=== (3) T controls the zero band: |w| <= T*ln3 -> 0 ===')
    probe = TernaryHyperplaneMultiHeadLUT(**kw, hyperplane_init='random',
                                          hyperplane_init_scale=1.0)
    with torch.no_grad():
        probe.hyperplane_weight.normal_(0.0, 1.0)
    # The claim is exact, so test it exactly: frac_zero must equal the fraction of
    # weights inside the analytic band |w| <= T*ln3, not merely trend the right way.
    # (A loose "T->0 gives zero zeros" bound would be wrong anyway: at T=0.01 the band
    # is 0.011 wide and a normal weight lands inside it ~1% of the time.)
    w_abs = probe.hyperplane_weight.detach().abs()
    print(f'  {"T":>10} {"band=T*ln3":>12} {"frac_zero":>10} {"predicted":>10} '
          f'{"frac_+1":>9} {"frac_-1":>9}')
    fr, exact_ok = [], True
    for T in (1e-4, 0.01, 0.1, 0.5, 1.0, 3.0, 10.0):
        with torch.no_grad():
            probe.log_ternary_temp.fill_(math.log(T))
        s = probe.ternary_stats()
        pred = float((w_abs <= T * _ZERO_BAND_PER_T).float().mean())
        exact_ok &= abs(pred - s['frac_zero']) < 1e-6
        fr.append(s['frac_zero'])
        print(f'  {T:>10.4f} {T * _ZERO_BAND_PER_T:>12.4f} {s["frac_zero"]:>10.4f} '
              f'{pred:>10.4f} {s["frac_pos"]:>9.4f} {s["frac_neg"]:>9.4f}')
    mono = all(b >= a - 1e-12 for a, b in zip(fr, fr[1:]))
    print(f'  frac_zero monotonically non-decreasing in T: {mono}')
    print(f'  matches the analytic band |w| <= T*ln3 at every T: {exact_ok}')
    print(f'  T->0 collapses the band ({fr[0]:.4f} zeros, weights go hard +-1); '
          f'large T swallows everything ({fr[-1]:.4f})')
    ok &= mono and exact_ok and fr[0] < 1e-6 and fr[-1] > 1.0 - 1e-6

    print('\n=== (4) drop-in: shapes match HyperplaneMultiHeadLUT ===')
    sm = {n: tuple(p.shape) for n, p in m.named_parameters()}
    sr = {n: tuple(p.shape) for n, p in ref.named_parameters()}
    shared = sorted(set(sm) & set(sr))
    for n in shared:
        flag = 'OK' if sm[n] == sr[n] else 'DIFFERS'
        print(f'  {n:<24} {str(sr[n]):<18} {str(sm[n]):<18} {flag}')
    extra = sorted(set(sm) - set(sr))
    missing = sorted(set(sr) - set(sm))
    print(f'  extra in ternary  : {[(n, sm[n]) for n in extra]}')
    print(f'  absent in ternary : {missing}  (hyperplane_bias — dropped by design)')
    same_shapes = all(sm[n] == sr[n] for n in shared)
    print(f'  all shared parameter shapes identical: {same_shapes}')
    yr, ym = ref(x), m(x)
    print(f'  output shape  ref {tuple(yr.shape)}  ternary {tuple(ym.shape)}  '
          f'{"OK" if yr.shape == ym.shape else "DIFFERS"}')
    ok &= (same_shapes and yr.shape == ym.shape
           and missing == ['hyperplane_bias']
           and extra == ['log_ternary_temp'])

    print('\n=== (4b) the bias is gone: no term, no parameter, frozen at zero ===')
    bufs = dict(m.named_buffers())
    print(f'  hyperplane_bias in parameters: '
          f'{"hyperplane_bias" in sm} (False expected)')
    print(f'  hyperplane_bias in buffers   : {"hyperplane_bias" in bufs} (True expected)')
    print(f'  all zero: {bool((bufs["hyperplane_bias"] == 0).all())}   '
          f'requires_grad: {bufs["hyperplane_bias"].requires_grad} (False expected)')
    m.zero_grad(set_to_none=True)
    m(x).sum().backward()
    print(f'  still zero after a backward+would-be update: '
          f'{bool((m.hyperplane_bias == 0).all())}')
    print(f'  routing test is <q, x> > 0 (no threshold term)')
    ok &= ('hyperplane_bias' not in sm and 'hyperplane_bias' in bufs
           and not bufs['hyperplane_bias'].requires_grad
           and bool((bufs['hyperplane_bias'] == 0).all()))

    print('\n=== (5) anchor_pairs init + default T=0.5 reduces to anchor pairs ===')
    a = TernaryHyperplaneMultiHeadLUT(**kw)          # default T=0.5, anchor_pairs init
    qa = a.hard_ternary_weight()
    exact = bool(torch.equal(qa, a.hyperplane_weight.detach()))
    print(f'  q == the anchor-pair weights exactly: {exact}  '
          f'(band {0.5 * _ZERO_BAND_PER_T:.4f} < 1)')
    print(f'  nonzeros per hyperplane: {a.ternary_stats()["nonzero_per_hyperplane"]:.2f} '
          f'(2 expected -- one +1 and one -1)')
    ok &= exact
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        TernaryHyperplaneMultiHeadLUT(**kw, ternary_temp_init=1.0)
        warned = any(issubclass(x_.category, RuntimeWarning) for x_ in w)
    print(f'  T=1.0 would zero an anchor-pairs init, and the constructor warns: {warned}')
    ok &= warned

    print('\n=== (6) lambda=0 is a verified NO-OP (the exp_g_0030 guarantee) ===')
    z = TernaryHyperplaneMultiHeadLUT(**kw)                       # default lambda=0
    pz = TernaryHyperplaneMultiHeadLUT(**kw, nonzero_penalty_weight=0.7)
    pen0 = z.sparsity_penalty()
    print(f'  default nonzero_penalty_weight: {z.nonzero_penalty_weight} (0.0 expected)')
    print(f'  sparsity_penalty() -> {float(pen0):.1f}  requires_grad={pen0.requires_grad} '
          f'grad_fn={pen0.grad_fn}  (0.0 / False / None expected)')
    same_params = ({n: tuple(p.shape) for n, p in z.named_parameters()}
                   == {n: tuple(p.shape) for n, p in pz.named_parameters()})
    same_sd = sorted(z.state_dict()) == sorted(pz.state_dict())
    print(f'  lambda adds no parameters: {same_params}')
    print(f'  lambda adds no state_dict entries: {same_sd}  '
          f'(so it never invalidates a checkpoint)')
    # forward is untouched by lambda: same inputs, same weights -> same output
    with torch.no_grad():
        pz.load_state_dict(z.state_dict())
    fz, fp = z(x), pz(x)
    print(f'  forward identical with lambda=0 vs lambda=0.7: '
          f'{bool(torch.equal(fz, fp))}  (penalty is loss-only)')
    # grads through the model must be untouched when the penalty is never added
    z.zero_grad(set_to_none=True); pz.zero_grad(set_to_none=True)
    z(x).sum().backward(); pz(x).sum().backward()
    grads_same = bool(torch.equal(z.hyperplane_weight.grad, pz.hyperplane_weight.grad))
    print(f'  forward-only grads identical: {grads_same}')
    noop = (z.nonzero_penalty_weight == 0.0 and float(pen0) == 0.0
            and not pen0.requires_grad and pen0.grad_fn is None
            and same_params and same_sd and bool(torch.equal(fz, fp)) and grads_same)
    print(f'  => lambda=0 no-op verified: {noop}')
    ok &= noop

    print('\n=== (7) lambda>0: the penalty is live and actually sparsifies ===')
    s = TernaryHyperplaneMultiHeadLUT(**kw, hyperplane_init='random',
                                      hyperplane_init_scale=1.0,
                                      nonzero_penalty_weight=1.0)
    with torch.no_grad():
        s.hyperplane_weight.normal_(0.0, 1.0)
    p0 = s.sparsity_penalty()
    print(f'  penalty value {float(p0):.6f}  finite={bool(torch.isfinite(p0))}  '
          f'scalar={p0.dim() == 0}  requires_grad={p0.requires_grad}')
    s.zero_grad(set_to_none=True)
    p0.backward()
    gw_ok = bool((s.hyperplane_weight.grad != 0).any())
    gT_ok = bool((s.log_ternary_temp.grad != 0).any())
    print(f'  penalty grad -> hyperplane_weight: {gw_ok}   -> log_ternary_temp: {gT_ok}')
    before = s.ternary_stats()
    opt = torch.optim.SGD([s.hyperplane_weight, s.log_ternary_temp], lr=0.5)
    for _ in range(30):
        opt.zero_grad(set_to_none=True)
        s.sparsity_penalty().backward()
        opt.step()
    after = s.ternary_stats()
    print(f'  30 steps of descent on the PENALTY ALONE:')
    print(f'    frac_zero  {before["frac_zero"]:.4f} -> {after["frac_zero"]:.4f}   '
          f'({after["frac_zero"] - before["frac_zero"]:+.4f})')
    print(f'    nonzeros/hyperplane {before["nonzero_per_hyperplane"]:.2f} -> '
          f'{after["nonzero_per_hyperplane"]:.2f}')
    print(f'    T {before["T_max"]:.4f} -> {after["T_max"]:.4f}  '
          f'(band widening is the second, legitimate route to sparsity)')
    sparsified = after['frac_zero'] > before['frac_zero']
    print(f'  penalty measurably increased the zero fraction: {sparsified}')
    ok &= (gw_ok and gT_ok and sparsified and bool(torch.isfinite(p0))
           and p0.dim() == 0)

    print('\n=== (8) the surrogate tracks actual density ===')
    t = TernaryHyperplaneMultiHeadLUT(**kw, hyperplane_init='random',
                                      hyperplane_init_scale=1.0,
                                      nonzero_penalty_weight=1.0)
    with torch.no_grad():
        t.hyperplane_weight.normal_(0.0, 1.0)
    print(f'  {"T":>8} {"surrogate":>10} {"frac_nonzero":>13}  (should move together)')
    pairs = []
    for T in (0.02, 0.1, 0.3, 1.0, 3.0, 20.0):
        with torch.no_grad():
            t.log_ternary_temp.fill_(math.log(T))
        sur = float(t.sparsity_surrogate().detach())
        nz = 1.0 - t.ternary_stats()['frac_zero']
        pairs.append((sur, nz))
        print(f'  {T:>8.2f} {sur:>10.4f} {nz:>13.4f}')
    mono_s = all(b[0] <= a[0] + 1e-12 for a, b in zip(pairs, pairs[1:]))
    print(f'  surrogate decreases monotonically as the band widens: {mono_s}')
    print(f'  deep in the dead zone (T=20) surrogate ~ {pairs[-1][0]:.4f}, '
          f'density {pairs[-1][1]:.4f}')
    print(f'  saturated (T=0.02)       surrogate ~ {pairs[0][0]:.4f}, '
          f'density {pairs[0][1]:.4f}')
    tracks = mono_s and pairs[-1][0] < 0.05 and pairs[0][0] > 0.9
    print(f'  surrogate tracks density: {tracks}')
    ok &= tracks

    print('\n=== (9) balanced_ternary init gives ~equal thirds at step 0 ===')
    # A bigger module: the split is a sample statistic, and 384 components is too few
    # to read a third off cleanly.
    bkw = dict(input_dim=256, n_heads=4, n_outputs=32, n_anchor_pairs=6,
               tables_per_head=32, random_seed=7)
    bal = TernaryHyperplaneHelper = TernaryHyperplaneMultiHeadLUT(
        **bkw, hyperplane_init='balanced_ternary', ternary_temp_init=0.5)
    bs = bal.ternary_stats()
    print(f'  T {0.5}  ->  band {0.5 * _ZERO_BAND_PER_T:.4f}, '
          f'sigma {bal.balanced_sigma:.4f}  (expected ~2.5504*T = 1.2752)')
    print(f'  step-0 split   -1 {bs["frac_neg"]:.4f}   0 {bs["frac_zero"]:.4f}   '
          f'+1 {bs["frac_pos"]:.4f}   over {bal.hyperplane_weight.numel():,} components')
    print(f'  measured w std {float(bal.hyperplane_weight.std().detach()):.4f}')
    print(f'  hyperplane_init reports: {bal.hyperplane_init!r}')
    thirds = all(abs(v - 1 / 3) < 0.02
                 for v in (bs['frac_neg'], bs['frac_zero'], bs['frac_pos']))
    print(f'  all three within 0.02 of 1/3: {thirds}')
    print(f'  nonzeros per hyperplane {bs["nonzero_per_hyperplane"]:.2f} '
          f'(vs anchor_pairs\' exactly 2.00 over {bkw["input_dim"]} components)')
    # the all-zero guard must still be satisfied
    print(f'  routing not all-zero: {bool((bal.hard_ternary_weight() != 0).any())}')
    # a different target must move the split predictably
    b2 = TernaryHyperplaneMultiHeadLUT(**bkw, hyperplane_init='balanced_ternary',
                                       ternary_temp_init=0.5,
                                       balanced_target_zero_frac=0.6)
    s2 = b2.ternary_stats()
    print(f'  target_zero_frac 0.60 -> measured frac_zero {s2["frac_zero"]:.4f} '
          f'(sigma {b2.balanced_sigma:.4f})')
    targeted = abs(s2['frac_zero'] - 0.6) < 0.02
    print(f'  retargeting works: {targeted}')
    # and the anchor_pairs path must be completely unaffected
    ap = TernaryHyperplaneMultiHeadLUT(**bkw)
    apq = ap.hard_ternary_weight()
    untouched = (bool(torch.equal(apq, ap.hyperplane_weight.detach()))
                 and ap.balanced_sigma is None
                 and ap.hyperplane_init == 'anchor_pairs')
    print(f'  anchor_pairs path untouched (still exact, sigma None): {untouched}')
    ok &= thirds and targeted and untouched and bool((bal.hard_ternary_weight() != 0).any())

    print('\nSANITY: ' + ('PASS' if ok else 'FAIL'))
    if not ok:
        raise SystemExit(1)


if __name__ == '__main__':
    _sanity()
