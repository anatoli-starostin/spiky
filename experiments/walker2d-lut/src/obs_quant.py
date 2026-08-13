"""Gaussian-companding observation quantizer for quantization-aware PPO fine-tuning.

WHY THIS EXISTS. The exp19 LUT actor addresses its tables by pairwise comparisons of the
NORMALISED observation vector -- `fast_multi_head_lut.py`:

    d    = x[:, anchor_a] - x[:, anchor_b]
    bits = (d > 0).to(torch.int64)

so the policy only ever consumes the *order* of the 17 coordinates, never their magnitudes
(magnitudes enter solely through the stored table values). A spiking implementation encodes
each coordinate as a first-spike TICK on a finite grid, which means the comparison is made
on quantised values. Every disagreement between the float policy and the spiking one is
therefore a pair that quantisation collapsed into the same bucket -- a near-tie. Training
the policy with the quantiser already in the loop lets it move its anchors and tables away
from those ties instead of merely tolerating them at deploy time.

THE MAP. Each normalised scalar is companded through the standard normal CDF and rounded to
one of `n_ticks` levels:

    tick = round((n_ticks - 1) * Phi(x / sigma))            (clamped to [0, n_ticks-1])
    xhat = DEQUANT[tick]                                    (bucket centre, precomputed)

Phi spends levels in proportion to Gaussian density, so a normalised (mean 0, unit variance)
observation gets fine resolution through the bulk and coarse resolution in the tails. That
is exactly where the near-ties are: values are dense in the core, so that is where a tick is
worth the most. `sigma` sets the companding strength -- sigma < 1 concentrates levels harder
into the core, sigma > 1 flattens toward a uniform grid.

*** THE MAP IS SHARED ACROSS ALL 17 COORDINATES, AND MUST STAY THAT WAY. ***
The address bits are comparisons BETWEEN coordinates. A per-coordinate scale or offset would
change `xhat[a] > xhat[b]` relative to `x[a] > x[b]` for every pair spanning two different
maps, and the address bit would stop meaning "coordinate a exceeds coordinate b". Only one
shared, strictly monotone map keeps the comparator semantics intact. Because this map IS
strictly monotone, the only comparisons it can change are those it collapses into a single
bucket (a tie), and the forward uses a strict `>`, so a tie deterministically yields bit 0.

GRADIENTS. Rounding has zero gradient almost everywhere, so the quantiser is applied with a
straight-through estimator: the forward value is the dequantised one, the backward pass sees
the identity. This is the standard STE and it is what makes the fine-tune "quantization
AWARE" rather than merely "quantization tested".

Self-contained: importing this module has no side effects, and nothing else in the chapter
imports it unless asked.
"""
import math

import torch

DEFAULT_N_TICKS = 128
DEFAULT_SIGMA = 1.0


def _phi(z):
    """Standard normal CDF."""
    return 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))


def _phi_inv(p):
    """Standard normal quantile function."""
    return math.sqrt(2.0) * torch.erfinv(2.0 * p - 1.0)


def build_dequant_table(n_ticks=DEFAULT_N_TICKS, sigma=DEFAULT_SIGMA,
                        device=None, dtype=torch.float32):
    """The `n_ticks` bucket-centre values, in normalised-observation units.

    Bucket t covers Phi in [(t-0.5)/(n-1), (t+0.5)/(n-1)] intersected with [0, 1]. The two
    end buckets are unbounded in x, so their Phi-interval midpoints are used; taking t/(n-1)
    there would give Phi_inv(0) = -inf and Phi_inv(1) = +inf.

    The result is strictly increasing, which is what preserves the comparator semantics.
    """
    n = int(n_ticks)
    if n < 2:
        raise ValueError(f"n_ticks must be >= 2, got {n}")
    span = n - 1
    t = torch.arange(n, device=device, dtype=torch.float64)
    p = t / span
    p[0] = 0.25 / span                  # midpoint of [0, 0.5/span]
    p[-1] = 1.0 - 0.25 / span           # midpoint of [1 - 0.5/span, 1]
    return (float(sigma) * _phi_inv(p)).to(dtype)


class GaussianCompandingQuantizer(torch.nn.Module):
    """Shared monotone `n_ticks`-level quantiser on the normalised observation vector.

    Stateless apart from the precomputed dequantisation table, which is a buffer so it
    follows `.to(device)` and is captured by `state_dict()`.
    """

    def __init__(self, n_ticks=DEFAULT_N_TICKS, sigma=DEFAULT_SIGMA, straight_through=True):
        super().__init__()
        self.n_ticks = int(n_ticks)
        self.sigma = float(sigma)
        self.straight_through = bool(straight_through)
        self.register_buffer("dequant", build_dequant_table(n_ticks, sigma), persistent=True)

    def ticks(self, x):
        """Integer tick index in [0, n_ticks-1]. No gradient; for analysis and for an
        eventual spiking encoder, which consumes exactly this."""
        with torch.no_grad():
            p = _phi(x / self.sigma)
            t = torch.round(p * (self.n_ticks - 1))
            return t.clamp_(0, self.n_ticks - 1).to(torch.int64)

    def forward(self, x):
        xq = self.dequant.to(x.dtype)[self.ticks(x)]
        if self.straight_through:
            # value = quantised, gradient = identity
            return x + (xq - x).detach()
        return xq

    def extra_repr(self):
        return (f"n_ticks={self.n_ticks}, sigma={self.sigma}, "
                f"straight_through={self.straight_through}, "
                f"range=[{float(self.dequant[0]):.3f}, {float(self.dequant[-1]):.3f}]")


def comparison_flip_rate(x, quant, anchor_a, anchor_b):
    """Fraction of the LUT's address bits that quantisation changes.

    Reproduces the forward's own convention exactly: bit = (x[a] - x[b] > 0), strict, so a
    collapsed pair yields 0. Returns (flip_rate, tie_rate) where tie_rate is the share of
    compared pairs landing in the same bucket -- the only way a flip can happen.
    """
    with torch.no_grad():
        xq = quant(x)
        bits = (x[:, anchor_a] - x[:, anchor_b]) > 0
        bits_q = (xq[:, anchor_a] - xq[:, anchor_b]) > 0
        ties = quant.ticks(x)[:, anchor_a] == quant.ticks(x)[:, anchor_b]
        return float((bits != bits_q).float().mean()), float(ties.float().mean())
