"""Uniform output (action-mean) quantizer, modelling the spiking Stage-3 readout.

WHY UNIFORM, AND WHY CLIP RATHER THAN TANH. The spiking actor decodes each action dim from
the FIRST SPIKE TICK of one anti-leaky output neuron:

    T  = first spike tick (an integer)
    mu = affine[o,0] * T + affine[o,1]
    return np.clip(mu, -1.0, 1.0)                  # spiking_lut.py:224-225

Because tick -> action is affine and T is an integer, the emitted action values form a
UNIFORM grid -- the step is |affine[o,0]| and is identical near zero, near +-1, and in the
tails. There is no companding to reproduce here (unlike the INPUT encoder, which is
deliberately companded). Measured on the shipped model the step is 0.276-0.283 action units
per tick, i.e. only ~7-8 emittable levels inside the actuator band, and the round-trip error
matches the textbook uniform prediction step/sqrt(12) to three decimals.

The readout ends in `np.clip`, not a tanh, and it physically cannot do otherwise: a tick is
decoded by an affine map. So the model here clips too. Clip's native gradient -- exactly zero
outside +-1 -- is the correct behaviour and is deliberately left alone: an action component
the readout cannot represent should receive no gradient pull from the quantizer.

STE. Rounding has zero gradient a.e., so the forward value is the quantized one and the
backward pass sees the (clipped) identity. Unlike the INPUT quantizer -- where the STE is
cosmetic because the input is a leaf with nothing learnable upstream -- here the STE is
genuinely load-bearing: the quantized mean is what the environment consumes and what the
Gaussian log-prob is centred on, so real parameter gradient flows through the rounding.
"""
import torch


class UniformActionQuantizer(torch.nn.Module):
    """Clip to +-`clip`, then snap to one of `levels` evenly spaced values, straight-through.

    Levels are `torch.linspace(-clip, +clip, levels)`, so the step is `2*clip/(levels-1)`
    and both endpoints are exactly representable -- matching a readout whose extreme ticks
    decode to the rail.
    """

    def __init__(self, levels, clip=1.0, straight_through=True):
        super().__init__()
        if levels < 2:
            raise ValueError(f"levels must be >= 2, got {levels}")
        self.levels = int(levels)
        self.clip = float(clip)
        # straight_through=False returns the quantised value EXACTLY. The STE form below is
        # `mu_c + (q - mu_c).detach()`, which is not value-exact in float32 — fine while
        # training (the perturbation is far below the grid step) but not what a deployed
        # artifact should reproduce, where determinism matters. Export uses False.
        self.straight_through = bool(straight_through)
        self.step = 2.0 * self.clip / (self.levels - 1)
        # diagnostics, refreshed on every forward; read at log time
        self.last_oob = 0.0                  # fraction of |mu| > clip BEFORE clipping
        self.last_oob_per_dim = None
        self.last_raw = None                 # pre-clip mean, graph intact (for the penalty)

    def forward(self, mu):
        with torch.no_grad():
            oob = (mu.abs() > self.clip)
            self.last_oob = float(oob.float().mean())
            self.last_oob_per_dim = oob.float().mean(0).detach().cpu()
        # keep the RAW pre-clip mean reachable: an out-of-band penalty has to differentiate
        # THIS, because everything downstream of the clamp has zero gradient out there.
        self.last_raw = mu
        mu_c = mu.clamp(-self.clip, self.clip)          # native grad: 1 inside, 0 outside
        with torch.no_grad():
            q = torch.round((mu_c + self.clip) / self.step) * self.step - self.clip
            q = q.clamp(-self.clip, self.clip)
        if not self.straight_through:
            return q
        return mu_c + (q - mu_c).detach()               # straight-through

    def extra_repr(self):
        return (f"levels={self.levels}, clip={self.clip}, step={self.step:.6f} "
                f"(spiking readout measures 0.276-0.283)")


def attach(ac, quant):
    """Quantize the action MEAN only, leaving log_std and the sampled action alone.

    A forward hook is used rather than editing the arch: `BaseActorCritic.act()` and
    `.evaluate()` both obtain the mean via `self(obs)`, so hooking the module's output
    covers the rollout and the update with one edit and touches no shared file.

    The Gaussian is then centred on the QUANTIZED mean while the sample stays continuous, so
    log_prob and the PPO importance ratio remain well-defined densities -- which they would
    not be if the sampled action itself were snapped to a grid.
    """
    def _hook(_module, _inputs, output):
        mean, value = output
        return quant(mean), value

    return ac.register_forward_hook(_hook)
