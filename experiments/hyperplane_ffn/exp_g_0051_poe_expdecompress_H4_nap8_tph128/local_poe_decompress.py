"""Escape 1: a deferred product-of-experts decompress on top of the CompressionMHL anchor.

WHY THIS IS CHEAP, and where the deferral actually lives. CompressionMultiHeadLUT.forward
ends with

    return self.decompress(torch.cat(parts, dim=-1))    # [N, H*inner_out] -> [N, output]

`parts` are the per-head gather-SUMS: each head has already summed its tph=128 gathered
cell codes. So that call site runs ONCE PER TOKEN on a 192-dim vector, not once per
gathered cell. Anything pointwise inserted there inherits the same deferral. Reading the
summed codes as log-space and exponentiating gives

    exp( sum_i log c_i ) = prod_i c_i

-- a product over the 128 cells a head gathered -- for the price of 192 exponentials per
token instead of 128*4*48 = 24,576. That 128x is the entire justification, and it is why
a logsumexp OVER cells is deliberately NOT implemented here: that would need exp per cell
BEFORE the sum, which is exactly the deferral this design exists to preserve.

Two placements, both deferred, selected by `poe_placement`:

    'a'  y = M @ exp(s*u + b)      exp on the summed 192-dim code, then the linear lift
    'b'  y = expm1(M @ u + ...)    linear first, then exp on the 384-d output

They are NOT equivalent, and (b) has a structural handicap worth stating plainly: exp of
anything is strictly positive, so `exp(M u)` could only ever ADD to the residual stream,
never subtract. expm1 is used for (b) so the output is >= -1 and is exactly 0 when its
argument is 0 -- signed, centred, and equal to the anchor in the linear regime. It is still
bounded below, which (a) is not: in (a) the exp output is positive but M carries signs, so
y is unrestricted.

For (a) exp vs expm1 makes no difference to what the model can express, since the following
affine map absorbs the constant into its bias; exp is kept there because it is the honest
product-of-experts form.

NUMERICAL STABILITY is the real risk. u is a sum of 128 cell codes, so its scale grows with
tph, and exp overflows fp32 at ~88. Three guards:
  * a learnable per-dim scale (init 1) and shift (init 0), so the model can shrink its own
    pre-exp argument rather than being clipped;
  * a hard clamp on the pre-exp argument, default +-10 -> exp in [4.5e-5, 2.2e4], far
    inside fp32 range, and its gradient is exactly zero outside so a runaway cannot feed
    back;
  * instrumentation: `last_stats` records the pre-clamp argument range and how often the
    clamp actually binds, so we find out rather than assume.
At init the anchor's decompress.weight is zero, and exp(0*u) = 1 with expm1(0) = 0, so both
placements start at the anchor's own output.
"""
import torch
import torch.nn as nn


class DeferredExpGate(nn.Module):
    """Pointwise exp on an already-summed code. Placement (a)'s first half."""

    def __init__(self, dim: int, clamp: float = 10.0, use_expm1: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))
        self.clamp, self.use_expm1 = float(clamp), bool(use_expm1)
        self.last_stats = {}

    def forward(self, u):
        a = u * self.scale + self.shift
        if not self.training or True:            # cheap; always worth knowing
            with torch.no_grad():
                self.last_stats = {
                    'pre_min': float(a.min()), 'pre_max': float(a.max()),
                    'pre_absmean': float(a.abs().mean()),
                    'clamp_frac': float((a.abs() > self.clamp).float().mean()),
                }
        a = a.clamp(-self.clamp, self.clamp)
        return torch.expm1(a) if self.use_expm1 else torch.exp(a)

    def extra_repr(self):
        return (f"dim={self.scale.numel()}, clamp=+-{self.clamp}, "
                f"{'expm1' if self.use_expm1 else 'exp'}")


class DeferredExpOut(nn.Module):
    """expm1 on the 384-d decompressed output. Placement (b)'s second half."""

    def __init__(self, dim: int, clamp: float = 10.0):
        super().__init__()
        self.clamp = float(clamp)
        self.last_stats = {}

    def forward(self, a):
        with torch.no_grad():
            self.last_stats = {
                'pre_min': float(a.min()), 'pre_max': float(a.max()),
                'pre_absmean': float(a.abs().mean()),
                'clamp_frac': float((a.abs() > self.clamp).float().mean()),
            }
        return torch.expm1(a.clamp(-self.clamp, self.clamp))

    def extra_repr(self):
        return f"clamp=+-{self.clamp}, expm1"


def attach_poe_decompress(slot, placement='a', clamp=10.0, use_expm1=False):
    """Wrap a CompressionMultiHeadLUT's decompress in place. Returns the slot.

    Swapping the attribute is enough: the parent forward calls self.decompress(...) at the
    one deferred call site, so nothing about routing, compress, the gather or the sum is
    touched, and no forward is overridden.
    """
    lin = slot.decompress
    if not isinstance(lin, nn.Linear):
        raise ValueError(f"expected a Linear decompress, got {type(lin).__name__} -- "
                         f"this variant needs inner_out_dim != -1")
    # The wrapper is attached AFTER model.to(DEVICE), so its fresh parameters are still on
    # CPU and would break the forward with a device mismatch. Follow the Linear we wrap.
    dev, dt = lin.weight.device, lin.weight.dtype
    if placement == 'a':
        gate = DeferredExpGate(lin.in_features, clamp=clamp, use_expm1=use_expm1)
        slot.decompress = nn.Sequential(gate.to(device=dev, dtype=dt), lin)
    elif placement == 'b':
        gate = DeferredExpOut(lin.out_features, clamp=clamp)
        slot.decompress = nn.Sequential(lin, gate.to(device=dev, dtype=dt))
    else:
        raise ValueError(f"poe_placement must be 'a' or 'b', got {placement!r}")
    slot.poe_placement = placement
    return slot


def poe_modules(model):
    return [m for m in model.modules()
            if isinstance(m, (DeferredExpGate, DeferredExpOut))]
