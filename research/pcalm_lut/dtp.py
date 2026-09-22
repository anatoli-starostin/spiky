"""Pure difference target propagation over paired forward/backward stacks.

WHAT MAKES THIS DIFFERENT FROM ARM A. Arm A also had a backward LUT, but its relaxation still took
grad_h of an energy by autograd, so the credit signal went through the layer Jacobians J_f and J_g.
Pure DTP must contain no cross-layer Jacobian anywhere: the top-down signal is two reads of g and a
subtraction, nothing else. Here the entire target computation runs under no_grad and calls only g's
forward, and `assert_no_jacobian` checks that no target carries a grad_fn. Autograd appears only inside
a layer's own local loss, with respect to that layer's own parameters -- which is a local quantity and
is the point of the method.

    1. forward, recording h_1 .. h_{L-1} and yhat
    2. t_top = y                                     (the output is pinned to the label)
    3. t_{i-1} = h_{i-1} + g_i(t_i) - g_i(h_i)       two g reads, a subtraction
    4. f_i minimises ||f_i(h_{i-1}) - t_i||^2 over f_i's OWN parameters (h_{i-1}, t_i detached)
    5. g_i minimises ||g_i(f_i(h+eps)) - (h+eps)||^2 over g_i's OWN parameters (f detached)

The models expose one interface -- h1, layer, readout, back_layer, back_readout -- so the identical DTP
code drives the LUT stack and the plain residual MLP, which is what makes the architectural comparison a
comparison rather than two implementations.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from paired import PairedLUTStack  # noqa: E402


class PairedMLP(nn.Module):
    """The gate-1 residual MLP with a mirrored backward stack, exposing the PairedLUTStack interface.

    f:  h_1 = a_1 W_1 x,  h_i = h_{i-1} + a_i W_i sigma(h_{i-1}),  yhat = a_L W_L sigma(h_{L-1})
    g:  mirrored at every level, independent parameters, including g_L: C -> N.
    """

    def __init__(self, in_dim=784, n_classes=10, width=64, depth=4, device='cuda', seed=0, act='tanh'):
        super().__init__()
        assert depth >= 3
        gen = torch.Generator(device='cpu').manual_seed(seed)
        self.in_dim, self.n_classes, self.width, self.depth = in_dim, n_classes, width, depth
        self.n_hidden = depth - 1
        self.act = {'id': lambda z: z, 'tanh': torch.tanh, 'relu': F.relu}[act]
        self.W1 = nn.Parameter(torch.randn(width, in_dim, generator=gen))
        self.Wi = nn.ParameterList([nn.Parameter(torch.randn(width, width, generator=gen))
                                    for _ in range(depth - 2)])
        self.WL = nn.Parameter(torch.randn(n_classes, width, generator=gen))
        self.Gi = nn.ParameterList([nn.Parameter(torch.randn(width, width, generator=gen))
                                    for _ in range(depth - 2)])
        self.GL = nn.Parameter(torch.randn(width, n_classes, generator=gen))
        self.a1 = 1.0 / math.sqrt(in_dim)
        self.ai = 1.0 / math.sqrt(depth * width)
        self.aL = 1.0 / width
        self.to(device)

    def h1(self, x):
        return self.a1 * F.linear(x, self.W1)

    def layer(self, i, h):
        return h + self.ai * F.linear(self.act(h), self.Wi[i])

    def readout(self, h):
        return self.aL * F.linear(self.act(h), self.WL)

    def back_layer(self, i, h):
        return h + self.ai * F.linear(self.act(h), self.Gi[i])

    def back_readout(self, y):
        return self.ai * F.linear(y, self.GL)

    def forward(self, x):
        h = self.h1(x)
        for i in range(self.n_hidden - 1):
            h = self.layer(i, h)
        return self.readout(h)

    # parameter groups, so a layer's local loss can only reach its own weights
    def f_params(self, i):
        return [self.W1] if i == 'in' else ([self.WL] if i == 'out' else [self.Wi[i]])

    def g_params(self, i):
        return [self.GL] if i == 'out' else [self.Gi[i]]


def lut_f_params(model, i):
    return [model.W1] if i == 'in' else (list(model.f_out.parameters()) if i == 'out'
                                         else list(model.f_lut[i].parameters()))


def lut_g_params(model, i):
    return list(model.g_out.parameters()) if i == 'out' else list(model.g_lut[i].parameters())


def f_params(model, i):
    return model.f_params(i) if isinstance(model, PairedMLP) else lut_f_params(model, i)


def g_params(model, i):
    return model.g_params(i) if isinstance(model, PairedMLP) else lut_g_params(model, i)


# --------------------------------------------------------------------------------- the algorithm ----
def forward_states(model, x):
    """h_1 .. h_{L-1} and yhat, detached. No graph is needed: every DTP loss rebuilds its own."""
    with torch.no_grad():
        hs = [model.h1(x)]
        for i in range(model.n_hidden - 1):
            hs.append(model.layer(i, hs[-1]))
        yhat = model.readout(hs[-1])
    return hs, yhat


def assert_no_jacobian(targets):
    """A target that carries a grad_fn would mean credit flowed through a layer Jacobian."""
    bad = [i for i, t in enumerate(targets) if t.requires_grad or t.grad_fn is not None]
    if bad:
        raise RuntimeError(f'DTP targets {bad} carry autograd history: the top-down signal went through '
                           f'a Jacobian, which is exactly what pure DTP forbids')


@torch.no_grad()
def make_targets(model, hs, yhat, y, difference=True):
    """t_{i-1} = h_{i-1} + g_i(t_i) - g_i(h_i), top-down, under no_grad and using only g's forward.

    difference=False is the falsification ablation: the plain target t_{i-1} = g_i(t_i), which drops the
    correction term that makes DTP exact when g is a perfect inverse."""
    n = model.n_hidden
    t = [None] * n
    t_top = y                                              # the output is pinned to the label
    if difference:
        t[n - 1] = hs[n - 1] + model.back_readout(t_top) - model.back_readout(yhat)
    else:
        t[n - 1] = model.back_readout(t_top)
    for i in range(n - 2, -1, -1):
        if difference:
            t[i] = hs[i] + model.back_layer(i, t[i + 1]) - model.back_layer(i, hs[i + 1])
        else:
            t[i] = model.back_layer(i, t[i + 1])
    return t


def f_losses(model, x, hs, t, y):
    """One local loss per forward layer, each from DETACHED inputs and targets."""
    out = {}
    out['in'] = (model.h1(x) - t[0]).pow(2).sum(-1).mean()
    for i in range(model.n_hidden - 1):
        out[i] = (model.layer(i, hs[i]) - t[i + 1]).pow(2).sum(-1).mean()
    out['out'] = (model.readout(hs[-1]) - y).pow(2).sum(-1).mean()
    return out


def g_losses(model, hs, yhat, sigma):
    """The DTP noisy-reconstruction objective, per backward layer, with f DETACHED.

    sigma is a FRACTION of each layer's own per-sample RMS, so the perturbation means the same thing at
    every layer and stays meaningful as the activations grow during training."""
    out = {}
    for i in range(model.n_hidden - 1):
        h = hs[i]
        scale = sigma * h.pow(2).mean(-1, keepdim=True).sqrt()
        hp = h + scale * torch.randn_like(h)
        with torch.no_grad():
            fx = model.layer(i, hp)
        out[i] = (model.back_layer(i, fx) - hp).pow(2).sum(-1).mean()
    h = hs[-1]
    scale = sigma * h.pow(2).mean(-1, keepdim=True).sqrt()
    hp = h + scale * torch.randn_like(h)
    with torch.no_grad():
        fx = model.readout(hp)
    out['out'] = (model.back_readout(fx) - hp).pow(2).sum(-1).mean()
    return out


def all_f_params(model):
    ps = list(f_params(model, 'in')) + list(f_params(model, 'out'))
    for i in range(model.n_hidden - 1):
        ps += list(f_params(model, i))
    return ps


def all_g_params(model):
    ps = list(g_params(model, 'out'))
    for i in range(model.n_hidden - 1):
        ps += list(g_params(model, i))
    return ps


def dtp_step(model, x, y, opt_f, opt_g, *, sigma=0.1, g_steps=1, difference=True, grad_clip=1.0,
             fp=None, gp=None):
    """One DTP weight update: targets by g reads only, then local f and g losses.

    fp/gp are the two parameter groups. They are clipped SEPARATELY -- clipping over all parameters
    would let the backward stack's gradient norm rescale the forward stack's step and vice versa."""
    fp = all_f_params(model) if fp is None else fp
    gp = all_g_params(model) if gp is None else gp
    hs, yhat = forward_states(model, x)
    t = make_targets(model, hs, yhat, y, difference=difference)
    assert_no_jacobian(t)

    for _ in range(g_steps):                               # inverses first, as in the DTP literature
        opt_g.zero_grad(set_to_none=True)
        gl = g_losses(model, hs, yhat, sigma)
        sum(gl.values()).backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(gp, grad_clip)
        opt_g.step()

    opt_f.zero_grad(set_to_none=True)
    fl = f_losses(model, x, hs, t, y)
    sum(fl.values()).backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(fp, grad_clip)
    opt_f.step()
    opt_f.zero_grad(set_to_none=True)
    opt_g.zero_grad(set_to_none=True)
    with torch.no_grad():
        data_loss = float(0.5 * (yhat - y).pow(2).sum(-1).mean())
    return data_loss, {'f_loss': {str(k): float(v) for k, v in fl.items()},
                       'g_loss': {str(k): float(v) for k, v in gl.items()}}


# ------------------------------------------------------------------------------ instrumentation -----
@torch.no_grad()
def inverse_quality(model, hs):
    """||g(f(h)) - h|| / ||h|| per layer. If g cannot invert f, DTP cannot route credit at all."""
    out = []
    for i in range(model.n_hidden - 1):
        h = hs[i]
        rec = model.back_layer(i, model.layer(i, h))
        out.append(float((rec - h).norm() / max(float(h.norm()), 1e-12)))
    h = hs[-1]
    rec = model.back_readout(model.readout(h))
    out.append(float((rec - h).norm() / max(float(h.norm()), 1e-12)))
    return out


@torch.no_grad()
def within_cell_fraction(model, hs):
    """How much of h's variance survives once you know only the addresses f assigns it.

    g o f can recover no more than a cell's centroid, so the variance of h WITHIN a cell is a floor on
    reconstruction error. The joint address over all 32 tables is near-unique per sample, which would
    make that floor vacuously zero, so this is measured PER TABLE and averaged: samples are grouped by
    one table's 8-bit address, and the within-group variance of h is compared with the total. It is an
    approximation in the optimistic direction for DTP -- the real read also uses continuous scores, so
    f keeps more about h than the address alone -- and is reported as such, not as an exact bound.
    Returns None for the MLP, which has no cells.
    """
    if isinstance(model, PairedMLP):
        return None
    luts = list(model.f_lut) + [model.f_out]
    zs = [hs[i] for i in range(model.n_hidden - 1)] + [hs[-1]]
    out = []
    for lut, z in zip(luts, zs):
        d = z[:, lut.anchor_a] - z[:, lut.anchor_b]
        idx = ((d > 0).to(torch.int64) * lut.powers.view(1, 1, -1)).sum(-1)        # [B, n_tables]
        tot = float(z.var(0, unbiased=False).sum())
        within = []
        for t in range(min(idx.shape[1], 8)):                # 8 tables is plenty for an average
            code = idx[:, t]
            uniq, inv = torch.unique(code, return_inverse=True)
            m = torch.zeros(len(uniq), z.shape[1], device=z.device)
            cnt = torch.zeros(len(uniq), device=z.device)
            m.index_add_(0, inv, z)
            cnt.index_add_(0, inv, torch.ones_like(inv, dtype=z.dtype))
            m = m / cnt.clamp_min(1).unsqueeze(-1)
            within.append(float((z - m[inv]).pow(2).sum(-1).mean()))
        out.append((sum(within) / len(within)) / max(tot, 1e-12))
    return out
