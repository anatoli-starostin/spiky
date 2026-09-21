"""A residual stack of LightMultiHeadLUT blocks, in the LayerChain form the PC / PC-ALM trainers need.

The LUT machinery is OURS, untouched (src/spiky/lutorch/light_multi_head_lut.py): anchor-pair margins, a
DETACHED integer sign address, and the differentiable confidence score that is the only gradient path into the
layer's input. read_top_n = 2 ("2 alternatives", the configuration the study is about).

Block i:  h_i = h_{i-1} + a_i * LUT_i(h_{i-1})          (residual, matching the paper's residual MLP topology)
Readout:  y_hat = a_L W_L h_{L-1}
with the paper's mean-field pre-multipliers a_i = 1/sqrt(L N), a_L = 1/(gamma0 N), and the LUT's own table init.

`addresses(h)` exposes the integer cell address each layer would read for a given state, which is what the
address-search hypothesis measures (flips during the inference phase, drift across training).
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402

from pcalm import LayerChain  # noqa: E402


class LUTStack(LayerChain):
    def __init__(self, in_dim, out_dim, width, depth, *, tables_per_layer=16, nap=8, read_top_n=2,
                 read_tau=0.5, confidence_form='margin', gamma0=1.0, device='cuda', seed=0,
                 initial_weights_noise=1e-3):
        super().__init__()
        assert depth >= 2
        self.depth, self.width, self.n_hidden = depth, width, depth - 1
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(width, in_dim, generator=g))
        self.WL = nn.Parameter(torch.randn(out_dim, width, generator=g))
        self.luts = nn.ModuleList([
            LightMultiHeadLUT(input_dim=width, n_tables=tables_per_layer, output_dim=width, n_anchor_pairs=nap,
                              confidence_form=confidence_form, read_top_n=read_top_n, read_tau=read_tau,
                              read_tau_learnable=True, random_seed=seed + 1000 * (i + 1),
                              initial_weights_noise=initial_weights_noise, device=torch.device(device))
            for i in range(depth - 2)])
        self.a1 = 1.0 / math.sqrt(in_dim)
        self.ai = 1.0 / math.sqrt(depth * width)
        self.aL = 1.0 / (gamma0 * width)
        self.to(device)

    def h1(self, x):
        return self.a1 * F.linear(x, self.W1)

    def layer(self, i, h):
        return h + self.ai * self.luts[i](h)

    def readout(self, h):
        return self.aL * F.linear(h, self.WL)

    @torch.no_grad()
    def addresses(self, hs):
        """Integer cell address per layer for states `hs`: list of [B, n_tables] int64 tensors."""
        out = []
        for i, lut in enumerate(self.luts):
            h = hs[i]
            d = h[:, lut.anchor_a] - h[:, lut.anchor_b]
            out.append(((d > 0).to(torch.int64) * lut.powers.view(1, 1, -1)).sum(-1))
        return out

    @torch.no_grad()
    def margins(self, hs):
        """|d| per (sample, table, anchor) -- how far each address bit is from flipping."""
        out = []
        for i, lut in enumerate(self.luts):
            h = hs[i]
            out.append((h[:, lut.anchor_a] - h[:, lut.anchor_b]).abs())
        return out


def address_stats(a0, a1, n_cells):
    """Compare two address snapshots: fraction of (sample, table) slots whose cell changed, and occupancy."""
    changed = torch.stack([(x != y).float().mean() for x, y in zip(a0, a1)])
    occ = []
    for a in a1:
        c = torch.bincount(a.reshape(-1), minlength=n_cells).double()
        p = c / c.sum()
        nz = p[p > 0]
        occ.append(( float((c > 0).float().mean()), float(torch.exp(-(nz * nz.log()).sum())) ))
    return changed.cpu(), occ
