"""Paired forward/backward LightMHL stacks and the three arms (BP, PC-A symmetric, PC-ALM-B constraint).

Implements DERIVATION_v2.md exactly; see that file for the math. Architecture (L layers total):

    layer 1      linear 784 -> N,                 h_1 = a_1 W_1 x              (a_1 = 1/sqrt(D))
    layers 2..L-1 residual LightMHL N -> N,       h_i = h_{i-1} + a_i LUT_i(h_{i-1})
    layer L      LUT readout N -> C, bare,        yhat = a_L LUT_L(h_{L-1})    (tables [n_tables, 256, C])

with a_i = b_i = 1/sqrt(L*N). Backward maps g mirror f at every level, INDEPENDENT (own tables, anchors,
log_tau): g_i: h_i -> h_{i-1} residual for the interior, g_L: C -> N bare for the readout.

Free states are h_1 .. h_{L-1} (the input is clamped; the target is NOT clamped -- it enters through the
squared-error data term). Residuals, with f_1(x) := a_1 W_1 x:

    r^f_i = h_i - f_i(h_{i-1})      i = 1..L-1
    r^b_i = h_{i-1} - g_i(h_i)      i = 2..L-1   (and r^b_L = h_{L-1} - g_L(yhat) for the readout level)

Arm A (symmetric PC):     E     = 1/2||yhat - y||^2 + sum_i ||r^f_i||^2 + sum_i ||r^b_i||^2
Arm B (constraint ALM):   L_rho = 1/2||yhat - y||^2 + sum_i [ lam_i . r^f_i + (rho/2)||r^f_i||^2 ]
                          g is absent from L_rho; it warm-starts the inner loop and trains on
                          R = mu sum_i ||h_{i-1} - g_i(f_i(h_{i-1}))||^2 with f DETACHED, on the FEEDFORWARD h.
"""
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.expanduser('~/projects/spiky/src'))
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402

LUT_KW = dict(n_anchor_pairs=8, read_top_n=2, read_tau=0.5, read_tau_learnable=True,
              confidence_form='margin', cell_mode='constant', forward_mode='scored',
              multi_head_input=False, initial_weights_noise=1e-3)


def make_lut(in_dim, out_dim, n_tables, seed, device, head_dropout_rate=0.0):
    return LightMultiHeadLUT(input_dim=in_dim, n_tables=n_tables, output_dim=out_dim,
                             random_seed=seed, device=torch.device(device),
                             head_dropout_rate=head_dropout_rate, **LUT_KW)


class PairedLUTStack(nn.Module):
    """f (input linear + residual LUT blocks + LUT readout) and the mirrored, independent g."""

    def __init__(self, in_dim=784, n_classes=10, width=32, depth=16, n_tables=16, device='cuda', seed=0,
                 table_dropout=0.0, residual_dropout=0.0):
        super().__init__()
        assert depth >= 3
        self.in_dim, self.n_classes, self.width, self.depth = in_dim, n_classes, width, depth
        self.n_hidden = depth - 1                      # free states h_1..h_{L-1}
        self.table_dropout, self.residual_dropout = float(table_dropout), float(residual_dropout)
        self._rmask_f, self._rmask_g = None, None      # residual-stream masks, one per interior block
        g = torch.Generator(device='cpu').manual_seed(seed)
        self.W1 = nn.Parameter(torch.randn(width, in_dim, generator=g))
        self.a1 = 1.0 / math.sqrt(in_dim)
        self.ai = 1.0 / math.sqrt(depth * width)       # = b_i
        p = self.table_dropout
        # forward: interior blocks f_2..f_{L-1} (indices 0..depth-3), readout f_L
        self.f_lut = nn.ModuleList([make_lut(width, width, n_tables, seed + 1000 * (i + 1), device, p)
                                    for i in range(depth - 2)])
        self.f_out = make_lut(width, n_classes, n_tables, seed + 999, device, p)
        # backward: g_i: h_i -> h_{i-1} for the same interior levels, and g_L: C -> N
        self.g_lut = nn.ModuleList([make_lut(width, width, n_tables, seed + 2000 * (i + 1) + 7, device, p)
                                    for i in range(depth - 2)])
        self.g_out = make_lut(n_classes, width, n_tables, seed + 4242, device, p)
        self.to(device)

    # ---- dropout: ONE mask per weight update, held fixed across the whole inner loop ----------------------
    def luts(self):
        return list(self.f_lut) + [self.f_out] + list(self.g_lut) + [self.g_out]

    def resample_dropout(self, batch_size):
        """Sample every dropout mask ONCE, for the update about to be taken.

        Both variants are pinned for the whole update, which is what the PC arms need: the inner loop
        runs T forward passes over the same batch and takes a vjp through them, so a mask resampled
        per call would make the energy non-stationary and differentiate a different network than the
        one that made the prediction. Call this at the top of each optimiser step, once."""
        for lut in self.luts():
            lut.resample_head_drop_mask(batch_size)
        if self.residual_dropout > 0.0:
            keep = 1.0 - self.residual_dropout
            dev, dt = self.W1.device, self.W1.dtype
            n = self.n_hidden - 1
            self._rmask_f = [torch.empty(batch_size, self.width, device=dev, dtype=dt).bernoulli_(keep) / keep
                             for _ in range(n)]
            self._rmask_g = [torch.empty(batch_size, self.width, device=dev, dtype=dt).bernoulli_(keep) / keep
                             for _ in range(n)]

    def clear_dropout(self):
        """Drop every pinned mask (back to the library default of per-call resampling)."""
        for lut in self.luts():
            lut.clear_head_drop_mask()
        self._rmask_f, self._rmask_g = None, None

    def _rdrop(self, i, u, masks):
        """Inverted dropout on a block's LUT output, BEFORE the a_i scaling. Train + grad only, like the
        table variant, so every no_grad diagnostic and the test-set forward see the full network."""
        if (self.residual_dropout <= 0.0 or masks is None or not self.training
                or not torch.is_grad_enabled()):
            return u
        m = masks[i]
        if m.shape[0] != u.shape[0]:
            raise RuntimeError(f'pinned residual mask is for batch {m.shape[0]}, forward is {u.shape[0]}')
        return u * m

    # ---- forward maps -------------------------------------------------------------------------------------
    def h1(self, x):
        return self.a1 * F.linear(x, self.W1)

    def layer(self, i, h):                              # f_{i+2}: h_{i+1} -> h_{i+2}, i = 0..depth-3
        return h + self.ai * self._rdrop(i, self.f_lut[i](h), self._rmask_f)

    def readout(self, h):
        return self.ai * self.f_out(h)

    def forward(self, x):
        h = self.h1(x)
        for i in range(self.n_hidden - 1):
            h = self.layer(i, h)
        return self.readout(h)

    # ---- backward maps ------------------------------------------------------------------------------------
    def back_layer(self, i, h):                         # g_{i+2}: h_{i+2} -> h_{i+1}
        return h + self.ai * self._rdrop(i, self.g_lut[i](h), self._rmask_g)

    def back_readout(self, yhat):                       # g_L: yhat -> h_{L-1}
        return self.ai * self.g_out(yhat)

    # ---- states, residuals, energies ----------------------------------------------------------------------
    @torch.no_grad()
    def init_states(self, x):
        hs = [self.h1(x)]
        for i in range(self.n_hidden - 1):
            hs.append(self.layer(i, hs[-1]))
        return hs

    @torch.no_grad()
    def init_states_warm(self, x, y, blend=0.5):
        """Arm B warm start: forward pass, then a downward sweep h_{i-1} <- g_i(h_i) seeded from the TARGET,
        blended with the forward states (blend = 1 -> pure downward, 0 -> pure forward)."""
        fwd = self.init_states(x)
        down = [None] * len(fwd)
        down[-1] = self.back_readout(y)
        for i in range(self.n_hidden - 2, -1, -1):
            down[i] = self.back_layer(i, down[i + 1])
        out = [(1 - blend) * a + blend * b for a, b in zip(fwd, down)]
        out[0] = fwd[0]                                 # h_1 is determined by the clamped input
        return out

    def residuals_f(self, x, hs):
        r = [hs[0] - self.h1(x)]
        for i in range(self.n_hidden - 1):
            r.append(hs[i + 1] - self.layer(i, hs[i]))
        return r

    def residuals_b(self, hs, yhat):
        """r^b_i = h_{i-1} - g_i(h_i) for the interior levels, plus the readout level h_{L-1} - g_L(yhat)."""
        r = [hs[i] - self.back_layer(i, hs[i + 1]) for i in range(self.n_hidden - 1)]
        r.append(hs[-1] - self.back_readout(yhat))
        return r

    def energy_A(self, x, y, hs):
        """E = 1/2||yhat - y||^2 + sum_i ||r^f_i||^2 + sum_i ||r^b_i||^2, summed over the batch."""
        yhat = self.readout(hs[-1])
        rf = self.residuals_f(x, hs)
        rb = self.residuals_b(hs, yhat)
        e = 0.5 * (yhat - y).pow(2).sum()
        for ri in rf:
            e = e + ri.pow(2).sum()
        for ri in rb:
            e = e + ri.pow(2).sum()
        return e, rf, rb

    def energy_B(self, x, y, hs, lam, rho):
        """L_rho = 1/2||yhat - y||^2 + sum_i [ lam_i . r^f_i + (rho/2)||r^f_i||^2 ], summed over the batch."""
        yhat = self.readout(hs[-1])
        rf = self.residuals_f(x, hs)
        e = 0.5 * (yhat - y).pow(2).sum()
        for ri, li in zip(rf, lam):
            e = e + (li * ri).sum() + 0.5 * rho * ri.pow(2).sum()
        return e, rf, None

    def recon_R(self, x, mu=1.0):
        """R = mu sum_i ||h_{i-1} - g_i(f_i(h_{i-1}))||^2 on the FEEDFORWARD states, f detached (arm B only).

        Includes the readout level: h_{L-1} - g_L(yhat). Gradient flows only into g (and, through the loss
        value, nothing else)."""
        with torch.no_grad():
            hs = self.init_states(x)
            yhat = self.readout(hs[-1])
        r = 0.0
        for i in range(self.n_hidden - 1):
            r = r + (hs[i] - self.back_layer(i, hs[i + 1])).pow(2).sum()
        r = r + (hs[-1] - self.back_readout(yhat)).pow(2).sum()
        return mu * r

    # ---- diagnostics --------------------------------------------------------------------------------------
    @torch.no_grad()
    def _lut_addr(self, lut, z):
        d = z[:, lut.anchor_a] - z[:, lut.anchor_b]
        return ((d > 0).to(torch.int64) * lut.powers.view(1, 1, -1)).sum(-1), d.abs()

    @torch.no_grad()
    def addresses(self, hs, yhat=None):
        """Addresses of every forward LUT for the given states: [n_layers][B, n_tables]."""
        out = [self._lut_addr(self.f_lut[i], hs[i])[0] for i in range(self.n_hidden - 1)]
        out.append(self._lut_addr(self.f_out, hs[-1])[0])
        return out

    @torch.no_grad()
    def margin_stats(self, hs):
        """Per forward LUT: quantiles of the smallest margin m_{j*} and of sum_j m_j (the conditioning
        diagnostic of DERIVATION_v2 section 5)."""
        qs = torch.tensor([0.1, 0.5, 0.9])
        rows = []
        zs = [hs[i] for i in range(self.n_hidden - 1)] + [hs[-1]]
        luts = list(self.f_lut) + [self.f_out]
        for lut, z in zip(luts, zs):
            _, m = self._lut_addr(lut, z)                      # [B, n_tables, nap]
            mmin = m.min(-1).values.flatten().float().cpu()
            msum = m.sum(-1).flatten().float().cpu()
            rows.append({'m_min_q': torch.quantile(mmin, qs).tolist(),
                         'm_sum_q': torch.quantile(msum, qs).tolist()})
        return rows

    def log_taus(self):
        return {'f': [float(l.read_tau) for l in list(self.f_lut) + [self.f_out]],
                'g': [float(l.read_tau) for l in list(self.g_lut) + [self.g_out]]}
