"""DominanceConfFast -- DominanceLUT on the ConfidenceLUT(n=1) cartridge.

Each table reads its single addressed row (``nop`` deltas, hard address ``c``) and multiplies it by the table's confidence
score of the library's ``ConfidenceLUT`` n=1,

    s_t = (sum_j m_j) * exp(gamma * sum_j log sigma(beta * m_j)),   m_j = |u_j| (the nap probe margins of table t),

``beta = exp(confidence_log_beta)``, ``gamma = exp(confidence_log_gamma)`` learned per layer (init 2, 1). The scaled rows are
aggregated by Borda exactly as in ``DominanceLUT`` (``y = (s * delta) @ M``, ``M`` the signed incidence matrix of
``dominance_fast``). There is NO two-cell rational surrogate: as in ConfidenceLUT n=1 the address is detached and the input
gradient reaches ``z`` only through ``s`` (all ``nap`` margins), plus beta/gamma get their gradients through ``s``.

Implementation: the score is computed in autograd with the library's own code (``ConfidenceLUT._score``, unbound), so its
backward into the margins, into the input anchors' difference ``z[a]-z[b]`` and into beta/gamma is the library's autograd one.
The row read * score * Borda is one ``autograd.Function`` (Triton row gather of ``dominance_fast``; backward: ``g_row = gy @ M^T``,
``dW = index_add(s * g_row)``, ``ds = <g_row, row>``). ``forward_reference`` is the same function in plain PyTorch
(index gather + scatter-add Borda), used by tests and as the non-CUDA fallback.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from spiky.lutorch_ex.cartridges.confidence import ConfidenceLUT

from dominance_fast import DominanceLUTFast, gather_rows
from dominance_lut_ref import borda_scatter


class _ConfRowFn(torch.autograd.Function):
    """``y[b,g,:] = sum_t s[b,g,t] * W[g,t,c[b,g,t]] @ M[g]`` (row read, score, Borda)."""

    @staticmethod
    def forward(ctx, s, W, idx_c, mod):
        B = s.shape[0]
        sp = mod.spec
        G, T, nop = sp.n_groups, sp.tph, mod.nop
        with torch.no_grad():
            W2 = W.detach().reshape(-1, nop).contiguous()
            v = gather_rows(W2, idx_c).reshape(B, G, T, nop)
            y = torch.einsum("bgj,gjd->bgd", (s.unsqueeze(-1) * v).reshape(B, G, T * nop), mod.M)
        ctx.mod = mod
        ctx.save_for_backward(s, idx_c, W)
        return y

    @staticmethod
    def backward(ctx, gy):
        mod = ctx.mod
        s, idx_c, W = ctx.saved_tensors
        sp = mod.spec
        B = gy.shape[0]
        G, T, K, nop = sp.n_groups, sp.tph, sp.n_cells, mod.nop
        g_row = torch.einsum("bgd,gjd->bgj", gy.contiguous(), mod.M).reshape(B, G, T, nop)
        gW = gs = None
        if ctx.needs_input_grad[1]:
            gW = torch.zeros(G * T * K, nop, device=gy.device, dtype=gy.dtype)
            gW.index_add_(0, idx_c.reshape(-1), (s.unsqueeze(-1) * g_row).reshape(-1, nop))
            gW = gW.reshape(W.shape)
        if ctx.needs_input_grad[0]:
            v = gather_rows(W.detach().reshape(-1, nop).contiguous(), idx_c).reshape(B, G, T, nop)
            gs = (g_row * v).sum(-1)
        return gs, gW, None, None


class DominanceConfFast(DominanceLUTFast):
    def __init__(self, spec, *, beta_init: float = 2.0, gamma_init: float = 1.0, **kw):
        if kw.get("vote", "delta") != "delta":
            raise ValueError("DominanceConfFast supports vote='delta' only")
        super().__init__(spec, **kw)
        if not (beta_init > 0 and gamma_init > 0):
            raise ValueError("beta_init, gamma_init must be > 0")
        # log-parametrised exactly as ConfidenceBase (one scalar each per layer)
        self.confidence_log_beta = nn.Parameter(torch.tensor(math.log(float(beta_init))))
        self.confidence_log_gamma = nn.Parameter(torch.tensor(math.log(float(gamma_init))))
        self.reset_score_stat()

    # the library's score, verbatim (it only reads confidence_log_beta/gamma from self)
    _score = ConfidenceLUT._score

    def reset_score_stat(self):
        self._s_sum, self._s_n = 0.0, 0

    def mean_score(self) -> float:
        return self._s_sum / max(self._s_n, 1)

    def _margins_and_cells(self, x):
        z = x[:, self.in_head, :]
        B = z.shape[0]
        G, T, nap = self.spec.n_groups, self.spec.tph, self.spec.nap
        ia = self.anchor_a.reshape(1, G, T * nap).expand(B, G, -1)
        ib = self.anchor_b.reshape(1, G, T * nap).expand(B, G, -1)
        u = (z.gather(2, ia) - z.gather(2, ib)).reshape(B, G, T, nap)
        c = ((u.detach() > 0).long() * self.powers).sum(-1)       # address: detached / integer
        return u, c

    def _finish(self, y, s):
        if not self.training:
            self._s_sum += float(s.detach().sum()); self._s_n += s.numel()
        if self.spec.h_out != self.spec.n_groups:
            y = y.sum(dim=1, keepdim=True)
        return y if self.bias is None else y + self.bias

    def forward_reference(self, x):
        """Plain PyTorch: index gather, score, scatter-add Borda (all autograd)."""
        self._check_input(x)
        u, c = self._margins_and_cells(x)
        s = self._score(u)                                         # [B, G, T]
        y = borda_scatter(s.unsqueeze(-1) * self._read(self.weights, c), self.out_a, self.out_b, self.spec.d_out)
        return self._finish(y, s)

    def forward(self, x):
        if not (x.is_cuda and x.dtype == torch.float32 and self.weights.dtype == torch.float32):
            return self.forward_reference(x)
        self._check_input(x)
        u, c = self._margins_and_cells(x)
        s = self._score(u)
        G, T, K = self.spec.n_groups, self.spec.tph, self.spec.n_cells
        off = (torch.arange(G * T, device=x.device, dtype=torch.long) * K).reshape(1, G, T)
        idx_c = (c + off).to(torch.int32)
        y = _ConfRowFn.apply(s, self.weights, idx_c, self)
        return self._finish(y, s)
