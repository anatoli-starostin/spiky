"""DominanceLUTFast -- an efficient (Triton row gather + dense-Borda bmm) drop-in for the reference DominanceLUT.

Same function, same buffers/parameters/state_dict, same value (fp32 round-off) and same gradients as
``dominance_lut_ref.DominanceLUT`` for ``vote='delta'``, ``normalize in {'none','count'}``, any head pattern, bias or
not. Anything else (``vote='sign'``, ``dense_D``, CPU input, fp64) falls back to the reference forward.

Why: at G=8, tph=64, B=6144 rows (device batch 12 x 512) the reference spends ~60% of its 9.5 ms (fwd+bwd) in three
torch ``index`` gathers of 100 MB each (1.9 ms apiece; a plain Triton gather does the same copy in 0.05 ms) and the
rest in scatter_add / index backward. Here:

* forward:  addresses (torch, no grad) -> Triton gather of the addressed delta rows -> Borda as a dense bmm with the
  fixed signed incidence matrix ``M[g] in {0,+-1}^{tph*nop x d_out}`` (``y = votes @ M``; exact for 0/+-1 entries);
* backward: ``g_row = gy @ M^T`` (the gradient of every selected delta; ``index_add_`` into the table gradient) and the
  rational two-cell surrogate for the input: ``dL/dU = <g_row, v_alt - v_c>`` (second Triton gather for the neighbour
  row), ``dU/d|u*| = -0.5/(1+|u*|)^2``, ``d|u*|/du* = sign(u*)``, then ``du* -> dz[a], dz[b]`` by two scatter_adds.
  Exactly the gradient autograd derives from the reference's ``y + (surr - surr.detach())``.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from dominance_lut_ref import DominanceLUT


@triton.jit
def _gather_rows(W, idx, out, N, nop, D: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    r = pid * BLOCK + tl.arange(0, BLOCK)
    m = r < N
    i = tl.load(idx + r, mask=m, other=0).to(tl.int64)
    cols = tl.arange(0, D)
    cm = m[:, None] & (cols[None, :] < nop)
    v = tl.load(W + i[:, None] * nop + cols[None, :], mask=cm, other=0.0)
    tl.store(out + r.to(tl.int64)[:, None] * nop + cols[None, :], v, mask=cm)


def gather_rows(W2: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``W2 [R, nop]`` (contiguous fp32), ``idx`` int32 flat -> ``[idx.numel(), nop]``."""
    nop = W2.shape[1]
    idx = idx.reshape(-1)
    N = idx.numel()
    out = torch.empty(N, nop, device=W2.device, dtype=W2.dtype)
    _gather_rows[(triton.cdiv(N, 256),)](W2, idx, out, N, nop, D=triton.next_power_of_2(nop), BLOCK=256)
    return out


class _DomFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, W, mod):
        B = z.shape[0]
        sp = mod.spec
        G, T, K, nap, nop = sp.n_groups, sp.tph, sp.n_cells, sp.nap, mod.nop
        with torch.no_grad():
            ia = mod.anchor_a.reshape(1, G, T * nap).expand(B, G, -1)
            ib = mod.anchor_b.reshape(1, G, T * nap).expand(B, G, -1)
            u = (z.gather(2, ia) - z.gather(2, ib)).reshape(B, G, T, nap)
            c = ((u > 0).long() * mod.powers).sum(-1)
            u_abs, j = u.abs().min(-1)
            c_alt = c ^ mod.powers[j]
            off = (torch.arange(G * T, device=z.device, dtype=torch.long) * K).reshape(1, G, T)
            idx_c = (c + off).to(torch.int32)
            idx_alt = (c_alt + off).to(torch.int32)
            W2 = W.detach().reshape(G * T * K, nop).contiguous()
            v = gather_rows(W2, idx_c).reshape(B, G, T * nop)
            y = torch.einsum("bgj,gjd->bgd", v, mod.M)
            sgn = torch.sign(u.gather(-1, j.unsqueeze(-1)).squeeze(-1))
        ctx.mod = mod
        ctx.need_z = ctx.needs_input_grad[0]
        ctx.save_for_backward(idx_c, idx_alt, u_abs, sgn, j, W)
        return y

    @staticmethod
    def backward(ctx, gy):
        mod = ctx.mod
        idx_c, idx_alt, u_abs, sgn, j, W = ctx.saved_tensors
        sp = mod.spec
        B = gy.shape[0]
        G, T, K, nap, nop, d_in = sp.n_groups, sp.tph, sp.n_cells, sp.nap, mod.nop, sp.d_in
        gy = gy.contiguous()
        g_row = torch.einsum("bgd,gjd->bgj", gy, mod.M)                    # [B, G, T*nop]
        gW = None
        if ctx.needs_input_grad[1]:
            gW = torch.zeros(G * T * K, nop, device=gy.device, dtype=gy.dtype)
            gW.index_add_(0, idx_c.reshape(-1), g_row.reshape(-1, nop))
            gW = gW.reshape(W.shape)
        gz = None
        if ctx.need_z:
            W2 = W.detach().reshape(G * T * K, nop).contiguous()
            dv = (gather_rows(W2, idx_alt) - gather_rows(W2, idx_c)).reshape(B, G, T, nop)
            dU = (g_row.reshape(B, G, T, nop) * dv).sum(-1)                # dL/dU
            du = dU * (-0.5 / (1.0 + u_abs) ** 2) * sgn                    # dL/du*  [B, G, T]
            a = mod.anchor_a.reshape(1, G, T, nap).expand(B, G, T, nap).gather(-1, j.unsqueeze(-1)).squeeze(-1)
            b = mod.anchor_b.reshape(1, G, T, nap).expand(B, G, T, nap).gather(-1, j.unsqueeze(-1)).squeeze(-1)
            gz = torch.zeros(B, G, d_in, device=gy.device, dtype=gy.dtype)
            gz.scatter_add_(2, a.reshape(B, G, T), du.reshape(B, G, T))
            gz.scatter_add_(2, b.reshape(B, G, T), -du.reshape(B, G, T))
        return gz, gW, None


class DominanceLUTFast(DominanceLUT):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        G, T, nop, d_out = self.spec.n_groups, self.spec.tph, self.nop, self.spec.d_out
        M = torch.zeros(G, T * nop, d_out)
        j = torch.arange(T * nop)
        for g in range(G):
            M[g].index_put_((j, self.out_a[g].reshape(-1)), torch.ones(T * nop), accumulate=True)
            M[g].index_put_((j, self.out_b[g].reshape(-1)), -torch.ones(T * nop), accumulate=True)
        self.register_buffer("M", M, persistent=False)   # signed incidence: Borda y = votes @ M

    def _fast_ok(self, x):
        return (self.vote == "delta" and not self.dense_D and x.is_cuda and x.dtype == torch.float32
                and self.weights.dtype == torch.float32)

    def forward(self, x):
        if not self._fast_ok(x):
            return super().forward(x)
        self._check_input(x)
        z = x[:, self.in_head, :]
        y = _DomFn.apply(z, self.weights, self)
        if self.normalize == "count":
            y = y * self.inv_count
        if self.spec.h_out != self.spec.n_groups:
            y = y.sum(dim=1, keepdim=True)
        return y if self.bias is None else y + self.bias
