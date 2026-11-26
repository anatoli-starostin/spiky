"""PORT (verbatim except the import block) of research/tally_transformer/src/tally_transformer/ternary_in.py of DeepSpike-AI/deepspike-ml, branch experiment/p384-ternin-5k16s (b7b7ca7f), made self-contained:
standardize / ternary_topk are copied from tally_transformer/symlut.py of that commit.

Ternary input matrix T [d, d] in front of the FFN TLUT (experiment/p384-ternin-5k16s): the TLUT reads ``T ln2(h)`` instead of ``ln2(h)``.

Rows of ``T`` are outputs. Forward value ``q in {-1, 0, +1}``, backward = real latents (shadow weights, standard-normal init as :class:`symlut.SymmetricLUT`), standardised per row (:func:`symlut.standardize`),
with the tanh straight-through backward ``d tanh(w / 2T)`` and a learned ``log_temp`` (one scalar per matrix, init log 0.5 as SymmetricLUT). Output ``y = (x q^T) * scale``, no bias.

* ``mode="topk"``: exactly ``k_plus`` entries ``+1`` at the largest latents of each row and ``k_minus`` entries ``-1`` at the smallest (:func:`symlut.ternary_topk`); ``scale = 1 / sqrt(k_plus + k_minus)``.
* ``mode="free"``: ``q = sign(w) * [|w| > thr]`` with a learned per-matrix dead-zone threshold ``thr`` (standardised units, init ``thr_init`` = 0.5, parametrised as ``log_thr``). Surrogate for the backward (the tanh STE with a dead zone):
  ``s = (tanh((w - thr) / 2T) + tanh((w + thr) / 2T)) / 2`` (for ``thr = 0`` exactly ``tanh(w / 2T)``); ``q_st = q + (s - s.detach())``, so the latents AND the threshold get gradients.
  ``scale = 1 / sqrt(mean nonzeros per row)`` computed on the forward, detached (clamped to >= 1 nonzero).

Stats (:meth:`stats`): nonzero fraction and the fraction of entries whose ternary value changed since the previous call (the first call compares with the initial matrix).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn



def standardize(w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    """Per-probe standardisation over the last dim (zero mean, unit std), differentiable (the spiky ``normalize_weights``)."""
    return (w - w.mean(-1, keepdim=True)) / (w.std(-1, keepdim=True) + eps)


def ternary_topk(w: torch.Tensor, k_plus: int, k_minus: int, temp=0.5):
    """``w [..., d]`` -> ``(q_st, q)``: ``q`` exactly ``k_plus`` of ``+1`` at the largest and ``k_minus`` of ``-1`` at the smallest entries (0 elsewhere);
    ``q_st = q`` in the forward and ``d tanh(w / 2temp)`` in the backward (``temp``: float or a tensor broadcastable to ``w[..., :1]``)."""
    d = w.shape[-1]
    if k_plus < 0 or k_minus < 0 or k_plus + k_minus > d:
        raise ValueError(f"need 0 <= k_plus, k_minus and k_plus + k_minus <= d_in = {d}, got {k_plus}, {k_minus}")
    q = torch.zeros_like(w)
    with torch.no_grad():
        if k_plus:
            q.scatter_(-1, w.topk(k_plus, dim=-1).indices, 1.0)
        if k_minus:
            q.scatter_(-1, (-w).topk(k_minus, dim=-1).indices, -1.0)
    s = torch.tanh(w / (2.0 * temp))
    return q + (s - s.detach()), q


class TernaryIn(nn.Module):
    def __init__(self, d: int, mode: str, k_plus: int = 0, k_minus: int = 0, temp: float = 0.5, thr_init: float = 0.5, seed: int = 0):
        super().__init__()
        if mode not in ("topk", "free"):
            raise ValueError(f"mode must be 'topk' or 'free', got {mode!r}")
        if mode == "topk" and (min(k_plus, k_minus) < 0 or not 1 <= k_plus + k_minus <= d):
            raise ValueError(f"need k_plus, k_minus >= 0 and 1 <= k_plus + k_minus <= d = {d}, got {k_plus}, {k_minus}")
        self.d, self.mode, self.k_plus, self.k_minus = d, mode, k_plus, k_minus
        gen = torch.Generator().manual_seed(int(seed))
        self.latent = nn.Parameter(torch.randn(d, d, generator=gen))                  # [out, in]
        self.log_temp = nn.Parameter(torch.full((1,), math.log(temp)))
        self.log_thr = nn.Parameter(torch.full((1,), math.log(thr_init))) if mode == "free" else None
        with torch.no_grad():
            self.register_buffer("_prev", self.ternary()[1].to(torch.int8), persistent=False)

    def ternary(self):
        """``(q_st, q)`` [d, d]: q the ternary value, q_st = q in the forward and the STE gradient in the backward."""
        w = standardize(self.latent)
        T = self.log_temp.exp()
        if self.mode == "topk":
            return ternary_topk(w, self.k_plus, self.k_minus, T)
        thr = self.log_thr.exp()
        with torch.no_grad():
            q = torch.sign(w) * (w.abs() > thr)
        s = 0.5 * (torch.tanh((w - thr) / (2.0 * T)) + torch.tanh((w + thr) / (2.0 * T)))
        return q + (s - s.detach()), q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_st, q = self.ternary()
        if self.mode == "topk":
            scale = 1.0 / math.sqrt(self.k_plus + self.k_minus)
        else:
            scale = q.ne(0).sum(-1).float().mean().clamp_min(1.0).rsqrt()           # detached tensor (no host sync in the train step)
        return (x @ q_st.to(x.dtype).t()) * scale

    @torch.no_grad()
    def stats(self) -> dict:
        q = self.ternary()[1].to(torch.int8)
        out = dict(nonzero_frac=float(q.ne(0).float().mean()), flip_rate=float(q.ne(self._prev).float().mean()), plus_frac=float(q.gt(0).float().mean()), minus_frac=float(q.lt(0).float().mean()))
        if self.log_thr is not None:
            out["thr"] = float(self.log_thr.exp())
        out["temp"] = float(self.log_temp.exp())
        self._prev = q
        return out
