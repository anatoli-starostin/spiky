"""ProductBucketLIFMHL — mixed-radix product generalization of BucketLIFDetectorsMHL.

BucketLIFDetectorsMHL uses ONE detector neuron per table addressing M buckets = M cells. This generalizes to
N_det detector neurons per head, each an independent M-way bucket detector (own input weights, delay, tau,
and bucket boundaries). The N_det per-detector bucket digits form a MIXED-RADIX index into a table of
M**N_det cells: hyperplane LUTs were the M=2 case, the plain bucket model is the N_det=1 case, this is the
M-way AND N_det-deep case.

- HARD forward: per detector d, hard bucket b_d = searchsorted(t_hard_d, boundaries_d); joint index
  idx = sum_d b_d * M**d; gather that row from the head's table (M**N_det, out).
- SOFT backward: each detector has a soft bucket distribution p_d (length M). The joint soft distribution over
  the M**N_det grid is the rank-1 tensor product P = p_0 (x) p_1 (x) ... (x) p_{N_det-1}. It is NOT
  materialized; the soft read P.table is computed by SEQUENTIALLY CONTRACTING the table tensor
  (shape (M, M, ..., M, out), N_det bucket axes) against each p_d along its axis (N_det einsum contractions).
  Decoupled straight-through: y_hard (full table grad -> selected cell), y_soft = contraction with
  table.detach() (address grad -> detector distributions, no table grad), forward == hard.
- Heads: each has its own N_det detectors + own M**N_det table; heads SUM into out=6.

Reuses BucketLIFDetectorsMHL's per-detector machinery: bounded excitatory w = w_max*sigmoid(w_raw) (w_max=2,
hot init), per-detector delay, tau = softplus(tau_raw)+1.0 (floor 1.0), the O(N) cumsum first-spike membrane,
t_window=32, trainable strictly-increasing boundaries beta = beta_base + cumsum(softplus(beta_raw)), and the
partition-of-unity soft bucket membership. (T_cross / T_bkt are fixed at 1.0 here, not trainable params.)

M**N_det is capped at MAX_CELLS=4096 per head (the table is exponential); configs beyond that are refused.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["ProductBucketLIFMHL"]

MAX_CELLS = 4096


class ProductBucketLIFMHL(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, n_det: int = 2, buckets: int = 16, *,
                 w_max: float = 2.0, t_window: float = 32.0, latency_c: float = 16.0,
                 latency_alpha: float = 3.0, device=None):
        super().__init__()
        cells = buckets ** n_det
        if cells > MAX_CELLS:
            raise ValueError(f"M**N_det = {buckets}**{n_det} = {cells} exceeds MAX_CELLS={MAX_CELLS}; "
                             f"reduce buckets or n_det.")
        self.in_dim = int(in_dim); self.out_dim = int(out_dim)
        self.n_heads = int(n_heads); self.n_det = int(n_det); self.buckets = int(buckets)
        self.cells = int(cells)
        self.w_max = float(w_max); self.t_window = float(t_window)
        self.latency_c = float(latency_c); self.latency_alpha = float(latency_alpha)

        H, Nd, M, I, O, dev = self.n_heads, self.n_det, self.buckets, self.in_dim, self.out_dim, (device or torch.device("cpu"))
        self.delay = nn.Parameter(torch.zeros(H, Nd, I, device=dev))
        self.w_raw = nn.Parameter(-2.2 + 0.5 * torch.randn(H, Nd, I, device=dev))     # bounded excitatory hot
        self.tau_raw = nn.Parameter(torch.ones(H, Nd, device=dev))
        step = self.t_window / M
        inv_softplus_step = math.log(math.expm1(step)) if step > 0 else 0.0
        self.beta_base = nn.Parameter(torch.zeros(H, Nd, 1, device=dev))
        self.beta_raw = nn.Parameter(torch.full((H, Nd, M - 1), float(inv_softplus_step), device=dev))
        self.table = nn.Parameter(0.1 * torch.randn(H, self.cells, O, device=dev))    # (H, M**N_det, out)
        self.register_buffer("theta_mem", torch.tensor(1.0, device=dev))
        self.register_buffer("T_cross", torch.tensor(1.0, device=dev))
        self.register_buffer("T_bkt", torch.tensor(1.0, device=dev))
        # mixed-radix place values (row-major: detector 0 is the most-significant digit) so the hard gather
        # addresses the SAME cell the soft contraction weights (which peels grid axis 0 = detector 0 first).
        self.register_buffer("radix", (M ** (Nd - 1 - torch.arange(Nd))).long())

    @property
    def w(self):
        return self.w_max * torch.sigmoid(self.w_raw)          # (H,Nd,Ninp)

    @property
    def tau(self):
        return F.softplus(self.tau_raw) + 1.0                  # (H,Nd)

    @property
    def boundaries(self):
        return self.beta_base + torch.cumsum(F.softplus(self.beta_raw), dim=-1)   # (H,Nd,M-1)

    def latency(self, x):
        return torch.clamp(self.latency_c - self.latency_alpha * x, 0.0, self.t_window)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def _first_spike(self, x):
        """Per-detector first-spike (hard, soft): (B,H,Nd) each."""
        B, H, Nd = x.shape[0], self.n_heads, self.n_det
        lat = self.latency(x)
        a = lat.view(B, 1, 1, -1) + self.delay.unsqueeze(0)               # (B,H,Nd,Ninp)
        a_srt, idx = torch.sort(a, dim=-1)
        w_srt = self.w.unsqueeze(0).expand(B, -1, -1, -1).gather(-1, idx)
        tv = self.tau.view(1, H, Nd, 1)
        V = torch.exp(-a_srt / tv) * torch.cumsum(w_srt * torch.exp(a_srt / tv), dim=-1)
        crossed = V >= self.theta_mem
        kstar = crossed.float().argmax(-1)
        t_hard = a_srt.gather(-1, kstar.unsqueeze(-1)).squeeze(-1)
        t_hard = torch.where(crossed.any(-1), t_hard, torch.full_like(t_hard, self.t_window))
        c = torch.sigmoid((V - self.theta_mem) / self.T_cross)
        surv = torch.cumprod(1.0 - c, dim=-1)
        surv_prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], dim=-1)
        p = c * surv_prev
        t_soft = (p * a_srt).sum(-1) + surv[..., -1] * self.t_window
        return t_hard, t_soft

    def _bucket(self, t_hard, t_soft):
        """Per-detector hard bucket index (B,H,Nd) and soft bucket distribution (B,H,Nd,M)."""
        H, Nd, M = self.n_heads, self.n_det, self.buckets
        b = self.boundaries.view(1, H, Nd, M - 1)
        S = torch.sigmoid((t_soft.unsqueeze(-1) - b) / self.T_bkt)        # (B,H,Nd,M-1)
        p = torch.cat([1.0 - S[..., :1], S[..., :-1] - S[..., 1:], S[..., -1:]], dim=-1)   # (B,H,Nd,M)
        b_hard = (t_hard.unsqueeze(-1) >= self.boundaries.view(1, H, Nd, M - 1)).sum(-1)   # (B,H,Nd)
        return b_hard, p

    def _hard_read(self, b_hard):
        B, H = b_hard.shape[0], self.n_heads
        idx = (b_hard * self.radix.view(1, 1, -1)).sum(-1)               # (B,H) joint mixed-radix index
        hh = torch.arange(H, device=b_hard.device).view(1, H).expand(B, H)
        return self.table[hh, idx]                                       # (B,H,out) full table grad -> selected cell

    def _soft_read(self, p, detach):
        """Contract the (M,...,M,out) table tensor against each p_d along its axis (no dense outer product)."""
        H, Nd, M = self.n_heads, self.n_det, self.buckets
        tab = (self.table.detach() if detach else self.table).reshape(H, *([M] * Nd), self.out_dim)
        cur = torch.einsum('hm...,bhm->bh...', tab, p[:, :, 0, :])       # (B,H, [M]*(Nd-1), out)
        for d in range(1, Nd):
            cur = torch.einsum('bhm...,bhm->bh...', cur, p[:, :, d, :])  # peel one bucket axis per detector
        return cur                                                      # (B,H,out)

    def forward(self, x, mode="st"):
        t_hard, t_soft = self._first_spike(x)
        b_hard, p = self._bucket(t_hard, t_soft)
        y_hard = self._hard_read(b_hard)                                 # (B,H,out)
        if mode == "hard":
            return y_hard.sum(1)
        if mode == "soft":
            return self._soft_read(p, detach=False).sum(1)
        if mode == "st":
            y_addr = self._soft_read(p, detach=True)                     # address grad, no table grad
            return (y_hard + y_addr - y_addr.detach()).sum(1)            # forward == hard
        raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
