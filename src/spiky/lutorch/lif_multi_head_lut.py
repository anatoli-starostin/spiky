"""LIFMultiHeadLUT — unified LIF-detector multi-head LUT that generalizes BucketLIFDetectorsMHL and
ProductBucketLIFMHL into one class.

Structure (three nested levels):
- n_heads output heads: the forward ALWAYS returns (B, n_heads, n_outputs); n_heads is the kept output axis.
- tables_per_head tables per head: the `tables_per_head` tables are SUMMED within each head (same as Bucket).
- n_det detectors per table: each detector is one LIF cell emitting one of M=n_buckets bucket digits; the
  n_det digits combine MIXED-RADIX into an index over M**n_det cells, each holding an n_outputs vector.
  n_det=1 => one LIF + M buckets => exactly the Bucket notion of a table.

n_tables = n_heads * tables_per_head. Reuses BucketLIFDetectorsMHL's machinery verbatim (latency coding,
bounded-excitatory w = w_max*sigmoid(w_raw) hot init, tau = softplus(tau_raw)+1.0 floor, the O(N) cumsum
first-spike membrane, per-detector delay, trainable strictly-increasing boundaries, trainable per-TABLE soft
temperatures log_T_cross / log_T_bkt init at 1.0, decoupled straight-through, table_init) and
ProductBucketLIFMHL's mixed-radix gather (hard) + rank-1 tensor-product contraction (soft, sequential einsums,
no dense outer product) for the n_det>1 combination. M**n_det is capped at 4096 cells/table.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LIFMultiHeadLUT"]

MAX_CELLS = 4096


class LIFMultiHeadLUT(nn.Module):
    def __init__(self, input_dim: int, n_heads: int, n_outputs: int, tables_per_head: int = 1, n_det: int = 1,
                 *, n_buckets: int = 16, w_max: float = 2.0, t_window: float = 32.0, latency_c: float = 16.0,
                 latency_alpha: float = 3.0, table_init: Optional[torch.Tensor] = None, device=None):
        super().__init__()
        if n_buckets < 2 or n_buckets > 256:
            raise ValueError(f"n_buckets must be in [2,256], got {n_buckets}")
        cells = n_buckets ** n_det
        if cells > MAX_CELLS:
            raise ValueError(f"n_buckets**n_det = {n_buckets}**{n_det} = {cells} exceeds MAX_CELLS={MAX_CELLS}")
        self.input_dim = int(input_dim); self.n_heads = int(n_heads); self.n_outputs = int(n_outputs)
        self.tables_per_head = int(tables_per_head); self.n_det = int(n_det); self.n_buckets = int(n_buckets)
        self.n_tables = self.n_heads * self.tables_per_head
        self.cells = int(cells); self.n_rows = self.cells
        self.w_max = float(w_max); self.t_window = float(t_window)
        self.latency_c = float(latency_c); self.latency_alpha = float(latency_alpha)

        T, D, M, N, O = self.n_tables, self.n_det, self.n_buckets, self.input_dim, self.n_outputs
        dev = device or torch.device("cpu")
        # Per-detector params. To make BucketLIFDetectorsMHL (n_det=1) a drop-in with identical param SHAPES,
        # the n_det axis is DROPPED when n_det==1 (so w_raw is (T,N) not (T,1,N)); it is present for n_det>1.
        # The forward normalizes both to (T, n_det, ...) via _nd().
        self._sq = (D == 1)
        dsh = (T, N) if self._sq else (T, D, N)
        bsh = (T, M - 1) if self._sq else (T, D, M - 1)
        tsh = (T,) if self._sq else (T, D)
        step = self.t_window / M
        inv_softplus_step = math.log(math.expm1(step)) if step > 0 else 0.0
        self.delay = nn.Parameter(torch.zeros(*dsh, device=dev))
        self.w_raw = nn.Parameter(-2.2 + 0.5 * torch.randn(*dsh, device=dev))         # bounded excitatory hot init
        self.tau_raw = nn.Parameter(torch.ones(*tsh, device=dev))
        self.beta_base = nn.Parameter(torch.zeros(*(tsh + (1,)), device=dev))
        self.beta_raw = nn.Parameter(torch.full(bsh, float(inv_softplus_step), device=dev))
        # per-TABLE trainable soft temperatures (init exp(0)=1.0 -> step-0 == Bucket)
        self.log_T_cross = nn.Parameter(torch.zeros(T, device=dev))
        self.log_T_bkt = nn.Parameter(torch.zeros(T, device=dev))
        if table_init is not None:
            if tuple(table_init.shape) != (T, self.cells, O):
                raise ValueError(f"table_init shape {tuple(table_init.shape)} != {(T, self.cells, O)}")
            self.table = nn.Parameter(table_init.clone().to(dev))
        else:
            self.table = nn.Parameter(0.1 * torch.randn(T, self.cells, O, device=dev))
        self.register_buffer("theta_mem", torch.tensor(1.0, device=dev))
        self.register_buffer("radix", (M ** (D - 1 - torch.arange(D))).long())    # row-major mixed-radix

    @property
    def w(self):
        return self.w_max * torch.sigmoid(self.w_raw)          # (T,D,N)

    @property
    def tau(self):
        return F.softplus(self.tau_raw) + 1.0                  # (T,D)

    @property
    def T_cross(self):
        return torch.exp(self.log_T_cross)                     # (T,)

    @property
    def T_bkt(self):
        return torch.exp(self.log_T_bkt)                       # (T,)

    @property
    def boundaries(self):
        return self.beta_base + torch.cumsum(F.softplus(self.beta_raw), dim=-1)   # (T,D,M-1)

    def latency(self, x):
        return torch.clamp(self.latency_c - self.latency_alpha * x, 0.0, self.t_window)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def _nd(self, t):
        """Insert the n_det axis at dim 1 when it was squeezed out (n_det==1)."""
        return t.unsqueeze(1) if self._sq else t

    def _first_spike(self, x):
        """Per-detector first-spike (hard, soft): (B, n_tables, n_det) each."""
        B, T, D = x.shape[0], self.n_tables, self.n_det
        lat = self.latency(x)
        a = lat.view(B, 1, 1, -1) + self._nd(self.delay).unsqueeze(0)      # (B,T,D,N)
        a_srt, idx = torch.sort(a, dim=-1)
        w_srt = self._nd(self.w).unsqueeze(0).expand(B, -1, -1, -1).gather(-1, idx)
        tv = self._nd(self.tau).view(1, T, D, 1)
        V = torch.exp(-a_srt / tv) * torch.cumsum(w_srt * torch.exp(a_srt / tv), dim=-1)   # O(N) cumsum membrane
        crossed = V >= self.theta_mem
        kstar = crossed.float().argmax(-1)
        t_hard = a_srt.gather(-1, kstar.unsqueeze(-1)).squeeze(-1)
        t_hard = torch.where(crossed.any(-1), t_hard, torch.full_like(t_hard, self.t_window))
        Tc = self.T_cross.view(1, T, 1, 1)
        c = torch.sigmoid((V - self.theta_mem) / Tc)
        surv = torch.cumprod(1.0 - c, dim=-1)
        surv_prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], dim=-1)
        p = c * surv_prev
        t_soft = (p * a_srt).sum(-1) + surv[..., -1] * self.t_window
        return t_hard, t_soft

    def _bucket(self, t_hard, t_soft):
        """Per-detector hard bucket (B,T,D) and soft partition (B,T,D,M)."""
        T, D, M = self.n_tables, self.n_det, self.n_buckets
        bnd = self._nd(self.boundaries)                                  # (T,D,M-1)
        b = bnd.view(1, T, D, M - 1)
        Tb = self.T_bkt.view(1, T, 1, 1)
        S = torch.sigmoid((t_soft.unsqueeze(-1) - b) / Tb)               # (B,T,D,M-1)
        p = torch.cat([1.0 - S[..., :1], S[..., :-1] - S[..., 1:], S[..., -1:]], dim=-1)   # (B,T,D,M)
        b_hard = (t_hard.unsqueeze(-1) >= b).sum(-1)                     # (B,T,D)
        return b_hard, p

    def _hard_read(self, b_hard):
        B, T = b_hard.shape[0], self.n_tables
        idx = (b_hard * self.radix.view(1, 1, -1)).sum(-1)              # (B,T) joint mixed-radix index
        tt = torch.arange(T, device=b_hard.device).view(1, T).expand(B, T)
        return self.table[tt, idx]                                      # (B,T,O) full table grad -> selected cell

    def _soft_read(self, p, detach):
        T, D, M = self.n_tables, self.n_det, self.n_buckets
        tab = (self.table.detach() if detach else self.table).reshape(T, *([M] * D), self.n_outputs)
        cur = torch.einsum('tm...,btm->bt...', tab, p[:, :, 0, :])      # peel detector 0's axis
        for d in range(1, D):
            cur = torch.einsum('btm...,btm->bt...', cur, p[:, :, d, :])
        return cur                                                     # (B,T,O)

    def address(self, x):
        t_hard, t_soft = self._first_spike(x)
        b_hard, _ = self._bucket(t_hard, t_soft)
        return (b_hard * self.radix.view(1, 1, -1)).sum(-1)            # (B, n_tables) joint index

    def forward(self, x, eps: float = 0.3, mode: str = "st"):
        """x:(B,input_dim) -> (B, n_heads, n_outputs). `eps` accepted for API parity (unused)."""
        B = x.shape[0]
        t_hard, t_soft = self._first_spike(x)
        b_hard, p = self._bucket(t_hard, t_soft)
        y_hard = self._hard_read(b_hard)                               # (B,T,O)
        if mode == "hard":
            rows = y_hard
        elif mode == "soft":
            rows = self._soft_read(p, detach=False)
        elif mode == "st":
            y_addr = self._soft_read(p, detach=True)                   # address grad, no table grad
            rows = y_hard + y_addr - y_addr.detach()                   # forward == hard
        else:
            raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
        return rows.view(B, self.n_heads, self.tables_per_head, self.n_outputs).sum(dim=2)
