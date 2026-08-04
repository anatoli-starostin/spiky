"""CompetitiveBucketLIFMHL — a lateral-inhibition (winner-take-all) population variant of the bucket LIF MHL.

Sibling of :class:`spiky.lutorch.bucket_lif_detectors_mhl.BucketLIFDetectorsMHL`. Instead of one LIF neuron
per table, each **head** is a *population* of ``N`` LIF neurons that share ONE bucket-time axis, with strict
**1-winner-take-all down each bucket column** (over the neuron axis): for every (head, bucket) the earliest-
spiking neuron assigned to that bucket wins and emits its row of the table. Heads are independent and their
contributions sum to the action.

Reuses the bucket-module machinery verbatim: bounded excitatory weights ``w = w_max*sigmoid(w_raw)``
(``w_max=2``, hot init ``w_raw ~ N(-2.2, 0.5)``), ``tau = softplus(tau_raw)+1.0`` (floor 1.0), the O(N)
cumsum membrane ``V = exp(-a/tau)*cumsum(w*exp(a/tau))`` over ascending-sorted arrivals, the soft first-spike
surrogate, latency input coding, trainable strictly-increasing per-head bucket boundaries, and the
FastMHL-style **decoupled straight-through** decode (weight grad -> winning cell via ``y_hard``; address grad
-> soft path via ``y_addr = C_soft @ table.detach()``; forward == hard).

Per head (stacked over ``n_heads`` on the leading axis, fully vectorized — no python loop over heads):
  delay (N, I), w_raw (N, I) [per-neuron synapses]; tau_raw (N,), log_T_cross (N,) [per-neuron];
  beta_base (1,), beta_raw (M-1,) [SHARED bucket boundaries for the head]; log_T_bkt (1,) [bucket softness];
  log_T_wta (1,) [lateral-inhibition temperature, trainable]; table (N, M, out_dim).

Competition (per head, per sample, down each bucket column m over neurons n):
  HARD: among neurons whose hard bucket == m, the winner is argmin_n t_hard (earliest spike; ties -> lowest
  n). One-hot ``C_hard`` with <=1 nonzero per (b, head, m) column; an empty column contributes nothing.
  SOFT: ``C_soft[..,n,m] = normalize_n( g[..,n,m] * exp(-t_soft[..,n]/T_wta) )`` — a soft winner distribution
  down the column, gated by soft bucket membership ``g`` and sharpened by earliness; -> hard argmin as
  T_wta -> 0, and ~0 for empty columns (``g -> 0`` there).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["CompetitiveBucketLIFMHL"]


class CompetitiveBucketLIFMHL(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, neurons_per_head: int = 8,
                 buckets: int = 16, *, w_max: float = 2.0, t_window: float = 32.0, latency_c: float = 16.0,
                 latency_alpha: float = 3.0, device=None):
        super().__init__()
        if buckets < 2 or buckets > 256:
            raise ValueError(f"buckets must be in [2,256], got {buckets}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.n_heads = int(n_heads)
        self.neurons_per_head = int(neurons_per_head)
        self.buckets = int(buckets)
        self.w_max = float(w_max)
        self.t_window = float(t_window)
        self.latency_c = float(latency_c)
        self.latency_alpha = float(latency_alpha)

        H, N, M, I, O = self.n_heads, self.neurons_per_head, self.buckets, self.in_dim, self.out_dim
        dev = device or torch.device("cpu")

        # --- trainable params (leading head axis) ---
        self.delay = nn.Parameter(torch.zeros(H, N, I, device=dev))              # per-neuron per-input
        self.w_raw = nn.Parameter(-2.2 + 0.5 * torch.randn(H, N, I, device=dev))  # bounded excitatory hot init
        self.tau_raw = nn.Parameter(torch.ones(H, N, device=dev))                # per-neuron (tau = softplus + 1.0)
        self.log_T_cross = nn.Parameter(torch.zeros(H, N, device=dev))           # per-neuron first-spike temp
        # SHARED bucket boundaries per head, strictly increasing via beta_base + cumsum(softplus(beta_raw))
        step = self.t_window / self.buckets
        inv_softplus_step = math.log(math.expm1(step)) if step > 0 else 0.0
        self.beta_base = nn.Parameter(torch.zeros(H, 1, device=dev))
        self.beta_raw = nn.Parameter(torch.full((H, M - 1), float(inv_softplus_step), device=dev))
        self.log_T_bkt = nn.Parameter(torch.zeros(H, 1, device=dev))             # per-head bucket softness
        self.log_T_wta = nn.Parameter(torch.zeros(H, 1, device=dev))             # per-head WTA temperature
        self.table = nn.Parameter(0.1 * torch.randn(H, N, M, O, device=dev))     # winner of bucket m emits table[n,m]

        self.register_buffer("theta_mem", torch.tensor(1.0, device=dev))

    # ---- constrained params ----
    @property
    def w(self):
        return self.w_max * torch.sigmoid(self.w_raw)          # (H,N,I) bounded excitatory in [0, w_max]

    @property
    def tau(self):
        return F.softplus(self.tau_raw) + 1.0                  # (H,N) floored at 1.0 (overflow-safe cumsum)

    @property
    def T_cross(self):
        return torch.exp(self.log_T_cross)                     # (H,N)

    @property
    def T_bkt(self):
        return torch.exp(self.log_T_bkt)                       # (H,1)

    @property
    def T_wta(self):
        return torch.exp(self.log_T_wta)                       # (H,1)

    @property
    def boundaries(self):
        return self.beta_base + torch.cumsum(F.softplus(self.beta_raw), dim=-1)   # (H, M-1) strictly increasing

    def latency(self, x):
        return torch.clamp(self.latency_c - self.latency_alpha * x, 0.0, self.t_window)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    # ---- LIF first-spike front-end: (B,H,N) hard/soft first-spike times ----
    def _first_spike(self, x):
        B, H, N = x.shape[0], self.n_heads, self.neurons_per_head
        lat = self.latency(x)                                          # (B,I)
        a = lat.view(B, 1, 1, -1) + self.delay.unsqueeze(0)            # (B,H,N,I)
        a_srt, idx = torch.sort(a, dim=-1)                            # ascending over inputs
        w_srt = self.w.unsqueeze(0).expand(B, -1, -1, -1).gather(-1, idx)   # (B,H,N,I)
        tau = self.tau.view(1, H, N, 1)
        # O(N) cumsum membrane: V_k = exp(-a_k/tau)*cumsum_{j<=k}(w_j*exp(a_j/tau)) (a_srt sorted -> causal;
        # overflow-safe because tau>=1.0 and a in [0,t_window] => exp(a/tau)<=exp(32)~8e13, inside float32).
        V = torch.exp(-a_srt / tau) * torch.cumsum(w_srt * torch.exp(a_srt / tau), dim=-1)   # (B,H,N,I)
        crossed = V >= self.theta_mem
        kstar = crossed.float().argmax(-1)
        t_hard = a_srt.gather(-1, kstar.unsqueeze(-1)).squeeze(-1)     # (B,H,N)
        t_hard = torch.where(crossed.any(-1), t_hard, torch.full_like(t_hard, self.t_window))
        T_cross = self.T_cross.view(1, H, N, 1)
        c = torch.sigmoid((V - self.theta_mem) / T_cross)
        surv = torch.cumprod(1.0 - c, dim=-1)
        surv_prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], dim=-1)
        p = c * surv_prev
        t_soft = (p * a_srt).sum(-1) + surv[..., -1] * self.t_window   # (B,H,N)
        return t_hard, t_soft

    # ---- bucket membership over the head's SHARED boundaries ----
    def _buckets(self, t_hard, t_soft):
        H, M = self.n_heads, self.buckets
        b = self.boundaries.view(1, H, 1, M - 1)                       # (1,H,1,M-1)
        S = torch.sigmoid((t_soft.unsqueeze(-1) - b) / self.T_bkt.view(1, H, 1, 1))   # (B,H,N,M-1)
        g = torch.cat([1.0 - S[..., :1], S[..., :-1] - S[..., 1:], S[..., -1:]], dim=-1)  # (B,H,N,M) partition of unity
        m_hard = (t_hard.unsqueeze(-1) >= self.boundaries.view(1, H, 1, M - 1)).sum(-1)   # (B,H,N)
        E = F.one_hot(m_hard.long(), M).to(g.dtype)                    # (B,H,N,M) hard bucket assignment
        return E, g

    # ---- lateral inhibition: strict 1-WTA down each bucket column (over neurons) ----
    def _compete(self, E, g, t_hard, t_soft):
        H, N = self.n_heads, self.neurons_per_head
        # HARD winner per (b,head,m): earliest-spiking neuron assigned to bucket m (ties -> lowest n).
        big = torch.finfo(t_hard.dtype).max
        t_col = torch.where(E > 0.5, t_hard.unsqueeze(-1), torch.full_like(E, big))   # (B,H,N,M)
        winner = t_col.argmin(dim=2)                                  # (B,H,M) neuron index (lowest n on ties)
        col_has = (E.sum(dim=2) > 0).to(E.dtype)                      # (B,H,M) is the column occupied?
        C_hard = F.one_hot(winner, N).to(E.dtype).permute(0, 1, 3, 2) * col_has.unsqueeze(2)  # (B,H,N,M)
        # SOFT winner distribution down the column: gated by membership g, sharpened by earliness.
        z = -t_soft / self.T_wta.view(1, H, 1)                        # (B,H,N)
        e = torch.exp(z - z.max(dim=2, keepdim=True).values).unsqueeze(-1)   # (B,H,N,1) stable
        s = g * e                                                    # (B,H,N,M)
        C_soft = s / (s.sum(dim=2, keepdim=True) + 1e-9)             # (B,H,N,M) ~soft one-hot down each column
        return C_hard, C_soft

    def _decode(self, C, table=None):
        W = self.table if table is None else table
        return torch.einsum('bhnm,hnmo->bo', C, W)                    # sum over heads/neurons/buckets -> (B,out_dim)

    def forward(self, x, mode="st"):
        t_hard, t_soft = self._first_spike(x)
        E, g = self._buckets(t_hard, t_soft)
        C_hard, C_soft = self._compete(E, g, t_hard, t_soft)
        if mode == "hard":
            return self._decode(C_hard)
        if mode == "soft":
            return self._decode(C_soft)
        if mode == "st":
            y_hard = self._decode(C_hard)                             # weight grad -> winning cell only
            y_addr = self._decode(C_soft, self.table.detach())        # address grad -> soft competition, no weight grad
            return y_hard + y_addr - y_addr.detach()                  # forward == hard
        raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
