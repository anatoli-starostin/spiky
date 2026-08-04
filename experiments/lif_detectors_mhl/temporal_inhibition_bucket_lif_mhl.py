"""TemporalInhibitionBucketLIFMHL — causal (recurrent) lateral-inhibition bucket-LIF population.

The causal counterpart of :class:`competitive_bucket_lif_mhl.CompetitiveBucketLIFMHL`. Rather
than computing each neuron's single first-spike independently and doing a post-hoc per-column WTA, the winner
of each bucket **inhibits the other still-unfired neurons before later buckets**, so a neuron that loses one
bucket can still spike (win) in a LATER bucket — a "second chance."

KEY REFRAMING: inhibition on a neuron = RAISING ITS THRESHOLD. A neuron crosses when V_n(t) >= theta_mem+I_n,
where I_n (>=0, only increases) is accumulated inhibition. Raising I_n pushes the neuron's first-crossing time
LATER, sliding it toward a later bucket. Forward is a SEQUENTIAL SCAN over the M buckets, carrying per-neuron
state I_n (inhibition, init 0) and fired_n (fired mask, init 0), per (batch, head, neuron):

  precompute membrane V at sorted arrivals (once; independent of inhibition)
  for m in 0..M-1 over window [b_{m-1}, b_m]:
    tau_n       = (soft) first-crossing time of threshold theta_mem+I_n
    in_bucket_n = soft window membership of tau_n in [b_{m-1}, b_m], gated by (1 - fired_n)  [unfired only]
    winner      = earliest tau among in-bucket unfired neurons
                  HARD: one-hot argmin_n tau over the in-bucket-unfired set (<=1; empty->none; ties->lowest n)
                  SOFT: p_n = (in_bucket_n * exp(-tau_n/T_wta)) / sum_n(same)   [membership x earliness]
    emit        winner of bucket m contributes table[head,n,m,:]  (decoupled straight-through, accumulated)
    update      fired_n += winner_n ; I_n += w_inh * winner_mass * (1 - fired_n)   [inhibit the still-unfired]

Reuses the established machinery: bounded excitatory w = w_max*sigmoid(w_raw) (w_max=2, hot init
w_raw~N(-2.2,0.5)); tau=softplus(tau_raw)+1.0 (floor 1.0); O(N) cumsum membrane; latency coding; trainable
strictly-increasing per-head boundaries; theta_mem=1.0, t_window=32.

NOTE: the spec for the state-update / ST-accumulation rules was truncated; the fired/inhibition updates and
the decoupled-ST accumulation here are reconstructed from the sibling modules' pattern.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["TemporalInhibitionBucketLIFMHL"]


class TemporalInhibitionBucketLIFMHL(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, neurons_per_head: int = 8,
                 buckets: int = 16, *, w_max: float = 2.0, t_window: float = 32.0, latency_c: float = 16.0,
                 latency_alpha: float = 3.0, device=None):
        super().__init__()
        if buckets < 2 or buckets > 256:
            raise ValueError(f"buckets must be in [2,256], got {buckets}")
        self.in_dim = int(in_dim); self.out_dim = int(out_dim)
        self.n_heads = int(n_heads); self.neurons_per_head = int(neurons_per_head); self.buckets = int(buckets)
        self.w_max = float(w_max); self.t_window = float(t_window)
        self.latency_c = float(latency_c); self.latency_alpha = float(latency_alpha)

        H, N, M, I, O = self.n_heads, self.neurons_per_head, self.buckets, self.in_dim, self.out_dim
        dev = device or torch.device("cpu")
        self.delay = nn.Parameter(torch.zeros(H, N, I, device=dev))
        self.w_raw = nn.Parameter(-2.2 + 0.5 * torch.randn(H, N, I, device=dev))
        self.tau_raw = nn.Parameter(torch.ones(H, N, device=dev))
        self.log_T_cross = nn.Parameter(torch.zeros(H, N, device=dev))
        step = self.t_window / self.buckets
        inv_softplus_step = math.log(math.expm1(step)) if step > 0 else 0.0
        self.beta_base = nn.Parameter(torch.zeros(H, 1, device=dev))
        self.beta_raw = nn.Parameter(torch.full((H, M - 1), float(inv_softplus_step), device=dev))
        self.log_T_bkt = nn.Parameter(torch.zeros(H, 1, device=dev))
        self.log_T_wta = nn.Parameter(torch.zeros(H, 1, device=dev))
        self.w_inh_raw = nn.Parameter(torch.zeros(H, 1, device=dev))          # inhibition strength (softplus>0)
        self.table = nn.Parameter(0.1 * torch.randn(H, N, M, O, device=dev))
        self.register_buffer("theta_mem", torch.tensor(1.0, device=dev))

    @property
    def w(self):
        return self.w_max * torch.sigmoid(self.w_raw)

    @property
    def tau(self):
        return F.softplus(self.tau_raw) + 1.0

    @property
    def T_cross(self):
        return torch.exp(self.log_T_cross)

    @property
    def T_bkt(self):
        return torch.exp(self.log_T_bkt)

    @property
    def T_wta(self):
        return torch.exp(self.log_T_wta)

    @property
    def w_inh(self):
        return F.softplus(self.w_inh_raw)

    @property
    def boundaries(self):
        return self.beta_base + torch.cumsum(F.softplus(self.beta_raw), dim=-1)   # (H, M-1)

    def latency(self, x):
        return torch.clamp(self.latency_c - self.latency_alpha * x, 0.0, self.t_window)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    # membrane V at sorted arrivals, computed once (independent of inhibition)
    def _membrane(self, x):
        B, H, N = x.shape[0], self.n_heads, self.neurons_per_head
        lat = self.latency(x)
        a = lat.view(B, 1, 1, -1) + self.delay.unsqueeze(0)              # (B,H,N,I)
        a_srt, idx = torch.sort(a, dim=-1)
        w_srt = self.w.unsqueeze(0).expand(B, -1, -1, -1).gather(-1, idx)
        tau = self.tau.view(1, H, N, 1)
        V = torch.exp(-a_srt / tau) * torch.cumsum(w_srt * torch.exp(a_srt / tau), dim=-1)   # (B,H,N,I)
        return V, a_srt

    def _tau_soft(self, V, a_srt, thr):
        # soft first-crossing time of threshold thr (B,H,N) over the arrivals
        H, N = self.n_heads, self.neurons_per_head
        T_cross = self.T_cross.view(1, H, N, 1)
        c = torch.sigmoid((V - thr.unsqueeze(-1)) / T_cross)             # (B,H,N,I)
        surv = torch.cumprod(1.0 - c, dim=-1)
        surv_prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], dim=-1)
        p = c * surv_prev
        return (p * a_srt).sum(-1) + surv[..., -1] * self.t_window       # (B,H,N)

    def _tau_hard(self, V, a_srt, thr):
        crossed = V >= thr.unsqueeze(-1)
        kstar = crossed.float().argmax(-1)
        t = a_srt.gather(-1, kstar.unsqueeze(-1)).squeeze(-1)
        return torch.where(crossed.any(-1), t, torch.full_like(t, self.t_window))

    def forward(self, x, mode="st", detach_state=False):
        # detach_state=True => truncated BPTT: the carried recurrent state (I, fired) is detached each bucket.
        # Full BPTT through the sequential scan (detach_state=False, default) is numerically unstable and NaNs
        # during distillation; truncated BPTT trains stably. See distill_walker2d_temporal.py.
        B, H, N, M = x.shape[0], self.n_heads, self.neurons_per_head, self.buckets
        V, a_srt = self._membrane(x)
        theta = self.theta_mem
        T_bkt = self.T_bkt.view(1, H, 1)
        T_wta = self.T_wta.view(1, H, 1)
        w_inh = self.w_inh.view(1, H, 1)
        # padded window edges: bucket m spans (bpad[m], bpad[m+1]); +-1e9 stand in for -+inf
        b = self.boundaries                                             # (H, M-1)
        BIG = 1e9
        bpad = torch.cat([torch.full((H, 1), -BIG, device=b.device), b, torch.full((H, 1), BIG, device=b.device)], dim=-1)  # (H,M+1)

        # two parallel scans: soft (differentiable, drives address grad) and hard (forward value)
        I_s = torch.zeros(B, H, N, device=x.device); fired_s = torch.zeros(B, H, N, device=x.device)
        I_h = torch.zeros(B, H, N, device=x.device); fired_h = torch.zeros(B, H, N, device=x.device)
        y_hard = x.new_zeros(B, self.out_dim); y_soft = x.new_zeros(B, self.out_dim); y_addr = x.new_zeros(B, self.out_dim)
        for m in range(M):
            lo = bpad[:, m].view(1, H, 1); hi = bpad[:, m + 1].view(1, H, 1)
            # SOFT winner of bucket m
            tau_s = self._tau_soft(V, a_srt, theta + I_s)               # (B,H,N)
            memb_s = torch.sigmoid((tau_s - lo) / T_bkt) * torch.sigmoid((hi - tau_s) / T_bkt) * (1.0 - fired_s)
            score = memb_s * torch.exp((-tau_s / T_wta) - (-tau_s / T_wta).max(dim=2, keepdim=True).values)
            p = score / (score.sum(dim=2, keepdim=True) + 1e-9)        # (B,H,N) soft winner
            y_soft = y_soft + torch.einsum('bhn,hno->bo', p, self.table[:, :, m, :])              # full soft (mode soft)
            y_addr = y_addr + torch.einsum('bhn,hno->bo', p, self.table[:, :, m, :].detach())     # address grad (st)
            win_mass = p.sum(dim=2, keepdim=True)                      # (B,H,1) ~column occupancy
            fired_s = torch.clamp(fired_s + p, 0.0, 1.0)
            I_s = I_s + w_inh * win_mass * (1.0 - fired_s)             # inhibit the still-unfired
            if detach_state:                                          # truncated BPTT (stabilizes training)
                fired_s = fired_s.detach(); I_s = I_s.detach()
            # HARD winner of bucket m
            tau_h = self._tau_hard(V, a_srt, theta + I_h)
            memb_h = ((tau_h >= lo) & (tau_h < hi) & (fired_h < 0.5)).float()
            t_masked = torch.where(memb_h > 0.5, tau_h, torch.full_like(tau_h, BIG))
            has = (memb_h.sum(dim=2) > 0).float()                     # (B,H)
            winner = t_masked.argmin(dim=2)                           # (B,H)
            C_h = F.one_hot(winner, N).float() * has.unsqueeze(-1)    # (B,H,N)
            y_hard = y_hard + torch.einsum('bhn,hno->bo', C_h, self.table[:, :, m, :])
            fired_h = torch.clamp(fired_h + C_h, 0.0, 1.0)
            I_h = I_h + self.w_inh.view(1, H, 1) * has.unsqueeze(-1) * (1.0 - fired_h)

        if mode == "hard":
            return y_hard
        if mode == "soft":
            return y_soft                                             # full soft forward+backward
        if mode == "st":
            return y_hard + y_addr - y_addr.detach()                 # forward == hard; address grad via soft (table detached)
        raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
