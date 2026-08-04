"""RolloutLIFGroupsMHL — a genuine 2-layer time-stepped spiking net (no bucket/table addressing).

Replaces the bucket-table front-end entirely with a REAL K-step LIF rollout + a linear latency readout:
inputs are injected as delayed temporal pulses, neurons integrate-and-fire over K discrete steps with
intra-group lateral inhibition (WTA + "second chance"), each neuron's differentiable first-spike time
becomes a latency feature, and a trainable readout projects those features to the action.

- P = M groups x N neurons/group neurons. Inhibition acts ONLY within a group.
- W = w_max*sigmoid(w_raw) (w_max=2, hot init w_raw~N(-2.2,0.5)) — bounded EXCITATORY synapses (P,Ninp).
- D = softplus(d_raw) >= 0 — per-synapse delays (P,Ninp): WHEN input i's current reaches neuron n.
- Input current at step k: I_n[k] = sum_i W_{n,i}*x_i*kernel(k - D_{n,i}), kernel = narrow Gaussian pulse
  (differentiable => gradient flows to the delays). K=32 steps (== t_window).
- LIF rollout: V_n[k] = beta*V_n[k-1]*(1-s_n[k-1]) + I_n[k] - inh_n[k]; beta=exp(-1/tau),
  tau=softplus(tau_raw)+1.0 (floor 1.0) per-neuron; s = Heaviside(V-theta_mem) (theta=1.0) HARD forward,
  fast-sigmoid SURROGATE backward; reset via the (1-s_prev) term.
- Intra-group inhibition (causal): inh_n[k] = w_inh * sum_{m in group, m!=n} trace_m[k], with
  trace[k]=gamma*trace[k-1]+s[k-1]; w_inh=softplus(w_inh_raw) per group. WTA within a group + second chance
  (a suppressed neuron can still fire later once the neighbours' inhibitory trace decays).
- Readout: differentiable first-spike time t_n (survivor-cumprod TTFS surrogate); latency feature
  phi(t)=1-t/K (earlier => larger, never-fired => 0); y = phi @ R + b (trainable readout R (P,out), bias b).
- Straight-through: forward uses HARD first-spike times; backward uses the surrogate (forward == hard).

NOTE: a few details left unspecified by the (truncated) task — Gaussian kernel width, surrogate steepness,
inhibitory-trace decay gamma, and the delay init — are set to sensible fixed defaults (documented below).
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["RolloutLIFGroupsMHL"]


class RolloutLIFGroupsMHL(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, groups: int = 8, neurons_per_group: int = 14,
                 steps: int = 32, *, w_max: float = 2.0, kernel_sigma: float = 2.0, surr_alpha: float = 4.0,
                 trace_gamma: float = 0.9, device=None):
        super().__init__()
        self.in_dim = int(in_dim); self.out_dim = int(out_dim)
        self.M = int(groups); self.N = int(neurons_per_group); self.P = self.M * self.N
        self.K = int(steps)
        self.w_max = float(w_max); self.sigma = float(kernel_sigma)
        self.surr_alpha = float(surr_alpha); self.gamma = float(trace_gamma)

        P, I, dev = self.P, self.in_dim, (device or torch.device("cpu"))
        self.w_raw = nn.Parameter(-2.2 + 0.5 * torch.randn(P, I, device=dev))   # bounded excitatory hot init
        # delays init: spread ~uniform in (0, K) so neurons respond across the whole window
        d0 = torch.rand(P, I, device=dev) * self.K
        self.d_raw = nn.Parameter(torch.log(torch.expm1(d0.clamp(min=1e-2))))
        self.tau_raw = nn.Parameter(torch.ones(P, device=dev))                  # tau = softplus + 1.0 (floor)
        self.w_inh_raw = nn.Parameter(torch.zeros(self.M, device=dev))          # per-group inhibition strength
        self.R = nn.Parameter(0.1 * torch.randn(P, self.out_dim, device=dev))   # latency-feature readout
        self.b = nn.Parameter(torch.zeros(self.out_dim, device=dev))
        self.register_buffer("theta_mem", torch.tensor(1.0, device=dev))

    @property
    def W(self):
        return self.w_max * torch.sigmoid(self.w_raw)          # (P,Ninp) excitatory in [0, w_max]

    @property
    def delays(self):
        return F.softplus(self.d_raw)                          # (P,Ninp) >= 0

    @property
    def tau(self):
        return F.softplus(self.tau_raw) + 1.0                  # (P,)

    @property
    def w_inh(self):
        return F.softplus(self.w_inh_raw)                      # (M,)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def _rollout(self, x):
        """K-step LIF rollout. Returns hard spikes SH (B,K,P) and surrogate crossing probs G (B,K,P)."""
        B, P, M, N, K = x.shape[0], self.P, self.M, self.N, self.K
        W, D, tau = self.W, self.delays, self.tau
        beta = torch.exp(-1.0 / tau)                          # (P,)
        w_inh = self.w_inh                                    # (M,)
        # input current I[b,k,n] = sum_i W_{n,i} x_{b,i} kernel(k - D_{n,i}); kernel = Gaussian pulse at D
        ks = torch.arange(K, device=x.device).view(K, 1, 1)
        kern = torch.exp(-0.5 * ((ks - D.unsqueeze(0)) / self.sigma) ** 2)     # (K,P,Ninp)
        WX = W.unsqueeze(0) * x.unsqueeze(1)                  # (B,P,Ninp)
        I = torch.einsum('bni,kni->bkn', WX, kern)           # (B,K,P)
        V = x.new_zeros(B, P); s_prev = x.new_zeros(B, P); trace = x.new_zeros(B, P)
        G, SH = [], []
        for k in range(K):
            tr = trace.view(B, M, N)
            neigh = tr.sum(dim=2, keepdim=True) - tr          # neighbours' trace (exclude self), (B,M,N)
            inh = (w_inh.view(1, M, 1) * neigh).reshape(B, P)
            V = beta.unsqueeze(0) * V * (1.0 - s_prev) + I[:, k, :] - inh
            g = torch.sigmoid(self.surr_alpha * (V - self.theta_mem))          # surrogate crossing prob
            s_hard = (V >= self.theta_mem).float()
            s = s_hard + (g - g.detach())                     # ST spike: forward hard, surrogate backward
            G.append(g); SH.append(s_hard)
            trace = self.gamma * trace + s
            s_prev = s
        return torch.stack(SH, dim=1), torch.stack(G, dim=1)  # (B,K,P) each

    def _phi(self, x):
        """Latency features (phi_hard, phi_soft) from first-spike times; phi = 1 - t/K."""
        B, P, K = x.shape[0], self.P, self.K
        SH, G = self._rollout(x)
        fired = SH.max(dim=1).values > 0
        t_hard = torch.where(fired, SH.argmax(dim=1).float(), torch.full((B, P), float(K), device=x.device))
        surv = torch.cumprod(1.0 - G, dim=1)
        surv_prev = torch.cat([torch.ones_like(surv[:, :1]), surv[:, :-1]], dim=1)
        p = G * surv_prev
        ks = torch.arange(K, device=x.device).view(1, K, 1).float()
        t_soft = (p * ks).sum(1) + surv[:, -1] * K            # (B,P)
        return 1.0 - t_hard / K, 1.0 - t_soft / K

    def forward(self, x, mode="st"):
        phi_hard, phi_soft = self._phi(x)
        if mode == "hard":
            phi = phi_hard
        elif mode == "soft":
            phi = phi_soft
        elif mode == "st":
            phi = phi_hard + phi_soft - phi_soft.detach()     # forward == hard; grad via surrogate first-spike
        else:
            raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
        return phi @ self.R + self.b
