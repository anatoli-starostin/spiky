"""BucketLIFDetectorsMHL — a single-LIF-neuron-per-table, time-bucket-addressed multi-head LUT front-end.

A sibling of :class:`spiky.lutorch.pure_lif_detectors_mhl.PureLIFDetectorsMHL`. It keeps the same LUT
skeleton (``n_tables`` tables, action = sum of per-table row lookups) and the FastMHL-style **decoupled
straight-through** decode, but replaces the per-detector *bit* address with a single neuron whose first
spike lands in one of ``n_buckets`` **time buckets** — the bucket index is the table row.

Per table (batched over ``n_tables`` and batch ``B``; exactly ONE LIF neuron per table, no nap / no P):
  arrivals   a[b,t,j] = latency(x)[b,j] + delay[t,j]           over j = 0..input_dim-1
  membrane   at the k-th (ascending-sorted) arrival: V_k = Σ_{j: a_j<=a_k} w_j · exp(-(a_k - a_j)/tau)
  first spike (hard) t* = first sorted arrival where V_k >= theta_mem (=1.0), else t_window
  first spike (soft) via a smooth first-success over the sorted arrivals (differentiable), same as PureLIF
  bucket     m* = #{boundaries <= t*}  (hard, one-hot) ; soft partition-of-unity g_m over t (below)

Buckets. Per-LUT boundaries are TRAINABLE and kept strictly increasing via a monotone parameterisation
``boundaries = beta_base + cumsum(softplus(beta_raw))`` (shape ``(n_tables, n_buckets-1)``). The soft
one-hot is a partition of unity built from the boundary sigmoids ``S_k = sigmoid((t - b_k)/T_bkt)``::

    g_0        = 1 - S_0
    g_m        = S_{m-1} - S_m           (0 < m < n_buckets-1)
    g_{last}   = S_{n_buckets-2}

which telescopes to ``Σ_m g_m = 1``. The hard bucket is ``m* = (t >= boundaries).sum(-1)`` (searchsorted).

No-spike handling: ``t_window`` is the largest possible time and initial boundaries all sit below it, so a
non-crossing neuron (t*=t_window) folds into the LAST bucket automatically (``m* = n_buckets-1``).

Synapses are bounded & excitatory: the effective weight is ``w = w_max * sigmoid(w_raw)`` (``0 <= w <=
w_max``, default ``w_max = 2``), with a hot init (``w_raw ~ N(-2.2, 0.5)``) so neurons actually spike at
step 0. This beat free-signed weights in distillation and yields an inherently sparse bank (~64% of
synapses prune to zero with ~no accuracy loss).

Trainable: delay (n_tables,N), w_raw (n_tables,N), per-LUT tau_raw / log_T_cross / log_T_bkt / beta_base
(n_tables,), beta_raw (n_tables, n_buckets-1), and the LUT table (n_tables, n_buckets, n_outputs).
theta_mem is fixed (1.0). Decode is IDENTICAL in spirit to PureLIFDetectorsMHL: weight grad -> selected
row only via ``y_hard``; address grad -> full partition via ``y_addr = g @ table.detach()``; forward == hard.
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["BucketLIFDetectorsMHL"]


class BucketLIFDetectorsMHL(nn.Module):
    def __init__(self, input_dim: int, n_heads: int, n_outputs: int, tables_per_head: int = 1, *,
                 n_buckets: int = 16, t_window: float = 32.0, latency_c: float = 16.0,
                 latency_alpha: float = 3.0, table_init: Optional[torch.Tensor] = None, device=None):
        super().__init__()
        if n_buckets < 2 or n_buckets > 256:
            raise ValueError(f"n_buckets must be in [2,256], got {n_buckets}")
        self.input_dim = int(input_dim)
        self.n_heads = int(n_heads)
        self.n_outputs = int(n_outputs)
        self.tables_per_head = int(tables_per_head)
        self.n_tables = self.n_heads * self.tables_per_head
        self.n_buckets = int(n_buckets)
        self.n_rows = self.n_buckets                       # rows per table (API parity with the bit variants)
        self.t_window = float(t_window)
        self.latency_c = float(latency_c)
        self.latency_alpha = float(latency_alpha)

        N, T, Mb = self.input_dim, self.n_tables, self.n_buckets
        dev = device or torch.device("cpu")

        # --- trainable params ---
        self.delay = nn.Parameter(torch.zeros(T, N, device=dev))            # per-table per-input
        # Bounded EXCITATORY synapses: w = w_max * sigmoid(w_raw), so 0 <= w <= w_max (biologically plausible;
        # beats free-signed weights here). HOT init w_raw ~ N(-2.2, 0.5) puts the eval-set init no-spike bucket
        # mass at ~0.03 (neurons actually spike at step 0 instead of ~97% collapsing into the no-spike bucket).
        self.w_max = 2.0
        self.w_raw = nn.Parameter(-2.2 + 0.5 * torch.randn(T, N, device=dev))   # per-table per-input
        self.tau_raw = nn.Parameter(torch.ones(T, device=dev))              # per-LUT (tau = softplus + 1.0 floor)
        self.log_T_cross = nn.Parameter(torch.zeros(T, device=dev))         # per-LUT (T_cross = exp; init 1.0)
        self.log_T_bkt = nn.Parameter(torch.zeros(T, device=dev))           # per-LUT bucket softness (T_bkt = exp; init 1.0)
        # Boundaries init: n_buckets-1 evenly spaced across (0, t_window). With beta_base=0 and each
        # softplus(beta_raw) == step, cumsum gives step, 2*step, ..., (n_buckets-1)*step (all < t_window).
        step = self.t_window / self.n_buckets
        inv_softplus_step = math.log(math.expm1(step)) if step > 0 else 0.0
        self.beta_base = nn.Parameter(torch.zeros(T, device=dev))           # per-LUT boundary offset
        self.beta_raw = nn.Parameter(torch.full((T, Mb - 1), float(inv_softplus_step), device=dev))
        if table_init is not None:
            if tuple(table_init.shape) != (T, Mb, self.n_outputs):
                raise ValueError(f"table_init shape {tuple(table_init.shape)} != {(T, Mb, self.n_outputs)}")
            self.table = nn.Parameter(table_init.clone().to(dev))
        else:
            self.table = nn.Parameter(0.1 * torch.randn(T, Mb, self.n_outputs, device=dev))

        # --- buffers ---
        self.register_buffer("theta_mem", torch.tensor(1.0, device=dev))

    @property
    def w(self):
        return self.w_max * torch.sigmoid(self.w_raw)   # bounded excitatory synapses in [0, w_max]

    # ---- positive-constrained per-LUT params ----
    @property
    def tau(self):
        # floored at 1.0 (not 1e-3): the LIF membrane time constant can never drop into the
        # fast-forgetter regime where the trace decays within a single arrival step.
        return F.softplus(self.tau_raw) + 1.0

    @property
    def T_cross(self):
        return torch.exp(self.log_T_cross)

    @property
    def T_bkt(self):
        return torch.exp(self.log_T_bkt)

    @property
    def boundaries(self):
        """Strictly increasing per-LUT bucket boundaries in time-space, shape (n_tables, n_buckets-1)."""
        return self.beta_base.unsqueeze(-1) + torch.cumsum(F.softplus(self.beta_raw), dim=-1)

    def latency(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.latency_c - self.latency_alpha * x, 0.0, self.t_window)

    # ---- LIF first-spike front-end: one neuron per table -> (t_hard, t_soft), each (B, n_tables) ----
    def _first_spike(self, x: torch.Tensor):
        B = x.shape[0]; T = self.n_tables
        t = self.latency(x)                                            # (B,N)
        a = t.unsqueeze(1) + self.delay.unsqueeze(0)                   # (B,T,N)
        a_srt, idx = torch.sort(a, dim=-1)                            # ascending; perm = constant subgradient
        w_srt = self.w.unsqueeze(0).expand(B, -1, -1).gather(-1, idx)  # (B,T,N)
        # Membrane V_k = Σ_{j<=k} w_j·exp(-(a_k-a_j)/tau). a_srt is sorted ascending so the causal prefix is
        # automatic (a_k-a_j >= 0 for j<=k); factor it as V_k = exp(-a_k/tau)·cumsum_{j<=k}(w_j·exp(a_j/tau)),
        # which is O(N) instead of the O(N^2) pairwise dt matrix. Overflow-safe ONLY because tau is floored at
        # 1.0 and arrivals a in [0, t_window]: exp(a/tau) <= exp(t_window) ~ 8e13 (well inside float32), and the
        # exp(-a_k/tau) rescale restores O(1) magnitude from the dominant prefix term, preserving relative
        # precision. (Do NOT lower the tau floor — that is what keeps this factorization stable.)
        tau = self.tau.view(1, T, 1)
        V = torch.exp(-a_srt / tau) * torch.cumsum(w_srt * torch.exp(a_srt / tau), dim=-1)   # (B,T,N)
        # hard first-spike time
        crossed = V >= self.theta_mem
        kstar = crossed.float().argmax(-1)                           # (B,T) first crossing (0 if none)
        t_hard = a_srt.gather(-1, kstar.unsqueeze(-1)).squeeze(-1)    # (B,T)
        t_hard = torch.where(crossed.any(-1), t_hard, torch.full_like(t_hard, self.t_window))
        # soft first-spike time (smooth first-success over sorted arrivals)
        T_cross = self.T_cross.view(1, T, 1)
        c = torch.sigmoid((V - self.theta_mem) / T_cross)           # (B,T,N)
        surv = torch.cumprod(1.0 - c, dim=-1)
        surv_prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], dim=-1)
        p = c * surv_prev
        p_never = surv[..., -1]
        t_soft = (p * a_srt).sum(-1) + p_never * self.t_window       # (B,T)
        return t_hard, t_soft

    # ---- bucket addressing ----
    def _bucket_soft(self, t: torch.Tensor) -> torch.Tensor:
        """Partition-of-unity soft one-hot over buckets. t:(B,T) -> g:(B,T,n_buckets), sums to 1 over buckets."""
        b = self.boundaries                                          # (T, n_buckets-1)
        T_bkt = self.T_bkt.view(1, self.n_tables, 1)
        S = torch.sigmoid((t.unsqueeze(-1) - b.unsqueeze(0)) / T_bkt)   # (B,T,n_buckets-1)
        g0 = 1.0 - S[..., :1]
        gmid = S[..., :-1] - S[..., 1:]                              # empty when n_buckets == 2
        glast = S[..., -1:]
        return torch.cat([g0, gmid, glast], dim=-1)                  # (B,T,n_buckets)

    def _bucket_hard(self, t: torch.Tensor) -> torch.Tensor:
        """Hard bucket one-hot. m* = #{boundaries <= t}; t_window (largest time) folds into the last bucket."""
        b = self.boundaries                                          # (T, n_buckets-1)
        mstar = (t.unsqueeze(-1) >= b.unsqueeze(0)).sum(-1)          # (B,T) in [0, n_buckets-1]
        return F.one_hot(mstar.long(), self.n_buckets).float()       # (B,T,n_buckets)

    def _rows(self, onehot: torch.Tensor, table: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.table if table is None else table
        return torch.einsum('btm,tmo->bto', onehot, W)               # sum selected/blended row per table

    def address(self, x: torch.Tensor) -> torch.Tensor:
        t_hard, _ = self._first_spike(x)
        b = self.boundaries
        return (t_hard.unsqueeze(-1) >= b.unsqueeze(0)).sum(-1)       # (B, n_tables) hard bucket index

    def forward(self, x: torch.Tensor, eps: float = 0.3, mode: str = "st") -> torch.Tensor:
        """x:(B,input_dim) -> (B,n_heads,n_outputs). `eps` accepted for API parity (unused: no gate temp).
        Modes 'st'|'hard'|'soft' as in PureLIFDetectorsMHL."""
        B = x.shape[0]
        t_hard, t_soft = self._first_spike(x)
        if mode == "hard":
            rows = self._rows(self._bucket_hard(t_hard))
        elif mode == "soft":
            rows = self._rows(self._bucket_soft(t_soft))
        elif mode == "st":
            y_hard = self._rows(self._bucket_hard(t_hard))               # weight grad -> selected row only
            y_addr = self._rows(self._bucket_soft(t_soft), self.table.detach())   # full-partition address grad
            rows = y_hard + y_addr - y_addr.detach()                     # forward == hard single bucket
        else:
            raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
        return rows.view(B, self.n_heads, self.tables_per_head, self.n_outputs).sum(dim=2)
