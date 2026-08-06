"""PureLIFDetectorsMHL — a pure time-to-first-spike (TTFS) LIF-detector front-end for the multi-head LUT.

A leaner sibling of :class:`spiky.lutorch.lif_detectors_mhl.LIFDetectorsMHL`: it keeps the same LUT skeleton
and the FastMHL-style decoupled straight-through decode, but each detector is a single leaky integrate-and-
fire cell read out by its **time of first spike** (no ordered-pair ``P`` matrix at all).

Per detector (M = n_tables * nap detectors over N=input_dim latency-coded inputs):
  arrivals   a[b,d,i] = latency(x)[b,i] + delay[d,i]
  membrane   at the k-th (ascending-sorted) arrival: V_k = Σ_{j: a_j<=a_k} w_j · exp(-(a_k - a_j)/tau)
  first spike (hard) t* = first sorted arrival time where V_k >= theta_mem (=1.0), else t_window
  first spike (soft) via a smooth first-success over the sorted arrivals (differentiable)
  detection (FLIPPED, early spike = detected): bit = 1[t* < L]; soft bit = sigmoid((L - t_soft)/temp_bit)

Trainable: delay (M,N), w (M,N), L (M,), plus PER-LUT tau_raw / log_T_cross / log_temp_bit (n_tables,) and
the LUT table (n_tables, 2**nap, n_outputs). theta_mem is fixed (1.0). No P block — ~19k params vs ~74k.

Decode is IDENTICAL to LIFDetectorsMHL: product-of-per-bit-Bernoullis cell distribution + decoupled
straight-through (weight grad -> selected row only via y_hard; address grad -> full-K softmax via
y_addr = prow_soft @ table.detach()); forward value == hard single cell.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["PureLIFDetectorsMHL"]


class PureLIFDetectorsMHL(nn.Module):
    def __init__(self, input_dim: int, n_heads: int, n_outputs: int, n_anchor_pairs: int,
                 tables_per_head: int = 1, *, t_window: float = 32.0, latency_c: float = 16.0,
                 latency_alpha: float = 3.0, temp_bit_init: float = 1.0,
                 table_init: Optional[torch.Tensor] = None, device=None):
        super().__init__()
        if n_anchor_pairs < 1 or n_anchor_pairs > 20:
            raise ValueError(f"n_anchor_pairs must be in [1,20], got {n_anchor_pairs}")
        self.input_dim = int(input_dim)
        self.n_heads = int(n_heads)
        self.n_outputs = int(n_outputs)
        self.n_anchor_pairs = int(n_anchor_pairs)
        self.tables_per_head = int(tables_per_head)
        self.n_tables = self.n_heads * self.tables_per_head
        self.n_rows = 1 << self.n_anchor_pairs
        self.t_window = float(t_window)
        self.latency_c = float(latency_c)
        self.latency_alpha = float(latency_alpha)

        N, T, NAP = self.input_dim, self.n_tables, self.n_anchor_pairs
        M = T * NAP
        self.n_detectors = M
        dev = device or torch.device("cpu")

        # --- trainable params ---
        self.delay = nn.Parameter(torch.zeros(M, N, device=dev))            # per-detector per-input
        self.w = nn.Parameter(0.2 * torch.randn(M, N, device=dev))         # per-detector per-input (match LIF w init)
        self.L = nn.Parameter(torch.full((M,), 0.5 * self.t_window, device=dev))   # per-detector deadline
        self.tau_raw = nn.Parameter(torch.ones(T, device=dev))            # per-LUT (tau = softplus + 1e-3)
        self.log_T_cross = nn.Parameter(torch.zeros(T, device=dev))       # per-LUT (T_cross = exp; init 1.0)
        self.log_temp_bit = nn.Parameter(torch.full((T,), float(torch.log(torch.tensor(float(temp_bit_init))))))  # per-LUT
        if table_init is not None:
            if tuple(table_init.shape) != (T, self.n_rows, self.n_outputs):
                raise ValueError(f"table_init shape {tuple(table_init.shape)} != {(T, self.n_rows, self.n_outputs)}")
            self.table = nn.Parameter(table_init.clone().to(dev))
        else:
            self.table = nn.Parameter(0.1 * torch.randn(T, self.n_rows, self.n_outputs, device=dev))

        # --- buffers ---
        self.register_buffer("theta_mem", torch.tensor(1.0, device=dev))
        self.register_buffer("pow2", (1 << torch.arange(NAP - 1, -1, -1, device=dev)).long())
        codes = torch.arange(self.n_rows, device=dev).unsqueeze(1)
        self.register_buffer("bit_matrix",
                             ((codes >> torch.arange(NAP - 1, -1, -1, device=dev).unsqueeze(0)) & 1).float())

    # ---- positive-constrained per-LUT params ----
    @property
    def tau(self):
        return F.softplus(self.tau_raw) + 1e-3

    @property
    def T_cross(self):
        return torch.exp(self.log_T_cross)

    @property
    def temp_bit(self):
        return torch.exp(self.log_temp_bit)

    def latency(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.latency_c - self.latency_alpha * x, 0.0, self.t_window)

    # ---- LUT decode helpers (identical to LIFDetectorsMHL) ----
    def _prow(self, bits: torch.Tensor) -> torch.Tensor:
        BM = self.bit_matrix.view(1, 1, self.n_rows, self.n_anchor_pairs)
        term = BM * bits.unsqueeze(2) + (1.0 - BM) * (1.0 - bits.unsqueeze(2))
        return term.prod(dim=-1)

    def _rows(self, prow: torch.Tensor, table: Optional[torch.Tensor] = None) -> torch.Tensor:
        W = self.table if table is None else table
        return torch.einsum('btc,tco->bto', prow, W)

    # ---- TTFS front-end: returns (hard_bits, soft_bits, t_hard) all grouped (B, n_tables, nap) / (B,M) ----
    def _spike_bits(self, x: torch.Tensor):
        B = x.shape[0]; M, NAP, T = self.n_detectors, self.n_anchor_pairs, self.n_tables
        t = self.latency(x)                                             # (B,N)
        a = t.unsqueeze(1) + self.delay.unsqueeze(0)                    # (B,M,N)
        a_srt, idx = torch.sort(a, dim=-1)                             # ascending
        w_srt = self.w.unsqueeze(0).expand(B, -1, -1).gather(-1, idx)  # (B,M,N)
        dt = a_srt.unsqueeze(-1) - a_srt.unsqueeze(-2)                 # dt[...,k,j] = a_k - a_j  (B,M,N,N)
        tau = self.tau.repeat_interleave(NAP).view(1, M, 1, 1)
        causal = (dt >= 0).float()
        contrib = w_srt.unsqueeze(-2) * torch.exp(-dt.clamp(min=0) / tau) * causal   # (B,M,N,N)
        V = contrib.sum(-1)                                            # (B,M,N) membrane at each sorted arrival
        # hard first-spike time
        crossed = V >= self.theta_mem
        kstar = crossed.float().argmax(-1)                            # (B,M) first crossing (0 if none)
        t_hard = a_srt.gather(-1, kstar.unsqueeze(-1)).squeeze(-1)     # (B,M)
        t_hard = torch.where(crossed.any(-1), t_hard, torch.full_like(t_hard, self.t_window))
        # soft first-spike time (smooth first-success over sorted arrivals)
        T_cross = self.T_cross.repeat_interleave(NAP).view(1, M, 1)
        c = torch.sigmoid((V - self.theta_mem) / T_cross)            # (B,M,N)
        surv = torch.cumprod(1.0 - c, dim=-1)
        surv_prev = torch.cat([torch.ones_like(surv[..., :1]), surv[..., :-1]], dim=-1)
        p = c * surv_prev
        p_never = surv[..., -1]
        t_soft = (p * a_srt).sum(-1) + p_never * self.t_window        # (B,M)
        # FLIPPED detection: early spike (t < L) = detected
        s_hard = (self.L.view(1, M) - t_hard).view(B, T, NAP)
        s_soft = (self.L.view(1, M) - t_soft).view(B, T, NAP)
        hard_bits = (s_hard > 0).float()
        soft_bits = torch.sigmoid(s_soft / self.temp_bit.view(1, T, 1)).clamp(1e-6, 1 - 1e-6)
        return hard_bits, soft_bits, t_hard

    def address(self, x: torch.Tensor) -> torch.Tensor:
        hard_bits, _, _ = self._spike_bits(x)
        return (hard_bits.long() * self.pow2.view(1, 1, -1)).sum(-1)   # (B, n_tables)

    def forward(self, x: torch.Tensor, eps: float = 0.3, mode: str = "st") -> torch.Tensor:
        """x:(B,input_dim) -> (B,n_heads,n_outputs). `eps` accepted for API parity with LIFDetectorsMHL
        (unused: TTFS has no gate temperature). Modes 'st'|'hard'|'soft' as in LIFDetectorsMHL."""
        B = x.shape[0]
        hard_bits, soft_bits, _ = self._spike_bits(x)
        if mode == "hard":
            rows = self._rows(self._prow(hard_bits))
        elif mode == "soft":
            rows = self._rows(self._prow(soft_bits))
        elif mode == "st":
            y_hard = self._rows(self._prow(hard_bits))                # weight grad -> selected row only
            y_addr = self._rows(self._prow(soft_bits), self.table.detach())   # full-K softmax address grad
            rows = y_hard + y_addr - y_addr.detach()                  # forward == hard single cell
        else:
            raise ValueError(f"mode must be 'st'|'hard'|'soft', got {mode!r}")
        return rows.view(B, self.n_heads, self.tables_per_head, self.n_outputs).sum(dim=2)
