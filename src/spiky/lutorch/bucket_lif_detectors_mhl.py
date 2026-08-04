"""BucketLIFDetectorsMHL — single-LIF-neuron-per-table, time-bucket-addressed multi-head LUT front-end.

Now a THIN WRAPPER over the unified :class:`spiky.lutorch.lif_multi_head_lut.LIFMultiHeadLUT` with n_det=1
(one LIF detector per table -> n_buckets buckets = n_buckets cells). The wrapper preserves the exact public
signature and (B, n_heads, n_outputs) output contract, and re-exposes the per-table single-detector view of
the internals (`_first_spike` -> (B, n_tables), `_bucket_soft`/`_bucket_hard` -> (B, n_tables, n_buckets)).
See LIFMultiHeadLUT for the shared machinery (bounded-excitatory weights, O(N) cumsum first-spike, trainable
boundaries and per-table soft temperatures, decoupled straight-through, mixed-radix product generalization).
"""
from typing import Optional

import torch
import torch.nn.functional as F

from .lif_multi_head_lut import LIFMultiHeadLUT

__all__ = ["BucketLIFDetectorsMHL"]


class BucketLIFDetectorsMHL(LIFMultiHeadLUT):
    def __init__(self, input_dim: int, n_heads: int, n_outputs: int, tables_per_head: int = 1, *,
                 n_buckets: int = 16, w_max: float = 2.0, t_window: float = 32.0, latency_c: float = 16.0,
                 latency_alpha: float = 3.0, table_init: Optional[torch.Tensor] = None, device=None):
        super().__init__(input_dim, n_heads, n_outputs, tables_per_head, n_det=1, n_buckets=n_buckets,
                         w_max=w_max, t_window=t_window, latency_c=latency_c, latency_alpha=latency_alpha,
                         table_init=table_init, device=device)

    # single-detector bucket views: accept t as (B, n_tables) or (B, n_tables, 1) and drop the n_det=1 axis.
    # (_first_spike is NOT overridden -- the inherited forward needs its (B, n_tables, 1) shape.)
    @staticmethod
    def _as_det(t):
        return t if t.dim() == 3 else t.unsqueeze(-1)         # (B, n_tables, 1)

    def _bucket_soft(self, t):
        _, p = super()._bucket(self._as_det(t), self._as_det(t))   # (B, T, 1, M)
        return p[:, :, 0, :]                                  # (B, n_tables, n_buckets), partition of unity

    def _bucket_hard(self, t):
        b_hard, _ = super()._bucket(self._as_det(t), self._as_det(t))
        return F.one_hot(b_hard[:, :, 0].long(), self.n_buckets).float()   # (B, n_tables, n_buckets)
