"""ProductBucketLIFMHL — mixed-radix product-of-detectors generalization of the bucket LIF detector.

Now a THIN WRAPPER over the unified :class:`spiky.lutorch.lif_multi_head_lut.LIFMultiHeadLUT`. Each of the
`n_heads` product-tables has `n_det` M-way detectors -> M**n_det joint cells; the tables are summed into the
`out_dim` action. In unified terms this is LIFMultiHeadLUT(n_heads=n_heads, tables_per_head=1, n_det=n_det)
with the (B, n_heads, out) output summed over the head axis to (B, out) -- preserving the original
ProductBucketLIFMHL signature and output contract. See LIFMultiHeadLUT for the mixed-radix gather + rank-1
tensor-product soft decode and the M**n_det <= 4096 cell cap.
"""
import torch

from .lif_multi_head_lut import LIFMultiHeadLUT, MAX_CELLS

__all__ = ["ProductBucketLIFMHL", "MAX_CELLS"]


class ProductBucketLIFMHL(LIFMultiHeadLUT):
    def __init__(self, in_dim: int, out_dim: int, n_heads: int = 4, n_det: int = 2, buckets: int = 16, *,
                 w_max: float = 2.0, t_window: float = 32.0, latency_c: float = 16.0,
                 latency_alpha: float = 3.0, device=None):
        super().__init__(input_dim=in_dim, n_heads=n_heads, n_outputs=out_dim, tables_per_head=1, n_det=n_det,
                         n_buckets=buckets, w_max=w_max, t_window=t_window, latency_c=latency_c,
                         latency_alpha=latency_alpha, device=device)
        # aliases to preserve the original public attribute names
        self.in_dim = self.input_dim
        self.out_dim = self.n_outputs
        self.buckets = self.n_buckets

    def forward(self, x, mode: str = "st"):
        # unified returns (B, n_heads, out); the product model SUMS the heads into the action
        return super().forward(x, mode=mode).sum(dim=1)       # (B, out_dim)
