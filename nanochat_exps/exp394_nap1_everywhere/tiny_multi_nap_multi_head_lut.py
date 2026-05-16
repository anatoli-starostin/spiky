"""Multi-NAP variant: a wrapper containing multiple TinyMultiHeadLut
modules with different NAPs but same n_heads/n_outputs/input_dim. Outputs
are summed elementwise.

Idea: different NAPs capture different routing granularity:
  - low NAP (e.g. 4): coarse pattern matching, 16 rows/table, always dense
  - high NAP (e.g. 8): fine pattern matching, 256 rows/table, can collapse

By summing them, the low-NAP component provides an always-active baseline
output while the high-NAP component contributes specificity. Directly
targets the row-collapse pathology documented in exp382 (L4/L5 out_proj
at 1% touch_frac).
"""
from __future__ import annotations
import torch
import torch.nn as nn
from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut


class TinyMultiNapMultiHeadLut(nn.Module):
    def __init__(
        self,
        input_dim: int,
        n_heads: int,
        n_outputs: int,
        nap_tph_pairs: list,  # [(nap, tph), ...]
        base_random_seed: int,
        device,
        **shared_lut_kwargs,
    ):
        super().__init__()
        self.nap_tph_pairs = list(nap_tph_pairs)
        self.luts = nn.ModuleList()
        for i, (nap, tph) in enumerate(self.nap_tph_pairs):
            self.luts.append(TinyMultiHeadLut(
                input_dim=input_dim,
                n_heads=n_heads,
                n_outputs=n_outputs,
                n_anchor_pairs=int(nap),
                tables_per_head=int(tph),
                random_seed=base_random_seed + 1000 * i,
                device=device,
                **shared_lut_kwargs,
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Each inner LUT outputs [B, n_heads, n_outputs]
        outputs = [lut(x) for lut in self.luts]
        return torch.stack(outputs, dim=0).sum(dim=0)
