"""CompressionMultiHeadLUT (short: CompressionMHL) — a compress / LUT / decompress bottleneck.

Wraps a FastMultiHeadLut inside a linear compress→decompress pair so the (expensive,
sparse-gradient) table lookup operates in a small `inner_dim` space instead of the full
`input_dim`:

    z   = compress(x)          # input_dim -> inner_dim   (dense Linear)
    y   = lut(z)               # FastMHL in the compressed space, inner -> inner
    out = decompress(y)        # inner_dim -> output_dim   (dense Linear)

This lets a LUT-based FFN-slot spend most of its parameters on the tables while keeping the
addressed vector low-dimensional. Reused across the CompressionMHL experiment series, which
varies `inner_dim` and `tph`.

Param count (n_heads defaults to 1):
    compress    = input_dim*inner_dim + inner_dim
    lut         = n_heads * tph * (2**nap) * inner_dim          # FastMHL: n_outputs = inner_dim
    decompress  = inner_dim*output_dim + output_dim
See `CompressionMultiHeadLUT.param_count(...)` for the exact formula.
"""
from typing import Optional

import torch
import torch.nn as nn

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut


class CompressionMultiHeadLUT(nn.Module):
    """Linear-compress → FastMultiHeadLut → linear-decompress bottleneck.

    Args:
        input_dim: dimension of x.
        output_dim: dimension of the returned vector.
        inner_dim: the compressed dimension the LUT operates in (both the LUT's
            input_dim and its per-head n_outputs).
        nap: FastMHL n_anchor_pairs (K = 2**nap rows per table).
        tph: FastMHL tables_per_head.
        n_heads: FastMHL output heads (default 1). Heads are summed before decompress,
            so the reduced vector is always [N, inner_dim] (a no-op squeeze when n_heads=1).
        forward_mode: "hard" (default) or "hybrid_smooth"; passed to FastMHL.
        weight_dtype: FastMHL table storage dtype (default fp32).
        use_bf16: FastMHL bf16-autocast flag (default False — these experiments run fp32).
        initial_weights_noise: FastMHL near-zero table init (default 1e-3).
        random_seed: FastMHL anchor/table seed.
        device: optional device for the submodules.

    Forward:
        x: float [N, input_dim]  ->  [N, output_dim].
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        inner_dim: int,
        *,
        nap: int,
        tph: int,
        n_heads: int = 1,
        forward_mode: str = "hard",
        weight_dtype: torch.dtype = torch.float32,
        use_bf16: bool = False,
        initial_weights_noise: float = 1e-3,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_dim = inner_dim
        self.nap = nap
        self.tph = tph
        self.n_heads = n_heads

        self.compress = nn.Linear(input_dim, inner_dim, device=device)
        self.lut = FastMultiHeadLut(
            input_dim=inner_dim,
            n_heads=n_heads,
            n_outputs=inner_dim,
            n_anchor_pairs=nap,
            tables_per_head=tph,
            forward_mode=forward_mode,
            weight_dtype=weight_dtype,
            use_bf16=use_bf16,
            initial_weights_noise=initial_weights_noise,
            random_seed=random_seed,
            device=device,
        )
        self.decompress = nn.Linear(inner_dim, output_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [N, {self.input_dim}], got {tuple(x.shape)}"
            )
        z = self.compress(x)                       # [N, inner]
        y = self.lut(z)                            # [N, n_heads, inner]
        y = y.sum(dim=1).to(z.dtype)               # combine heads -> [N, inner]
        return self.decompress(y)                  # [N, output_dim]

    @staticmethod
    def param_count(input_dim: int, output_dim: int, inner_dim: int,
                    *, nap: int, tph: int, n_heads: int = 1) -> dict:
        """Exact parameter breakdown (dict of the three parts + total)."""
        compress = input_dim * inner_dim + inner_dim
        lut = n_heads * tph * (2 ** nap) * inner_dim
        decompress = inner_dim * output_dim + output_dim
        return {
            "compress": compress,
            "lut": lut,
            "decompress": decompress,
            "total": compress + lut + decompress,
        }


# Short alias — both names refer to the same class.
CompressionMHL = CompressionMultiHeadLUT
