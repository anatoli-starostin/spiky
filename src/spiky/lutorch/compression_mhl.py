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
        inner_residual: if True, add the compressed input `z` to the LUT output before
            decompress (`y = lut(z) + z`), so the LUT learns a residual over the compressed
            vector. Adds ZERO parameters. Requires the LUT output dim == inner_dim (always
            true here). Default False (plain feed-forward bottleneck).
        joint_head_compression: how the compress/decompress projections are shared across
            heads (only matters for n_heads > 1; at n_heads=1 both modes are numerically
            identical). True -> JOINT: one shared compress feeds all heads, a single
            FastMHL(n_heads) reads the shared compressed vector, heads are summed, one
            shared decompress maps back. False (DEFAULT) -> INDEPENDENT: each head has its
            OWN compress and decompress and its OWN single-head FastMHL; head h reads
            z_h = compress_h(x), and the per-head decompressed outputs are summed. The LUT
            table budget is identical in both modes; only the projection params differ.
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
        inner_residual: bool = False,
        joint_head_compression: bool = False,
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
        self.inner_residual = bool(inner_residual)
        self.joint_head_compression = bool(joint_head_compression)

        _lut_kw = dict(
            n_anchor_pairs=nap, tables_per_head=tph, forward_mode=forward_mode,
            weight_dtype=weight_dtype, use_bf16=use_bf16,
            initial_weights_noise=initial_weights_noise, device=device,
        )
        if self.joint_head_compression:
            # JOINT / shared: one compress feeds all heads, a single FastMHL with n_heads
            # table-groups reads the shared z, heads are summed, one shared decompress.
            self.compress = nn.Linear(input_dim, inner_dim, device=device)
            self.lut = FastMultiHeadLut(
                input_dim=inner_dim, n_heads=n_heads, n_outputs=inner_dim,
                random_seed=random_seed, **_lut_kw,
            )
            self.decompress = nn.Linear(inner_dim, output_dim, device=device)
        else:
            # INDEPENDENT per-head: each head gets its OWN compress and decompress and its
            # OWN single-head FastMHL. The per-head compress maps are the row-blocks of one
            # Linear(input_dim -> n_heads*inner_dim); the per-head decompress maps + summed
            # bias are one Linear(n_heads*inner_dim -> output_dim) over the concatenated
            # per-head outputs. Per-head anchor seeds = random_seed + h (so head 0 == the
            # joint single-head seed -> exact numerical match at n_heads=1).
            self.compress = nn.Linear(input_dim, n_heads * inner_dim, device=device)
            self.luts = nn.ModuleList([
                FastMultiHeadLut(
                    input_dim=inner_dim, n_heads=1, n_outputs=inner_dim,
                    random_seed=(None if random_seed is None else random_seed + h),
                    **_lut_kw,
                )
                for h in range(n_heads)
            ])
            self.decompress = nn.Linear(n_heads * inner_dim, output_dim, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [N, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.joint_head_compression:
            z = self.compress(x)                       # [N, inner]
            y = self.lut(z)                            # [N, n_heads, inner]
            y = y.sum(dim=1).to(z.dtype)               # combine heads -> [N, inner]
            if self.inner_residual:
                y = y + z                              # inner skip over the shared z
            return self.decompress(y)                  # [N, output_dim]

        # INDEPENDENT per-head path.
        N = x.shape[0]
        z = self.compress(x).view(N, self.n_heads, self.inner_dim)   # [N, H, inner]
        parts = []
        for h, lut in enumerate(self.luts):
            z_h = z[:, h, :]                            # [N, inner] — this head's own z
            y_h = lut(z_h).sum(dim=1).to(z_h.dtype)     # [N, inner] (single head -> squeeze)
            if self.inner_residual:
                y_h = y_h + z_h                         # per-head inner skip
            parts.append(y_h)
        y = torch.cat(parts, dim=-1)                    # [N, H*inner]
        return self.decompress(y)                       # sum_h decompress_h(y_h) -> [N, output_dim]

    @staticmethod
    def param_count(input_dim: int, output_dim: int, inner_dim: int,
                    *, nap: int, tph: int, n_heads: int = 1,
                    joint_head_compression: bool = False) -> dict:
        """Exact parameter breakdown (dict of the three parts + total).

        The LUT budget is the same in both modes (n_heads table-groups either way). Only
        the compress/decompress projections differ: JOINT shares one pair across heads;
        INDEPENDENT gives each head its own pair (n_heads x the compress weight, and a
        decompress over the concatenated per-head vectors). At n_heads=1 the two agree.
        """
        lut = n_heads * tph * (2 ** nap) * inner_dim
        if joint_head_compression:
            compress = input_dim * inner_dim + inner_dim
            decompress = inner_dim * output_dim + output_dim
        else:
            compress = input_dim * (n_heads * inner_dim) + n_heads * inner_dim
            decompress = (n_heads * inner_dim) * output_dim + output_dim
        return {
            "compress": compress,
            "lut": lut,
            "decompress": decompress,
            "total": compress + lut + decompress,
        }


# Short alias — both names refer to the same class.
CompressionMHL = CompressionMultiHeadLUT
