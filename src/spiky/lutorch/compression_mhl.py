"""CompressionMultiHeadLUT (short: CompressionMHL) — a compress / LUT / decompress bottleneck.

Wraps a FastMultiHeadLut inside a linear compress→decompress pair so the (expensive,
sparse-gradient) table lookup can operate in a chosen inner space rather than the full
input/output space:

    z   = compress(x)          # input_dim  -> inner_in_dim    (dense Linear)
    y   = lut(z)               # FastMHL     inner_in_dim -> inner_out_dim
    out = decompress(y)        # inner_out_dim -> output_dim    (dense Linear)

The LUT's input width (`inner_in_dim`) and output width (`inner_out_dim`) are set
independently. Either projection can be dropped with a `-1` sentinel:
  * inner_in_dim = -1  -> NO compress; the LUT reads x directly (its input_dim = input_dim).
  * inner_out_dim = -1 -> NO decompress; the LUT emits output_dim and heads are summed to
    the final output.
  * both -1            -> a pure FastMHL FFN slot (no projections).
The legacy `inner_dim` kwarg is still accepted and sets both to the same value.

Reused across the CompressionMHL experiment series (which varies the inner dims and `tph`).
See `CompressionMultiHeadLUT.param_count(...)` for the exact parameter formula.
"""
from typing import Optional

import torch
import torch.nn as nn

from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut


def _resolve_inner(inner_dim, inner_in_dim, inner_out_dim):
    """Resolve the (inner_in, inner_out) pair from the legacy `inner_dim` shim or the two
    explicit dims. Returns the RAW values (which may be the -1 'no projection' sentinel)."""
    if inner_dim is not None:
        if inner_in_dim is not None or inner_out_dim is not None:
            raise ValueError(
                "pass either inner_dim (sets both) OR inner_in_dim/inner_out_dim, not both"
            )
        return inner_dim, inner_dim
    if inner_in_dim is None or inner_out_dim is None:
        raise ValueError("pass inner_dim, or BOTH inner_in_dim and inner_out_dim")
    return inner_in_dim, inner_out_dim


class CompressionMultiHeadLUT(nn.Module):
    """Linear-compress → FastMultiHeadLut → linear-decompress bottleneck.

    Args:
        input_dim: dimension of x.
        output_dim: dimension of the returned vector.
        inner_dim: legacy convenience — sets inner_in_dim = inner_out_dim = inner_dim.
            Mutually exclusive with inner_in_dim/inner_out_dim.
        inner_in_dim: the LUT's input width. -1 means "no compress" (LUT reads x directly,
            so its input width = input_dim).
        inner_out_dim: the LUT's per-head output width. -1 means "no decompress" (LUT emits
            output_dim and heads are summed to the output directly).
        nap: FastMHL n_anchor_pairs (K = 2**nap rows per table).
        tph: FastMHL tables_per_head.
        n_heads: FastMHL output heads (default 1). Heads are summed before decompress.
        inner_residual: if True, add the LUT input to its output (`y = lut(z) + z`). Only
            valid when the LUT's effective input width == effective output width (the skip
            lives in one space); raises otherwise. Adds ZERO parameters.
        joint_head_compression: at n_heads>1, whether compress/decompress are shared across
            heads. True -> JOINT (one shared pair + one FastMHL(n_heads)); False (DEFAULT) ->
            INDEPENDENT (per-head compress/decompress + one single-head FastMHL per head).
            At n_heads=1 the two modes are numerically identical.
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
        inner_dim: Optional[int] = None,
        *,
        inner_in_dim: Optional[int] = None,
        inner_out_dim: Optional[int] = None,
        nap: int,
        tph: int,
        n_heads: int = 1,
        inner_residual: bool = False,
        joint_head_compression: bool = False,
        forward_mode: str = "hard",
        weight_dtype: torch.dtype = torch.float32,
        use_bf16: bool = False,
        initial_weights_noise: float = 1e-3,
        learnable_temps: bool = True,
        pre_lut_meanabsnorm: bool = False,
        batched_multi_head_input: bool = True,
        random_seed: Optional[int] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        in_raw, out_raw = _resolve_inner(inner_dim, inner_in_dim, inner_out_dim)
        eff_in = input_dim if in_raw == -1 else in_raw
        eff_out = output_dim if out_raw == -1 else out_raw
        if eff_in < 1 or eff_out < 1:
            raise ValueError(f"effective inner dims must be >= 1 (or -1); got "
                             f"inner_in={in_raw}, inner_out={out_raw}")
        if inner_residual and eff_in != eff_out:
            raise ValueError(
                "inner_residual requires the LUT's effective input width == output width "
                f"(the skip lives in one space); got eff_in={eff_in}, eff_out={eff_out}"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.inner_in_dim = in_raw
        self.inner_out_dim = out_raw
        self.eff_in = eff_in
        self.eff_out = eff_out
        self.has_compress = in_raw != -1
        self.has_decompress = out_raw != -1
        # legacy attribute (only meaningful when the two inner dims coincide)
        self.inner_dim = in_raw if in_raw == out_raw else None
        self.nap = nap
        self.tph = tph
        self.n_heads = n_heads
        self.inner_residual = bool(inner_residual)
        self.joint_head_compression = bool(joint_head_compression)
        self.pre_lut_meanabsnorm = bool(pre_lut_meanabsnorm)   # MeanAbsNorm on compressed z before FastMHL
        # Default ON: replace the independent path's per-head ModuleList with a
        # single batched multi_head_input FastMHL (bit-identical init, ~2x faster
        # hard eval). It only applies to the independent path with per-head
        # compression (each head reads its own inner_in slice); for any other
        # config -- joint_head_compression, or no per-head compress -- we fall
        # back GRACEFULLY to the per-head loop path instead of raising, so the
        # default is safe for every existing CompressionMHL config.
        self._batched_multi_head_input_requested = bool(batched_multi_head_input)
        self.batched_multi_head_input = (
            self._batched_multi_head_input_requested
            and not self.joint_head_compression
            and self.has_compress
        )

        _lut_kw = dict(
            n_anchor_pairs=nap, tables_per_head=tph, forward_mode=forward_mode,
            weight_dtype=weight_dtype, use_bf16=use_bf16,
            initial_weights_noise=initial_weights_noise, learnable_temps=learnable_temps,
            device=device,
        )
        if self.joint_head_compression:
            # JOINT: one shared compress feeds all heads; a single FastMHL(n_heads) reads the
            # shared z; heads summed; one shared decompress.
            self.compress = (nn.Linear(input_dim, in_raw, device=device)
                             if self.has_compress else nn.Identity())
            self.lut = FastMultiHeadLut(
                input_dim=eff_in, n_heads=n_heads, n_outputs=eff_out,
                random_seed=random_seed, **_lut_kw,
            )
            self.decompress = (nn.Linear(out_raw, output_dim, device=device)
                               if self.has_decompress else nn.Identity())
        else:
            # INDEPENDENT per-head: per-head compress (row-blocks of one Linear ->
            # n_heads*inner_in), one single-head FastMHL per head (seed + h so head 0 matches
            # the joint single-head seed -> exact match at n_heads=1), and a decompress over
            # the concatenated per-head outputs (== summed per-head decompress).
            self.compress = (nn.Linear(input_dim, n_heads * in_raw, device=device)
                             if self.has_compress else nn.Identity())
            if self.batched_multi_head_input:
                # One batched multi_head_input FastMHL replacing the ModuleList:
                # block-diagonal per-head routing with the SAME seed+h convention,
                # so this is a bit-identical, faster drop-in for self.luts.
                self.lut_batched = FastMultiHeadLut(
                    input_dim=eff_in, n_heads=n_heads, n_outputs=eff_out,
                    multi_head_input=True, random_seed=random_seed, **_lut_kw,
                )
            else:
                self.luts = nn.ModuleList([
                    FastMultiHeadLut(
                        input_dim=eff_in, n_heads=1, n_outputs=eff_out,
                        random_seed=(None if random_seed is None else random_seed + h),
                        **_lut_kw,
                    )
                    for h in range(n_heads)
                ])
            self.decompress = (nn.Linear(n_heads * out_raw, output_dim, device=device)
                               if self.has_decompress else nn.Identity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(
                f"x shape must be [N, {self.input_dim}], got {tuple(x.shape)}"
            )
        if self.joint_head_compression:
            z = self.compress(x)                       # [N, eff_in]  (Identity -> x)
            if self.pre_lut_meanabsnorm:
                z = z / (z.abs().mean(-1, keepdim=True) + 1e-6)   # MeanAbsNorm before FastMHL
            y = self.lut(z).sum(dim=1).to(z.dtype)     # [N, eff_out]
            if self.inner_residual:
                y = y + z                              # eff_in == eff_out guaranteed
            return self.decompress(y)                  # [N, output_dim]  (Identity -> y)

        # INDEPENDENT per-head path.
        N = x.shape[0]
        if self.has_compress:
            z = self.compress(x).view(N, self.n_heads, self.inner_in_dim)   # [N, H, inner_in]
        if self.batched_multi_head_input:
            # Batched drop-in equivalent of the per-head loop below (has_compress
            # guaranteed True by the constructor check).
            z3 = z
            if self.pre_lut_meanabsnorm:
                z3 = z3 / (z3.abs().mean(-1, keepdim=True) + 1e-6)   # per-head MeanAbsNorm
            y = self.lut_batched(z3).to(z3.dtype)              # [N, H, eff_out]
            if self.inner_residual:
                y = y + z3                                      # eff_in == eff_out
            if self.has_decompress:
                return self.decompress(y.reshape(N, self.n_heads * self.eff_out))
            return y.sum(dim=1)                                 # [N, eff_out]
        parts = []
        for h, lut in enumerate(self.luts):
            z_h = z[:, h, :] if self.has_compress else x        # [N, eff_in] (all heads read x if -1)
            if self.pre_lut_meanabsnorm:
                z_h = z_h / (z_h.abs().mean(-1, keepdim=True) + 1e-6)   # MeanAbsNorm before FastMHL
            y_h = lut(z_h).sum(dim=1).to(z_h.dtype)             # [N, eff_out]
            if self.inner_residual:
                y_h = y_h + z_h
            parts.append(y_h)
        if self.has_decompress:
            return self.decompress(torch.cat(parts, dim=-1))    # [N, H*inner_out] -> [N, output]
        return sum(parts)                                       # sum per-head [N, eff_out=output]

    @staticmethod
    def param_count(input_dim: int, output_dim: int, inner_dim: Optional[int] = None,
                    *, inner_in_dim: Optional[int] = None, inner_out_dim: Optional[int] = None,
                    nap: int, tph: int, n_heads: int = 1,
                    joint_head_compression: bool = False) -> dict:
        """Exact parameter breakdown (dict of the three parts + total).

        A -1 inner dim drops that projection (0 params). The LUT budget uses the effective
        output width. JOINT shares one compress/decompress pair; INDEPENDENT gives each head
        its own (n_heads x the compress weight, decompress over concatenated per-head
        vectors). At n_heads=1 the two modes agree.
        """
        in_raw, out_raw = _resolve_inner(inner_dim, inner_in_dim, inner_out_dim)
        eff_out = output_dim if out_raw == -1 else out_raw
        lut = n_heads * tph * (2 ** nap) * eff_out
        if in_raw == -1:
            compress = 0
        elif joint_head_compression:
            compress = input_dim * in_raw + in_raw
        else:
            compress = input_dim * (n_heads * in_raw) + n_heads * in_raw
        if out_raw == -1:
            decompress = 0
        elif joint_head_compression:
            decompress = out_raw * output_dim + output_dim
        else:
            decompress = (n_heads * out_raw) * output_dim + output_dim
        return {
            "compress": compress,
            "lut": lut,
            "decompress": decompress,
            "total": compress + lut + decompress,
        }


# Short alias — both names refer to the same class.
CompressionMHL = CompressionMultiHeadLUT
