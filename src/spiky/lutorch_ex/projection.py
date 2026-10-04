"""ProjectionMHL - a compress -> cartridge -> decompress bottleneck.

Wraps any cartridge (a :class:`~spiky.lutorch_ex.lut_base.MultiHeadLUT`) in a dense
linear ``compress`` / ``decompress`` pair so the (cheap, discrete) lookup runs in a
narrow per-head space while the module presents a plain ``d_model -> d_model`` map::

    z   = compress(x)       # [B, d_model] -> per-head [B, h_in, d_in]
    y   = cartridge(z)      # [B, h_in, d_in] -> [B, h_out, d_out]   (opaque)
    out = decompress(y)     # [B, h_out*d_out] -> [B, d_model]

Either projection may be switched off (``compress=False`` / ``decompress=False``), in
which case that side is an identity and the corresponding width must already match
``d_model``. Switching **both** off is forbidden - the wrapper would then do nothing.

The wrapper treats the cartridge as an **opaque box** reached only through the shape
contract. The one thing a cartridge may tell the wrapper (a quantised cartridge wanting a
per-output-channel scale folded into decompress) is advertised as a *capability*, the
:class:`SupportsDecompressBake` protocol, not detected by ``isinstance`` on a concrete
class.
"""
from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lut_base import MultiHeadLUT


@runtime_checkable
class SupportsDecompressBake(Protocol):
    """Capability: a cartridge that wants a per-output-channel scale baked into decompress.

    Implementers return a 1-D tensor of length ``spec.out_features`` (= ``h_out * d_out``),
    or ``None`` to opt out. Cartridges without this need (e.g. :class:`ManifestoHardLUT`)
    simply do not implement the method, so ``isinstance(cart, SupportsDecompressBake)`` is
    ``False``.
    """

    def decompress_scale(self) -> Optional[torch.Tensor]:
        ...


class ProjectionMHL(nn.Module):
    """Linear-compress -> cartridge -> linear-decompress wrapper around a cartridge."""

    def __init__(
        self,
        cartridge: MultiHeadLUT,
        d_model: int,
        *,
        compress: bool = True,
        decompress: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        if not compress and not decompress:
            raise ValueError(
                "ProjectionMHL requires at least one of compress/decompress; switching "
                "both off (both -1) leaves the wrapper with nothing to do."
            )
        self.cartridge = cartridge
        self.d_model = d_model
        self.has_compress = bool(compress)
        self.has_decompress = bool(decompress)
        spec = cartridge.spec

        if compress:
            self.compress = nn.Linear(d_model, spec.in_features, bias=bias, device=device)
            if self.compress.weight.device.type != "meta":
                nn.init.normal_(self.compress.weight, std=0.02)   # faithful init: OLD CompressionMHL ~0.02
        else:
            if spec.in_features != d_model:
                raise ValueError(
                    f"compress=False requires h_in*d_in == d_model, got {spec.in_features} != {d_model}"
                )
            self.compress = nn.Identity()

        if decompress:
            self.decompress = nn.Linear(spec.out_features, d_model, bias=bias, device=device)
            if self.decompress.weight.device.type != "meta":
                nn.init.zeros_(self.decompress.weight)            # faithful init: FFN starts ~0 (OLD zeroes decompress)
        else:
            if spec.out_features != d_model:
                raise ValueError(
                    f"decompress=False requires h_out*d_out == d_model, got {spec.out_features} != {d_model}"
                )
            self.decompress = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.d_model:
            raise ValueError(f"expected input [B, {self.d_model}], got {tuple(x.shape)}")
        spec = self.cartridge.spec
        B = x.shape[0]

        z = self.compress(x).reshape(B, spec.h_in, spec.d_in)  # [B, h_in, d_in]
        y = self.cartridge(z)                                  # [B, h_out, d_out]
        y = y.reshape(B, spec.out_features)                    # [B, h_out*d_out]

        # Capability check - advertised, not an isinstance-on-concrete-class probe.
        scale = None
        if isinstance(self.cartridge, SupportsDecompressBake):
            scale = self.cartridge.decompress_scale()
            if scale is not None and scale.shape != (spec.out_features,):
                raise ValueError(
                    f"decompress_scale() must be 1-D of length out_features="
                    f"{spec.out_features}, got shape {tuple(scale.shape)}"
                )

        if self.has_decompress and scale is not None:
            # Fold the per-output-channel scale into the decompress matrix: scaling column
            # k of W is exactly scaling cartridge output channel k, with no extra op.
            weight = self.decompress.weight * scale.reshape(1, -1)
            return F.linear(y, weight, self.decompress.bias)
        if scale is not None:
            # No decompress matrix to fold into -> apply the scale directly.
            y = y * scale.reshape(1, -1)
        return self.decompress(y)
