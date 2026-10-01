"""ProjectionMHL — a compress -> cartridge -> decompress bottleneck.

Wraps any cartridge (a :class:`~spiky.lutorch_ex.lut_base.MultiHeadLUT`) in a dense
linear ``compress`` / ``decompress`` pair so the (cheap, discrete) lookup runs in a
narrow per-head space while the module presents a plain ``d_model -> d_model`` map::

    z   = compress(x)       # [B, d_model] -> per-head [B, H, d_in]
    y   = cartridge(z)      # [B, H, d_in] -> [B, H, d_out]   (opaque)
    out = decompress(y)     # [B, H*d_out] -> [B, d_model]

The wrapper treats the cartridge as an **opaque box** reached only through the shape
contract — it never inspects the cartridge's internals or its concrete class. The one
thing a cartridge may need to tell the wrapper (a quantised cartridge that emits
integer-valued cells and wants a per-output-channel scale folded into the decompress
matrix) is advertised as a *capability*, the :class:`SupportsDecompressBake` protocol,
rather than detected by ``isinstance`` on a concrete class.
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

    A quantised cartridge reads integer-valued cells and carries the real scale
    separately; folding that scale into the decompress weight columns lets the read stay
    pure-integer while the dequantisation happens for free inside the existing matmul.

    Implementers return a 1-D tensor of length ``spec.out_features`` (= ``H * d_out``,
    one entry per flattened cartridge output channel), or ``None`` to opt out at this
    step. Cartridges without this need (e.g. :class:`ManifestoHardLUT`) simply do not
    implement the method, so ``isinstance(cart, SupportsDecompressBake)`` is ``False``.
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
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.cartridge = cartridge
        self.d_model = d_model
        spec = cartridge.spec
        # compress: d_model -> flattened per-head input; decompress: flattened per-head
        # output -> d_model. The cartridge sees [B, H, d_in] and returns [B, H, d_out].
        self.compress = nn.Linear(d_model, spec.in_features, bias=bias, device=device)
        self.decompress = nn.Linear(spec.out_features, d_model, bias=bias, device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2 or x.shape[1] != self.d_model:
            raise ValueError(f"expected input [B, {self.d_model}], got {tuple(x.shape)}")
        spec = self.cartridge.spec
        B = x.shape[0]

        z = self.compress(x).reshape(B, spec.n_heads, spec.d_in)  # [B, H, d_in]
        y = self.cartridge(z)                                     # [B, H, d_out]
        y = y.reshape(B, spec.out_features)                       # [B, H*d_out]

        # Capability check — advertised, not an isinstance-on-concrete-class probe.
        if isinstance(self.cartridge, SupportsDecompressBake):
            scale = self.cartridge.decompress_scale()
            if scale is not None:
                if scale.shape != (spec.out_features,):
                    raise ValueError(
                        f"decompress_scale() must be 1-D of length out_features="
                        f"{spec.out_features}, got shape {tuple(scale.shape)}"
                    )
                # Fold the per-output-channel scale into the decompress matrix: scaling
                # column k of W is exactly scaling cartridge output channel k before the
                # matmul, with no extra op on the activation path.
                weight = self.decompress.weight * scale.reshape(1, -1)
                return F.linear(y, weight, self.decompress.bias)

        return self.decompress(y)
