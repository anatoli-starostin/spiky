"""ProjectionMHL - a compress -> cartridge -> decompress bottleneck.

Wraps any cartridge (a :class:`~spiky.lutorch_ex.lut_base.MultiHeadLUT`) in a dense
linear ``compress`` / ``decompress`` pair so the (cheap, discrete) lookup runs in a
narrow per-head space while the module presents a plain ``input_dim -> output_dim``
map::

    z   = compress(x)       # [B, input_dim]  -> per-head [B, h_in, d_in]
    y   = cartridge(z)      # [B, h_in, d_in] -> [B, h_out, d_out]   (opaque)
    out = decompress(y)     # [B, h_out*d_out] -> [B, output_dim]

``input_dim`` and ``output_dim`` may differ: the two projections are independent
``nn.Linear`` maps and nothing in between cares about the model widths. ``d_model`` is
kept as a convenience alias that sets both to the same value - the square case, and the
only form earlier versions accepted - so ``ProjectionMHL(cart, 384)`` and
``ProjectionMHL(cart, d_model=384)`` keep working unchanged and mean
``input_dim == output_dim == 384``. The parameter names (``compress.weight``,
``decompress.weight``, ``cartridge.*``) are the same in both forms, so checkpoints saved
by the square form load into the square form as before.

Either projection may be switched off (``compress=False`` / ``decompress=False``), in
which case that side is an identity and the corresponding width must already match
(``h_in*d_in == input_dim``, resp. ``h_out*d_out == output_dim``). Switching **both** off
is refused: nothing could then change the width, so the wrapper would only be legal for
``input_dim == output_dim == h_in*d_in == h_out*d_out`` and would do nothing there - wrap
the cartridge directly instead.

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
    """Linear-compress -> cartridge -> linear-decompress wrapper around a cartridge.

    Args:
        cartridge: the :class:`MultiHeadLUT` to wrap.
        d_model: convenience alias setting ``input_dim == output_dim == d_model`` (positional
            or keyword; the form all earlier callers use). Mutually consistent with
            ``input_dim`` / ``output_dim`` if those are given too.
        input_dim: width of the input ``x`` (keyword-only). Required unless ``d_model`` is given.
        output_dim: width of the output (keyword-only). Required unless ``d_model`` is given.
        compress / decompress: switch the corresponding projection off (identity); see the
            module docstring for the width requirements that then apply.
        bias: bias on both projections.
        device: device (``"meta"`` builds an allocation-free skeleton).
    """

    def __init__(
        self,
        cartridge: MultiHeadLUT,
        d_model: Optional[int] = None,
        *,
        input_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        compress: bool = True,
        decompress: bool = True,
        bias: bool = True,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        input_dim, output_dim = self._resolve_dims(d_model, input_dim, output_dim)
        if not compress and not decompress:
            raise ValueError(
                "ProjectionMHL requires at least one of compress/decompress: with both switched "
                "off nothing could change the width (it would need input_dim == output_dim == "
                f"h_in*d_in == h_out*d_out; got input_dim={input_dim}, output_dim={output_dim}, "
                f"h_in*d_in={cartridge.spec.in_features}, h_out*d_out={cartridge.spec.out_features}) "
                "and the wrapper would do nothing - wrap the cartridge directly instead."
            )
        self.cartridge = cartridge
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.has_compress = bool(compress)
        self.has_decompress = bool(decompress)
        spec = cartridge.spec

        if compress:
            self.compress = nn.Linear(input_dim, spec.in_features, bias=bias, device=device)
            if self.compress.weight.device.type != "meta":
                nn.init.normal_(self.compress.weight, std=0.02)   # faithful init: OLD CompressionMHL ~0.02
        else:
            if spec.in_features != input_dim:
                raise ValueError(
                    f"compress=False requires h_in*d_in == input_dim, got {spec.in_features} != {input_dim}"
                )
            self.compress = nn.Identity()

        if decompress:
            self.decompress = nn.Linear(spec.out_features, output_dim, bias=bias, device=device)
            if self.decompress.weight.device.type != "meta":
                nn.init.zeros_(self.decompress.weight)            # faithful init: FFN starts ~0 (OLD zeroes decompress)
        else:
            if spec.out_features != output_dim:
                raise ValueError(
                    f"decompress=False requires h_out*d_out == output_dim, got {spec.out_features} != {output_dim}"
                )
            self.decompress = nn.Identity()

    @staticmethod
    def _resolve_dims(d_model, input_dim, output_dim) -> tuple[int, int]:
        """``d_model`` sets both widths; explicit widths must agree with it if both are given."""
        if d_model is not None:
            if input_dim is not None and input_dim != d_model:
                raise ValueError(f"input_dim={input_dim} contradicts d_model={d_model}")
            if output_dim is not None and output_dim != d_model:
                raise ValueError(f"output_dim={output_dim} contradicts d_model={d_model}")
            input_dim = output_dim = d_model
        if input_dim is None or output_dim is None:
            raise ValueError(
                "ProjectionMHL needs either d_model (sets input_dim == output_dim) or both "
                f"input_dim and output_dim; got d_model={d_model}, input_dim={input_dim}, output_dim={output_dim}"
            )
        for name, v in (("input_dim", input_dim), ("output_dim", output_dim)):
            if not isinstance(v, int) or isinstance(v, bool) or v < 1:
                raise ValueError(f"ProjectionMHL {name} must be a positive int, got {v!r}")
        return int(input_dim), int(output_dim)

    @property
    def d_model(self) -> int:
        """The common width of a square wrapper (``input_dim == output_dim``). A rectangular wrapper
        has no single ``d_model``; ask for ``input_dim`` / ``output_dim`` instead."""
        if self.input_dim != self.output_dim:
            raise AttributeError(
                f"ProjectionMHL is rectangular (input_dim={self.input_dim}, output_dim={self.output_dim}); "
                "it has no single d_model - use input_dim / output_dim"
            )
        return self.input_dim

    def extra_repr(self) -> str:
        return (f"input_dim={self.input_dim}, output_dim={self.output_dim}, "
                f"compress={self.has_compress}, decompress={self.has_decompress}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, input_dim] -> [B, output_dim]``."""
        if x.dim() != 2 or x.shape[1] != self.input_dim:
            raise ValueError(f"expected input [B, {self.input_dim}], got {tuple(x.shape)}")
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
