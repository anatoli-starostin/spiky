"""ProjectionMHL: compress/decompress round-trip and the decompress-bake capability."""
from typing import Optional

import torch
import torch.nn as nn

from spiky.lutorch_ex import (
    LUTSpec,
    ManifestoHardLUT,
    MultiHeadLUT,
    ProjectionMHL,
    SupportsDecompressBake,
)


def test_round_trip_shapes_and_backward():
    spec = LUTSpec(n_heads=3, tph=2, nap=4, d_in=8, d_out=6)
    proj = ProjectionMHL(ManifestoHardLUT(spec, seed=20), d_model=32).train()
    x = torch.randn(10, 32, requires_grad=True)
    y = proj(x)
    assert y.shape == (10, 32), "wrapper presents a plain d_model -> d_model map"
    y.pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert proj.compress.weight.grad is not None
    assert proj.decompress.weight.grad is not None


def test_manifesto_does_not_advertise_bake():
    spec = LUTSpec(n_heads=2, tph=2, nap=3, d_in=4, d_out=3)
    cart = ManifestoHardLUT(spec, seed=21)
    assert not isinstance(cart, SupportsDecompressBake), (
        "a non-quant cartridge must not implement the bake capability"
    )


class _FakeQuantCartridge(MultiHeadLUT):
    """Minimal bake-capable cartridge: emits a fixed per-head output and a known scale."""

    def __init__(self, spec: LUTSpec):
        super().__init__(spec)
        self.register_buffer("_scale", torch.arange(1, spec.out_features + 1, dtype=torch.float))

    def decompress_scale(self) -> Optional[torch.Tensor]:
        return self._scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z, was_flat = self._as_per_head(x)
        B = z.shape[0]
        # Deterministic integer-ish per-head output (stands in for a quantised read).
        y = z[..., : self.spec.d_out] * 0 + torch.arange(
            1, self.spec.d_out + 1, dtype=z.dtype
        )
        y = y.expand(B, self.spec.n_heads, self.spec.d_out).contiguous()
        return self._restore_rank(y, was_flat)


def test_decompress_bake_is_detected_and_folded():
    spec = LUTSpec(n_heads=2, tph=1, nap=2, d_in=4, d_out=3)
    cart = _FakeQuantCartridge(spec)
    assert isinstance(cart, SupportsDecompressBake)

    proj = ProjectionMHL(cart, d_model=16, bias=True).eval()
    x = torch.randn(5, 16)
    got = proj(x)

    # Independent reference: scale the cartridge output, THEN decompress (plain).
    z = proj.compress(x).reshape(5, spec.n_heads, spec.d_in)
    y = cart(z).reshape(5, spec.out_features) * cart.decompress_scale().reshape(1, -1)
    ref = proj.decompress(y)
    assert torch.allclose(got, ref, atol=1e-6), "baking the scale into W must match scale-then-decompress"


def test_decompress_scale_none_is_plain_path():
    class _NoneScale(_FakeQuantCartridge):
        def decompress_scale(self):
            return None

    spec = LUTSpec(n_heads=2, tph=1, nap=2, d_in=4, d_out=3)
    proj = ProjectionMHL(_NoneScale(spec), d_model=16).eval()
    x = torch.randn(4, 16)
    got = proj(x)
    z = proj.compress(x).reshape(4, spec.n_heads, spec.d_in)
    ref = proj.decompress(proj.cartridge(z).reshape(4, spec.out_features))
    assert torch.allclose(got, ref, atol=1e-6)
