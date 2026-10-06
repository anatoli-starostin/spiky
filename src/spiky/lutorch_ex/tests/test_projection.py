"""ProjectionMHL: round-trip across H patterns, compress/decompress toggles, bake."""
from typing import Optional

import pytest
import torch

from spiky.lutorch_ex import (
    LUTSpec,
    ManifestoHardLUT,
    MultiHeadLUT,
    ProjectionMHL,
    SupportsDecompressBake,
)


@pytest.mark.parametrize("h_in,h_out", [(3, 3), (1, 4), (4, 1), (1, 1)])
def test_round_trip_shapes_and_backward(h_in, h_out):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=2, nap=4, d_in=8, d_out=6)
    proj = ProjectionMHL(ManifestoHardLUT(spec, seed=20), d_model=32).train()
    # The default zero-inits decompress (the FFN starts as a zero contribution), which gates first-step gradient to the input side; nudge it off zero so this test
    # exercises what it is about - that ProjectionMHL routes gradient through the composition.
    torch.nn.init.normal_(proj.decompress.weight, std=0.02)
    x = torch.randn(10, 32, requires_grad=True)
    y = proj(x)
    assert y.shape == (10, 32)
    y.pow(2).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert proj.compress.weight.grad is not None
    assert proj.decompress.weight.grad is not None


def test_both_projections_off_forbidden():
    spec = LUTSpec(h_in=2, h_out=2, tph=1, nap=2, d_in=8, d_out=8)  # in=out=16
    with pytest.raises(ValueError):
        ProjectionMHL(ManifestoHardLUT(spec, seed=1), d_model=16,
                      compress=False, decompress=False)


def test_compress_off_identity():
    spec = LUTSpec(h_in=2, h_out=2, tph=1, nap=3, d_in=8, d_out=4)  # in_features=16
    proj = ProjectionMHL(ManifestoHardLUT(spec, seed=5), d_model=16, compress=False).eval()
    assert isinstance(proj.compress, torch.nn.Identity)
    assert proj(torch.randn(4, 16)).shape == (4, 16)


def test_decompress_off_identity():
    spec = LUTSpec(h_in=2, h_out=2, tph=1, nap=3, d_in=4, d_out=8)  # out_features=16
    proj = ProjectionMHL(ManifestoHardLUT(spec, seed=6), d_model=16, decompress=False).eval()
    assert isinstance(proj.decompress, torch.nn.Identity)
    assert proj(torch.randn(4, 16)).shape == (4, 16)


def test_off_side_dimension_mismatch_rejected():
    spec = LUTSpec(h_in=2, h_out=2, tph=1, nap=3, d_in=8, d_out=4)  # in_features=16
    with pytest.raises(ValueError):
        ProjectionMHL(ManifestoHardLUT(spec, seed=7), d_model=20, compress=False)


def test_manifesto_does_not_advertise_bake():
    spec = LUTSpec(h_in=2, h_out=2, tph=2, nap=3, d_in=4, d_out=3)
    assert not isinstance(ManifestoHardLUT(spec, seed=21), SupportsDecompressBake)


class _FakeQuantCartridge(MultiHeadLUT):
    """Minimal bake-capable cartridge: fixed per-head output and a known scale."""

    def __init__(self, spec: LUTSpec):
        super().__init__(spec)
        self.register_buffer("_scale", torch.arange(1, spec.out_features + 1, dtype=torch.float))

    def decompress_scale(self) -> Optional[torch.Tensor]:
        return self._scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._check_input(x)
        B = x.shape[0]
        y = torch.arange(1, self.spec.d_out + 1, dtype=x.dtype)
        return y.expand(B, self.spec.h_out, self.spec.d_out).contiguous()


def test_decompress_bake_is_detected_and_folded():
    spec = LUTSpec(h_in=2, h_out=2, tph=1, nap=2, d_in=4, d_out=3)
    cart = _FakeQuantCartridge(spec)
    assert isinstance(cart, SupportsDecompressBake)

    proj = ProjectionMHL(cart, d_model=16).eval()
    x = torch.randn(5, 16)
    got = proj(x)

    z = proj.compress(x).reshape(5, spec.h_in, spec.d_in)
    y = cart(z).reshape(5, spec.out_features) * cart.decompress_scale().reshape(1, -1)
    ref = proj.decompress(y)
    assert torch.allclose(got, ref, atol=1e-6)


def test_decompress_scale_none_is_plain_path():
    class _NoneScale(_FakeQuantCartridge):
        def decompress_scale(self):
            return None

    spec = LUTSpec(h_in=2, h_out=2, tph=1, nap=2, d_in=4, d_out=3)
    proj = ProjectionMHL(_NoneScale(spec), d_model=16).eval()
    x = torch.randn(4, 16)
    got = proj(x)
    z = proj.compress(x).reshape(4, spec.h_in, spec.d_in)
    ref = proj.decompress(proj.cartridge(z).reshape(4, spec.out_features))
    assert torch.allclose(got, ref, atol=1e-6)
