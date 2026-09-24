"""Regression test: CompressionMultiHeadLUT input_multi_head (pre-split multi-head input, NO compress).

Used by the lut_out_proj variant to feed the head-separated attention output straight into the
out_proj-replacement LUT: inner_in_dim=-1 (no compress), each head routes its own input_dim//n_heads
slice. Default input_multi_head=False is byte-identical to every existing config.
"""
import torch

from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT


def _outproj_lut():
    return CompressionMultiHeadLUT(
        input_dim=384, output_dim=384, inner_in_dim=-1, inner_out_dim=64,
        nap=8, tph=64, n_heads=6, lut_impl="light", forward_confidence=True,
        confidence_form="margin", light_forward_mode="scored", read_top_n=2,
        z_norm=False, input_multi_head=True, random_seed=1)


def test_input_multi_head_builds_no_compress():
    op = _outproj_lut()
    assert isinstance(op.compress, torch.nn.Identity)          # NO compress
    assert op.eff_in == 64 and op.eff_out == 64                # eff_in = input_dim // n_heads
    assert op.light_multi_head_input is True
    x = torch.randn(8, 384)
    y = op(x)
    assert y.shape == (8, 384)


def test_input_multi_head_routes_per_head():
    op = _outproj_lut()
    # tables are ~0 at init -> give them signal so a re-addressed cell actually changes the read
    for m in op.modules():
        if isinstance(m, LightMultiHeadLUT):
            with torch.no_grad():
                m.tables.normal_(0.0, 1.0)
    op.eval()
    x = torch.randn(8, 384)
    with torch.no_grad():
        y0 = op(x)
        xa = x.clone(); xa[:, 0:64] = xa[:, 0:64] * 3.0 + 2.0        # perturb head 0's slice
        ya = op(xa)
        xb = x.clone(); xb[:, 320:384] = xb[:, 320:384] * 3.0 + 2.0  # perturb head 5's slice
        yb = op(xb)
    assert not torch.allclose(y0, ya), "perturbing a head's own slice must change the output"
    assert not torch.allclose(y0, yb)
    # different heads route separately -> their deltas differ (not one shared global pool)
    assert not torch.allclose(ya - y0, yb - y0)


def test_input_multi_head_validation():
    # requires no compress (inner_in_dim=-1)
    import pytest
    with pytest.raises(ValueError):
        CompressionMultiHeadLUT(input_dim=384, output_dim=384, inner_in_dim=64, inner_out_dim=64,
                                nap=8, tph=64, n_heads=6, lut_impl="light",
                                input_multi_head=True, random_seed=1)
    # requires input_dim divisible by n_heads
    with pytest.raises(ValueError):
        CompressionMultiHeadLUT(input_dim=380, output_dim=384, inner_in_dim=-1, inner_out_dim=64,
                                nap=8, tph=64, n_heads=6, lut_impl="light",
                                input_multi_head=True, random_seed=1)


def test_default_off_unchanged_multi_head_with_compress():
    # the ordinary light multi_head_input path (has compress) still builds/runs; default off
    ffn = CompressionMultiHeadLUT(input_dim=384, output_dim=384, inner_in_dim=64, inner_out_dim=64,
                                  nap=8, tph=64, n_heads=6, lut_impl="light", forward_confidence=True,
                                  confidence_form="margin", light_forward_mode="scored", read_top_n=2,
                                  z_norm=False, random_seed=1)
    assert ffn.input_multi_head is False
    assert isinstance(ffn.compress, torch.nn.Linear)
    assert ffn.eff_in == 64 and ffn.light_multi_head_input is True
    assert ffn(torch.randn(8, 384)).shape == (8, 384)
