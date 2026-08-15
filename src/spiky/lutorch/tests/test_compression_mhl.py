"""Tests for CompressionMultiHeadLUT (CompressionMHL) — the compress / FastMHL / decompress bottleneck.

Covers: forward shape+dtype; the param_count formula matches the built module; gradients
flow through compress, the LUT tables, decompress, and back to the input; input-shape
validation; the n_heads>1 head-sum keeps the [N, inner] contract; determinism.
"""
import pytest
import torch

from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT


def _mk(**kw):
    base = dict(input_dim=32, output_dim=32, inner_dim=8, nap=4, tph=6,
                forward_mode="hard", use_bf16=False, random_seed=0)
    base.update(kw)
    return CompressionMultiHeadLUT(**base)


def test_forward_shape_dtype():
    m = _mk()
    x = torch.randn(16, 32)
    out = m(x)
    assert out.shape == (16, 32)
    assert out.dtype == torch.float32


def test_param_count_matches_module():
    for inner, nap, tph, nh in [(8, 4, 6, 1), (16, 6, 10, 1), (8, 5, 7, 3)]:
        m = _mk(inner_dim=inner, nap=nap, tph=tph, n_heads=nh)
        measured = sum(p.numel() for p in m.parameters())
        formula = CompressionMultiHeadLUT.param_count(32, 32, inner, nap=nap, tph=tph, n_heads=nh)
        assert measured == formula["total"]
        assert formula["compress"] == 32 * inner + inner
        assert formula["lut"] == nh * tph * (2 ** nap) * inner
        assert formula["decompress"] == inner * 32 + 32


def test_grads_flow_through_all_three_parts():
    m = _mk()
    x = torch.randn(64, 32, requires_grad=True)
    m(x).pow(2).mean().backward()
    for name, p in [("compress.w", m.compress.weight), ("lut.tables", m.lut.weights),
                    ("decompress.w", m.decompress.weight)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"no grad for {name}"
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_input_shape_validation():
    m = _mk()
    with pytest.raises(ValueError):
        m(torch.randn(4, 999))
    with pytest.raises(ValueError):
        m(torch.randn(4, 5, 32))          # must be 2-D [N, input_dim]


def test_multihead_sum_keeps_inner_contract():
    m = _mk(n_heads=3)
    out = m(torch.randn(10, 32))
    assert out.shape == (10, 32)          # heads summed before decompress


def test_determinism():
    m = _mk()
    x = torch.randn(8, 32)
    with torch.no_grad():
        assert torch.equal(m(x), m(x))
