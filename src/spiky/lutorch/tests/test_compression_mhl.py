"""Tests for CompressionMultiHeadLUT (CompressionMHL) — the compress / FastMHL / decompress bottleneck.

Covers: forward shape+dtype; the param_count formula matches the built module in BOTH
head-compression modes and for several n_heads; gradients flow through compress, the LUT
tables, decompress and back to the input; input-shape validation; the n_heads>1 outputs sum
to output_dim; determinism; inner_residual; and the joint_head_compression flag (default
False, the n_heads=1 identity between modes, and shapes for n_heads>1 under both modes).
"""
import pytest
import torch

from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT, CompressionMHL


def _mk(**kw):
    base = dict(input_dim=32, output_dim=32, inner_dim=8, nap=4, tph=6,
                forward_mode="hard", use_bf16=False, random_seed=0)
    base.update(kw)
    return CompressionMultiHeadLUT(**base)


def _lut_tables(m):
    """The LUT weight tensors, across all three attribute shapes the module can expose:
    joint (single self.lut), independent+batched (self.lut_batched, the default since
    batched_multi_head_input=True), and independent+unbatched (self.luts, a ModuleList)."""
    if hasattr(m, "lut"):
        return [m.lut.weights]
    if hasattr(m, "lut_batched"):
        return [m.lut_batched.weights]
    return [lut.weights for lut in m.luts]


# ----------------------------- basics -----------------------------

def test_alias_is_same_class():
    assert CompressionMHL is CompressionMultiHeadLUT


def test_forward_shape_dtype():
    m = _mk()
    out = m(torch.randn(16, 32))
    assert out.shape == (16, 32)
    assert out.dtype == torch.float32


def test_grads_flow_through_all_three_parts():
    m = _mk()
    x = torch.randn(64, 32, requires_grad=True)
    m(x).pow(2).mean().backward()
    assert m.compress.weight.grad is not None and m.compress.weight.grad.abs().sum() > 0
    assert m.decompress.weight.grad is not None and m.decompress.weight.grad.abs().sum() > 0
    for w in _lut_tables(m):
        assert w.grad is not None and w.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_input_shape_validation():
    m = _mk()
    with pytest.raises(ValueError):
        m(torch.randn(4, 999))
    with pytest.raises(ValueError):
        m(torch.randn(4, 5, 32))          # must be 2-D [N, input_dim]


def test_multihead_output_is_output_dim():
    for joint in (True, False):
        out = _mk(n_heads=3, joint_head_compression=joint)(torch.randn(10, 32))
        assert out.shape == (10, 32)      # per-head outputs summed -> output_dim


def test_determinism():
    m = _mk()
    x = torch.randn(8, 32)
    with torch.no_grad():
        assert torch.equal(m(x), m(x))


# ----------------------------- inner_residual -----------------------------

def test_inner_residual_shapes_and_grads():
    m = _mk(inner_residual=True)
    assert m.inner_residual is True
    x = torch.randn(64, 32, requires_grad=True)
    out = m(x)
    assert out.shape == (64, 32)
    out.pow(2).mean().backward()
    assert m.compress.weight.grad.abs().sum() > 0
    assert m.decompress.weight.grad.abs().sum() > 0
    for w in _lut_tables(m):
        assert w.grad is not None and w.grad.abs().sum() > 0
    assert x.grad is not None and x.grad.abs().sum() > 0


def test_inner_residual_adds_zero_params():
    off = sum(p.numel() for p in _mk(inner_residual=False).parameters())
    on = sum(p.numel() for p in _mk(inner_residual=True).parameters())
    assert off == on                              # the inner skip is parameter-free


def test_inner_residual_changes_output():
    # n_heads=1 (default): compress -> [N, inner]; the skip adds z, reaching decompress via
    # both the skip and the lut path. Toggling the flag must change the output by exactly
    # decompress applied to the skipped z.
    m = _mk(inner_residual=True)
    x = torch.randn(16, 32)
    with torch.no_grad():
        out_on = m(x)
        m.inner_residual = False
        out_off = m(x)
    assert not torch.allclose(out_on, out_off)
    with torch.no_grad():
        z = m.compress(x)
        assert torch.allclose(out_on - out_off, m.decompress(z) - m.decompress.bias, atol=1e-4)


# ----------------------------- joint_head_compression -----------------------------

def test_joint_head_compression_default_is_false():
    assert _mk().joint_head_compression is False
    assert _mk(joint_head_compression=True).joint_head_compression is True


def test_nheads1_identical_output_both_modes():
    # INVARIANT: at n_heads=1 the two modes are numerically identical (so exp036-exp039,
    # which use n_heads=1, are unaffected by the default flip to independent).
    x = torch.randn(16, 32)
    torch.manual_seed(123)
    m_joint = CompressionMultiHeadLUT(32, 32, 8, nap=4, tph=6, n_heads=1,
                                      joint_head_compression=True, use_bf16=False, random_seed=7)
    torch.manual_seed(123)
    m_indep = CompressionMultiHeadLUT(32, 32, 8, nap=4, tph=6, n_heads=1,
                                      joint_head_compression=False, use_bf16=False, random_seed=7)
    with torch.no_grad():
        assert torch.allclose(m_joint(x), m_indep(x), atol=1e-6)
    # ...and inner_residual=True keeps the identity at n_heads=1 too
    torch.manual_seed(5)
    a = CompressionMultiHeadLUT(32, 32, 8, nap=4, tph=6, n_heads=1, inner_residual=True,
                                joint_head_compression=True, use_bf16=False, random_seed=3)
    torch.manual_seed(5)
    b = CompressionMultiHeadLUT(32, 32, 8, nap=4, tph=6, n_heads=1, inner_residual=True,
                                joint_head_compression=False, use_bf16=False, random_seed=3)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6)


def test_nheads_gt1_shapes_and_grads_both_modes():
    x = torch.randn(12, 32, requires_grad=True)
    for joint in (True, False):
        m = _mk(n_heads=3, joint_head_compression=joint)
        out = m(x if x.grad is None else x.detach().requires_grad_(True))
        assert out.shape == (12, 32)
        m.zero_grad()
        xx = torch.randn(12, 32, requires_grad=True)
        m(xx).pow(2).mean().backward()
        for w in _lut_tables(m):
            assert w.grad is not None and w.grad.abs().sum() > 0
        assert m.compress.weight.grad.abs().sum() > 0
        assert m.decompress.weight.grad.abs().sum() > 0
        assert xx.grad is not None and xx.grad.abs().sum() > 0


def test_param_count_matches_module_both_modes():
    cases = [(8, 4, 6, 1), (16, 6, 10, 1), (8, 5, 7, 3), (12, 6, 9, 2)]
    for inner, nap, tph, nh in cases:
        for joint in (True, False):
            m = CompressionMultiHeadLUT(32, 32, inner, nap=nap, tph=tph, n_heads=nh,
                                        joint_head_compression=joint, use_bf16=False,
                                        learnable_temps=False, random_seed=0)
            measured = sum(p.numel() for p in m.parameters())
            f = CompressionMultiHeadLUT.param_count(32, 32, inner, nap=nap, tph=tph,
                                                    n_heads=nh, joint_head_compression=joint)
            assert measured == f["total"], (inner, nap, tph, nh, joint, measured, f)
            assert f["lut"] == nh * tph * (2 ** nap) * inner
            if joint:
                assert f["compress"] == 32 * inner + inner
                assert f["decompress"] == inner * 32 + 32
            else:
                assert f["compress"] == 32 * (nh * inner) + nh * inner
                assert f["decompress"] == (nh * inner) * 32 + 32
    # at n_heads=1 the two modes have identical param counts
    for inner, nap, tph in [(8, 4, 6), (16, 6, 10)]:
        pj = CompressionMultiHeadLUT.param_count(32, 32, inner, nap=nap, tph=tph, n_heads=1,
                                                 joint_head_compression=True)["total"]
        pi = CompressionMultiHeadLUT.param_count(32, 32, inner, nap=nap, tph=tph, n_heads=1,
                                                 joint_head_compression=False)["total"]
        assert pj == pi


# ----------------------------- separate inner_in/inner_out + -1 no-projection -----------------------------

def test_inner_dim_shim_equals_explicit_in_out():
    x = torch.randn(8, 32)
    torch.manual_seed(1)
    a = CompressionMultiHeadLUT(32, 32, inner_dim=8, nap=4, tph=6, use_bf16=False, random_seed=2)
    torch.manual_seed(1)
    b = CompressionMultiHeadLUT(32, 32, inner_in_dim=8, inner_out_dim=8, nap=4, tph=6,
                                use_bf16=False, random_seed=2)
    with torch.no_grad():
        assert torch.allclose(a(x), b(x), atol=1e-6)
    assert sum(p.numel() for p in a.parameters()) == sum(p.numel() for p in b.parameters())


def test_conflicting_or_missing_inner_args_raise():
    with pytest.raises(ValueError):                       # both shim and explicit
        CompressionMultiHeadLUT(32, 32, 8, inner_in_dim=8, nap=4, tph=6)
    with pytest.raises(ValueError):                       # neither given
        CompressionMultiHeadLUT(32, 32, nap=4, tph=6)
    with pytest.raises(ValueError):                       # only one explicit
        CompressionMultiHeadLUT(32, 32, inner_in_dim=8, nap=4, tph=6)


def test_separate_in_out_dims_shapes_grads_params():
    for joint in (True, False):
        m = CompressionMultiHeadLUT(32, 40, inner_in_dim=8, inner_out_dim=12, nap=4, tph=6,
                                    n_heads=2, joint_head_compression=joint,
                                    use_bf16=False, learnable_temps=False, random_seed=0)
        x = torch.randn(16, 32, requires_grad=True)
        out = m(x)
        assert out.shape == (16, 40)
        out.pow(2).mean().backward()
        assert m.compress.weight.grad.abs().sum() > 0
        assert m.decompress.weight.grad.abs().sum() > 0
        for w in _lut_tables(m):
            assert w.grad is not None and w.grad.abs().sum() > 0
        assert x.grad is not None and x.grad.abs().sum() > 0
        f = CompressionMultiHeadLUT.param_count(32, 40, inner_in_dim=8, inner_out_dim=12,
                                                nap=4, tph=6, n_heads=2,
                                                joint_head_compression=joint)
        assert sum(p.numel() for p in m.parameters()) == f["total"]


def test_minus1_no_projection_shapes_grads_params():
    cases = [dict(inner_in_dim=-1, inner_out_dim=8),    # no compress
             dict(inner_in_dim=8, inner_out_dim=-1),    # no decompress
             dict(inner_in_dim=-1, inner_out_dim=-1)]   # pure FastMHL slot
    for joint in (True, False):
        for kw in cases:
            m = CompressionMultiHeadLUT(32, 32, nap=4, tph=6, n_heads=2,
                                        joint_head_compression=joint, use_bf16=False,
                                        learnable_temps=False, random_seed=0, **kw)
            x = torch.randn(12, 32, requires_grad=True)
            out = m(x)
            assert out.shape == (12, 32), (joint, kw)
            out.pow(2).mean().backward()
            for w in _lut_tables(m):
                assert w.grad is not None and w.grad.abs().sum() > 0
            assert x.grad is not None and x.grad.abs().sum() > 0
            if kw["inner_in_dim"] == -1:
                assert isinstance(m.compress, torch.nn.Identity)
            else:
                assert m.compress.weight.grad.abs().sum() > 0
            if kw["inner_out_dim"] == -1:
                assert isinstance(m.decompress, torch.nn.Identity)
            else:
                assert m.decompress.weight.grad.abs().sum() > 0
            f = CompressionMultiHeadLUT.param_count(32, 32, nap=4, tph=6, n_heads=2,
                                                    joint_head_compression=joint, **kw)
            assert sum(p.numel() for p in m.parameters()) == f["total"], (joint, kw)
            if kw["inner_in_dim"] == -1:
                assert f["compress"] == 0
            if kw["inner_out_dim"] == -1:
                assert f["decompress"] == 0


def test_both_minus1_equals_plain_fastmhl():
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    m = CompressionMultiHeadLUT(32, 32, inner_in_dim=-1, inner_out_dim=-1, nap=4, tph=6,
                                n_heads=1, use_bf16=False, random_seed=9)
    lut = FastMultiHeadLut(input_dim=32, n_heads=1, n_outputs=32, n_anchor_pairs=4,
                           tables_per_head=6, forward_mode="hard", use_bf16=False, random_seed=9)
    x = torch.randn(8, 32)
    with torch.no_grad():
        assert torch.allclose(m(x), lut(x).sum(dim=1), atol=1e-6)
    # no projection params at all
    f = CompressionMultiHeadLUT.param_count(32, 32, inner_in_dim=-1, inner_out_dim=-1,
                                            nap=4, tph=6, n_heads=1)
    assert f["compress"] == 0 and f["decompress"] == 0


def test_inner_residual_dim_validation():
    with pytest.raises(ValueError):                      # mismatched in/out
        CompressionMultiHeadLUT(32, 32, inner_in_dim=8, inner_out_dim=12, nap=4, tph=6,
                                inner_residual=True)
    # matching (incl -1/-1 when input_dim==output_dim) works and flows grad
    for kw in [dict(inner_in_dim=8, inner_out_dim=8), dict(inner_in_dim=-1, inner_out_dim=-1)]:
        m = CompressionMultiHeadLUT(32, 32, nap=4, tph=6, inner_residual=True,
                                    use_bf16=False, random_seed=0, **kw)
        x = torch.randn(8, 32, requires_grad=True)
        out = m(x)
        assert out.shape == (8, 32)
        out.pow(2).mean().backward()
        assert x.grad is not None and x.grad.abs().sum() > 0
