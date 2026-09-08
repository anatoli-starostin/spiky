"""Tests for the single-anchor addressing mode on LightMultiHeadLUT.

anchor_mode="single": each address bit is sign(x[c_i]) (one pooled coordinate) instead
of the default pair difference sign(x[a_i]-x[b_i]). Everything downstream is identical.
These check (1) the single path constructs with the right shapes and disables the
pair-only native kernel, (2) its addressing really is the single-coordinate sign,
(3) grad flows to tables and to x, (4) the default pair path is untouched.
"""
import torch
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT


def _mk(anchor_mode, mh=True, seed=1, nap=8, tph=8, H=4, din=48, dout=48):
    return LightMultiHeadLUT(
        input_dim=din, n_tables=H * tph, output_dim=dout, n_anchor_pairs=nap,
        confidence_form="margin", random_seed=seed, device=torch.device("cpu"),
        n_heads=H, multi_head_input=mh, anchor_mode=anchor_mode,
    )


def test_single_constructs_with_right_shapes():
    nap, tph, H, din = 8, 8, 4, 48
    m = _mk("single", nap=nap, tph=tph, H=H, din=din)
    assert m.anchor_mode == "single"
    assert m.table_size == (1 << nap) == 256
    assert m.pool_size == din
    assert hasattr(m, "anchor_c") and not hasattr(m, "anchor_a")
    assert tuple(m.anchor_c.shape) == (H, tph, nap)          # [H, T, NAP]
    assert int(m.anchor_c.max()) < din and int(m.anchor_c.min()) >= 0
    assert tuple(m.tables.shape) == (H * tph, 256, din)
    # pair-only native kernel must be OFF in single mode
    assert m._native_msb is None and m._native_msb_scored is None


def test_single_addressing_is_single_coord_sign():
    """The packed index must equal MSB-pack of sign(x[c]) for the single path."""
    torch.manual_seed(0)
    m = _mk("single", mh=True, nap=6, tph=4, H=2, din=16)
    B, H, din = 3, m.n_heads, m.input_dim
    x = torch.randn(B, H, din)
    # reference: per (b,h,t) index = sum_j (x[b,h,c_{h,t,j}]>0) * 2^(NAP-1-j)
    T, NAP = m.tables_per_head, m.n_anchor_pairs
    idx_c = m.anchor_c.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    d = torch.gather(x, 2, idx_c).view(B, H, T, NAP)
    ref = ((d > 0).to(torch.int64) * m.powers.view(1, 1, 1, -1)).sum(-1)   # [B,H,T]
    got = m._pack_index(x.reshape(B, H * din), d).view(B, H, T)
    assert torch.equal(ref, got)


def test_single_forward_and_grad():
    for mh in (True, False):
        m = _mk("single", mh=mh, nap=6, tph=4, H=(2 if mh else 1), din=16, dout=16)
        if mh:
            x = torch.randn(5, m.n_heads, m.input_dim, requires_grad=True)
        else:
            x = torch.randn(5, m.input_dim, requires_grad=True)
        out = m(x)
        assert out.shape[0] == 5 and out.shape[-1] == m.output_dim
        out.sum().backward()
        assert m.tables.grad is not None and torch.isfinite(m.tables.grad).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()   # via score(|d|)


def test_single_is_deterministic_in_seed():
    a = _mk("single", seed=7, nap=6, tph=4, H=2, din=16)
    b = _mk("single", seed=7, nap=6, tph=4, H=2, din=16)
    assert torch.equal(a.anchor_c, b.anchor_c)
    x = torch.randn(4, 2, 16)
    assert torch.equal(a(x), b(x))


def test_pair_mode_untouched():
    """Default pair path still constructs and runs (regression guard)."""
    m = _mk("pair", nap=6, tph=4, H=2, din=16, dout=16)
    assert m.anchor_mode == "pair"
    assert hasattr(m, "anchor_a") and hasattr(m, "anchor_b") and not hasattr(m, "anchor_c")
    x = torch.randn(4, 2, 16)
    out = m(x)
    assert out.shape == (4, 2, 16)
    # pair addressing reference: sign(x[a]-x[b])
    B, H, din = 4, 2, 16
    T, NAP = m.tables_per_head, m.n_anchor_pairs
    ia = m.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    ib = m.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    d = (torch.gather(x, 2, ia) - torch.gather(x, 2, ib)).view(B, H, T, NAP)
    ref = ((d > 0).to(torch.int64) * m.powers.view(1, 1, 1, -1)).sum(-1)
    got = m._pack_index(x.reshape(B, H * din), d).view(B, H, T)
    assert torch.equal(ref, got)


def test_single_pool_size_mismatch_raises():
    import pytest
    with pytest.raises(NotImplementedError):
        LightMultiHeadLUT(input_dim=48, n_tables=8, output_dim=48, n_anchor_pairs=6,
                          device=torch.device("cpu"), anchor_mode="single", pool_size=96)


def test_output_heads_global_addressing_and_grouping():
    """output_heads>1: global single-index addressing over the FULL input, output grouped
    into G head-bags -> [B, G, output_dim]; the heads partition the whole-ensemble sum."""
    G, tph, nap, pool, dout = 8, 4, 6, 32, 5
    m = LightMultiHeadLUT(
        input_dim=pool, n_tables=G * tph, output_dim=dout, n_anchor_pairs=nap,
        confidence_form="margin", random_seed=3, device=torch.device("cpu"),
        n_heads=1, multi_head_input=False, anchor_mode="single",
        pool_size=pool, output_heads=G,
    )
    assert m.output_heads == G and m.anchor_mode == "single"
    assert tuple(m.anchor_c.shape) == (G * tph, nap)          # GLOBAL indices, not per-head
    x = torch.randn(4, pool)
    y = m(x)
    assert tuple(y.shape) == (4, G, dout)                    # per-head grouped output
    m.output_heads = 1                                       # read as one ensemble
    full = m(x)                                              # [4, dout] = sum of ALL tables
    m.output_heads = G
    assert torch.allclose(y.sum(dim=1), full, atol=1e-5)     # heads partition the ensemble sum
