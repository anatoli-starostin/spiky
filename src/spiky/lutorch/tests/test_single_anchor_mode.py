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


def test_unique_partition_anchor():
    """anchor_unique_partition: n_tables*nap == pool_size, indices are a permutation of
    [0,pool_size) (every hyperplane used exactly once)."""
    pool, nt, nap = 384, 48, 8
    m = LightMultiHeadLUT(
        input_dim=pool, n_tables=nt, output_dim=pool, n_anchor_pairs=nap,
        confidence_form="margin", random_seed=1, device=torch.device("cpu"),
        n_heads=1, multi_head_input=False, anchor_mode="single", pool_size=pool,
        anchor_unique_partition=True,
    )
    assert tuple(m.anchor_c.shape) == (nt, nap)
    flat = m.anchor_c.flatten().sort().values
    assert torch.equal(flat, torch.arange(pool))              # genuine partition of [0,384)
    x = torch.randn(3, pool)
    assert tuple(m(x).shape) == (3, pool)                     # 48 tables summed -> [B, pool]
    # mismatch (n_tables*nap != pool_size) must raise
    import pytest
    with pytest.raises(ValueError):
        LightMultiHeadLUT(input_dim=384, n_tables=40, output_dim=384, n_anchor_pairs=8,
                          device=torch.device("cpu"), anchor_mode="single", pool_size=384,
                          anchor_unique_partition=True)


def _mk_gated(mode, din=8, mh=True):
    return LightMultiHeadLUT(
        input_dim=din, n_tables=(2 if mh else 1) * 4, output_dim=din, n_anchor_pairs=5,
        confidence_form="margin", random_seed=2, device=torch.device("cpu"),
        n_heads=(2 if mh else 1), multi_head_input=mh, cell_mode=mode)


def test_gated_tables_doubled_and_dim_guard():
    import pytest
    m = _mk_gated("gated_affine", din=8)
    assert m.cell_mode == "gated_affine" and m._gated
    assert tuple(m.tables.shape) == (8, 32, 16)          # last dim = 2*output_dim (u|v)
    c = _mk_gated("constant", din=8)
    assert tuple(c.tables.shape) == (8, 32, 8)           # unchanged
    with pytest.raises(ValueError):                       # gated requires d_in==d_out
        LightMultiHeadLUT(input_dim=8, n_tables=4, output_dim=6, n_anchor_pairs=4,
                          device=torch.device("cpu"), cell_mode="gated_affine")


def test_apply_cell_is_u_plus_v_times_x():
    """The core gated math: bag=[U|V], out = U + V*x (affine) / V*x (multiply), elementwise."""
    for mode in ("gated_affine", "gated_multiply"):
        m = _mk_gated(mode, din=4)
        U = torch.tensor([[1., 2., 3., 4.]]); V = torch.tensor([[10., 20., 30., 40.]])
        x = torch.tensor([[0.5, -1., 2., 0.]])
        bagged = torch.cat([U, V], dim=-1)               # [1, 8]
        got = m._apply_cell(bagged, x)
        want = (U + V * x) if mode == "gated_affine" else (V * x)
        assert torch.allclose(got, want)
    # constant mode returns the bag unchanged
    c = _mk_gated("constant", din=4)
    b = torch.randn(1, 4)
    assert torch.equal(c._apply_cell(b, torch.randn(1, 4)), b)


def test_margin_readout_is_v_plus_W_times_signed_margins():
    """margin_readout: output = v_c + W_t · m with SIGNED margins m = x[a]-x[b].
    Zero the bias store v so the output isolates term2 = Σ_t score_t (W_t · m_t); compare to
    a hand-computed einsum from the module's own W, margins, and score."""
    import torch
    from spiky.lutorch.fast_multi_head_lut import _confidence_score
    nap, nt, din, dout = 5, 6, 8, 4
    m = LightMultiHeadLUT(
        input_dim=din, n_tables=nt, output_dim=dout, n_anchor_pairs=nap,
        confidence_form="margin", random_seed=4, device=torch.device("cpu"),
        n_heads=1, multi_head_input=False, cell_mode="margin_readout", margin_signed=True)
    assert m.cell_mode == "margin_readout" and m._margin
    assert tuple(m.margin_W.shape) == (nt, dout, nap)     # per-table W [n_tables, d_out, nap]
    with torch.no_grad(): m.tables.zero_()                 # isolate the margin term
    x = torch.randn(3, din)
    out = m(x)
    # reference: SIGNED margins, score = confidence(|d|), term2 = Σ_t score·(W_t·d_t)
    d = x[:, m.anchor_a] - x[:, m.anchor_b]                # [B, nt, nap] signed
    score = _confidence_score(d, "margin", 1.0)            # [B, nt]
    Wm = torch.einsum('tdn,btn->btd', m.margin_W, d)
    ref = torch.einsum('bt,btd->bd', score, Wm)
    assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()
    # signed vs abs really differ
    m2 = LightMultiHeadLUT(input_dim=din, n_tables=nt, output_dim=dout, n_anchor_pairs=nap,
        confidence_form="margin", random_seed=4, device=torch.device("cpu"),
        n_heads=1, multi_head_input=False, cell_mode="margin_readout", margin_signed=False)
    with torch.no_grad(): m2.tables.zero_()
    assert not torch.allclose(m2(x), out)                  # |m| path differs from signed


def test_codebook_readout_is_M_times_score_weighted_scalars():
    """codebook: each cell stores a SCALAR w_c; per table g_t = s_t·w_{c_t} (NO cross-table
    sum); y = M · g with M [d_model, n_tables]. Assert y equals that, from the module's own
    tensors, and that the store is scalar-shaped."""
    import torch
    from spiky.lutorch.fast_multi_head_lut import _confidence_score
    H, tph, nap, din, D = 2, 4, 6, 16, 10
    nt, ts = H * tph, 1 << nap
    m = LightMultiHeadLUT(
        input_dim=din, n_tables=nt, output_dim=din, n_anchor_pairs=nap,
        confidence_form="margin", random_seed=5, device=torch.device("cpu"),
        n_heads=H, multi_head_input=True, cell_mode="codebook", codebook_out_dim=D)
    assert m.cell_mode == "codebook" and m._codebook
    assert tuple(m.tables.shape) == (nt, ts, 1)             # SCALAR store (r=1)
    assert tuple(m.codebook_M.shape) == (D, nt)             # M [d_model, T]
    B = 3
    x = torch.randn(B, H, din)
    out = m(x)
    assert tuple(out.shape) == (B, D)                       # decoded to d_model, no head axis
    # reference: signed margins -> address + score, scalar gather, g=s*w, y = g @ M^T
    T = tph
    ia = m.anchor_a.reshape(1, H, T * nap).expand(B, H, T * nap)
    ib = m.anchor_b.reshape(1, H, T * nap).expand(B, H, T * nap)
    d = (torch.gather(x, 2, ia) - torch.gather(x, 2, ib)).view(B, H, T, nap)
    idx = m._pack_index(x.reshape(B, H * din), d).view(B, H, T)
    score = _confidence_score(d, "margin", 1.0)             # [B,H,T]
    flat = m.tables.reshape(nt * ts, 1)
    gcell = (idx + m.table_offset.view(1, H, T)).reshape(-1)
    w = flat[gcell].view(B, H, T)
    g = (score * w).reshape(B, H * T)                       # [B, n_tables]
    ref = g @ m.codebook_M.t()
    assert torch.allclose(out, ref, atol=1e-6), (out - ref).abs().max()
    # non-multi-head codebook must be rejected (mh-only)
    import pytest
    with pytest.raises(NotImplementedError):
        LightMultiHeadLUT(input_dim=din, n_tables=8, output_dim=din, n_anchor_pairs=nap,
                          device=torch.device("cpu"), n_heads=1, multi_head_input=False,
                          cell_mode="codebook", codebook_out_dim=D)


def test_gated_forward_runs_and_grad():
    for mh in (True, False):
        m = _mk_gated("gated_affine", din=8, mh=mh)
        x = (torch.randn(3, m.n_heads, 8) if mh else torch.randn(3, 8)).requires_grad_(True)
        out = m(x)
        assert out.shape == ((3, m.n_heads, 8) if mh else (3, 8))
        out.sum().backward()
        assert m.tables.grad is not None and torch.isfinite(m.tables.grad).all()
        assert x.grad is not None and torch.isfinite(x.grad).all()   # grad via BOTH score and V⊙x
