"""Single-anchor addressing mode (``anchor_mode="single"``) across every cartridge.

Single mode replaces each table's two-coordinate sign test ``[z[a_j] - z[b_j] > eps]`` with
a single coordinate vs zero, ``[z[a_j] > eps]``. The two-cell structure, routing, and
uncertainty backward are untouched, so every cartridge supports it. Composed with a learned
input projection (a :class:`ProjectionMHL` compress with bias, ``z = W x + b``) each bit
becomes a learned hyperplane ``[⟨w_j, x⟩ + b_j > 0]``. The final test checks that end to end
against an independent torch reference, in both anchor modes.
"""
import pytest
import torch

from spiky.lutorch_ex import (
    FusedManifestoHardLUT,
    FusedManifestoSoftLUT,
    LUTSpec,
    ManifestoHardLUT,
    ManifestoSoftLUT,
    ProjectionMHL,
)
from spiky.lutorch_ex.addressing import msb_first_powers

ALL_CARTRIDGES = [ManifestoHardLUT, ManifestoSoftLUT, FusedManifestoHardLUT, FusedManifestoSoftLUT]
PURE_FUSED = [(ManifestoHardLUT, FusedManifestoHardLUT), (ManifestoSoftLUT, FusedManifestoSoftLUT)]
PATTERNS = [(4, 4), (1, 4), (4, 1)]


# ---------------------------------------------------------------------------- spec / build

def test_spec_rejects_bad_anchor_mode():
    with pytest.raises(ValueError):
        LUTSpec(h_in=2, h_out=2, tph=2, nap=2, d_in=4, d_out=3, anchor_mode="triple")


def test_spec_default_is_pairs():
    assert LUTSpec(h_in=1, h_out=1, tph=1, nap=1, d_in=2, d_out=2).anchor_mode == "pairs"


@pytest.mark.parametrize("Cls", ALL_CARTRIDGES)
def test_single_mode_has_no_anchor_b_and_right_shapes(Cls):
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=3, d_in=5, d_out=4, anchor_mode="single")
    m = Cls(spec)
    assert m.single is True
    assert m.anchor_b is None, "single mode must not carry an anchor_b buffer"
    assert tuple(m.anchor_a.shape) == (2, 3, 3)
    assert bool((m.anchor_a >= 0).all()) and bool((m.anchor_a < 5).all())


def test_single_mode_allows_d_in_1_but_pairs_needs_2():
    # A single anchor vs zero needs only one coordinate; a pair needs two.
    ManifestoHardLUT(LUTSpec(h_in=1, h_out=1, tph=1, nap=1, d_in=1, d_out=2, anchor_mode="single"))
    with pytest.raises(ValueError):
        ManifestoHardLUT(LUTSpec(h_in=1, h_out=1, tph=1, nap=1, d_in=1, d_out=2, anchor_mode="pairs"))


# ---------------------------------------------------------------------------- addressing

def test_addressing_is_single_coordinate_vs_zero():
    """The packed cell index equals the MSB-first pack of ``[z[a_j] > 0]`` (single vs zero)."""
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=4, d_in=7, d_out=5, anchor_mode="single")
    m = ManifestoHardLUT(spec, seed=1)
    torch.manual_seed(0)
    x = torch.randn(9, 2, 7)
    z, u, c, j_star, u_abs_star, c_alt = m._addresses(x)
    # margin is the coordinate itself, not a difference.
    assert torch.equal(z, x)  # per-head routing is identity when h_in == h_out
    expect_u = z.gather(2, m.anchor_a.reshape(1, 2, 12).expand(9, 2, 12)).reshape(9, 2, 3, 4)
    assert torch.allclose(u, expect_u)
    powers = msb_first_powers(4)
    expect_c = ((expect_u > 0).to(torch.long) * powers).sum(-1)
    assert torch.equal(c, expect_c)
    # neighbour differs from c by exactly the least-confident bit.
    assert torch.equal(c_alt, c ^ powers[j_star])


# ---------------------------------------------------------------------------- all cartridges

@pytest.mark.parametrize("Cls", ALL_CARTRIDGES)
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
def test_every_cartridge_single_forward_eval_grad(Cls, h_in, h_out):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=4, nap=3, d_in=6, d_out=5, anchor_mode="single")
    m = Cls(spec, seed=0, weight_init_std=1.0)
    torch.manual_seed(3)
    x = torch.randn(32, h_in, 6, requires_grad=True)
    m.train()
    y = m(x)
    assert y.shape == (32, h_out, 5)
    gx, gw = torch.autograd.grad(y.pow(2).sum(), (x, m.weights))
    assert torch.isfinite(gx).all() and torch.isfinite(gw).all()
    assert gx.abs().sum() > 0 and gw.abs().sum() > 0
    m.eval()
    with torch.no_grad():
        assert m(x).shape == (32, h_out, 5)


@pytest.mark.parametrize("PureCls,FusedCls", PURE_FUSED)
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("backend", ["auto", "tier1"])
def test_single_fused_equals_pure_f64(PureCls, FusedCls, h_in, h_out, backend):
    """Fused single-mode cartridges are bit-exact (fp64) against the pure oracle."""
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=4, nap=4, d_in=9, d_out=6, anchor_mode="single")
    pure = PureCls(spec, seed=0, weight_init_std=1.0).double()
    fused = FusedCls(spec, seed=0, weight_init_std=1.0, backend=backend).double()
    pure.train(); fused.train()
    torch.manual_seed(5)
    x = torch.randn(64, h_in, 9, dtype=torch.float64)
    xp = x.clone().requires_grad_(True); xf = x.clone().requires_grad_(True)
    yp, yf = pure(xp), fused(xf)
    assert torch.allclose(yp, yf, atol=1e-10)
    go = torch.randn_like(yp)
    gxp, gwp = torch.autograd.grad(yp, (xp, pure.weights), go, retain_graph=True)
    gxf, gwf = torch.autograd.grad(yf, (xf, fused.weights), go)
    assert torch.allclose(gxp, gxf, atol=1e-9)
    assert torch.allclose(gwp, gwf, atol=1e-9)


def test_projection_mhl_wraps_single_cartridge_and_trains():
    """ProjectionMHL ("big input projection") composes with a single-anchor cartridge and
    propagates gradient into the input projection, the LUT table, and decompress."""
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=3, d_in=6, d_out=4, anchor_mode="single")
    cart = ManifestoHardLUT(spec, seed=0, weight_init_std=1.0)
    proj = ProjectionMHL(cart, d_model=10, compress=True, decompress=True, bias=True)
    # The default zero-inits decompress (the FFN starts as a zero contribution), which gates first-step gradient to the input side; nudge it off zero so this test
    # exercises what it is about - that gradient reaches the input projection and the LUT table.
    torch.nn.init.normal_(proj.decompress.weight, std=0.02)
    proj.train()
    torch.manual_seed(7)
    x = torch.randn(16, 10, requires_grad=True)
    y = proj(x)
    assert y.shape == (16, 10)
    y.pow(2).sum().backward()
    assert proj.compress.weight.grad.abs().sum() > 0, "input projection got no gradient"
    assert cart.weights.grad.abs().sum() > 0, "LUT table got no gradient"
    assert proj.decompress.weight.grad.abs().sum() > 0, "decompress got no gradient"


# ------------------------------------------------- end-to-end value check against a reference

def _reference_hard_output(x, Wc, bc, anchor_a, anchor_b, tables):
    """Independent plain-torch reference for ProjectionMHL(compress -> hard cartridge), decompress off.

    The compress row feeding head g's coordinate i is ``g * d_in + i`` (head-major). Each (head g,
    table t, bit j) is an affine function of x: a = <w, x> + b, with (w, b) the compress row of its
    anchor (single), or the difference of its two anchors' rows (pairs). Then bits = a > 0, MSB-first
    packing, and each head sums its tables' rows: y[:, g] = sum_t W[g, t, idx]."""
    G, tph, nap = anchor_a.shape
    d_in = Wc.shape[0] // G
    rows_a = anchor_a + d_in * torch.arange(G).view(G, 1, 1)                      # [G, tph, nap]
    w, b = Wc[rows_a], bc[rows_a]
    if anchor_b is not None:
        rows_b = anchor_b + d_in * torch.arange(G).view(G, 1, 1)
        w, b = w - Wc[rows_b], b - bc[rows_b]
    a = torch.einsum("bd,gtjd->bgtj", x, w) + b                                    # [B, G, tph, nap]
    idx = ((a > 0).long() * (1 << torch.arange(nap - 1, -1, -1))).sum(-1)          # [B, G, tph]
    return tables[torch.arange(G).view(1, G, 1), torch.arange(tph).view(1, 1, tph), idx].sum(2)


@pytest.mark.parametrize("anchor_mode", ["single", "pairs"])
@pytest.mark.parametrize("H,O,nap,tph", [(2, 3, 3, 2), (1, 4, 4, 1), (3, 2, 2, 3)])
def test_projection_end_to_end_matches_reference(H, O, nap, tph, anchor_mode):
    """ProjectionMHL(compress with bias -> ManifestoHardLUT, decompress off) equals an independent
    torch reference of the hyperplane-LSH hard read, in both anchor modes.

    ``d_in = tph*nap`` makes each group's canonical single-anchor coverage a permutation of
    ``range(d_in)`` (a bijection (table, bit) -> coordinate), so in single mode each compress row
    is set to exactly one seeded random hyperplane (w, b) and the bit is ``[<w, x> + b > 0]``. In
    pairs mode the compress rows are seeded random and each bit compares two of them. ``d_model =
    H*O`` with decompress off makes the ProjectionMHL output the cartridge output itself."""
    gen = torch.Generator().manual_seed(1000 * H + 100 * O + 10 * nap + tph)
    d_in, D = tph * nap, H * O
    spec = LUTSpec(h_in=H, h_out=H, tph=tph, nap=nap, d_in=d_in, d_out=O, anchor_mode=anchor_mode)
    cart = ManifestoHardLUT(spec, seed=0)
    cart.weights.data = torch.randn(cart.weights.shape, generator=gen)
    proj = ProjectionMHL(cart, d_model=D, compress=True, decompress=False, bias=True).eval()
    if anchor_mode == "single":
        for g in range(H):          # the canonical coverage is a per-group permutation of range(d_in)
            assert sorted(cart.anchor_a[g].reshape(-1).tolist()) == list(range(d_in))
        w = torch.randn(H, tph, nap, D, generator=gen)          # one hyperplane per (head, table, bit)
        b = torch.randn(H, tph, nap, generator=gen)
        Wc, bc = torch.zeros(H * d_in, D), torch.zeros(H * d_in)
        rows = cart.anchor_a + d_in * torch.arange(H).view(H, 1, 1)
        Wc[rows], bc[rows] = w, b
    else:
        Wc, bc = torch.randn(H * d_in, D, generator=gen), torch.randn(H * d_in, generator=gen)
    proj.compress.weight.data, proj.compress.bias.data = Wc, bc
    x = torch.randn(128, D, generator=gen)
    with torch.no_grad():
        y = proj(x).view(128, H, O)
    ref = _reference_hard_output(x, Wc, bc, cart.anchor_a, cart.anchor_b, cart.weights.detach())
    assert torch.allclose(y, ref, atol=1e-5), (y - ref).abs().max().item()
