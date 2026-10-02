"""Single-anchor addressing mode (``anchor_mode="single"``) across every cartridge.

Single mode replaces each table's two-coordinate sign test ``[z[a_j] - z[b_j] > eps]`` with
a single coordinate vs zero, ``[z[a_j] > eps]``. The two-cell structure, routing, and
uncertainty backward are untouched, so every cartridge supports it. Composed with a learned
input projection (a :class:`ProjectionMHL` compress with bias, ``z = W x + b``) each bit
becomes a learned hyperplane ``[⟨w_j, x⟩ + b_j > 0]`` — exactly the old
``HyperplaneMultiHeadLUT`` front-end, which the final test reproduces bit-for-bit.
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
    proj.train()
    torch.manual_seed(7)
    x = torch.randn(16, 10, requires_grad=True)
    y = proj(x)
    assert y.shape == (16, 10)
    y.pow(2).sum().backward()
    assert proj.compress.weight.grad.abs().sum() > 0, "input projection got no gradient"
    assert cart.weights.grad.abs().sum() > 0, "LUT table got no gradient"
    assert proj.decompress.weight.grad.abs().sum() > 0, "decompress got no gradient"


# ------------------------------------------------- HyperplaneMHL redundancy (the deliverable)

def _build_hyperplane_equivalent(H, O, nap, tph, seed=0, noise=0.5):
    """Construct a HyperplaneMultiHeadLUT and a ProjectionMHL(single-anchor) that match it.

    ``d_in = tph*nap`` makes each group's canonical single-anchor coverage a permutation of
    ``range(d_in)`` (a bijection (table,bit) -> coordinate), and ``d_model = H*O`` lets the
    decompress be the identity, so the ProjectionMHL output IS the cartridge output. The
    input projection (compress, with bias) carries the hyperplane weights/biases to the
    coordinate each (table, bit) addresses, so ``z[coord] = ⟨w,x⟩ + b`` and the single-anchor
    sign bit ``[z[coord] > 0]`` equals the hyperplane bit ``[⟨w,x⟩ + b > 0]``.
    """
    from spiky.lutorch.hyperplane_multi_head_lut import HyperplaneMultiHeadLUT

    d_in = tph * nap
    D = H * O
    K = 1 << nap
    hyper = HyperplaneMultiHeadLUT(
        input_dim=D, n_heads=H, n_outputs=O, n_anchor_pairs=nap, tables_per_head=tph,
        forward_mode="hard", use_bf16=False, hyperplane_init="random",
        weight_dtype=torch.float32, hyperplane_dtype=torch.float32,
        random_seed=seed, initial_weights_noise=noise,
    )
    spec = LUTSpec(h_in=H, h_out=H, tph=tph, nap=nap, d_in=d_in, d_out=O, anchor_mode="single")
    cart = ManifestoHardLUT(spec, seed=seed)
    # LUT table weights: table_global = g*tph + t (head-major), same flat order both sides.
    cart.weights.data = hyper.weights.data.view(H, tph, K, O).clone()
    proj = ProjectionMHL(cart, d_model=D, compress=True, decompress=False, bias=True)
    Wc = torch.zeros(H * d_in, D)
    bc = torch.zeros(H * d_in)
    for g in range(H):
        for t in range(tph):
            tg = g * tph + t
            for j in range(nap):
                coord = int(cart.anchor_a[g, t, j])
                Wc[g * d_in + coord] = hyper.hyperplane_weight[tg, j]
                bc[g * d_in + coord] = hyper.hyperplane_bias[tg, j]
    proj.compress.weight.data = Wc
    proj.compress.bias.data = bc
    return hyper, proj, (H, O)


@pytest.mark.parametrize("H,O,nap,tph", [(2, 3, 3, 2), (1, 4, 4, 1), (3, 2, 2, 3)])
def test_projection_single_reproduces_hyperplane_mhl(H, O, nap, tph):
    """ProjectionMHL(big input projection + single-anchor canonical coverage) reproduces the
    HyperplaneMultiHeadLUT hard output exactly — the evidence it can be dropped later."""
    hyper, proj, (H, O) = _build_hyperplane_equivalent(H, O, nap, tph)
    # Canonical single coverage is a per-group permutation (d_in == tph*nap), so the
    # (table, bit) -> coordinate map used above is a bijection.
    for g in range(H):
        assert sorted(proj.cartridge.anchor_a[g].reshape(-1).tolist()) == list(range(tph * nap))
    hyper.eval(); proj.eval()
    torch.manual_seed(11)
    x = torch.randn(128, H * O)
    with torch.no_grad():
        yh = hyper(x)                           # [B, H, O]
        yp = proj(x).view(128, H, O)            # compress -> single-anchor cartridge -> identity
    assert torch.allclose(yh, yp, atol=1e-6), (yh - yp).abs().max().item()
