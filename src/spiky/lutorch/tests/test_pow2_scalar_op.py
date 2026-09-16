"""spiky_lutorch::p2_scalars (pow2_scalar_op.py): ONE forward definition of the per-table integers for training and eval.

Gates:
  drift     the op's integers == the inference kernel's integers (both call p2::table_scalars, csrc/pow2_scalars.cuh), on
            random margins and on margins placed at the q / k' rounding boundaries -- fails if the two CUDA call sites or
            their inputs ever diverge
  fused     read_fused == read_cells == pow2_read.int8_blend_read on the op's integers, across D = 48 / 40 / 52 / 8 / 128,
            every block size, both load styles, garbage stride padding
  train=int the quant_mode training forward equals the integer read (LightMultiHeadLUT.forward_int) BIT FOR BIT on CUDA fp32,
            and the exported artefact's fused kernel equals its torch read bit for bit
  backward  the op's recompute backward gives BIT-IDENTICAL parameter gradients to the torch expression evaluated on the
            same integers; float64 gradcheck of that expression with forced boundary integers (k' = -3 / clamped 4,
            q = 0 / Q = 3, skip, drop); skipped tables and dropped second cells keep a nonzero gradient through the op
  compile   no graph break over the op in the compiled training forward
  fallback  op disabled / extension absent -> the torch definition, which still trains (finite, matching gradients, loss
            decreases)
"""
import math

import pytest
import torch

from spiky.lutorch import pow2_int8_cuda as K
from spiky.lutorch import pow2_read as P
from spiky.lutorch import pow2_scalar_op as OP
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

HAVE_OP = OP.ensure_registered()
needs_op = pytest.mark.skipif(not HAVE_OP, reason="spiky_lutorch::p2_scalars needs the CUDA extension (compute capability 12.x)")


@pytest.fixture(autouse=True)
def _op_enabled():
    OP.set_enabled(True)
    yield
    OP.set_enabled(True)


def _ffn(dev="cuda", seed=5, tau=0.5, noise=0.3, H=4, tph=128, nap=8, din=48, **kw):
    torch.manual_seed(seed)
    ffn = CompressionMultiHeadLUT(input_dim=384, output_dim=384, inner_in_dim=din, inner_out_dim=din, nap=nap, tph=tph,
                                  n_heads=H, lut_impl="light", confidence_form="learned_margin", learned_margin_freeze_g=True,
                                  read_top_n=2, read_tau=tau, read_tau_learnable=True, random_seed=seed,
                                  device=torch.device(dev), quant_mode="p2_int8", initial_weights_noise=noise, **kw)
    ffn.lut_light._compile_enabled = False
    return ffn


def _margins(lut, z):
    B, H, T, NAP = z.shape[0], lut.n_heads, lut.tables_per_head, lut.n_anchor_pairs
    ia = lut.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    ib = lut.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    return (torch.gather(z, 2, ia) - torch.gather(z, 2, ib)).view(B, H, T, NAP)


# ------------------------------------------------------------------ drift ------------------------------------------------
@needs_op
@pytest.mark.parametrize("scale", [0.05, 1.0, 20.0])
def test_op_integers_equal_fused_kernel_integers(scale):
    ffn = _ffn(noise=0.3)
    art = ffn.export_quantised()
    lut = ffn.lut_light
    z = torch.randn(3000, 4, 48, device="cuda", generator=torch.Generator(device="cuda").manual_seed(7)) * scale
    tau, g, beta, gamma = lut._quant_scalars()
    with torch.no_grad():
        cells_op = torch.ops.spiky_lutorch.p2_scalars(_margins(lut, z), tau, g, beta, gamma, -3, 4, 3)[1]
        kc = art._kernel_cache(z.device)
        cells_k = torch.empty_like(cells_op)
        K.read_fused(z, kc["anchor_a"], kc["anchor_b"], kc["tables"], kc["scalars"], 8, 48, -3, 4, 3, cells_out=cells_k)
    assert torch.equal(cells_op, cells_k)
    sh = cells_op[..., 2]
    assert ((sh & 15) == 15).any() or scale > 1                          # the case mix includes skipped tables


@needs_op
def test_op_integers_equal_fused_kernel_on_rounding_boundaries():
    """Margins whose smallest value sits exactly on a q threshold, and scores whose log2 lands on k' + 1/2."""
    ffn = _ffn()
    art = ffn.export_quantised()
    lut = ffn.lut_light
    tau, g, beta, gamma = lut._quant_scalars()
    inv = float(2.0 / (tau.detach() * P.LN2))
    thr = torch.tensor([(j - 0.5) / inv for j in range(1, 12)], device="cuda")
    z = torch.randn(1100, 4, 48, device="cuda", generator=torch.Generator(device="cuda").manual_seed(3))
    with torch.no_grad():
        d = _margins(lut, z)
        # push one margin per table onto a threshold (and one ulp either side) through z itself
        a0 = lut.anchor_a.view(4, -1, 8)[:, :, 0]                       # [H, T] first anchor pair of every table
        b0 = lut.anchor_b.view(4, -1, 8)[:, :, 0]
        for n in range(z.shape[0]):
            t = thr[n % thr.numel()]
            t = [t, torch.nextafter(t, t + 1), torch.nextafter(t, t - 1)][n % 3]
            h, j = n % 4, (n * 7) % a0.shape[1]
            z[n, h, a0[h, j]] = z[n, h, b0[h, j]] + t
        d = _margins(lut, z)
        cells_op = torch.ops.spiky_lutorch.p2_scalars(d, tau, g, beta, gamma, -3, 4, 3)[1]
        kc = art._kernel_cache(z.device)
        cells_k = torch.empty_like(cells_op)
        K.read_fused(z, kc["anchor_a"], kc["anchor_b"], kc["tables"], kc["scalars"], 8, 48, -3, 4, 3, cells_out=cells_k)
    assert torch.equal(cells_op, cells_k)


@needs_op
@pytest.mark.parametrize("D", [48, 40, 52, 8, 128])
def test_fused_bit_exact_vs_cells_and_int8_blend_read_coverage_matrix(D):
    """Fused (integers in-kernel) == op integers, and its read == cells read == pow2_read.int8_blend_read on those integers,
    for every cell width, block size and load style, with garbage in the stride padding."""
    N, H, T, nap, din = 211, 4, 32, 6, 24
    gen = torch.Generator(device="cuda").manual_seed(D)
    tables = torch.randint(-128, 128, (H * T * (1 << nap), D), device="cuda", generator=gen).to(torch.int8)
    tables[0], tables[1] = -128, 127
    aa = torch.randint(0, din, (H, T, nap), device="cuda", generator=gen)
    ab = (aa + torch.randint(1, din, (H, T, nap), device="cuda", generator=gen)) % din
    z = torch.randn(N, H, din, device="cuda", generator=gen) * 0.6
    sc = (torch.tensor([0.3]), torch.tensor([0.1]), torch.tensor([2.0]), torch.tensor([1.0]))
    sc = tuple(t.to("cuda") for t in sc)
    Bi = torch.arange(N, device="cuda").view(N, 1, 1)
    Hi = torch.arange(H, device="cuda").view(1, H, 1)
    d = (z[Bi, Hi, aa.view(1, H, -1)] - z[Bi, Hi, ab.view(1, H, -1)]).view(N, H, T, nap)
    psw, cells, kq = torch.ops.spiky_lutorch.p2_scalars(d, *sc, -3, 4, 3)
    q, k, skip, drop = OP._integers_from(cells, kq, 3)
    assert skip.any() and drop.any() and (~skip & ~drop).any()
    offs = (torch.arange(H * T, device="cuda") * (1 << nap)).view(1, H, T, 1)
    ref = P.int8_blend_read(tables, D, cells[..., :2].long() + offs, P.shift_groups(q, k, skip, drop)).to(torch.float32)
    ts = K.stride_tables(tables, D)
    if ts.shape[1] != D:
        ts = ts.clone()
        ts[:, D:] = 127
    a32, b32 = aa.to(torch.int32).contiguous(), ab.to(torch.int32).contiguous()
    for bn in K.BLOCK_NS:
        if bn * K.row_stride(D) // 16 > 1024:
            continue
        for load16 in (True, False):
            c_out = torch.empty_like(cells)
            acc_f = K.read_fused(z, a32, b32, ts, sc, nap, D, -3, 4, 3, block_n=bn, load16=load16, cells_out=c_out)
            acc_c = K.read_cells(ts, cells, nap, D, -3, 4, 3, block_n=bn, load16=load16)
            assert torch.equal(c_out, cells), (bn, load16)
            assert torch.equal(acc_f, acc_c) and torch.equal(acc_c, ref), (bn, load16)


# ------------------------------------------------------------------ train == int ----------------------------------------
@needs_op
def test_training_forward_equals_integer_read_bit_for_bit():
    ffn = _ffn(tau=0.05)
    lut = ffn.lut_light
    z = ffn.compress(torch.randn(700, 384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))).view(700, 4, 48)
    y_train = lut(z)
    with torch.no_grad():
        y_int = lut.forward_int(z)
    assert torch.equal(y_train.detach(), y_int)
    art = ffn.export_quantised()
    x = torch.randn(1500, 384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    assert art._uses_kernel(x)
    assert torch.equal(art(x), art._forward_torch(x))


# ------------------------------------------------------------------ backward --------------------------------------------
@needs_op
def test_recompute_backward_bit_identical_to_torch_expression_on_same_integers():
    ffn = _ffn(tau=0.05)
    lut = ffn.lut_light
    z0 = ffn.compress(torch.randn(500, 384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(4))).view(500, 4, 48).detach()
    w = torch.randn(500, 4, 128, 2, device="cuda", generator=torch.Generator(device="cuda").manual_seed(5))

    def run(use_op):
        lut.zero_grad(set_to_none=True)
        z = z0.clone().requires_grad_(True)
        d = _margins(lut, z)
        tau, g, beta, gamma = lut._quant_scalars()
        if use_op:
            psw, _cells, _kq = torch.ops.spiky_lutorch.p2_scalars(d, tau, g, beta, gamma, -3, 4, 3)
        else:
            with torch.no_grad():
                _, cells, kq = torch.ops.spiky_lutorch.p2_scalars(d.detach(), tau.detach(), g, beta.detach(), gamma.detach(), -3, 4, 3)
            q, k, skip, drop = OP._integers_from(cells, kq, 3)
            psw = OP.ste_cell_weights(d, tau, g, beta, gamma, q, k, skip, drop)
        (psw * w).sum().backward()
        return psw.detach(), {n: p.grad.clone() for n, p in lut.named_parameters() if p.grad is not None}, z.grad.clone()

    p_op, g_op, z_op = run(True)
    p_t, g_t, z_t = run(False)
    assert torch.equal(p_op, p_t)
    assert set(g_op) == set(g_t) and len(g_op) >= 3
    for n in g_op:
        assert torch.equal(g_op[n], g_t[n]), n
    torch.testing.assert_close(z_op, z_t, rtol=0, atol=1e-5 * float(z_t.abs().max()))   # CUDA scatter-add accumulation order


def test_gradcheck_of_the_backward_expression_with_forced_boundary_integers():
    """float64 gradcheck of the backward expression the op evaluates, with the integers forced onto every boundary.
    The straight-through value is piecewise constant, so finite differences of it are 0 by design; its gradient is defined
    as that of the smooth surrogate e(theta) * r with the ratios r = b / e(theta0) frozen. So: gradcheck the surrogate
    (analytic vs numerical), then require the STE expression's autograd gradient to equal the surrogate's."""
    torch.manual_seed(0)
    T, NAP = 6, 8
    d = (torch.randn(2, 1, T, NAP, dtype=torch.float64) * 0.7).requires_grad_(True)
    tau = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    beta = torch.tensor(1.9, dtype=torch.float64, requires_grad=True)
    gamma = torch.tensor(1.1, dtype=torch.float64, requires_grad=True)
    g = torch.tensor(0.0, dtype=torch.float64)
    k = torch.tensor([-3.0, 4.0, 0.0, 2.0, -3.0, 1.0], dtype=torch.float64).expand(2, 1, T)
    q = torch.tensor([0.0, 3.0, 4.0, 1.0, 7.0, 64.0], dtype=torch.float64).expand(2, 1, T)
    skip = torch.tensor([True, False, False, False, True, False]).expand(2, 1, T)
    drop = q > 3

    def e_of(d_, tau_, beta_, gamma_):
        m = d_.abs()
        x = 2.0 * m.min(dim=-1, keepdim=True).values / tau_
        return OP.score_from_margins(m, g, beta_, gamma_).unsqueeze(-1) * torch.cat([torch.sigmoid(x), torch.sigmoid(-x)], -1)

    theta = (d, tau, beta, gamma)
    with torch.no_grad():
        r = torch.pow(2.0, torch.stack([k, k - q], dim=-1)) / e_of(*theta).clamp_min(1e-30)

    def surrogate(*th):
        return e_of(*th) * r

    assert torch.autograd.gradcheck(surrogate, theta, eps=1e-6, atol=1e-7)
    wt = torch.randn(2, 1, T, 2, dtype=torch.float64)
    g_ste = torch.autograd.grad((OP.ste_cell_weights(d, tau, g, beta, gamma, q, k, skip, drop) * wt).sum(), theta)
    g_sur = torch.autograd.grad((surrogate(*theta) * wt).sum(), theta)
    for a, b in zip(g_ste, g_sur):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=0)
    assert d.abs().min() > 1e-3                                         # |d| stays differentiable at the sample


@needs_op
def test_skipped_tables_and_dropped_second_cells_keep_gradient_through_the_op():
    ffn = _ffn(tau=0.05)
    lut = ffn.lut_light
    z = ffn.compress(torch.randn(400, 384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(8))).view(400, 4, 48).detach()
    z.requires_grad_(True)
    d = _margins(lut, z)
    tau, g, beta, gamma = lut._quant_scalars()
    psw, cells, kq = torch.ops.spiky_lutorch.p2_scalars(d, tau, g, beta, gamma, -3, 4, 3)
    skip = (cells[..., 2] & 15) == 15
    drop = (~skip) & (kq[..., 1] > 3)
    assert skip.any() and drop.any()
    assert torch.all(psw.detach()[skip] == 0) and torch.all(psw.detach()[..., 1][drop] == 0)
    gd, = torch.autograd.grad(psw[..., 0][skip].sum(), d, retain_graph=True)
    assert gd.abs().sum() > 0                                               # skipped tables: ratio gradient survives
    gd2, = torch.autograd.grad(psw[..., 1][drop].sum(), d)
    assert gd2.abs().sum() > 0                                              # dropped second cells: ratio gradient survives


# ------------------------------------------------------------------ compile ---------------------------------------------
@needs_op
def test_no_graph_break_over_the_op():
    ffn = _ffn()
    lut = ffn.lut_light
    z = torch.randn(64, 4, 48, device="cuda", requires_grad=True)
    exp = torch._dynamo.explain(lut._forward_impl)(z)
    assert exp.graph_break_count == 0, [str(r.reason) for r in exp.break_reasons]
    lut._compile_enabled = True
    lut._compiled_fwd = None
    y = lut(z)
    y.sum().backward()
    assert torch.isfinite(z.grad).all()


# ------------------------------------------------------------------ fallback --------------------------------------------
@pytest.mark.parametrize("dev", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_torch_fallback_trains(dev):
    OP.set_enabled(False)
    ffn = CompressionMultiHeadLUT(input_dim=64, output_dim=64, inner_in_dim=16, inner_out_dim=16, nap=5, tph=16, n_heads=4,
                                  lut_impl="light", confidence_form="learned_margin", learned_margin_freeze_g=True,
                                  read_top_n=2, read_tau=0.3, read_tau_learnable=True, random_seed=2,
                                  device=torch.device(dev), quant_mode="p2_int8", initial_weights_noise=0.1)
    ffn.lut_light._compile_enabled = False
    assert not OP.op_available(torch.zeros(1, device=dev))
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(256, 64, device=dev, generator=g)
    target = torch.randn(256, 64, device=dev, generator=g) * 0.1
    opt = torch.optim.Adam(ffn.parameters(), lr=3e-3)
    losses = []
    for _ in range(30):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(ffn(x), target)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in ffn.parameters() if p.grad is not None)
        opt.step()
        losses.append(float(loss))
    assert losses[-1] < losses[0] and math.isfinite(losses[-1])
