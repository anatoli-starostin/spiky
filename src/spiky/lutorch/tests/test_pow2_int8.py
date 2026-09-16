"""spiky/lutorch/pow2_int8.py: the CUDA extension for the int8 power-of-two read (csrc/pow2_int8_read.cu) and the
spiky_lutorch::p2_scalars custom op -- ONE forward definition of the per-table integers for training and eval.

Gates:
  cells     the kernel's int32 accumulation on supplied integers (read_cells, the reference) is BIT-EXACT against
            pow2_read.int8_blend_read (eager and torch.compile) and against an explicit per-cell (row << shift) sum -- for
            cell widths D = 48, 40, 52, 8, 128 (16-aligned, stride-padded, smaller than one load, several units), with
            garbage in the padding bytes, skipped tables, dropped second cells, k' at -3 and at the clamp 4, q at 0 and at the
            Q = 3 boundary, and saturated -128 / 127 rows; every block size and both load styles
  headroom  the worst case (all rows -128 or 127, every shift 10, 2T rows) stays exact in int32
  fused     read_fused's in-kernel integers equal the reference integers and its read equals read_cells on them (the full
            fused coverage matrix and the drift gates are below)
  artefact  QuantisedLightFFN's CUDA forward (the fused kernel) is bit-identical to its torch read, at every block size and
            for small batches; without the extension, or for CPU inputs, it silently uses the torch read
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

The kernel builds only on compute capability 12.x (RTX 5090); elsewhere these tests skip, except the fallback ones.
"""
import math
import os

import pytest
import torch

from spiky.lutorch import pow2_int8 as K
from spiky.lutorch import pow2_read as P
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

HAVE_KERNEL = K.ensure_registered()             # the extension loads and the op is registered (one and the same condition)
needs_kernel = pytest.mark.skipif(not HAVE_KERNEL, reason=f"pow2 int8 CUDA kernel unavailable: {K.available()[1]}")


@pytest.fixture(autouse=True)
def _op_enabled():
    K.set_enabled(True)
    yield
    K.set_enabled(True)


def _random_case(N, H, T, nap, D, seed, dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(seed)
    Kc = 1 << nap
    tables = torch.randint(-128, 128, (H * T * Kc, D), device=dev, generator=g).to(torch.int8)
    tables[0] = -128                                                    # saturated rows, read by table 0 below
    tables[1] = 127
    idx = torch.randint(0, Kc, (N, H, T, 2), device=dev, generator=g)
    idx[:, 0, 0] = torch.tensor([0, 1], device=dev)
    q = torch.randint(0, 9, (N, H, T), device=dev, generator=g).float()
    k = torch.randint(-3, 5, (N, H, T), device=dev, generator=g).float()
    q[:, :, 1], q[:, :, 2] = 0, 3                                       # q at 0 and at the Q = 3 boundary (second cell kept)
    q[:, :, 3] = 4                                                      # first q that drops the second cell
    k[:, :, 4], k[:, :, 5] = -3, 4                                      # both window ends
    skip = torch.rand(N, H, T, device=dev, generator=g) < 0.15
    skip[:, :, 1:6] = False
    drop = q > 3
    offs = (torch.arange(H * T, device=dev) * Kc).view(1, H, T, 1)
    return tables, idx, offs, q, k, skip, drop


@needs_kernel
@pytest.mark.parametrize("D", [48, 40, 52, 8, 128])
def test_cells_bit_exact_against_pr1_paths(D):
    N, H, T, nap = 67, 4, 32, 6
    tables, idx, offs, q, k, skip, drop = _random_case(N, H, T, nap, D, seed=D)
    assert skip.any() and drop.any() and (k == -3).any() and (k == 4).any() and (q == 3).any() and (q == 0).any()
    group = P.shift_groups(q, k, skip, drop)
    ref_eager = P.int8_blend_read(tables, D, idx + offs, group, chunk_bags=7)
    ref_comp = torch.compile(P.int8_accumulate, dynamic=True)(tables, idx + offs, group)
    rows = tables[idx + offs].to(torch.int64)                          # explicit per-cell (row << shift) sum
    w = torch.where(group < P.N_SHIFTS, torch.pow(2, group.clamp(max=62)), torch.zeros_like(group))
    explicit = (rows * w.unsqueeze(-1)).sum(dim=(2, 3))
    assert torch.equal(ref_eager, ref_comp) and torch.equal(ref_eager.to(torch.int64), explicit)
    ts = K.stride_tables(tables, D)
    if ts.shape[1] != D:
        ts = ts.clone()
        ts[:, D:] = 127                                                 # garbage padding: must be masked by the kernel
    cells = K.pack_cells(idx, q, k, skip, drop)
    for bn in K.BLOCK_NS:
        if bn * K.row_stride(D) // 16 > 1024:
            continue
        for load16 in (True, False):
            out = K.read_cells(ts, cells, nap, D, -3, 4, 3, block_n=bn, load16=load16)
            assert out.dtype == torch.float32 and out.shape == (N, H, D)
            assert torch.equal(out.to(torch.int64), explicit), (bn, load16)
            assert torch.equal(out, ref_eager.to(torch.float32)), (bn, load16)


def test_row_stride_and_padding():
    assert [K.row_stride(D) for D in (1, 8, 16, 17, 40, 48, 52, 64, 128)] == [16, 16, 16, 32, 48, 48, 64, 64, 128]
    t = torch.randint(-128, 128, (5, 40)).to(torch.int8)
    s = K.stride_tables(t, 40)
    assert s.shape == (5, 48) and torch.equal(s[:, :40], t) and torch.all(s[:, 40:] == 0)   # zero padding at pack time
    t48 = torch.randint(-128, 128, (5, 48)).to(torch.int8)
    assert K.stride_tables(t48, 48).data_ptr() == t48.data_ptr()                              # no copy when aligned
    with pytest.raises(ValueError):
        K.stride_tables(t.float(), 40)


@needs_kernel
def test_int32_headroom_worst_case():
    """Every read cell at the largest shift (k' = 4, q = 0 -> 10) and every row saturated: |acc| = 2T * 128 << 10."""
    N, H, T, nap, D = 3, 2, 128, 8, 48
    Kc = 1 << nap
    for fill in (-128, 127):
        tables = torch.full((H * T * Kc, D), fill, dtype=torch.int8, device="cuda")
        idx = torch.zeros(N, H, T, 2, dtype=torch.long, device="cuda")
        q = torch.zeros(N, H, T, device="cuda")
        k = torch.full((N, H, T), 4.0, device="cuda")
        skip = torch.zeros(N, H, T, dtype=torch.bool, device="cuda")
        out = K.read_cells(tables, K.pack_cells(idx, q, k, skip, q > 3), nap, D, -3, 4, 3)
        expect = 2 * T * fill * (1 << 10)
        assert abs(expect) <= 2 ** 25 < 2 ** 31
        assert torch.all(out == float(expect))


def _artefact(dev="cuda", din=48, nap=8, tph=128, seed=5, E=384):
    torch.manual_seed(0)
    ffn = CompressionMultiHeadLUT(input_dim=E, output_dim=E, inner_in_dim=din, inner_out_dim=din, nap=nap, tph=tph, n_heads=4,
                                  lut_impl="light", confidence_form="learned_margin", learned_margin_freeze_g=True,
                                  read_top_n=2, read_tau=0.5, read_tau_learnable=True, random_seed=seed,
                                  device=torch.device(dev), quant_mode="p2_int8", initial_weights_noise=0.3)
    return ffn.export_quantised()


@needs_kernel
def test_fused_integers_and_read_equal_the_reference():
    """read_fused computes the integers in the kernel with p2::table_scalars; they equal the reference integers (the
    p2_scalars op, the same function) and its read equals read_cells on them."""
    art = _artefact()
    x = torch.randn(2048, 384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    cells_ref = art._reference_cells(x)
    kc = art._kernel_cache(x.device)
    z = torch.nn.functional.linear(x, art.compress_weight, art.compress_bias).view(2048, 4, 48)
    cells_k = torch.empty_like(cells_ref)
    acc_f = K.read_fused(z, kc["anchor_a"], kc["anchor_b"], kc["tables"], kc["scalars"], 8, 48, -3, 4, 3, cells_out=cells_k)
    assert torch.equal(cells_k, cells_ref)
    assert torch.equal(acc_f, K.read_cells(kc["tables"], cells_ref, 8, 48, -3, 4, 3))


@needs_kernel
def test_artefact_cuda_forward_is_the_fused_kernel_and_equals_the_torch_read():
    art = _artefact(din=40, nap=7, tph=64, seed=9)
    x = torch.randn(777, 384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    assert art._uses_kernel(x)
    ref = art._forward_torch(x)
    assert torch.equal(art(x), ref)
    kc = art._kernel_cache(x.device)
    z = torch.nn.functional.linear(x, art.compress_weight, art.compress_bias).view(777, 4, 40)
    for bn in K.BLOCK_NS:
        for l16 in (True, False):
            acc = K.read_fused(z, kc["anchor_a"], kc["anchor_b"], kc["tables"], kc["scalars"], 7, 40, -3, 4, 3,
                               block_n=bn, load16=l16)
            out = torch.nn.functional.linear(acc.reshape(777, 160), art.decompress_weight, art.decompress_bias)
            assert torch.equal(out, ref), (bn, l16)
    for n in (1, 3):                           # small calls too (same batch: cuBLAS matmuls are batch-size dependent)
        assert torch.equal(art(x[:n]), art._forward_torch(x[:n])), n


def test_fallback_when_extension_unavailable(monkeypatch):
    """No extension (simulated): the artefact silently uses the torch read; outputs unchanged."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    art = _artefact(dev, din=16, nap=5, tph=8, seed=3, E=64)
    x = torch.randn(50, 64, device=dev)
    ref = art(x)
    monkeypatch.setattr(K, "load", lambda: None)
    K.set_enabled(False)                                   # an absent extension never registers the op
    try:
        assert not art._uses_kernel(x) and K.available()[0] is False
        assert torch.equal(art(x), art._forward_torch(x))
        torch.testing.assert_close(art(x), ref, rtol=0, atol=1e-5)      # torch integers: ulp-level boundary flips at most
    finally:
        K.set_enabled(True)


def test_disable_env_and_cpu_inputs(monkeypatch):
    monkeypatch.setenv("SPIKY_P2_CUDA_DISABLE", "1")
    saved = (K._ext, K._error, K._tried)
    try:
        K._reset_for_tests()
        assert K.load() is None and "SPIKY_P2_CUDA_DISABLE" in K.available()[1]
    finally:
        K._ext, K._error, K._tried = saved
    art = _artefact("cpu", din=16, nap=5, tph=8, seed=3, E=64)
    x = torch.randn(20, 64)
    assert not art._uses_kernel(x)                                      # CPU input: never the kernel
    assert torch.equal(art(x), art._forward_torch(x))


# ======================================================================================================================
# the spiky_lutorch::p2_scalars op
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
@needs_kernel
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


@needs_kernel
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


@needs_kernel
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
    q, k, skip, drop = K._integers_from(cells, kq, 3)
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
@needs_kernel
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
@needs_kernel
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
            q, k, skip, drop = K._integers_from(cells, kq, 3)
            psw = K.ste_cell_weights(d, tau, g, beta, gamma, q, k, skip, drop)
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
        return K.score_from_margins(m, g, beta_, gamma_).unsqueeze(-1) * torch.cat([torch.sigmoid(x), torch.sigmoid(-x)], -1)

    theta = (d, tau, beta, gamma)
    with torch.no_grad():
        r = torch.pow(2.0, torch.stack([k, k - q], dim=-1)) / e_of(*theta).clamp_min(1e-30)

    def surrogate(*th):
        return e_of(*th) * r

    assert torch.autograd.gradcheck(surrogate, theta, eps=1e-6, atol=1e-7)
    wt = torch.randn(2, 1, T, 2, dtype=torch.float64)
    g_ste = torch.autograd.grad((K.ste_cell_weights(d, tau, g, beta, gamma, q, k, skip, drop) * wt).sum(), theta)
    g_sur = torch.autograd.grad((surrogate(*theta) * wt).sum(), theta)
    for a, b in zip(g_ste, g_sur):
        torch.testing.assert_close(a, b, rtol=1e-12, atol=0)
    assert d.abs().min() > 1e-3                                         # |d| stays differentiable at the sample


@needs_kernel
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
@needs_kernel
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
    K.set_enabled(False)
    ffn = CompressionMultiHeadLUT(input_dim=64, output_dim=64, inner_in_dim=16, inner_out_dim=16, nap=5, tph=16, n_heads=4,
                                  lut_impl="light", confidence_form="learned_margin", learned_margin_freeze_g=True,
                                  read_top_n=2, read_tau=0.3, read_tau_learnable=True, random_seed=2,
                                  device=torch.device(dev), quant_mode="p2_int8", initial_weights_noise=0.1)
    ffn.lut_light._compile_enabled = False
    assert not K.op_available(torch.zeros(1, device=dev))
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
