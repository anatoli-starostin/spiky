"""CUDA extension for the int8 power-of-two read (spiky/lutorch/pow2_int8_cuda.py, csrc/pow2_int8_read.cu).

Gates:
  cells     the kernel's int32 accumulation on supplied integers (read_cells, the reference) is BIT-EXACT against
            pow2_read.int8_blend_read (eager and torch.compile) and against an explicit per-cell (row << shift) sum -- for
            cell widths D = 48, 40, 52, 8, 128 (16-aligned, stride-padded, smaller than one load, several units), with
            garbage in the padding bytes, skipped tables, dropped second cells, k' at -3 and at the clamp 4, q at 0 and at the
            Q = 3 boundary, and saturated -128 / 127 rows; every block size and both load styles
  headroom  the worst case (all rows -128 or 127, every shift 10, 2T rows) stays exact in int32
  fused     read_fused's in-kernel integers equal the reference integers and its read equals read_cells on them (the full
            fused coverage matrix and the drift gates are in test_pow2_scalar_op.py)
  artefact  QuantisedLightFFN's CUDA forward (the fused kernel) is bit-identical to its torch read, at every block size and
            for small batches; without the extension, or for CPU inputs, it silently uses the torch read

The kernel builds only on compute capability 12.x (RTX 5090); elsewhere these tests skip, except the fallback ones.
"""
import os

import pytest
import torch

from spiky.lutorch import pow2_int8_cuda as K
from spiky.lutorch import pow2_read as P
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

HAVE_KERNEL = K.load() is not None
needs_kernel = pytest.mark.skipif(not HAVE_KERNEL, reason=f"pow2 int8 CUDA kernel unavailable: {K.available()[1]}")


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
    from spiky.lutorch import pow2_scalar_op
    ref = art(x)
    monkeypatch.setattr(K, "load", lambda: None)
    pow2_scalar_op.set_enabled(False)                                   # an absent extension never registers the op
    try:
        assert not art._uses_kernel(x) and K.available()[0] is False
        assert torch.equal(art(x), art._forward_torch(x))
        torch.testing.assert_close(art(x), ref, rtol=0, atol=1e-5)      # torch integers: ulp-level boundary flips at most
    finally:
        pow2_scalar_op.set_enabled(True)


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
