"""Fused CUDA kernel for the int8 power-of-two read (spiky/lutorch/pow2_int8_cuda.py, csrc/pow2_int8_read.cu).

Gates:
  cells     the kernel's int32 accumulation is BIT-EXACT against the PR 1 integer path (pow2_read.int8_blend_read, eager and
            torch.compile) and against an explicit per-cell (row << shift) sum -- for cell widths D = 48, 40, 52, 8, 128
            (16-aligned, stride-padded, smaller than one load, several units), with garbage in the padding bytes,
            skipped tables, dropped second cells, k' at -3 and at the clamp 4, q at 0 and at the Q = 3 boundary, and
            saturated -128 / 127 rows; every block size and both load styles
  headroom  the worst case (all rows -128 or 127, every shift 10, 2T rows) stays exact in int32
  fused     the one-launch regime computes exactly the cells regime's integers (one definition, p2::table_scalars) and
            the same read; the full fused coverage matrix and drift gates are in test_pow2_scalar_op.py
  artefact  QuantisedLightFFN "cells" and "fused" ("auto"'s default) are bit-identical to the compiled "off" path; with
            the extension unavailable the artefact silently keeps the torch path; CPU inputs never touch the kernel

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


@needs_kernel
def test_fused_regime_integers_and_read():
    """Fused computes the integers in the kernel with the same p2::table_scalars the op runs for the cells regime, so its
    integers and its read equal the cells regime's exactly (drift gates: test_pow2_scalar_op.py)."""
    torch.manual_seed(0)
    ffn = CompressionMultiHeadLUT(input_dim=384, output_dim=384, inner_in_dim=48, inner_out_dim=48, nap=8, tph=128, n_heads=4,
                                  lut_impl="light", confidence_form="learned_margin", learned_margin_freeze_g=True,
                                  read_top_n=2, read_tau=0.5, read_tau_learnable=True, random_seed=5,
                                  device=torch.device("cuda"), quant_mode="p2_int8", initial_weights_noise=0.3)
    art = ffn.export_quantised()
    x = torch.randn(2048, 384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
    cells_t = torch.compile(art._cells_impl, dynamic=True)(x)
    kc = art._kernel_cache(x.device)
    mt = art.meta
    z = torch.nn.functional.linear(x, art.compress_weight, art.compress_bias).view(2048, 4, 48)
    cells_k = torch.empty_like(cells_t)
    acc_f = K.read_fused(z, kc["anchor_a"], kc["anchor_b"], kc["tables"], kc["scalars"], 8, 48, -3, 4, 3, cells_out=cells_k)
    assert torch.equal(cells_k, cells_t)                                               # one definition: exact
    acc_c = K.read_cells(kc["tables"], cells_t, 8, 48, -3, 4, 3)
    assert torch.equal(acc_f, acc_c)


@needs_kernel
def test_artefact_cells_kernel_bit_identical_to_compiled_torch_path():
    torch.manual_seed(0)
    ffn = CompressionMultiHeadLUT(input_dim=384, output_dim=384, inner_in_dim=40, inner_out_dim=40, nap=7, tph=64, n_heads=4,
                                  lut_impl="light", confidence_form="learned_margin", learned_margin_freeze_g=True,
                                  read_top_n=2, read_tau=0.5, read_tau_learnable=True, random_seed=9,
                                  device=torch.device("cuda"), quant_mode="p2_int8", initial_weights_noise=0.3)
    art = ffn.export_quantised()
    x = torch.randn(777, 384, device="cuda", generator=torch.Generator(device="cuda").manual_seed(2))
    art.kernel = "off"
    ref = art(x)
    art.kernel = "cells"
    assert art.kernel_regime(x) == "cells"
    for bn in K.BLOCK_NS:
        art.kernel_block_n = bn
        assert torch.equal(art(x), ref), bn
    art.kernel = "fused"
    for bn in K.BLOCK_NS:
        art.kernel_block_n = bn
        assert torch.equal(art(x), ref), bn
    art.kernel = "auto"
    assert art.kernel_min_tokens == 1 and art.kernel_regime(x[:1]) == "fused"          # default: fused at every size
    small = art(x[:3])
    art.kernel = "off"
    ref_small = art(x[:3])                     # same batch: compress/decompress matmuls are batch-size dependent in cuBLAS
    art.kernel = "auto"
    assert torch.equal(art(x), ref) and torch.equal(small, ref_small)
    art.kernel_min_tokens = 1000
    assert art.kernel_regime(x) is None and torch.equal(art(x), ref)                  # below the threshold: torch path
    art.kernel = "bogus"
    with pytest.raises(ValueError):
        art(x)


def test_fallback_when_extension_unavailable(monkeypatch):
    """No extension (simulated): "auto", "cells" and "fused" all silently use the torch path; outputs unchanged."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    ffn = CompressionMultiHeadLUT(input_dim=64, output_dim=64, inner_in_dim=16, inner_out_dim=16, nap=5, tph=8, n_heads=4,
                                  lut_impl="light", confidence_form="learned_margin", learned_margin_freeze_g=True,
                                  read_top_n=2, read_tau=0.5, read_tau_learnable=True, random_seed=3,
                                  device=torch.device(dev), quant_mode="p2_int8", initial_weights_noise=0.3)
    art = ffn.export_quantised()
    x = torch.randn(50, 64, device=dev)
    from spiky.lutorch import pow2_scalar_op
    monkeypatch.setattr(K, "load", lambda: None)
    pow2_scalar_op.set_enabled(False)                                   # an absent extension never registers the op
    try:
        art.kernel = "off"
        ref = art(x)
        for mode in ("auto", "cells", "fused"):
            art.kernel = mode
            assert art.kernel_regime(x) is None
            assert torch.equal(art(x), ref)
        assert K.available()[0] is False
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
    torch.manual_seed(0)
    ffn = CompressionMultiHeadLUT(input_dim=64, output_dim=64, inner_in_dim=16, inner_out_dim=16, nap=5, tph=8, n_heads=4,
                                  lut_impl="light", confidence_form="learned_margin", learned_margin_freeze_g=True,
                                  read_top_n=2, read_tau=0.5, read_tau_learnable=True, random_seed=3,
                                  quant_mode="p2_int8", initial_weights_noise=0.3)
    art = ffn.export_quantised()
    x = torch.randn(20, 64)
    art.kernel = "cells"
    assert art.kernel_regime(x) is None                                               # CPU input: never the kernel
    art.kernel = "off"
    ref = art(x)
    art.kernel = "cells"
    assert torch.equal(art(x), ref)
