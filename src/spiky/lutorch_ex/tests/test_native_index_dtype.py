"""The native lprojection kernels read int32 OR int64 stored cell indices directly (templated on index_t).

Only the stored index tensors take index_t; every offset -- (table * n_entries + entry) * n_outputs + o -- is computed in
int64 inside the kernels. Checked here: exact int32/int64 parity per host entry point (integer-valued data, so the
atomic sums are exact whatever their order), rejection of any other or mixed index dtype, that int32 training stores
no int64 cell-index copy, and the offset-overflow trap at > 2^31 weight elements.
"""
import subprocess
import sys
import textwrap

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex import LUTSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the native kernels are CUDA-only")


def _mgr():
    from spiky.lutorch_ex.cartridges._native_ops import native_manager
    mgr = native_manager()
    if mgr is None:
        pytest.skip("native lprojection extension not available")
    return mgr


def _inputs(B=64, nt=12, K=32, D=24, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    W = torch.randint(-4, 5, (nt, K, D), device="cuda", generator=g).to(dtype)        # integer-valued: exact sums
    li = torch.randint(0, K, (B, nt), device="cuda", generator=g)
    lai = torch.randint(0, K, (B, nt, 1), device="cuda", generator=g)
    tif = torch.arange(nt, device="cuda").repeat(B)
    grad = torch.randint(-3, 4, (B, nt, D), device="cuda", generator=g).to(dtype)
    lad = torch.randint(-2, 3, (B, nt, 1), device="cuda", generator=g).to(dtype)
    return W, li, lai, tif, grad, lad


def _as(dt, *ts):
    return [t.to(dt).contiguous() for t in ts]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_backward_na1_nonsmooth_int32_matches_int64_exactly(dtype):
    mgr = _mgr()
    W, li, lai, tif, grad, _ = _inputs(dtype=dtype)
    outs = [mgr.lprojection_backward_na1_nonsmooth(grad, W, *_as(dt, li, lai, tif, tif), 256)
            for dt in (torch.int64, torch.int32)]
    for a, b in zip(*outs):
        assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_backward_na1_smooth_and_forward_smooth_int32_match_int64_exactly(dtype):
    mgr = _mgr()
    W, li, lai, tif, grad, lad = _inputs(dtype=dtype)
    fwd = [mgr.lprojection_forward_smooth(W, *_as(dt, li, lai), lad, *_as(dt, tif, tif), True, 256)
           for dt in (torch.int64, torch.int32)]
    for a, b in zip(*fwd):
        assert torch.equal(a, b)
    _, mw, aw = fwd[0]
    mw, aw = torch.full_like(mw, 0.75), torch.full_like(aw, 0.25)          # exact binary fractions
    bwd = [mgr.lprojection_backward_na1_smooth(grad, W, *_as(dt, li, lai, tif, tif), mw, aw, 256)
           for dt in (torch.int64, torch.int32)]
    for a, b in zip(*bwd):
        assert torch.equal(a, b)


@pytest.mark.parametrize("bad", ["int16", "float", "mixed_lai", "mixed_tif"])
def test_other_or_mixed_index_dtypes_rejected(bad):
    mgr = _mgr()
    W, li, lai, tif, grad, lad = _inputs()
    li, lai, tif_m, tif_a = _as(torch.int32, li, lai, tif, tif)
    if bad == "int16":
        li, lai, tif_m, tif_a = _as(torch.int16, li, lai, tif_m, tif_a)
    elif bad == "float":
        li = li.float()
    elif bad == "mixed_lai":
        lai = lai.long()
    else:
        tif_m = tif_m.long()
    with pytest.raises(ValueError, match="int32 or all be int64"):
        mgr.lprojection_backward_na1_nonsmooth(grad, W, li, lai, tif_m, tif_a, 256)
    with pytest.raises(ValueError, match="int32 or all be int64"):
        mgr.lprojection_forward_smooth(W, li, lai, lad, tif_m, tif_a, True, 256)


@pytest.mark.parametrize("cls", ["FusedManifestoHardLUT", "FusedManifestoSoftLUT", "FusedSoftSignHardLUT",
                                 "FusedSoftSignSmoothLUT"])
def test_int32_native_training_stores_no_int64_cell_index(cls):
    """The per-(token, table) index tensors the native Functions save: going from index_dtype=int64 to int32 moves
    exactly three of them -- the cell indices li / lai and the table ids -- to int32 (no int64 copy is made). What stays
    int64 is not a cell index: j* (torch.min's argmin) and the anchor-derived a_glob / b_glob / batch_offset."""
    _mgr()
    spec = LUTSpec(h_in=4, h_out=4, tph=16, nap=8, d_in=16, d_out=24)
    B, nt = 256, 4 * 16
    counts = {}
    for dt in (torch.int64, torch.int32):
        m = getattr(lx, cls)(spec, seed=1, backend="native", index_dtype=dt).cuda().train()
        x = torch.randn(B, 4, 16, device="cuda", requires_grad=True)
        m(x).sum().backward()
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            y = m(x)
        y.sum().backward()
        per_table = [t.dtype for t in saved if not t.dtype.is_floating_point and t.numel() == B * nt]
        counts[dt] = (per_table.count(torch.int64), per_table.count(torch.int32))
    (i64_a, i32_a), (i64_b, i32_b) = counts[torch.int64], counts[torch.int32]
    assert i32_a == 0 and i32_b == 3 and i64_a - i64_b == 3, counts


_OVERFLOW = textwrap.dedent("""
    import torch
    from spiky.lutorch_ex.cartridges._native_ops import native_manager
    mgr = native_manager()
    E, O, B = (2 ** 20) + 1024, 2048, 8            # E * O = 2,149,580,800 > 2^31 weight elements, one table
    W = torch.zeros(1, E, O, device="cuda")
    W[0, E - 1] = 1.0
    li = torch.full((B, 1), E - 1, device="cuda", dtype=torch.int32)    # widx = (E - 1) * O + o > 2^31
    lai = torch.full((B, 1, 1), E - 2, device="cuda", dtype=torch.int32)
    tif = torch.zeros(B, device="cuda", dtype=torch.int32)
    grad = torch.ones(B, 1, O, device="cuda")
    wg, gm, ga = mgr.lprojection_backward_na1_nonsmooth(grad, W, li, lai, tif, tif, 256)
    torch.cuda.synchronize()
    ok = bool((wg[0, E - 1] == B).all()) and bool((gm == O).all()) and bool((ga == 0).all()) \\
        and float(wg.abs().sum()) == B * O
    print("EXACT" if ok else "WRONG")
""")


def test_int32_indices_keep_int64_offsets_past_2_31_weight_elements():
    """The overflow trap: with int32 STORED indices, (table * n_entries + entry) * n_outputs + o must still be computed
    in int64 -- here it reaches (2^20 + 1023) * 2048 > 2^31. If it were index_t arithmetic it would wrap (a crash or
    wrong rows), so this runs in a subprocess (an illegal access poisons the CUDA context). ~17 GB of GPU memory."""
    _mgr()
    import gc
    gc.collect()
    torch.cuda.empty_cache()             # the subprocess needs ~17.2 GiB; free what earlier tests left cached here
    if torch.cuda.mem_get_info()[0] < 18 * 2 ** 30:
        pytest.skip("needs ~18 GiB of free GPU memory")
    r = subprocess.run([sys.executable, "-c", _OVERFLOW], capture_output=True, text=True, timeout=900)
    assert r.returncode == 0 and "EXACT" in r.stdout, (r.stdout[-400:], r.stderr[-800:])
