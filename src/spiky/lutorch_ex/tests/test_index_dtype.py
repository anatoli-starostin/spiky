"""index_dtype (every cartridge): int32 (the default when the table fits) vs int64 stored cell indices, produced once in
ManifestoLUT._addresses.

What is bit-identical (torch.equal on the output and on EVERY gradient: tables, score / blend / temperature
parameters, ProjectionMHL compress/decompress, input), measured 2026-10-09:
  * TRAINING, every cartridge, CPU and CUDA, under torch.use_deterministic_algorithms(True). Without it the int64
    baseline itself is not run-to-run reproducible (atomic scatters in the input-grad and, on the pure CPU path,
    the table-grad index_put accumulate, ~1e-7 relative), so two int64 runs already differ.
  * The fused twins on their tier1 (embedding_bag) path. Their NATIVE CUDA kernels read int32 indices directly
    (templated on the index type) and are nondeterministic in their own baseline (custom-kernel atomics), so there
    only closeness within that baseline noise is asserted; exact per-kernel parity is in test_native_index_dtype.py.
  * EVAL: bit-identical, CPU and CUDA (one compiled CUDA case is asserted with a tolerance for an Inductor
    autotuning reason, not an index-dtype one -- see test_eval_int32_vs_int64).
"""
import contextlib

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex import LUTSpec, ProjectionMHL

GEOM = dict(h_in=4, h_out=4, tph=16, nap=8, d_in=16, d_out=12)     # flat index up to 4*16*256-1 = 16383
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# (class name, extra kwargs). Fused twins pinned to tier1 here; their native path is checked separately below.
CARTRIDGES = [
    ("ManifestoHardLUT", {}),
    ("ManifestoSoftLUT", {}),
    ("SoftSignHardLUT", {}),
    ("SoftSignSmoothLUT", {}),
    ("ConfidenceLUT", dict(read_top_n=1)),
    ("ConfidenceLUT", dict(read_top_n=2)),
    ("QuantisedConfidenceLUT", dict(read_top_n=1)),
    ("QuantisedConfidenceLUT", dict(read_top_n=2)),
    ("FusedManifestoHardLUT", dict(backend="tier1")),
    ("FusedManifestoSoftLUT", dict(backend="tier1")),
    ("FusedSoftSignHardLUT", dict(backend="tier1")),
    ("FusedSoftSignSmoothLUT", dict(backend="tier1")),
]
FUSED = [n for n, _ in CARTRIDGES if n.startswith("Fused")]


@contextlib.contextmanager
def _deterministic(on):
    prev = torch.are_deterministic_algorithms_enabled()
    if on:
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def _run(name, kw, index_dtype, device, wrap, spec):
    torch.manual_seed(0)
    cart = getattr(lx, name)(spec, seed=1, table_dropout_rate=0.2, index_dtype=index_dtype, **kw)
    gen = torch.Generator().manual_seed(1)
    if wrap:
        m = ProjectionMHL(cart, d_model=64).to(device).train()
        torch.nn.init.normal_(m.decompress.weight, std=0.02)
        x = torch.randn(256, 64, generator=gen)
    else:
        m = cart.to(device).train()
        x = torch.randn(256, spec.h_in, spec.d_in, generator=gen)
    x = x.to(device).requires_grad_(True)
    torch.manual_seed(5)                                         # same table-dropout mask in both runs
    y = m(x)
    y.float().pow(2).sum().backward()
    grads = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    return y.detach(), x.grad.clone(), grads


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,kw", CARTRIDGES, ids=[f"{n}{''.join(f'-{k}{v}' for k, v in kw.items())}"
                                                       for n, kw in CARTRIDGES])
@pytest.mark.parametrize("wrap", [False, True], ids=["bare", "projection"])
def test_int32_bit_identical_to_int64(device, name, kw, wrap):
    spec = LUTSpec(**GEOM)
    with _deterministic(True):
        y64, gx64, g64 = _run(name, kw, torch.int64, device, wrap, spec)
        y32, gx32, g32 = _run(name, kw, torch.int32, device, wrap, spec)
    assert torch.equal(y64, y32), "forward output differs"
    assert torch.equal(gx64, gx32), "input gradient differs"
    assert g64.keys() == g32.keys() and g64
    for n in g64:
        assert torch.equal(g64[n], g32[n]), f"grad of {n} differs"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the native kernels are CUDA-only")
@pytest.mark.parametrize("name", FUSED)
def test_fused_native_close_within_baseline_noise(name):
    """Native kernels, end to end: they now read the int32 indices directly. int64 is not reproducible run-to-run
    (custom-kernel atomics); int32 must stay within the same noise (measured: both ~4-7e-9 absolute on table grads
    ~5e-2). Exact per-kernel parity: test_native_index_dtype.py."""
    from spiky.lutorch_ex.cartridges._native_ops import native_available
    if not native_available(torch.device("cuda")):
        pytest.skip("native lutorch_cuda ops not available")
    spec = LUTSpec(**GEOM)
    y64, gx64, g64 = _run(name, dict(backend="native"), torch.int64, "cuda", False, spec)
    y32, gx32, g32 = _run(name, dict(backend="native"), torch.int32, "cuda", False, spec)
    assert torch.equal(y64, y32)
    torch.testing.assert_close(gx32, gx64, rtol=1e-6, atol=1e-7)
    for n in g64:
        torch.testing.assert_close(g32[n], g64[n], rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,kw", CARTRIDGES, ids=[n + str(kw.get("read_top_n", "")) for n, kw in CARTRIDGES])
def test_eval_int32_vs_int64(device, name, kw):
    spec = LUTSpec(**GEOM)
    outs = []
    for dt in (torch.int64, torch.int32):
        torch.manual_seed(0)
        m = getattr(lx, name)(spec, seed=1, index_dtype=dt, **kw).to(device).eval()
        with torch.no_grad():
            outs.append(m(torch.randn(128, spec.h_in, spec.d_in, generator=torch.Generator().manual_seed(2)).to(device)))
    if name == "ConfidenceLUT" and kw.get("read_top_n") == 1 and device == "cuda":
        # Not an index-dtype issue: Inductor's autotuner may pick different XBLOCK / num_warps winners for the int32-
        # and int64-indexed eval kernels, which changes the fp32 summation order (<= 1 ulp of the output scale at this
        # geometry). Bit-identical with triton.autotune_pointwise=False or under use_deterministic_algorithms(True),
        # and zero at the production geometry (h=16, tph=64, nap=8, d=48).
        torch.testing.assert_close(outs[1], outs[0], rtol=1e-6, atol=1e-7)
    else:
        assert torch.equal(*outs)


@pytest.mark.parametrize("device", DEVICES)
def test_deployed_int8_read_bit_identical(device):
    """The deploy-only packed-int8 object (n=2): int32 indices reach its int8 shift-add read too."""
    from spiky.lutorch_ex.cartridges.quantised_confidence import DeployedQuantisedConfidenceLUT
    spec = LUTSpec(**GEOM)
    torch.manual_seed(0)
    q = lx.QuantisedConfidenceLUT(spec, seed=1, read_top_n=2).to(device)
    payload = q.to_deployment()
    x = torch.randn(128, spec.h_in, spec.d_in, generator=torch.Generator().manual_seed(3)).to(device)
    outs = []
    for dt in (torch.int64, torch.int32):
        d = DeployedQuantisedConfidenceLUT(spec, payload["tensors"], payload["meta"], device=device, index_dtype=dt)
        with torch.no_grad():
            outs.append(d.eval()(x))
    assert torch.equal(*outs)


def test_default_is_int32_everywhere_when_the_table_fits():
    from spiky.lutorch_ex.cartridges.quantised_confidence import DeployedQuantisedConfidenceLUT
    spec = LUTSpec(**GEOM)
    for name, kw in CARTRIDGES:
        m = getattr(lx, name)(spec, seed=1, **kw)
        assert m.index_dtype == torch.int32 and m._index_dtype_auto, name
        assert getattr(lx, name)(spec, seed=1, index_dtype=torch.int64, **kw).index_dtype == torch.int64, name
    p = lx.QuantisedConfidenceLUT(spec, seed=1, read_top_n=2).to_deployment()
    assert DeployedQuantisedConfidenceLUT(spec, p["tensors"], p["meta"]).index_dtype == torch.int32


def test_resolve_index_dtype_rule():
    """None -> int32 iff n_groups * tph * 2^nap - 1 fits int32, else int64; explicit choices honoured; an explicit
    int32 that does not fit raises (never silently widened). Pure arithmetic on the geometry, no allocation."""
    from types import SimpleNamespace

    from spiky.lutorch_ex.cartridges.manifesto_base import resolve_index_dtype
    fits = SimpleNamespace(n_groups=16, tph=64, n_cells=256)                    # 262,144 rows
    edge = SimpleNamespace(n_groups=1, tph=1, n_cells=2 ** 31)                  # max flat index 2^31 - 1: fits
    over = SimpleNamespace(n_groups=2, tph=1, n_cells=2 ** 31)                  # max flat index 2^32 - 1: does not
    assert resolve_index_dtype(None, fits) == torch.int32
    assert resolve_index_dtype(None, edge) == torch.int32
    assert resolve_index_dtype(None, over) == torch.int64
    assert resolve_index_dtype(torch.int64, fits) == torch.int64
    assert resolve_index_dtype(torch.int64, over) == torch.int64
    with pytest.raises(ValueError, match="cannot hold the flat cell index"):
        resolve_index_dtype(torch.int32, over)


def test_auto_widens_a_call_too_large_for_int32_positions(monkeypatch):
    """Auto mode reads in int64 for a call with more than 2^31 - 1 possible index entries (PyTorch's CUDA embedding_bag
    forward asserts above that with int32); an explicit int32 is never widened. The limit is patched down so a small
    batch exercises it."""
    import spiky.lutorch_ex.cartridges.manifesto_base as mb
    spec = LUTSpec(**GEOM)                                         # G * tph * 2 = 128 entries per row
    monkeypatch.setattr(mb, "_INT32_MAX_ENTRIES", 128 * 10)
    x_small, x_big = torch.randn(10, spec.h_in, spec.d_in), torch.randn(11, spec.h_in, spec.d_in)
    auto = lx.ConfidenceLUT(spec, seed=1)
    assert auto._addresses(x_small)[2].dtype == torch.int32
    assert auto._addresses(x_big)[2].dtype == torch.int64
    explicit = lx.ConfidenceLUT(spec, seed=1, index_dtype=torch.int32)
    assert explicit._addresses(x_big)[2].dtype == torch.int32


@pytest.mark.parametrize("bad", [torch.int16, torch.int8, torch.uint8, torch.float32, "int32"])
def test_narrow_or_bad_index_dtype_rejected(bad):
    with pytest.raises(ValueError, match="index_dtype"):
        lx.ManifestoHardLUT(LUTSpec(**GEOM), seed=1, index_dtype=bad)


def test_addressing_produces_the_requested_dtype():
    spec = LUTSpec(**GEOM)
    m = lx.ConfidenceLUT(spec, seed=1, index_dtype=torch.int32)
    _, _, c, _, _, c_alt = m._addresses(torch.randn(8, spec.h_in, spec.d_in))
    assert c.dtype == torch.int32 and c_alt.dtype == torch.int32
    assert m.powers.dtype == torch.int64                         # the buffer (state_dict) is unchanged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="saved-tensor size check uses the CUDA compiled path")
def test_int32_halves_the_saved_index():
    # Fresh dynamo state: after the many configurations above, the recompile limit would send this to eager, where
    # min()'s int64 argmin (same numel as the flat index) is saved too and muddles the comparison.
    import torch._dynamo
    torch._dynamo.reset()
    spec = LUTSpec(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48)
    sizes = {}
    for dt in (torch.int64, torch.int32):
        m = lx.ConfidenceLUT(spec, seed=1, read_top_n=1, index_dtype=dt).cuda().train()
        x = torch.randn(1024, 16, 48, device="cuda", requires_grad=True)
        m(x).sum().backward()                                    # warm the compile
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(lambda t: saved.append(t) or t, lambda t: t):
            y = m(x)
        flat = [t for t in saved if t.dtype == dt and t.numel() == 1024 * 16 * 64]
        assert flat, f"expected the {dt} flat cell index among the saved tensors"
        # token-scaled integer tensors only (the fixed int64 anchor / head buffers are not cell indices)
        sizes[dt] = sum(t.numel() * t.element_size() for t in saved
                        if not t.dtype.is_floating_point and t.numel() >= 1024 * 16)
        y.sum().backward()
    assert sizes[torch.int32] * 2 == sizes[torch.int64], sizes
