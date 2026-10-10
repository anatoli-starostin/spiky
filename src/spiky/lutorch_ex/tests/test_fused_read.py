"""ConfidenceLUT fused_read: the fused gather -> x score -> sum read from the fp32 master table.

Oracle: the default embedding_bag read. Output, score gradient and input gradient agree to fp32 re-association (~1e-7,
measured 2026-10-09). The table gradient depends on the accumulation mode (cartridges/_fused_read.py TABLE_GRAD):
  * embedding_bag: BIT-IDENTICAL to the default read (both go through aten._embedding_bag_dense_backward from the same
    fp32 output grad and scores);
  * index_add (the default) / scatter_add: atomic fp32 adds in a nondeterministic order, so they agree with the sorted
    embedding_bag reduction to fp32 re-association (measured 1.1e-7 rel-norm), not bit for bit.
"""
import contextlib
import os
import re
import warnings

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex import LUTSpec
from spiky.lutorch_ex.cartridges import _fused_read as fr

SPEC = LUTSpec(h_in=4, h_out=4, tph=16, nap=8, d_in=16, d_out=12)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
REL_TOL = 1e-5


@contextlib.contextmanager
def _deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


@contextlib.contextmanager
def _table_grad(mode):
    prev = fr.TABLE_GRAD
    fr.TABLE_GRAD = mode
    try:
        yield
    finally:
        fr.TABLE_GRAD = prev


def _pair(device, read_top_n=1):
    torch.manual_seed(0)
    ref = lx.ConfidenceLUT(SPEC, seed=1, read_top_n=read_top_n).to(device).train()
    with torch.no_grad():
        ref.weights.normal_(0, 0.05)                              # trained-like spread, not the 1e-3 init
    m = lx.ConfidenceLUT(SPEC, seed=1, read_top_n=read_top_n, fused_read=True).to(device).train()
    m.load_state_dict(ref.state_dict())
    return ref, m


def _run(m, x, g):
    m.zero_grad(set_to_none=True)
    xi = x.clone().requires_grad_(True)
    y = m(xi)
    y.backward(g)
    return y.detach(), xi.grad, {n: p.grad.clone() for n, p in m.named_parameters()}


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def test_default_is_off():
    assert lx.ConfidenceLUT(SPEC, seed=1).fused_read is False


@pytest.mark.skipif("LUTORCH_EX_TABLE_GRAD" in os.environ, reason="LUTORCH_EX_TABLE_GRAD is set for this run")
def test_default_table_grad_is_index_add():
    assert fr.TABLE_GRAD == "index_add"


def test_quantised_refuses_fused_read():
    with pytest.raises(ValueError, match="fused_read"):
        lx.QuantisedConfidenceLUT(SPEC, seed=1, fused_read=True)


@pytest.mark.parametrize("mode", fr.TABLE_GRAD_MODES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("read_top_n", [1, 2])
def test_against_embedding_bag_oracle(device, read_top_n, mode):
    # Both sides must run in the same (compiled) mode: after the suite's many configurations the dynamo recompile
    # limit would send one side to eager, where the n=2 blend weights s*(1-v), s*v round differently by 1 ulp - and
    # the table grad (sum of psw * grad) is only bit-identical given identical scores.
    import torch._dynamo
    torch._dynamo.reset()
    ref, m = _pair(device, read_top_n)
    x = torch.randn(512, SPEC.h_in, SPEC.d_in, device=device)
    g = torch.randn(512, SPEC.h_out, SPEC.d_out, device=device)
    with _deterministic(), _table_grad(mode):
        yr, gxr, gr = _run(ref, x, g)
        yn, gxn, gn = _run(m, x, g)
    if mode == "embedding_bag":
        assert torch.equal(gn["weights"], gr["weights"]), "embedding_bag table grad must be bit-identical"
    else:
        # index_add_ / scatter_add_ accumulate with atomic fp32 adds whose ORDER is nondeterministic, so bit-equality
        # with the sorted embedding_bag reduction is not a meaningful assertion here. Measured: 1.1e-7 rel-norm; the
        # tolerance is ~100x that (any wrong row, index or weight is O(1) relative and fails it).
        ref_w = gr["weights"]
        torch.testing.assert_close(gn["weights"], ref_w, rtol=1e-5, atol=1e-5 * ref_w.abs().max().item())
        assert _rel(gn["weights"], ref_w) < REL_TOL
    assert _rel(yn, yr) < REL_TOL
    assert _rel(gxn, gxr) < REL_TOL
    for n in gr:
        if n != "weights":
            assert _rel(gn[n], gr[n]) < REL_TOL, n
    assert gn["weights"].dtype == torch.float32


# --------------------------------------------------------------------------------------- eager-memory warning (CUDA)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="the eager-rows warning is CUDA-only")
def test_eager_cuda_backward_warns_once_about_the_materialised_rows(monkeypatch):
    monkeypatch.setattr(fr, "EAGER_WARN_GIB", 0.0)
    monkeypatch.setattr(fr, "_EAGER_WARNED", False)
    idx = torch.randint(0, 64, (32, 8), device="cuda", dtype=torch.int32)
    w = torch.randn(64, 12, device="cuda", requires_grad=True)
    s = torch.rand(32, 8, device="cuda", requires_grad=True)
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        for _ in range(2):
            fr.fused_scored_read(idx, s, w).sum().backward()
    ours = [x for x in wl if "fused read running EAGER" in str(x.message)]
    assert len(ours) == 1 and issubclass(ours[0].category, RuntimeWarning)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the eager-rows warning is CUDA-only")
def test_compiled_backward_does_not_warn(monkeypatch):
    import torch._dynamo
    torch._dynamo.reset()
    monkeypatch.setattr(fr, "EAGER_WARN_GIB", 0.0)
    monkeypatch.setattr(fr, "_EAGER_WARNED", False)
    m = lx.ConfidenceLUT(SPEC, seed=1, fused_read=True).cuda().train()   # compiled train forward on CUDA
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always")
        for _ in range(2):
            m(torch.randn(256, SPEC.h_in, SPEC.d_in, device="cuda")).sum().backward()
    assert not [x for x in wl if "fused read running EAGER" in str(x.message)]


# --------------------------------------------------------------------------------------- fusion property (CUDA)

_KERNEL_RE = re.compile(r"async_compile\.triton\(\s*'([^']+)'\s*,\s*'''(.*?)'''", re.S)


def _compiled_step_code(read_top_n):
    """(dynamo counters snapshot, every Inductor output-code string) for one compiled fwd+bwd of the fused read,
    from a fresh dynamo state so no cached graph can mask a graph break."""
    import torch._dynamo
    from torch._dynamo.utils import counters
    from torch._inductor.utils import run_and_get_code
    torch._dynamo.reset()
    counters.clear()
    m = lx.ConfidenceLUT(SPEC, seed=1, read_top_n=read_top_n, fused_read=True).cuda().train()
    x = torch.randn(512, SPEC.h_in, SPEC.d_in, device="cuda", requires_grad=True)
    _, code = run_and_get_code(lambda: m(x).sum().backward())
    return ({k: dict(v) for k, v in counters.items()}, code)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused read compiles (and fuses) only on CUDA")
@pytest.mark.parametrize("read_top_n", [1, 2])
def test_compiled_fused_read_is_one_graph_with_the_scatter_fused(read_top_n):
    """PERFORMANCE property, asserted hard. The fused read's speed comes from Inductor fusing the score-gradient
    re-gather and the index_add_ table-gradient scatter into ONE generated kernel. A graph break (e.g. an untraceable
    call in _FusedScoredRead.backward) or a change that keeps them apart silently loses the whole win while every
    correctness test still passes - this happened once during development (an is_fake() check graph-broke the step).
    So: 0 graph breaks, 1 graph, and exactly one generated kernel that does the atomic scatter AND the score-grad
    reduction. Matched on kernel CONTENT (atomic_add / tl.sum), not on Inductor's generated kernel names."""
    if fr.TABLE_GRAD != "index_add":
        pytest.skip(f"LUTORCH_EX_TABLE_GRAD={fr.TABLE_GRAD} for this run; the fusion property is about index_add")
    counters, code = _compiled_step_code(read_top_n)
    breaks = counters.get("graph_break", {})
    assert sum(breaks.values()) == 0, f"graph break(s) in the compiled fused read: {list(breaks)[:2]}"
    assert counters.get("stats", {}).get("unique_graphs") == 1, counters.get("stats")
    src = "\n".join(code)
    kernels = _KERNEL_RE.findall(src)
    assert kernels, "no Triton kernels parsed from Inductor's output code - its format changed; update _KERNEL_RE"
    # The backward also has other, legitimate atomic scatters (the input-grad scatter into the margins, the gather
    # backward's index_put) - pointwise kernels with no reduction. The fused table-grad kernel is the one that does
    # BOTH the atomic scatter and the score-grad re-gather's reduction; if the two were split there would be none.
    fused = [name for name, body in kernels if "atomic_add" in body and "tl.sum(" in body]
    assert len(fused) == 1, (f"expected exactly one kernel holding both the table-grad scatter (atomic_add) and the "
                             f"score-grad reduction (tl.sum), got {fused}; kernels: {[n for n, _ in kernels]}")
    assert "_embedding_bag_dense_backward" not in src


# --------------------------------------------------------------------------------------- large sizes (CUDA)

_LARGE_SPEC = LUTSpec(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48)   # 262,144 rows x 48: the d24 geometry
_EB_CLIFF = 332_922_890          # embedding_bag dense backward crashes from here at this geometry (README)


def _need_gib(gib):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
    if total < gib:
        pytest.skip(f"needs ~{gib} GiB of GPU memory, this GPU has {total:.0f} GiB")
    torch.cuda.empty_cache()


def test_real_compiled_n2_path_past_the_embedding_bag_cliff():
    """The real training path (ConfidenceLUT n=2, fused read, compiled train forward with the library's dynamic=True,
    warmed at 32k tokens) at 180,000 tokens = 368.6M index entries: past embedding_bag's cliff (332.9M) AND past the
    numel * d_out = 2^34 point where a STATIC-shape bare scatter fails to compile (see the next test). Table grad
    checked against a chunked reference: embedding_bag's dense backward per 32k-token chunk (each below the cliff),
    summed, from the same addresses."""
    _need_gib(30)
    from spiky.lutorch_ex.cartridges._fused_ops import _global_cells
    import torch._dynamo
    torch._dynamo.reset()
    G, TPH, K, D = 16, 64, 256, 48
    tokens = 180_000
    assert tokens * G * TPH * 2 > max(_EB_CLIFF, 2 ** 34 // D)
    torch.manual_seed(0)
    m = lx.ConfidenceLUT(_LARGE_SPEC, seed=1, read_top_n=2, fused_read=True).cuda().train()
    with torch.no_grad():
        m.weights.normal_(0, 0.05)
    m(torch.randn(32_768, 16, 48, device="cuda")).sum().backward()          # warm: compile at the training size
    m.zero_grad(set_to_none=True)
    x = torch.randn(tokens, 16, 48, device="cuda")
    m(x).sum().backward()
    got = m.weights.grad.reshape(-1, D)
    with torch.no_grad():
        ref = torch.zeros_like(got)
        for a in range(0, tokens, 32_768):
            z, u, c, j, ua, ca = m._addresses(x[a:a + 32_768])
            s, v = m._score(u), m._blend_v(ua)
            idx = torch.cat([_global_cells(c, G, TPH, K), _global_cells(ca, G, TPH, K)], 2).reshape(-1, 2 * TPH)
            psw = torch.cat([s * (1 - v), s * v], 2).reshape(-1, 2 * TPH)
            nb, nper = idx.shape
            flat = idx.reshape(-1)
            ref += torch.ops.aten._embedding_bag_dense_backward(
                torch.ones(nb, D, device="cuda"), flat,
                torch.arange(nb, device="cuda", dtype=flat.dtype).repeat_interleave(nper),
                torch.full((nb,), nper, device="cuda", dtype=flat.dtype), torch.empty(0, device="cuda", dtype=flat.dtype),
                G * TPH * K, False, 0, psw.reshape(-1).float(), -1)
    assert torch.isfinite(got).all()
    assert _rel(got, ref) < 1e-5                                             # measured <= 1.1e-6 (2026-10-09)


def test_static_shape_bare_scatter_fails_loudly_above_2_pow_34():
    """Pinned PyTorch behaviour (torch 2.9.1): a bare index_add_ scatter compiled with STATIC shapes fails to compile
    once numel * d_out > 2^34 (Triton 'XBLOCK' too large) - a loud error, never silent corruption. The library's real
    path compiles with dynamic shapes and is unaffected (previous test, same order of size). If this ever stops
    raising, the static-shape behaviour changed: re-check it, then drop or invert this test."""
    _need_gib(20)
    import torch._dynamo
    torch._dynamo.reset()
    ROWS, D, PER = 262_144, 48, 64
    N = 358_000_000 // PER * PER                                             # numel * d_out = 1.718e10 > 2^34
    assert N * D > 2 ** 34

    def tablegrad(idx, s, go):
        src = (s.unsqueeze(-1) * go.unsqueeze(1)).reshape(-1, D)
        return go.new_zeros(ROWS, D).index_add_(0, idx.reshape(-1), src)

    idx = torch.empty(N, dtype=torch.int32, device="cuda")
    for a in range(0, N, 1 << 28):
        b = min(N, a + (1 << 28))
        idx[a:b] = (torch.arange(a, b, device="cuda") % ROWS).to(torch.int32)
    idx = idx.view(N // PER, PER)
    s = torch.ones(1, 1, device="cuda").expand(N // PER, PER)
    go = torch.ones(1, D, device="cuda").expand(N // PER, D)
    with pytest.raises(Exception, match="XBLOCK"):
        torch.compile(tablegrad, dynamic=False)(idx, s, go)
        torch.cuda.synchronize()
