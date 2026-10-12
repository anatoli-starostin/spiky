"""The ConfidenceLUT read: compiled, the fused gather × weight → sum (cartridges/_fused_read.py); eager, one
F.embedding_bag. The two forms of the same maths agree to fp32 re-association, not bit for bit: the fused table
gradient is an index_add_ with atomic fp32 adds in a nondeterministic order (measured 1.1e-7 rel-norm against
embedding_bag's sorted reduction, 2026-10-09).
"""
import re

import pytest
import torch
import torch.nn.functional as F

import spiky.lutorch_ex as lx
from spiky.lutorch_ex import LUTSpec
from spiky.lutorch_ex.cartridges import confidence as conf
from spiky.lutorch_ex.cartridges import _fused_read as fr

SPEC = LUTSpec(h_in=4, h_out=4, tph=16, nap=8, d_in=16, d_out=12)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


@pytest.mark.parametrize("dtype,tol", [(torch.float32, 1e-5), (torch.float64, 1e-12)])
@pytest.mark.parametrize("device", DEVICES)
def test_the_two_forms_agree(device, dtype, tol):
    """fused_scored_read and F.embedding_bag: same value, weight gradient and table gradient. The tolerance is far
    above re-association (fp32 measured 1.1e-7) and far below any wrong row, index or weight (O(1) relative)."""
    torch.manual_seed(0)
    rows = torch.randint(0, 300, (64, 24), device=device)
    table0 = torch.randn(300, 12, device=device, dtype=dtype)
    w0 = torch.rand(64, 24, device=device, dtype=dtype)
    go = torch.randn(64, 12, device=device, dtype=dtype)
    got, want = [], []
    for read, out in ((lambda r, w, t: fr.fused_scored_read(r, w, t), got),
                      (lambda r, w, t: F.embedding_bag(r, t, per_sample_weights=w, mode="sum"), want)):
        table, w = table0.clone().requires_grad_(True), w0.clone().requires_grad_(True)
        y = read(rows, w, table)
        y.backward(go)
        out += [y.detach(), w.grad, table.grad]
    for g, r in zip(got, want):
        assert _rel(g, r) < tol


def test_eager_reads_through_embedding_bag_only(monkeypatch):
    """CPU (never compiled): train and eval run without ever touching the fused read."""
    def boom(*a, **k):
        raise AssertionError("the fused read ran eagerly")
    monkeypatch.setattr(conf, "fused_scored_read", boom)
    for n in (1, 2):
        m = lx.ConfidenceLUT(SPEC, seed=1, read_top_n=n)
        x = torch.randn(32, SPEC.h_in, SPEC.d_in, requires_grad=True)
        m.train()(x).sum().backward()
        with torch.no_grad():
            m.eval()(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the cartridge forward compiles on CUDA only")
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("read_top_n", [1, 2])
def test_compiled_cartridge_matches_its_eager_form(read_top_n, anchor_mode):
    """m(x) on CUDA is compiled (the fused read); m._forward_impl(x) is eager (embedding_bag). Value, input gradient
    and every parameter gradient, train mode with table dropout under the same keep mask."""
    import torch._dynamo
    torch._dynamo.reset()
    spec = LUTSpec(h_in=4, h_out=4, tph=16, nap=8, d_in=16, d_out=12, anchor_mode=anchor_mode)
    m = lx.ConfidenceLUT(spec, seed=1, read_top_n=read_top_n, table_dropout_rate=0.2).cuda().train()
    with torch.no_grad():
        m.weights.normal_(0, 0.05)                                   # trained-like spread, not the 1e-3 init
    torch.manual_seed(0)
    x = torch.randn(512, spec.h_in, spec.d_in, device="cuda")
    go = torch.randn(512, spec.h_out, spec.d_out, device="cuda")
    keep = torch.rand(512, spec.n_groups, spec.tph, device="cuda") < 0.8
    m._table_dropout_mask = lambda B, dev, dt: keep.to(dt) / 0.8      # same mask on both sides (patched before tracing)
    res = []
    for fwd in (m, m._forward_impl):
        m.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        fwd(xi).backward(go)
        res.append((fwd(x).detach(), xi.grad, {n: p.grad.clone() for n, p in m.named_parameters()}))
    (yc, gxc, gc), (ye, gxe, ge) = res
    assert _rel(yc, ye) < 1e-5 and _rel(gxc, gxe) < 1e-5
    for n in ge:
        assert _rel(gc[n], ge[n]) < 1e-5, n


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
    m = lx.ConfidenceLUT(SPEC, seed=1, read_top_n=read_top_n).cuda().train()
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
    """The real training path (ConfidenceLUT n=2, compiled train forward = the fused read with the library's dynamic=True,
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
    m = lx.ConfidenceLUT(_LARGE_SPEC, seed=1, read_top_n=2).cuda().train()
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
