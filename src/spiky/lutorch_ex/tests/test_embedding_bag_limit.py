"""The embedding_bag backward size guard (manifesto_base.EMBEDDING_BAG_*), and a check of the PyTorch limit it encodes.

PyTorch's CUDA embedding_bag backward overflows a 32-bit thread index when
    (numel // 10 + min(numel, num_weights)) * 32 * ceil(d_out / 32) > 2**31
(measured on torch 2.9.1+cu130; identical for int32 and int64 indices). ManifestoLUT.forward refuses a training call
above 2**30 for every cartridge whose training route reads through a differentiable embedding_bag.
"""
import subprocess
import sys
import textwrap

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex import LUTSpec
from spiky.lutorch_ex.cartridges.manifesto_base import (
    EMBEDDING_BAG_MAX_THREADS, EMBEDDING_BAG_THREAD_CLIFF, embedding_bag_backward_threads, max_safe_embedding_bag_rows)
from spiky.lutorch_ex.cartridges.quantised_confidence import DeployedQuantisedConfidenceLUT

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="the limit is in the CUDA embedding_bag backward")
GEOM_1536 = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48)   # 262,144 rows of width 48
SMALL = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8)


# ------------------------------------------------------------------------------------------------ predicate arithmetic

# Measured boundary pairs (last good numel, num_weights, d_out); first bad = last good + 1. torch 2.9.1, one RTX 5090.
MEASURED_LAST_GOOD = [
    (332_922_889, 262_144, 48), (332_922_889, 262_144, 33), (332_922_889, 262_144, 64),
    (221_074_779, 262_144, 65), (165_150_729, 262_144, 128), (668_467_209, 262_144, 16),
    (325_544_329, 1_000_000, 48), (335_544_169, 16, 48),
]


@pytest.mark.parametrize("numel,num_weights,d_out", MEASURED_LAST_GOOD)
def test_predicate_reproduces_every_measured_cliff(numel, num_weights, d_out):
    assert embedding_bag_backward_threads(numel, num_weights, d_out) <= EMBEDDING_BAG_THREAD_CLIFF
    assert embedding_bag_backward_threads(numel + 1, num_weights, d_out) > EMBEDDING_BAG_THREAD_CLIFF


def test_guard_keeps_a_2x_margin():
    assert EMBEDDING_BAG_MAX_THREADS * 2 == EMBEDDING_BAG_THREAD_CLIFF


@pytest.mark.parametrize("entries_per_row,num_weights,d_out", [(1024, 262_144, 48), (2048, 262_144, 48), (8, 64, 8)])
def test_max_safe_rows_is_the_exact_boundary(entries_per_row, num_weights, d_out):
    r = max_safe_embedding_bag_rows(entries_per_row, num_weights, d_out)
    assert embedding_bag_backward_threads(r * entries_per_row, num_weights, d_out) <= EMBEDDING_BAG_MAX_THREADS
    assert embedding_bag_backward_threads((r + 1) * entries_per_row, num_weights, d_out) > EMBEDDING_BAG_MAX_THREADS


# ------------------------------------------------------------------------------------------------ which cartridges opt in

def _x(spec, device="cpu", n=4):
    return torch.zeros(n, spec.h_in, spec.d_in, device=device)


@pytest.mark.parametrize("make,expected", [
    (lambda: lx.ConfidenceLUT(SMALL, seed=1, read_top_n=1), 1),
    (lambda: lx.ConfidenceLUT(SMALL, seed=1, read_top_n=2), 2),
    (lambda: lx.ConfidenceLUT(SMALL, seed=1, read_top_n=2, table_dtype=torch.float32), 2),
    (lambda: lx.QuantisedConfidenceLUT(SMALL, seed=1, read_top_n=1), 1),
    (lambda: lx.QuantisedConfidenceLUT(SMALL, seed=1, read_top_n=2), 2),
    (lambda: lx.SoftSignHardLUT(SMALL, seed=1), 1),          # value read is over c only; the surrogate is detached
    (lambda: lx.SoftSignSmoothLUT(SMALL, seed=1), 2),
    (lambda: lx.ManifestoHardLUT(SMALL, seed=1), 0),         # advanced indexing
    (lambda: lx.ManifestoSoftLUT(SMALL, seed=1), 0),
    (lambda: lx.FusedManifestoHardLUT(SMALL, seed=1), 0),    # FusedHardSTE (index_add_) / native
    (lambda: lx.FusedManifestoSoftLUT(SMALL, seed=1, backend="tier1"), 2),
    (lambda: lx.FusedManifestoSoftLUT(SMALL, seed=1, backend="native"), 0),
    (lambda: lx.FusedManifestoSoftLUT(SMALL, seed=1, backend="pure"), 0),
    (lambda: lx.FusedSoftSignHardLUT(SMALL, seed=1, backend="tier1"), 1),
    (lambda: lx.FusedSoftSignHardLUT(SMALL, seed=1, backend="native"), 0),
    (lambda: lx.FusedSoftSignSmoothLUT(SMALL, seed=1, backend="tier1"), 2),
    (lambda: lx.FusedSoftSignSmoothLUT(SMALL, seed=1, backend="native"), 0),
], ids=["conf-n1", "conf-n2", "conf-n2-fused-read", "quant-n1", "quant-n2", "softsign-hard", "softsign-smooth",
        "manifesto-hard", "manifesto-soft", "fused-manifesto-hard", "fused-manifesto-soft-tier1",
        "fused-manifesto-soft-native", "fused-manifesto-soft-pure", "fused-softsign-hard-tier1",
        "fused-softsign-hard-native", "fused-softsign-smooth-tier1", "fused-softsign-smooth-native"])
def test_cells_per_table_opt_in(make, expected):
    m = make().train()
    assert m._embedding_bag_cells_per_table(_x(SMALL)) == expected
    assert m._embedding_bag_cells_per_table_max() >= expected
    assert (m.max_safe_microbatch_tokens is None) == (m._embedding_bag_cells_per_table_max() == 0)


def test_auto_routes_follow_pick():
    """auto backends opt in only on the route that reads through embedding_bag (CPU: no native extension)."""
    x = _x(SMALL)
    assert lx.FusedSoftSignSmoothLUT(SMALL, seed=1).train()._embedding_bag_cells_per_table(x) == 2   # auto -> tier1
    assert lx.FusedSoftSignHardLUT(SMALL, seed=1).train()._embedding_bag_cells_per_table(x) == 1     # no native: tier1
    assert lx.FusedManifestoSoftLUT(SMALL, seed=1).train()._embedding_bag_cells_per_table(x) == 0    # CPU auto: pure


def test_deployed_never_opts_in():
    q = lx.QuantisedConfidenceLUT(SMALL, seed=1, read_top_n=2)
    p = q.to_deployment()
    d = DeployedQuantisedConfidenceLUT(SMALL, p["tensors"], p["meta"])
    assert d._embedding_bag_cells_per_table(_x(SMALL)) == 0 and d.max_safe_microbatch_tokens is None


def test_max_safe_microbatch_tokens_at_geom_1536():
    spec = LUTSpec(**GEOM_1536)
    n1 = lx.ConfidenceLUT(spec, seed=1, read_top_n=1).max_safe_microbatch_tokens
    n2 = lx.ConfidenceLUT(spec, seed=1, read_top_n=2).max_safe_microbatch_tokens
    assert n1 == max_safe_embedding_bag_rows(1024, 262_144, 48) and n2 == max_safe_embedding_bag_rows(2048, 262_144, 48)
    assert 32_768 < n2 < n1                                      # today's 32k micro-batch is admitted by both


# ------------------------------------------------------------------------------------------------ the guard itself

def test_guard_is_cuda_training_grad_only():
    m = lx.ConfidenceLUT(LUTSpec(**GEOM_1536), seed=1, read_top_n=2).train()
    huge = torch.empty(10 ** 7, 16, 48, device="meta")           # far above the limit, but not a CUDA tensor
    m._check_embedding_bag_limits(huge)                          # CPU/meta: the limit is a CUDA kernel's


@needs_cuda
@pytest.mark.parametrize("read_top_n", [1, 2])
def test_guard_refuses_just_above_and_admits_at_the_limit(read_top_n):
    m = lx.ConfidenceLUT(LUTSpec(**GEOM_1536), seed=1, read_top_n=read_top_n).cuda().train()
    limit = m.max_safe_microbatch_tokens
    m._check_embedding_bag_limits(torch.empty(limit, 16, 48, device="cuda"))
    over = torch.empty(limit + 1, 16, 48, device="cuda")
    with pytest.raises(RuntimeError, match="gradient accumulation"):
        m._check_embedding_bag_limits(over)
    with pytest.raises(RuntimeError, match=f"{limit:,} tokens per micro-batch"):
        m(over)                                                  # through forward, before any compute
    with torch.no_grad():
        m._check_embedding_bag_limits(over)                      # no backward -> no limit
    m.eval()._check_embedding_bag_limits(over)


@needs_cuda
def test_guard_under_an_outer_compile():
    """nanochat compiles the whole model: the guard must add no graph break and still raise its own message (the
    sizes are symbolic while dynamo traces; the message is built in a compiler-disabled helper)."""
    import torch._dynamo
    from spiky.lutorch_ex import ProjectionMHL
    torch._dynamo.reset()
    m = ProjectionMHL(lx.ConfidenceLUT(LUTSpec(**GEOM_1536), seed=1, read_top_n=2), d_model=1536).cuda().train()
    assert torch._dynamo.explain(m)(torch.randn(64, 1536, device="cuda")).graph_break_count == 0
    torch._dynamo.reset()
    cm = torch.compile(m, dynamic=True)
    cm(torch.randn(96, 1536, device="cuda")).sum().backward()
    with pytest.raises(RuntimeError, match="gradient accumulation"):
        cm(torch.empty(m.cartridge.max_safe_microbatch_tokens + 1, 1536, device="cuda"))


@needs_cuda
def test_guard_skips_routes_without_embedding_bag():
    m = lx.FusedManifestoHardLUT(LUTSpec(**GEOM_1536), seed=1).cuda().train()
    m._check_embedding_bag_limits(torch.empty(10 ** 6, 16, 48, device="cuda"))   # FusedHardSTE / native: unaffected


# ------------------------------------------------------------------------------------------------ the PyTorch limit itself

_REPRO = textwrap.dedent("""
    import sys, torch, torch.nn.functional as F
    N, NW, D = int(sys.argv[1]), 16, 1024
    idx = torch.arange(N, device="cuda") % NW
    offsets = torch.arange(0, N, 64, device="cuda")
    w = torch.ones(NW, D, device="cuda", requires_grad=True)
    F.embedding_bag(idx, w, offsets, mode="sum").sum().backward()
    torch.cuda.synchronize()
    expect = torch.full((NW,), N // NW, device="cuda", dtype=torch.float32)
    expect[: N % NW] += 1
    sys.exit(0 if bool((w.grad == expect.unsqueeze(1)).all()) else 3)   # 3 = ran but the gradient is wrong
""")


def _run_repro(numel):
    return subprocess.run([sys.executable, "-c", _REPRO, str(numel)], capture_output=True, text=True, timeout=600)


@needs_cuda
def test_installed_torch_still_has_the_measured_cliff():
    """Guards the guard: if PyTorch changes compute_grad_weight_bags, EMBEDDING_BAG_* may be wrong in EITHER direction.

    16 rows of width 1024 put the predicted cliff at 20,971,370 index entries (~1.3 GB): the last good size must
    still pass with an exact gradient and the first bad one must still fail (crash, or a wrong gradient). The failing
    half runs in a SUBPROCESS: an illegal memory access poisons the CUDA context of the process that hits it. If this
    test fails, re-measure the cliff on the new torch and update the constants in manifesto_base.py."""
    last_good, first_bad = 20_971_369, 20_971_370
    assert embedding_bag_backward_threads(last_good, 16, 1024) <= EMBEDDING_BAG_THREAD_CLIFF
    assert embedding_bag_backward_threads(first_bad, 16, 1024) > EMBEDDING_BAG_THREAD_CLIFF
    ok = _run_repro(last_good)
    assert ok.returncode == 0, f"the predicted-safe size failed (rc={ok.returncode}): {ok.stderr[-600:]}"
    bad = _run_repro(first_bad)
    assert bad.returncode != 0, ("the predicted crash size now runs with an exact gradient: PyTorch's embedding_bag "
                                 "backward limit has moved -- re-measure and update EMBEDDING_BAG_* in manifesto_base")
