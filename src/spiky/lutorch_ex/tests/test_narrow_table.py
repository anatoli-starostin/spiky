"""ConfidenceLUT table_dtype: the fused gather read from an fp32 / bf16 / fp8 copy of the fp32 master.

Oracle: the default fp32 embedding_bag read. What is exact and what is approximate (measured 2026-10-09):
  * table gradient: BIT-IDENTICAL for every table_dtype (it goes to the fp32 master from the fp32 output grad and the
    fp32 scores; the narrow copy is not involved);
  * output / score grad / input grad: table_dtype=float32 agrees to fp32 re-association (~1e-7); bf16 ~1.6e-3 and
    fp8 (tensorwise e4m3) ~2.6e-2 relative - the rounding of the table values.
The cache tests are the main safety net: a narrow copy that survives a weight update trains on stale weights.
"""
import contextlib

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex import LUTSpec
from spiky.lutorch_ex.cartridges._narrow_table import invalidate_narrow_tables

SPEC = LUTSpec(h_in=4, h_out=4, tph=16, nap=8, d_in=16, d_out=12)
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
NARROW = [torch.bfloat16, torch.float8_e4m3fn]
# Measured (CPU + one CUDA GPU, this fixture): output / input grad ~1.6e-3 bf16, ~2.6e-2 fp8. The single-scalar score
# parameters (beta, gamma, tau) each sum many terms and cancel, so their relative error is larger and seed/device
# dependent: up to 3.9e-2 (tau, bf16) and 2.9e-1 (tau, fp8) measured.
REL_TOL = {torch.float32: 1e-5, torch.bfloat16: 5e-3, torch.float8_e4m3fn: 6e-2}
SCALAR_TOL = {torch.float32: 1e-5, torch.bfloat16: 0.1, torch.float8_e4m3fn: 0.5}


@contextlib.contextmanager
def _deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def _pair(device, table_dtype, read_top_n=1):
    torch.manual_seed(0)
    ref = lx.ConfidenceLUT(SPEC, seed=1, read_top_n=read_top_n).to(device).train()
    with torch.no_grad():
        ref.weights.normal_(0, 0.05)                              # trained-like spread, not the 1e-3 init
    m = lx.ConfidenceLUT(SPEC, seed=1, read_top_n=read_top_n, table_dtype=table_dtype).to(device).train()
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


def test_default_is_none_and_unchanged():
    m = lx.ConfidenceLUT(SPEC, seed=1)
    assert m.table_dtype is None and m._forward_extra_args() == ()


@pytest.mark.parametrize("bad", [torch.float16, torch.int8, "bf16"])
def test_bad_table_dtype_rejected(bad):
    with pytest.raises(ValueError, match="table_dtype"):
        lx.ConfidenceLUT(SPEC, seed=1, table_dtype=bad)


def test_quantised_refuses_table_dtype():
    with pytest.raises(ValueError, match="table_dtype"):
        lx.QuantisedConfidenceLUT(SPEC, seed=1, table_dtype=torch.bfloat16)


@pytest.mark.parametrize("dt", NARROW)
def test_copy_is_genuinely_narrow(dt):
    m = lx.ConfidenceLUT(SPEC, seed=1, table_dtype=dt)
    table, _ = m._forward_extra_args()
    assert table.dtype == dt and table.element_size() == torch.tensor([], dtype=dt).element_size() < 4
    assert table.untyped_storage().nbytes() * (4 // table.element_size()) == m.weights.untyped_storage().nbytes()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dt", [torch.float32] + NARROW)
@pytest.mark.parametrize("read_top_n", [1, 2])
def test_against_fp32_embedding_bag_oracle(device, dt, read_top_n):
    # Both sides must run in the same (compiled) mode: after the suite's many configurations the dynamo recompile
    # limit would send one side to eager, where the n=2 blend weights s*(1-v), s*v round differently by 1 ulp - and
    # the table grad (sum of psw * grad) is only bit-identical given identical scores.
    import torch._dynamo
    torch._dynamo.reset()
    ref, m = _pair(device, dt, read_top_n)
    x = torch.randn(512, SPEC.h_in, SPEC.d_in, device=device)
    g = torch.randn(512, SPEC.h_out, SPEC.d_out, device=device)
    with _deterministic():
        yr, gxr, gr = _run(ref, x, g)
        yn, gxn, gn = _run(m, x, g)
    assert torch.equal(gn["weights"], gr["weights"]), "table grad must be bit-identical (fp32 master path)"
    assert _rel(yn, yr) < REL_TOL[dt]
    assert _rel(gxn, gxr) < REL_TOL[dt]
    for n in gr:
        if n != "weights":
            assert _rel(gn[n], gr[n]) < SCALAR_TOL[dt], n
    assert gn["weights"].dtype == torch.float32


@pytest.mark.parametrize("dt", NARROW)
def test_cache_casts_once_across_forwards(dt):
    m = lx.ConfidenceLUT(SPEC, seed=1, table_dtype=dt).train()
    x = torch.randn(64, SPEC.h_in, SPEC.d_in)
    for _ in range(16):                                          # 16 micro-batches, no update in between
        m(x).sum().backward()
    assert m._narrow_cache.n_casts == 1


def _fresh(m, dt):
    W = m.weights.detach().float()
    return W.to(dt) if dt == torch.bfloat16 else (W * (448.0 / W.abs().max().clamp_min(1e-30))).to(dt)


@pytest.mark.parametrize("dt", NARROW)
@pytest.mark.parametrize("make_opt", [
    lambda ps: torch.optim.SGD(ps, lr=1e-2),
    lambda ps: torch.optim.AdamW(ps, lr=1e-2),
    pytest.param(lambda ps: torch.optim.AdamW(ps, lr=1e-2, fused=True),
                 marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="fused AdamW needs CUDA")),
], ids=["SGD", "AdamW-foreach", "AdamW-fused"])
def test_cache_refreshed_after_every_optimizer_step(dt, make_opt):
    """Fused AdamW updates params WITHOUT bumping _version (measured, torch 2.9.1); the global step hook must still
    invalidate, and the next forward must see the updated master - exactly one recast per step."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = lx.ConfidenceLUT(SPEC, seed=1, table_dtype=dt).to(dev).train()
    opt = make_opt(m.parameters())
    x = torch.randn(64, SPEC.h_in, SPEC.d_in, device=dev)
    for step in range(3):
        for _ in range(4):
            m(x).sum().backward()
        before = m._narrow_cache.n_casts
        opt.step()
        opt.zero_grad(set_to_none=True)
        m(x)
        assert m._narrow_cache.n_casts == before + 1, "exactly one recast per optimizer step"
        assert torch.equal(m._narrow_cache.table.float(), _fresh(m, dt).float()), "stale narrow copy"


def test_copy_under_no_grad_recasts_and_data_write_needs_invalidate():
    m = lx.ConfidenceLUT(SPEC, seed=1, table_dtype=torch.bfloat16)
    x = torch.randn(8, SPEC.h_in, SPEC.d_in)
    m(x)
    with torch.no_grad():
        m.weights.copy_(m.weights * 2)                           # bumps _version -> recast
    m(x)
    assert m._narrow_cache.n_casts == 2 and torch.equal(m._narrow_cache.table, m.weights.detach().bfloat16())
    m.weights.data.mul_(0.5)                                     # does NOT bump _version: documented, needs invalidate
    m(x)
    assert m._narrow_cache.n_casts == 2
    invalidate_narrow_tables(m)
    m(x)
    assert m._narrow_cache.n_casts == 3 and torch.equal(m._narrow_cache.table, m.weights.detach().bfloat16())


def test_float32_reads_the_master_without_a_copy():
    m = lx.ConfidenceLUT(SPEC, seed=1, table_dtype=torch.float32)
    table, _ = m._forward_extra_args()
    assert table.data_ptr() == m.weights.data_ptr() and m._narrow_cache.n_casts == 0
