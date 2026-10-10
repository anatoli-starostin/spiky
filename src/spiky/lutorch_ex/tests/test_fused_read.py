"""ConfidenceLUT fused_read: the fused gather -> x score -> sum read from the fp32 master table.

Oracle: the default embedding_bag read. The table gradient is BIT-IDENTICAL (both go through
aten._embedding_bag_dense_backward from the same fp32 output grad and scores); output, score gradient and input gradient
agree to fp32 re-association (~1e-7, measured 2026-10-09).
"""
import contextlib

import pytest
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex import LUTSpec

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


def test_quantised_refuses_fused_read():
    with pytest.raises(ValueError, match="fused_read"):
        lx.QuantisedConfidenceLUT(SPEC, seed=1, fused_read=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("read_top_n", [1, 2])
def test_against_embedding_bag_oracle(device, read_top_n):
    # Both sides must run in the same (compiled) mode: after the suite's many configurations the dynamo recompile
    # limit would send one side to eager, where the n=2 blend weights s*(1-v), s*v round differently by 1 ulp - and
    # the table grad (sum of psw * grad) is only bit-identical given identical scores.
    import torch._dynamo
    torch._dynamo.reset()
    ref, m = _pair(device, read_top_n)
    x = torch.randn(512, SPEC.h_in, SPEC.d_in, device=device)
    g = torch.randn(512, SPEC.h_out, SPEC.d_out, device=device)
    with _deterministic():
        yr, gxr, gr = _run(ref, x, g)
        yn, gxn, gn = _run(m, x, g)
    assert torch.equal(gn["weights"], gr["weights"]), "table grad must be bit-identical"
    assert _rel(yn, yr) < REL_TOL
    assert _rel(gxn, gxr) < REL_TOL
    for n in gr:
        if n != "weights":
            assert _rel(gn[n], gr[n]) < REL_TOL, n
    assert gn["weights"].dtype == torch.float32
