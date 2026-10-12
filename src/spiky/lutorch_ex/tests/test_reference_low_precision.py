"""The reference cartridges in bf16 / fp16 (spec amendment C): the same maths in fp32, exactly one rounding per leaf.

For a low-precision table and/or input, the reference class must give exactly the fp32 run on the same
(low-precision-representable) values, rounded once: the output to the input dtype, the input gradient to the input
dtype, and each parameter's gradient to that parameter's dtype. "Exactly" is bitwise: the low-precision forward runs
the identical fp32 computation, so anything beyond the single final rounding (a bf16 intermediate, a bf16
accumulator, a second rounding) breaks equality. Every reference class, both weights, both read_top_n, CPU and CUDA.
On CUDA the gradient scatters use atomics, so the fp32 baseline is itself not reproducible run to run; the test runs
under ``torch.use_deterministic_algorithms`` there (as test_index_dtype.py does), which makes both runs bitwise
comparable.
"""
import contextlib

import pytest
import torch

from spiky.lutorch_ex import ConfidenceLUT, LUTSpec, ManifestoHardLUT, ManifestoSoftLUT, ProjectionMHL
from spiky.lutorch_ex.tests._weights import softsign

CLASSES = [ManifestoHardLUT, ManifestoSoftLUT, softsign(ManifestoHardLUT), softsign(ManifestoSoftLUT),
           ("ConfidenceLUT n=1", lambda s, **kw: ConfidenceLUT(s, read_top_n=1, **kw)),
           ("ConfidenceLUT n=2", lambda s, **kw: ConfidenceLUT(s, read_top_n=2, **kw))]
IDS = [c[0] if isinstance(c, tuple) else c.__name__ for c in CLASSES]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
LOW = [torch.bfloat16, torch.float16]
#: (table low?, input low?): both, the table only, the input only.
CASES = [(True, True), (True, False), (False, True)]


def _make(C, spec):
    make = C[1] if isinstance(C, tuple) else C
    return make(spec, seed=3, weight_init_std=0.5, table_dropout_rate=0.0)


def _run(m, x, go):
    xi = x.clone().requires_grad_(True)
    y = m(xi)
    y.backward(go.to(y.dtype))
    out = {"y": y.detach(), "grad x": xi.grad}
    out.update({f"grad {n}": p.grad for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return out


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("low", LOW, ids=["bf16", "fp16"])
@pytest.mark.parametrize("table_low,input_low", CASES, ids=["both", "table", "input"])
@pytest.mark.parametrize("pattern", [(4, 4), (4, 1)], ids=["per-head", "fan-in"])
@pytest.mark.parametrize("C", CLASSES, ids=IDS)
def test_low_precision_is_the_fp32_run_rounded_once(C, pattern, table_low, input_low, low, device):
    h_in, h_out = pattern
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=8, nap=5, d_in=12, d_out=10)
    lo = _make(C, spec).to(device)
    if table_low:
        lo = lo.to(low)
    hi = _make(C, spec).to(device)
    hi.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in lo.state_dict().items()})
    torch.manual_seed(0)
    x = torch.randn(64, h_in, 12, device=device)
    go = torch.randn(64, h_out, 10, device=device)
    if input_low:
        x, go = x.to(low), go.to(low)
    with _deterministic():
        _compare(lo, hi, x, go)


@contextlib.contextmanager
def _deterministic():
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(prev)


def _compare(lo, hi, x, go):
    for train in (True, False):
        lo.train(train); hi.train(train)
        if train:
            a, b = _run(lo, x, go), _run(hi, x.float(), go.float())
        else:
            with torch.no_grad():
                a, b = {"y": lo(x)}, {"y": hi(x.float())}
        assert a.keys() == b.keys()
        for k in b:
            leaf_dtype = x.dtype if k in ("y", "grad x") else dict(lo.named_parameters())[k[5:]].dtype
            assert a[k].dtype == leaf_dtype, f"{k}: dtype {a[k].dtype}, want {leaf_dtype}"
            assert torch.equal(a[k], b[k].to(leaf_dtype)), \
                f"{k} ({'train' if train else 'eval'}): max |diff| {(a[k].float() - b[k].to(leaf_dtype).float()).abs().max():.3e}"


@pytest.mark.parametrize("device", DEVICES)
def test_projection_wrapping_a_reference_runs_in_bf16(device):
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=4, d_in=6, d_out=5)
    proj = ProjectionMHL(ManifestoHardLUT(spec, seed=0), d_model=10, bias=True).to(device).to(torch.bfloat16)
    x = torch.randn(4, 10, device=device, dtype=torch.bfloat16, requires_grad=True)
    y = proj(x)
    assert y.dtype == torch.bfloat16 and y.shape == (4, 10)
    y.float().sum().backward()
    assert x.grad.dtype == torch.bfloat16 and proj.cartridge.weights.grad.dtype == torch.bfloat16
