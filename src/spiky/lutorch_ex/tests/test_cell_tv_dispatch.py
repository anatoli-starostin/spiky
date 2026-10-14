"""ManifestoLUT.cell_tv's dispatch: on CUDA the cell-TV penalty runs through torch.compile (the default), on the CPU or
with LUTORCH_EX_NO_COMPILE=1 through the plain eager formula (_cell_tv_eager). The compiled tier must give the eager
formula's value AND gradient -- the gradient ACCUMULATED into an existing .grad, as the trainers call it, once per
optimiser step after the micro-batch backwards.

fp32 / fp64: compiled vs eager on the same tables, to round-off. bf16 / fp16: compiled vs an fp64 evaluation of the same
tables, within a few units of the dtype's rounding (the compiled kernels accumulate in fp32; eager accumulates in the
table dtype, so the two need not agree with each other more closely than each agrees with fp64).

Every compiled test resets Dynamo first and checks that a graph was captured: _cell_tv_eager's compile entries are
shared by every instance and keyed by (table shape, nap, dtype), so past Dynamo's recompile limit a call silently runs
eager -- which
would make a compiled-vs-eager test compare eager with itself."""
import os
import subprocess
import sys
import textwrap

import pytest
import torch
import torch._dynamo
import torch.nn as nn
from torch._dynamo.utils import counters

from spiky.lutorch_ex import ConfidenceLUT, FusedManifestoHardLUT, LUTSpec, ManifestoHardLUT, cell_tv_penalty
from spiky.lutorch_ex.cartridges import manifesto_base
from spiky.lutorch_ex.cartridges.manifesto_base import _cell_tv_eager

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")

#: max |a - b| / max |b| allowed against fp64, per dtype (low precision: 4 units of the dtype's rounding).
LOWP_TOL = {torch.bfloat16: 4 * 2.0 ** -8, torch.float16: 4 * 2.0 ** -11}
DTYPES = [torch.float32, torch.float64, torch.bfloat16, torch.float16]


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).abs().max() / b.abs().max()).item()


def _cartridge(nap, d, dtype, device, H=4, tph=8, seed=1):
    spec = LUTSpec(h_in=H, h_out=H, tph=tph, nap=nap, d_in=max(d, 2), d_out=d)
    return ManifestoHardLUT(spec, seed=seed, weight_init_std=0.1).to(device=device, dtype=dtype)


def _eager_value_and_grad(weights: torch.Tensor, nap: int, dtype=None):
    w = weights.detach().to(dtype or weights.dtype).reshape(-1, *weights.shape[2:]).requires_grad_(True)
    v = _cell_tv_eager(w, nap)
    v.backward()
    return v.detach(), w.grad.reshape(weights.shape)


@pytest.fixture
def fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


@needs_cuda
@pytest.mark.parametrize("dtype", DTYPES, ids=str)
@pytest.mark.parametrize("nap,d", [(1, 3), (4, 5), (8, 48)])
def test_compiled_cell_tv_matches_eager_value_and_accumulated_grad(nap, d, dtype, fresh_dynamo):
    m = _cartridge(nap, d, dtype, "cuda")
    # An existing gradient the TV backward must add to (fp32 / fp64; in a low-precision .grad, adding to and then
    # subtracting a larger g0 would itself round away the TV gradient's low bits, so there it starts empty).
    g0 = torch.randn_like(m.weights) * 1e-5 if dtype not in LOWP_TOL else torch.zeros_like(m.weights)
    m.weights.grad = g0.clone() if dtype not in LOWP_TOL else None
    graphs = counters["stats"]["unique_graphs"]
    v = m.cell_tv()
    (2.5 * v).backward()
    assert m._compiled_tv is not None
    assert counters["stats"]["unique_graphs"] > graphs, "no graph captured: the call ran eager"
    assert v.dtype == dtype and v.shape == ()
    grad = (m.weights.grad.double() - g0.double()) / 2.5
    if dtype in LOWP_TOL:
        ref_v, ref_g = _eager_value_and_grad(m.weights, nap, torch.float64)
        assert _rel(v, ref_v) <= LOWP_TOL[dtype]
        assert _rel(grad, ref_g) <= LOWP_TOL[dtype]
    else:
        tol = 1e-6 if dtype == torch.float32 else 1e-12
        ref_v, ref_g = _eager_value_and_grad(m.weights, nap)
        assert _rel(v, ref_v) <= tol
        assert _rel(grad, ref_g) <= tol


@needs_cuda
def test_compiled_cell_tv_one_entry_per_table_shape(fresh_dynamo):
    """Instances with the same (table shape, nap, dtype) -- the layers of one model -- share one compiled entry; a
    different table shape adds exactly one (the compile is static-shaped)."""
    graphs = counters["stats"]["unique_graphs"]
    for seed in range(6):                                          # six layers of one model
        m = _cartridge(8, 48, torch.float32, "cuda", seed=seed)
        ref_v, _ = _eager_value_and_grad(m.weights, 8)
        assert _rel(m.cell_tv(), ref_v) <= 1e-6
    assert counters["stats"]["unique_graphs"] - graphs == 1
    m = _cartridge(8, 7, torch.float32, "cuda", H=2, tph=2)
    ref_v, _ = _eager_value_and_grad(m.weights, 8)
    assert _rel(m.cell_tv(), ref_v) <= 1e-6
    assert counters["stats"]["unique_graphs"] - graphs == 2


@needs_cuda
def test_compiled_cell_tv_penalty_over_a_model(fresh_dynamo):
    """The model-level mean over several cartridges (reference and fused) goes through each one's compiled cell_tv."""
    spec = LUTSpec(h_in=4, h_out=4, tph=8, nap=8, d_in=48, d_out=48)
    mods = nn.ModuleList([ManifestoHardLUT(spec, seed=1, weight_init_std=0.1),
                          ConfidenceLUT(spec, seed=2, weight_init_std=0.1, read_top_n=2),
                          FusedManifestoHardLUT(spec, seed=3, weight_init_std=0.1)]).cuda()
    p = cell_tv_penalty(mods)
    p.backward()
    ref = torch.stack([_eager_value_and_grad(m.weights, 8)[0] for m in mods]).mean()
    assert _rel(p, ref) <= 1e-6
    for m in mods:
        assert m._compiled_tv is not None
        assert _rel(m.weights.grad, _eager_value_and_grad(m.weights, 8)[1] / len(mods)) <= 1e-6


def test_cpu_cell_tv_is_the_eager_formula():
    """CPU: never compiled; bit-identical to _cell_tv_eager (value and gradient)."""
    m = _cartridge(4, 5, torch.float64, "cpu")
    v = m.cell_tv()
    v.backward()
    ref_v, ref_g = _eager_value_and_grad(m.weights, 4)
    assert m._compiled_tv is None
    assert torch.equal(v.detach(), ref_v) and torch.equal(m.weights.grad, ref_g)


@needs_cuda
def test_compile_disabled_runs_the_eager_formula_on_cuda(monkeypatch):
    """With compilation off (what LUTORCH_EX_NO_COMPILE=1 sets), CUDA runs the eager formula: no compiled handle,
    bit-identical value and gradient."""
    monkeypatch.setattr(manifesto_base, "_COMPILE_ENABLED", False)
    m = _cartridge(8, 48, torch.float32, "cuda")
    v = m.cell_tv()
    v.backward()
    ref_v, ref_g = _eager_value_and_grad(m.weights, 8)
    assert m._compiled_tv is None
    assert torch.equal(v.detach(), ref_v) and torch.equal(m.weights.grad, ref_g)


_PROBE = textwrap.dedent("""
    import torch
    from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT
    from spiky.lutorch_ex.cartridges.manifesto_base import _cell_tv_eager
    m = ManifestoHardLUT(LUTSpec(h_in=2, h_out=2, tph=4, nap=4, d_in=6, d_out=6), seed=1).cuda()
    v = m.cell_tv()
    same = torch.equal(v, _cell_tv_eager(m.weights.reshape(-1, 16, 6), 4))
    print("compiled" if m._compiled_tv is not None else "eager", same)
""")


@needs_cuda
@pytest.mark.parametrize("env,expected", [("1", "eager True"), (None, "compiled")])
def test_no_compile_env_var_selects_the_eager_path(env, expected):
    """The real switch, read at import: LUTORCH_EX_NO_COMPILE=1 -> eager on CUDA; unset -> compiled (the default)."""
    e = {k: v for k, v in os.environ.items() if k != "LUTORCH_EX_NO_COMPILE"}
    if env is not None:
        e["LUTORCH_EX_NO_COMPILE"] = env
    out = subprocess.run([sys.executable, "-c", _PROBE], env=e, capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1].startswith(expected)
