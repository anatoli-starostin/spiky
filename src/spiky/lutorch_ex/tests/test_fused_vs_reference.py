"""Spec amendment D: the fused-vs-reference equivalence harness, the one REQUIRED check binding each fused class to its
reference class.

Per cartridge (FusedManifestoHardLUT / FusedManifestoSoftLUT with both weights, FusedConfidenceLUT with read_top_n 1
and 2), per dtype (fp32; bf16 table + input; fp16 input on an fp32 table), per anchor mode and head pattern: the fused
class and its reference class, loaded with the same state_dict, are fed the same input and output gradient, in
training and in eval, and compared on every leaf: the value, the input gradient, the table gradient and every learned
scalar's gradient (soft-sign temperatures; confidence beta / gamma / tau).

Tolerances (relative norm error ``||a - b|| / ||b||``), each from the one way the two can legitimately differ:
* Both sides take the same discrete decisions: addressing runs in fp32 on the same values in both (a margin is one
  fp32 subtraction), so the cells read are identical and only the arithmetic around them can differ.
* fp32 leaf -- summation order. The kernels reduce in a different order (and the table gradient with atomics), so
  each sum carries a different fp32 round-off: O(sqrt(K) * 2**-24) relative for K terms, ~1e-6 for the K <= 10**4
  here. ``FP32_TOL = 1e-5`` per element-wise leaf; ``FP32_SCALAR_TOL = 1e-4`` for a learned scalar, whose gradient is
  ONE sum over every (sample, group, table, output) and so the worst-conditioned reduction in the cartridge.
* bf16 / fp16 leaf -- one rounding. Both classes compute in fp32 and round each leaf once (fused: in the kernel's
  epilogue; reference: tests/test_reference_low_precision.py proves it bit-exact). Two roundings of fp32 values that
  differ only by the fp32 tolerance above land on the same low-precision value except where they straddle a
  rounding boundary, and then differ by one ulp. One ulp everywhere would be a relative norm error of at most
  u = 2**-8 (bf16) / 2**-11 (fp16), so ``u`` bounds it with a wide margin. A learned scalar in a bf16 module is a
  bf16 leaf as well, so the same bound applies.
* Leaves that stay fp32 in a low-precision run (every parameter's gradient when only the input is fp16) take the
  fp32 tolerances, widened to the input dtype's u: they are computed from the rounded output gradient and input.

Measured 2026-10-11 (RTX 5090), worst error as a share of its tolerance over the whole matrix: fp32 value 3%, grad x
2%, grad W 9%, learned scalars 21%; bf16 value 5%, grad x 5%, grad W 3%; fp16-input grad x 7%. (The harness found one
real double rounding: a low-precision input gathered for several groups (fan-out) summed their input gradients in the
input dtype, 78% of u; the fused classes now cast before the gather.)

Kernel-specific checks (launch knobs, determinism, the d_out launch boundary, bf16 accumulation under row collision,
table dropout) live in test_fused_manifesto_cuda.py, test_fused_confidence.py and test_table_dropout_and_cell_tv.py.
"""
import pytest
import torch

from spiky.lutorch_ex import (ConfidenceLUT, FusedConfidenceLUT, FusedManifestoHardLUT, FusedManifestoSoftLUT, LUTSpec,
                              ManifestoHardLUT, ManifestoSoftLUT)
from spiky.lutorch_ex.tests._requires_extension import requires_extension

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

FP32_TOL = 1e-5
FP32_SCALAR_TOL = 1e-4
U = {torch.bfloat16: 2.0 ** -8, torch.float16: 2.0 ** -11}

#: (id, fused class, reference class, kwargs, extension)
CARTRIDGES = [
    ("ManifestoHard", FusedManifestoHardLUT, ManifestoHardLUT, {}, "manifesto"),
    ("ManifestoSoft", FusedManifestoSoftLUT, ManifestoSoftLUT, {}, "manifesto"),
    ("ManifestoHard[softsign]", FusedManifestoHardLUT, ManifestoHardLUT, dict(weight="softsign"), "manifesto"),
    ("ManifestoSoft[softsign]", FusedManifestoSoftLUT, ManifestoSoftLUT, dict(weight="softsign"), "manifesto"),
    ("Confidence n=1", FusedConfidenceLUT, ConfidenceLUT,
     dict(read_top_n=1, beta_init=1.7, gamma_init=0.8), "confidence"),
    ("Confidence n=2", FusedConfidenceLUT, ConfidenceLUT,
     dict(read_top_n=2, beta_init=1.7, gamma_init=0.8, read_tau_init=0.6), "confidence"),
]
#: (id, table dtype, input dtype)
DTYPES = [("fp32", torch.float32, torch.float32), ("bf16", torch.bfloat16, torch.bfloat16),
          ("fp16-input", torch.float32, torch.float16)]
#: (id, h_in, h_out, d, tph, nap, batch)
GEOMS = [("small-per-head", 4, 4, 12, 6, 5, 128), ("fan-in", 4, 1, 12, 6, 5, 128), ("fan-out", 1, 4, 12, 6, 5, 128),
         ("batch-1", 4, 4, 12, 6, 5, 1), ("canonical", 16, 16, 48, 64, 8, 512), ("large-batch", 4, 4, 12, 6, 5, 24576)]


def _rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def _tol(leaf: str, leaf_dtype: torch.dtype, input_dtype: torch.dtype) -> float:
    scalar = leaf not in ("value", "grad x", "grad weights")
    if leaf_dtype in U:
        return U[leaf_dtype]
    base = FP32_SCALAR_TOL if scalar else FP32_TOL
    return max(base, U.get(input_dtype, 0.0))


def _leaves(m, x, go):
    xi = x.clone().requires_grad_(True)
    y = m(xi)
    y.backward(go)
    out = {"value": y.detach(), "grad x": xi.grad}
    out.update({f"grad {n}": p.grad for n, p in m.named_parameters() if p.grad is not None})
    m.zero_grad(set_to_none=True)
    return out


@pytest.mark.parametrize("geom", GEOMS, ids=[g[0] for g in GEOMS])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("dt", DTYPES, ids=[d[0] for d in DTYPES])
@pytest.mark.parametrize("cart", CARTRIDGES, ids=[c[0] for c in CARTRIDGES])
def test_fused_matches_reference(cart, dt, anchor_mode, geom):
    name, Fused, Ref, kw, ext = cart
    _, table_dtype, input_dtype = dt
    _, h_in, h_out, d, tph, nap, B = geom
    requires_extension(ext)
    torch._dynamo.reset()          # every case compiles the reference fresh (else Dynamo's recompile limit -> eager)
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode=anchor_mode)
    ref = Ref(spec, seed=7, weight_init_std=0.5, **kw).cuda().to(table_dtype)
    fused = Fused(spec, seed=7, weight_init_std=0.5, **kw).cuda().to(table_dtype)
    fused.load_state_dict(ref.state_dict())
    g = torch.Generator(device="cuda").manual_seed(B + h_in)
    x = torch.randn(B, h_in, d, device="cuda", generator=g).to(input_dtype)
    go = torch.randn(B, h_out, d, device="cuda", generator=g).to(input_dtype)

    ref.train(); fused.train()
    a, b = _leaves(fused, x, go), _leaves(ref, x, go)
    assert a.keys() == b.keys(), f"{name}: leaves {sorted(a)} vs {sorted(b)}"
    ref.eval(); fused.eval()
    with torch.no_grad():
        a["eval value"], b["eval value"] = fused(x), ref(x)
    for leaf in b:
        assert a[leaf].dtype == b[leaf].dtype, f"{name} {leaf}: dtype {a[leaf].dtype} vs {b[leaf].dtype}"
        tol = _tol(leaf.replace("eval ", ""), b[leaf].dtype, input_dtype)
        e = _rel(a[leaf], b[leaf])
        assert e < tol, f"{name} {leaf}: rel-norm error {e:.2e} >= {tol:.1e}"
