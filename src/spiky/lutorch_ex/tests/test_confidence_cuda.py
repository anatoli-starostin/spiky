"""ConfidenceLUTCuda (hand-written CUDA) vs ConfidenceLUT: forward value and every gradient, both read_top_n, both
anchor modes, with and without table dropout, across the launch knobs."""
import pytest
import torch

from spiky.lutorch_ex.cartridges.confidence import ConfidenceLUT
from spiky.lutorch_ex.cartridges.confidence_cuda import ConfidenceLUTCuda, CudaKnobs, confidence_cuda_ext
from spiky.lutorch_ex.lut_spec import LUTSpec

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or confidence_cuda_ext() is None,
                                reason="needs CUDA and the confidence_cuda extension")


def _pair(n, anchor_mode="pairs", rate=0.0, h_in=4, h_out=4, tph=16, nap=6, d=24, knobs=None):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode=anchor_mode)
    kw = dict(seed=3, read_top_n=n, table_dropout_rate=rate, weight_init_std=0.5, beta_init=1.7, gamma_init=0.8,
              read_tau_init=0.6)
    ref = ConfidenceLUT(spec, **kw).cuda().train()
    cud = ConfidenceLUTCuda(spec, knobs=knobs, **kw).cuda().train()
    cud.load_state_dict(ref.state_dict())
    return ref, cud


def _run(ref, cud, B=96, keep=None):
    torch.manual_seed(0)
    x = torch.randn(B, ref.spec.h_in, ref.spec.d_in, device="cuda")
    go = torch.randn(B, ref.spec.h_out, ref.spec.d_out, device="cuda")
    if keep is not None:
        p = 1.0 - ref.table_dropout_rate
        ref._table_dropout_mask = lambda B_, dev, dt: keep.to(dt) / p
        cud._keep_flags = lambda B_, dev: keep
    out = {}
    for name, mod in (("ref", ref), ("cuda", cud)):
        xi = x.clone().requires_grad_(True)
        y = mod._forward_impl(xi) if name == "ref" else mod(xi)    # ref: eager (same numerics as compiled)
        y.backward(go)
        grads = {k: p.grad.clone() for k, p in mod.named_parameters() if p.grad is not None}
        out[name] = (y.detach(), xi.grad.clone(), grads)
        mod.zero_grad(set_to_none=True)
    return out


def _close(a, b, what, rtol=1e-4):
    err = (a - b).norm() / b.norm().clamp_min(1e-30)
    assert err < rtol, f"{what}: rel-norm error {err:.3e}"


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("rate", [0.0, 0.3])
def test_matches_confidence_lut(n, anchor_mode, rate):
    ref, cud = _pair(n, anchor_mode, rate)
    keep = None
    if rate > 0:
        g = torch.Generator(device="cuda").manual_seed(1)
        keep = torch.rand(96, ref.spec.n_groups, ref.spec.tph, device="cuda", generator=g) < 1 - rate
    o = _run(ref, cud, keep=keep)
    (yr, gxr, pr), (yc, gxc, pc) = o["ref"], o["cuda"]
    _close(yc, yr, "forward")
    _close(gxc, gxr, "grad x")
    assert pr.keys() == pc.keys()
    for k in pr:
        _close(pc[k], pr[k], f"grad {k}")


@pytest.mark.parametrize("knobs", [
    CudaKnobs(fwd_threads=32, bwd_threads=32, rows_per_cta=1, vec=1, vec_atomics=False),
    CudaKnobs(fwd_threads=64, bwd_threads=96, rows_per_cta=3, vec=2, vec_atomics=True),
    CudaKnobs(fwd_threads=256, bwd_threads=512, rows_per_cta=7, vec=4, vec_atomics=True),
])
@pytest.mark.parametrize("n", [1, 2])
def test_knobs_do_not_change_the_result(knobs, n):
    ref, cud = _pair(n, knobs=knobs)
    o = _run(ref, cud, B=50)                                     # B not a multiple of rows_per_cta
    (yr, gxr, pr), (yc, gxc, pc) = o["ref"], o["cuda"]
    _close(yc, yr, "forward")
    _close(gxc, gxr, "grad x")
    for k in pr:
        _close(pc[k], pr[k], f"grad {k}")


def test_fan_in_and_d24_geometry():
    """Fan-in routing (h_out = 1) and the d24 LUT-FFN geometry (h 16, d 48, tph 64, nap 8)."""
    ref, cud = _pair(1, h_in=4, h_out=1, tph=8, nap=4, d=16)
    o = _run(ref, cud)
    _close(o["cuda"][0], o["ref"][0], "fan-in forward")
    _close(o["cuda"][1], o["ref"][1], "fan-in grad x")
    for n in (1, 2):
        ref, cud = _pair(n, h_in=16, h_out=16, tph=64, nap=8, d=48)
        o = _run(ref, cud, B=64)
        _close(o["cuda"][0], o["ref"][0], f"d24 n={n} forward")
        _close(o["cuda"][1], o["ref"][1], f"d24 n={n} grad x")
        for k in o["ref"][2]:
            _close(o["cuda"][2][k], o["ref"][2][k], f"d24 n={n} grad {k}")


def test_falls_back_off_cuda_fp32():
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8, anchor_mode="pairs")
    cud = ConfidenceLUTCuda(spec, seed=0).double()
    ref = ConfidenceLUT(spec, seed=0).double()
    x = torch.randn(5, 2, 8, dtype=torch.float64)
    torch.testing.assert_close(cud.eval()(x), ref.eval()(x))
