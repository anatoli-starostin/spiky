"""Gen-3 ConfidenceLUT (LightMHL learned-margin score, scored-only).

Covers: fp64 gradcheck of the scored backward (input, weights, β, γ, and τ for n=2); both anchor
modes; read_top_n ∈ {1,2}; gradients reach β/γ/τ; the address is detached (x gets gradient ONLY
through the score / blend, no routing grad); bf16/fp16 raises (pure fp32/fp64 only); single-anchor
canonical coverage; train (embedding_bag) == eval (gather) forward value; shape/dtype invariants.
"""
import pytest
import torch

from spiky.lutorch_ex import ConfidenceLUT, LUTSpec

PATTERNS = [(4, 4), (1, 4), (4, 1)]
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _normrel(a, b):
    return (a.double() - b.double()).norm().item() / max(b.double().norm().item(), 1e-12)


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_gradcheck_fp64(read_top_n, anchor_mode):
    """fp64 gradcheck wrt input, weights, and the score/blend temperatures — the scored forward
    is value==gradient, so finite differences verify the analytic backward."""
    torch.set_default_dtype(torch.float64)
    try:
        spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=3, d_in=5, d_out=4, anchor_mode=anchor_mode)
        m = ConfidenceLUT(spec, seed=1, weight_init_std=1.0, read_top_n=read_top_n).train()
        x = torch.randn(6, 2, 5, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(lambda xx: m(xx).pow(2).sum(), (x,), eps=1e-6, atol=1e-5)

        names = ["weights", "confidence_log_beta", "confidence_log_gamma"]
        if read_top_n == 2:
            names.append("log_read_tau")
        leaves = [getattr(m, n).detach().clone().requires_grad_(True) for n in names]
        xd = x.detach()

        def f(*ps):
            return torch.func.functional_call(m, dict(zip(names, ps)), (xd,)).pow(2).sum()

        assert torch.autograd.gradcheck(f, tuple(leaves), eps=1e-6, atol=1e-5)
    finally:
        torch.set_default_dtype(torch.float32)


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("h_in,h_out", PATTERNS)
@pytest.mark.parametrize("device", DEVICES)
def test_forward_backward_runs(read_top_n, anchor_mode, h_in, h_out, device):
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=6, nap=5, d_in=12, d_out=8, anchor_mode=anchor_mode)
    m = ConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=read_top_n).to(device)
    torch.manual_seed(5)
    x = torch.randn(64, h_in, 12, device=device, requires_grad=True)
    m.train()
    y = m(x)
    assert y.shape == (64, h_out, 8) and y.dtype == x.dtype
    targets = [x, m.weights, m.confidence_log_beta, m.confidence_log_gamma]
    if read_top_n == 2:
        targets.append(m.log_read_tau)
    grads = torch.autograd.grad(y.pow(2).sum(), targets)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    m.eval()
    with torch.no_grad():
        assert m(x).shape == (64, h_out, 8)


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
@pytest.mark.parametrize("device", DEVICES)
def test_train_eval_value_match(read_top_n, anchor_mode, device):
    """The embedding_bag train read and the plain-gather eval read are the same scored value."""
    spec = LUTSpec(h_in=3, h_out=3, tph=4, nap=4, d_in=6, d_out=5, anchor_mode=anchor_mode)
    m = ConfidenceLUT(spec, seed=2, weight_init_std=1.0, read_top_n=read_top_n).to(device)
    x = torch.randn(32, 3, 6, device=device)
    with torch.no_grad():
        m.train(); yt = m(x)
        m.eval(); ye = m(x)
    assert _normrel(yt, ye) < 1e-5


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_input_grad_is_score_only_no_routing(read_top_n, anchor_mode):
    """The address is detached: x receives gradient ONLY through the score (and, for n=2, the
    blend). Detaching the score/blend must zero the input gradient — proving no routing grad
    flows through the (discrete) addressed cell."""
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5, anchor_mode=anchor_mode)
    m = ConfidenceLUT(spec, seed=3, weight_init_std=1.0, read_top_n=read_top_n).train()
    x0 = torch.randn(16, 2, 6)
    go = torch.randn(16, 2, 5)

    x = x0.clone().requires_grad_(True)
    (m(x) * go).sum().backward()
    assert x.grad.abs().sum() > 0, "expected a nonzero score-path input gradient"

    # Detach the score and the blend weight -> the only remaining x path is the discrete index.
    orig_score, orig_blend = m._score, m._blend_v
    m._score = lambda u: orig_score(u).detach()
    m._blend_v = lambda ua: orig_blend(ua).detach()
    try:
        xd = x0.clone().requires_grad_(True)
        out = m(xd)
        g = torch.autograd.grad((out * go).sum(), xd, allow_unused=True)[0]
        assert g is None or g.abs().max().item() == 0.0, "routing gradient leaked through the index"
    finally:
        m._score, m._blend_v = orig_score, orig_blend


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rejects_low_precision(read_top_n, dtype):
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5)
    m = ConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=read_top_n).to(dtype)
    with pytest.raises(TypeError, match="does not support low precision"):
        m(torch.randn(8, 2, 6, dtype=dtype))
    m32 = ConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=read_top_n)
    with pytest.raises(TypeError, match="does not support low precision"):
        m32(torch.randn(8, 2, 6, dtype=dtype))


def test_score_params_and_tau_wiring():
    spec = LUTSpec(h_in=2, h_out=2, tph=3, nap=3, d_in=5, d_out=4)
    m1 = ConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=1)
    names1 = {n for n, _ in m1.named_parameters()}
    assert {"confidence_log_beta", "confidence_log_gamma"} <= names1
    assert "log_read_tau" not in dict(m1.named_buffers()) and not hasattr(m1, "log_read_tau")
    import math
    assert abs(m1.confidence_log_beta.exp().item() - 2.0) < 1e-5
    assert abs(m1.confidence_log_gamma.exp().item() - 1.0) < 1e-5

    m2 = ConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=2)
    assert "log_read_tau" in {n for n, _ in m2.named_parameters()}
    assert abs(m2.log_read_tau.exp().item() - 0.5) < 1e-5

    frozen = ConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=2,
                           learnable_score=False, read_tau_learnable=False)
    fnames = {n for n, _ in frozen.named_parameters()}
    assert not ({"confidence_log_beta", "confidence_log_gamma", "log_read_tau"} & fnames)

    with pytest.raises(ValueError, match="read_top_n"):
        ConfidenceLUT(spec, seed=0, weight_init_std=1.0, read_top_n=3)


@pytest.mark.parametrize("anchor_mode", ["pairs", "single"])
def test_single_and_pairs_canonical_coverage(anchor_mode):
    """Both anchor modes run; single mode uses only anchor_a (coordinate-vs-zero margins)."""
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=7, d_out=5, anchor_mode=anchor_mode)
    m = ConfidenceLUT(spec, seed=1, weight_init_std=1.0, read_top_n=2).train()
    if anchor_mode == "single":
        assert m.single and m.anchor_b is None
    else:
        assert not m.single and m.anchor_b is not None
    x = torch.randn(8, 2, 7, requires_grad=True)
    y = m(x)
    assert y.shape == (8, 2, 5)
    torch.autograd.grad(y.sum(), x)  # runs, no error
