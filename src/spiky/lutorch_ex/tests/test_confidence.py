"""Gen-3 ConfidenceLUT (learned-margin confidence score, scored-only).

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
@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("device", DEVICES)
def test_bag_eval_read_matches_the_gather_eval_read(read_top_n, anchor_mode, pattern, dtype, device):
    """bag_eval_read=True reads eval through embedding_bag: same value as the plain-gather eval read
    (bag_eval_read=False) up to summation order (fp32 re-association), and the flag changes nothing in training.
    The default (None) resolves per n: on for n=2, off for n=1, and its value equals the forced read's bitwise."""
    h_in, h_out = pattern
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=8, nap=4, d_in=6, d_out=5, anchor_mode=anchor_mode)
    mk = lambda **kw: ConfidenceLUT(spec, seed=2, weight_init_std=1.0, read_top_n=read_top_n, **kw).to(
        device=device, dtype=dtype)
    gather, bag, default = mk(bag_eval_read=False), mk(bag_eval_read=True), mk()
    assert gather.bag_eval_read is False and bag.bag_eval_read is True
    assert default.bag_eval_read is (read_top_n == 2)
    x = torch.randn(64, h_in, 6, device=device, dtype=dtype)
    with torch.no_grad():
        ye_g, ye_b, ye_d = gather.eval()(x), bag.eval()(x), default.eval()(x)
        yt_g, yt_b = gather.train()(x), bag.train()(x)
    # max |a-b| / max |b|, the measure the n-dependent default was decided on (fp32 6.7e-7 measured at scale)
    maxrel = ((ye_b.double() - ye_g.double()).abs().max() / ye_g.double().abs().max()).item()
    assert maxrel < (1e-6 if dtype == torch.float32 else 1e-14)
    assert _normrel(ye_b, ye_g) < (1e-6 if dtype == torch.float32 else 1e-14)
    assert torch.equal(ye_d, ye_b if read_top_n == 2 else ye_g)
    assert torch.equal(yt_b, yt_g)


@pytest.mark.parametrize("read_top_n", [1, 2])
@pytest.mark.parametrize("flag", [None, True, False])
def test_the_eval_read_taken_per_n_and_override(read_top_n, flag, monkeypatch):
    """Which read eval actually runs: the default takes embedding_bag at n=2 and the plain gather at n=1; an explicit
    bag_eval_read=True / False forces either for both n; fused_read=True overrides all of them. CPU (eager), so the
    spied methods are really called."""
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=5)
    calls = []
    for name in ("_scored_read", "_scored_blend_read", "_read", "_read_pair", "_fused_read"):
        orig = getattr(ConfidenceLUT, name)
        monkeypatch.setattr(ConfidenceLUT, name,
                            lambda self, *a, _n=name, _o=orig, **k: (calls.append(_n), _o(self, *a, **k))[1])
    kw = {} if flag is None else {"bag_eval_read": flag}
    m = ConfidenceLUT(spec, seed=0, read_top_n=read_top_n, **kw).eval()
    x = torch.randn(8, 2, 6)
    with torch.no_grad():
        m(x)
    bag = (read_top_n == 2) if flag is None else flag
    want = ({1: "_scored_read", 2: "_scored_blend_read"} if bag else {1: "_read", 2: "_read_pair"})[read_top_n]
    assert calls == [want]
    calls.clear()
    with torch.no_grad():
        ConfidenceLUT(spec, seed=0, read_top_n=read_top_n, fused_read=True, **kw).eval()(x)
    assert calls == ["_fused_read"]


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


def test_quantised_refuses_bag_eval_read():
    """QuantisedConfidenceLUT has its own fake-quant read, so the ConfidenceLUT eval-read option is refused."""
    from spiky.lutorch_ex import QuantisedConfidenceLUT
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=6, d_out=4, anchor_mode="pairs")
    with pytest.raises(ValueError, match="bag_eval_read"):
        QuantisedConfidenceLUT(spec, seed=1, bag_eval_read=True)
