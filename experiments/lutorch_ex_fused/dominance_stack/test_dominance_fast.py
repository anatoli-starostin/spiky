"""DominanceLUTFast vs the reference DominanceLUT (CUDA): value, table grad, input grad, bias grad."""
import os, sys
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dominance_lut_ref import DominanceLUT, LUTSpec
from dominance_fast import DominanceLUTFast

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA (Triton gather)")


def pair(spec, **kw):
    ref = DominanceLUT(spec, **kw).cuda()
    fast = DominanceLUTFast(spec, **kw).cuda()
    for p in (ref.weights, fast.weights):      # non-trivial table values so every gradient term is exercised
        p.data.normal_(0, 0.5, generator=torch.Generator(device="cuda").manual_seed(3))
    fast.weights.data.copy_(ref.weights.data)
    if ref.bias is not None:
        ref.bias.data.normal_(); fast.bias.data.copy_(ref.bias.data)
    return ref, fast


def run(m, x):
    x = x.clone().requires_grad_(True)
    m.zero_grad()
    y = m(x)
    w = torch.linspace(-1, 1, y.numel(), device=y.device).reshape(y.shape)
    (y * w).sum().backward()
    return y.detach(), m.weights.grad.clone(), x.grad.clone(), (m.bias.grad.clone() if m.bias is not None else None)


@pytest.mark.parametrize("cfg", [
    dict(hh=(8, 8), tph=64, nap=8, d=48, B=6144),         # the experiment geometry
    dict(hh=(8, 8), tph=64, nap=8, d=48, B=1000, normalize="count"),
    dict(hh=(3, 3), tph=5, nap=4, d=12, B=129),
    dict(hh=(1, 3), tph=5, nap=4, d=12, B=129),            # fan-out
    dict(hh=(3, 1), tph=5, nap=4, d=12, B=129),            # fan-in
    dict(hh=(2, 2), tph=7, nap=3, d=8, B=50, bias=True),
    dict(hh=(2, 2), tph=7, nap=3, d=8, B=50, out="random", nop=5),
])
def test_matches_reference(cfg):
    h_in, h_out = cfg["hh"]
    d = cfg["d"]
    spec = LUTSpec(h_in=h_in, h_out=h_out, tph=cfg["tph"], nap=cfg["nap"], d_in=d, d_out=d)
    kw = dict(seed=11, normalize=cfg.get("normalize", "none"), add_bias=cfg.get("bias", False),
              output_anchors=cfg.get("out", "same"))
    if "nop" in cfg:
        kw["nop"] = cfg["nop"]
    ref, fast = pair(spec, **kw)
    x = torch.randn(cfg["B"], h_in, d, device="cuda")
    yr, gwr, gxr, gbr = run(ref, x)
    yf, gwf, gxf, gbf = run(fast, x)
    scale = max(1.0, yr.abs().max().item())
    assert (yr - yf).abs().max() < 1e-5 * scale
    assert (gwr - gwf).abs().max() < 1e-4 * max(1.0, gwr.abs().max().item())
    assert (gxr - gxf).abs().max() < 1e-4 * max(1.0, gxr.abs().max().item())
    if gbr is not None:
        assert torch.allclose(gbr, gbf, atol=1e-4)
    # untouched rows have exactly zero gradient in both
    assert torch.equal(gwr == 0, gwf == 0) or (gwr - gwf).abs().max() < 1e-4


def test_no_input_grad_path_and_fallback():
    spec = LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8)
    ref, fast = pair(spec, seed=1)
    x = torch.randn(10, 2, 8, device="cuda")                  # x does not require grad
    assert torch.allclose(ref(x), fast(x), atol=1e-6)
    fast.vote = "sign"; ref.vote = "sign"                      # falls back to the reference forward
    assert torch.equal(ref(x), fast(x))
