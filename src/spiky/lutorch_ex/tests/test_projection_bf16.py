"""ProjectionMHL projection_dtype: bf16 compress/decompress GEMMs from fp32 masters, cartridge stays fp32.

The flip-rate bound is measured: at GEOM_1536 (h=16, d=48, tph=64, nap=8) with the default init, bf16
compress flips ~0.07% of sign bits / ~0.6% of cells vs an exact-fp32 compress (measured 2026-10-09).
"""
import pytest
import torch

from spiky.lutorch_ex import ConfidenceLUT, LUTSpec, ProjectionMHL

GEOM_1536 = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48)
needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 GEMM measurement needs CUDA")


def _pair(device, d_model=1536, **kw):
    """(fp32 reference, twin with `kw`) sharing cartridge and weights; decompress nudged off its zero init."""
    torch.manual_seed(0)
    cart = ConfidenceLUT(LUTSpec(**GEOM_1536), seed=1, read_top_n=1)
    ref = ProjectionMHL(cart, d_model=d_model).to(device)
    torch.nn.init.normal_(ref.decompress.weight, std=0.02)
    m = ProjectionMHL(cart, d_model=d_model, **kw).to(device)
    m.load_state_dict(ref.state_dict())
    return ref, m


def _cartridge_input(m, x):
    seen = {}
    h = m.cartridge.register_forward_pre_hook(lambda mod, args: seen.setdefault("z", args[0].detach().clone()))
    try:
        with torch.no_grad():
            m.eval()(x)
    finally:
        h.remove()
    return seen["z"]


def test_default_projection_dtype_is_none_and_unchanged():
    ref, m = _pair("cpu")
    assert m.projection_dtype is None and "projection_dtype" not in m.extra_repr()
    x = torch.randn(64, 1536)
    torch.manual_seed(3)
    y_ref = ref.eval()(x)
    torch.manual_seed(3)
    assert torch.equal(m.eval()(x), y_ref)


@pytest.mark.parametrize("bad", [torch.float16, torch.float8_e4m3fn, "bfloat16", 16])
def test_invalid_projection_dtype_rejected(bad):
    spec = LUTSpec(h_in=2, h_out=2, tph=2, nap=3, d_in=8, d_out=8)
    with pytest.raises(ValueError, match="projection_dtype"):
        ProjectionMHL(ConfidenceLUT(spec, seed=1), d_model=16, projection_dtype=bad)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=needs_cuda)])
def test_cartridge_gets_fp32_and_masters_stay_fp32(device):
    _, m = _pair(device, projection_dtype=torch.bfloat16)
    x = torch.randn(512, 1536, device=device, requires_grad=True)
    z = _cartridge_input(m, x.detach())
    assert z.dtype == torch.float32                                    # addressing input is fp32, not bf16
    m.train()
    y = m(x)
    assert y.dtype == torch.float32                                    # returned in the input's dtype
    y.pow(2).sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.float32
    for lin in (m.compress, m.decompress):
        for p in (lin.weight, lin.bias):
            assert p.dtype == torch.float32 and p.grad is not None and p.grad.dtype == torch.float32
        assert lin.weight.grad.abs().sum() > 0
    assert m.cartridge.weights.dtype == torch.float32 and m.cartridge.weights.grad.dtype == torch.float32
    before = m.compress.weight.detach().clone()
    torch.optim.SGD(m.parameters(), lr=1e-3).step()
    assert m.compress.weight.dtype == torch.float32 and not torch.equal(before, m.compress.weight)
    assert all(t.dtype in (torch.float32, torch.int64, torch.bool) for t in m.state_dict().values())


@needs_cuda
def test_bf16_compress_flip_rate_within_measured_bound():
    ref, m = _pair("cuda", projection_dtype=torch.bfloat16)
    x = torch.randn(4096, 1536, device="cuda").bfloat16().float()     # the island's input: a bf16 stream cast up
    cart, spec = ref.cartridge, ref.cartridge.spec
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.no_grad():
            z_exact = ref.compress(x).reshape(-1, spec.h_in, spec.d_in)
    finally:
        torch.set_float32_matmul_precision(prev)
    bits = lambda z: cart._addresses(z)[1] > cart.cmp_eps             # noqa: E731
    d = bits(_cartridge_input(m, x)) != bits(z_exact)
    bit_flip, cell_flip = d.float().mean().item(), d.any(-1).float().mean().item()
    assert bit_flip < 0.002, bit_flip      # measured ~0.0007
    assert cell_flip < 0.015, cell_flip    # measured ~0.006


@needs_cuda
def test_composes_with_fp8_decompress():
    from spiky.lutorch_ex.fp8 import fp8_available
    if not fp8_available(torch.device("cuda"))[0]:
        pytest.skip("needs torch._scaled_mm")
    ref, m = _pair("cuda", projection_dtype=torch.bfloat16, fp8_projections=("decompress",))
    _, m_bf = _pair("cuda", projection_dtype=torch.bfloat16)
    x = torch.randn(1024, 1536, device="cuda")
    assert torch.equal(_cartridge_input(m, x), _cartridge_input(m_bf, x))   # compress ran in bf16 in both
    assert "projection_dtype=torch.bfloat16" in m.extra_repr() and "fp8_projections=('decompress',)" in m.extra_repr()
