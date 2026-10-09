"""ProjectionMHL fp8_projections: opt-in fp8 GEMMs for compress / decompress (fp32 masters, fp32 accumulate).

The flip-rate bound is measured, not assumed: at the d24 geometry (h=16, d=48, tph=64, nap=8) with the default
init, e4m3 compress operands flip ~1.2% of sign bits / ~9.1% of cells vs an exact-fp32 compress (RTX 5090,
2026-10-09) - the flips come from the operand quantisation, so the output dtype (fp32 vs bf16) does not matter.
"""
import pytest
import torch

from spiky.lutorch_ex import ConfidenceLUT, LUTSpec, MultiHeadLUT, ProjectionMHL
from spiky.lutorch_ex.fp8 import fp8_available

D24 = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48)
_FP8_OK = torch.cuda.is_available() and fp8_available(torch.device("cuda"))[0]
needs_fp8 = pytest.mark.skipif(not _FP8_OK, reason="needs a CUDA device that runs torch._scaled_mm")


def _proj(fp8=(), out_dtype=torch.float32, d_model=1536, device="cuda", seed=0):
    torch.manual_seed(seed)
    cart = ConfidenceLUT(LUTSpec(**D24), seed=1, read_top_n=1)
    m = ProjectionMHL(cart, d_model=d_model, fp8_projections=fp8, compress_fp8_out_dtype=out_dtype).to(device)
    torch.nn.init.normal_(m.decompress.weight, std=0.02)       # off zero, so the output and backward are non-trivial
    return m


def _twin(ref, fp8, out_dtype=torch.float32):
    """Same cartridge and weights as `ref`, with fp8 projections."""
    m = ProjectionMHL(ref.cartridge, d_model=ref.input_dim, fp8_projections=fp8,
                      compress_fp8_out_dtype=out_dtype).to(ref.compress.weight.device)
    m.load_state_dict(ref.state_dict())
    return m


def _cartridge_input(m, x):
    seen = {}
    h = m.cartridge.register_forward_pre_hook(lambda mod, args: seen.setdefault("z", args[0].detach().clone()))
    try:
        with torch.no_grad():
            m.eval()(x)
    finally:
        h.remove()
    return seen["z"]


# ------------------------------------------------------------------------------------------------ CPU-valid

def test_default_is_unchanged_fp32_path():
    m = _proj(device="cpu")
    assert m.fp8_projections == frozenset() and "fp8" not in m.extra_repr()
    assert all(p.dtype == torch.float32 for p in m.parameters())


@pytest.mark.parametrize("kw,match", [
    (dict(fp8_projections=("compress", "lut")), "subset"),
    (dict(fp8_projections=("decompress",), decompress=False), "decompress=False"),
    (dict(fp8_projections=("compress",), compress_fp8_out_dtype=torch.float16), "compress_fp8_out_dtype"),
])
def test_bad_options_rejected(kw, match):
    spec = LUTSpec(h_in=2, h_out=2, tph=2, nap=3, d_in=8, d_out=8)
    with pytest.raises(ValueError, match=match):
        ProjectionMHL(ConfidenceLUT(spec, seed=1), d_model=16, **kw)


def test_cpu_forward_raises_clearly():
    spec = LUTSpec(h_in=2, h_out=2, tph=2, nap=3, d_in=8, d_out=8)
    m = ProjectionMHL(ConfidenceLUT(spec, seed=1), d_model=16, fp8_projections=("decompress",))
    with pytest.raises(RuntimeError, match="CUDA"):
        m(torch.randn(4, 16))


# ------------------------------------------------------------------------------------------------ CUDA + fp8

@needs_fp8
def test_decompress_only_leaves_addressing_bit_identical():
    ref = _proj()
    m = _twin(ref, ("decompress",))
    x = torch.randn(2048, 1536, device="cuda")
    assert torch.equal(_cartridge_input(ref, x), _cartridge_input(m, x))   # compress untouched -> same addresses
    with torch.no_grad():
        y_ref, y = ref.eval()(x), m.eval()(x)
    rel = (y - y_ref).pow(2).mean().sqrt() / y_ref.pow(2).mean().sqrt()
    assert rel < 0.1, rel


@needs_fp8
@pytest.mark.parametrize("out_dtype", [torch.float32, torch.bfloat16])
def test_compress_fp8_sign_bit_flips_within_measured_bound(out_dtype):
    ref = _proj()
    m = _twin(ref, ("compress", "decompress"), out_dtype)
    x = torch.randn(4096, 1536, device="cuda").bfloat16().float()          # the island's input: a bf16 stream cast up
    cart, spec = ref.cartridge, ref.cartridge.spec
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.no_grad():
            z_exact = ref.compress(x).reshape(-1, spec.h_in, spec.d_in)
    finally:
        torch.set_float32_matmul_precision(prev)
    bits = lambda z: cart._addresses(z)[1] > cart.cmp_eps                  # noqa: E731
    d = bits(_cartridge_input(m, x)) != bits(z_exact)
    bit_flip, cell_flip = d.float().mean().item(), d.any(-1).float().mean().item()
    assert bit_flip < 0.025, bit_flip      # measured ~0.012
    assert cell_flip < 0.15, cell_flip     # measured ~0.091


@needs_fp8
@pytest.mark.parametrize("fp8", [("compress",), ("decompress",), ("compress", "decompress")])
def test_gradients_reach_the_fp32_masters(fp8):
    m = _twin(_proj(), fp8).train()
    x = torch.randn(1000, 1536, device="cuda", requires_grad=True)       # 1000: not a multiple of 16
    m(x).pow(2).sum().backward()
    assert x.grad is not None and x.grad.dtype == torch.float32 and x.grad.abs().sum() > 0
    for name in ("compress", "decompress"):
        lin = getattr(m, name)
        for p in (lin.weight, lin.bias):
            assert p.dtype == torch.float32 and p.grad is not None and p.grad.dtype == torch.float32
            assert torch.isfinite(p.grad).all()
        assert lin.weight.grad.abs().sum() > 0
    assert all(not t.dtype.is_floating_point or t.dtype in (torch.float32, torch.float64)
               for t in list(m.parameters()) + list(m.buffers()))        # nothing fp8 is stored
    before = m.compress.weight.detach().clone()
    torch.optim.SGD(m.parameters(), lr=1e-3).step()
    assert m.compress.weight.dtype == torch.float32 and not torch.equal(before, m.compress.weight)


@needs_fp8
def test_fp8_grad_close_to_fp32_grad():
    ref = _proj().train()
    m = _twin(ref, ("decompress",)).train()
    x = torch.randn(2048, 1536, device="cuda")
    g = torch.randn(2048, 1536, device="cuda")
    grads = []
    for mod in (ref, m):
        mod.zero_grad(set_to_none=True)
        torch.manual_seed(7)                                                # same table-dropout mask
        mod(x).backward(g)
        grads.append(mod.decompress.weight.grad.clone())
    rel = (grads[1] - grads[0]).pow(2).mean().sqrt() / grads[0].pow(2).mean().sqrt()
    assert rel < 0.1, rel


class _ScaledCartridge(MultiHeadLUT):
    """Identity-like cartridge that asks for a per-channel decompress scale (exercises the fold + fp8 path)."""

    def __init__(self, spec):
        super().__init__(spec)
        self.register_buffer("_s", torch.linspace(0.5, 2.0, spec.out_features))

    def decompress_scale(self):
        return self._s

    def forward(self, x):
        return x.reshape(x.shape[0], self.spec.h_out, self.spec.d_out)


@needs_fp8
def test_decompress_scale_fold_with_fp8():
    spec = LUTSpec(h_in=4, h_out=4, tph=1, nap=2, d_in=32, d_out=32)
    torch.manual_seed(0)
    ref = ProjectionMHL(_ScaledCartridge(spec), d_model=128).cuda()
    torch.nn.init.normal_(ref.decompress.weight, std=0.05)
    m = ProjectionMHL(_ScaledCartridge(spec), d_model=128, fp8_projections=("decompress",)).cuda()
    m.load_state_dict(ref.state_dict())
    x = torch.randn(64, 128, device="cuda")
    with torch.no_grad():
        y_ref, y = ref(x), m(x)
    assert (y - y_ref).abs().max() / y_ref.abs().max() < 0.1
