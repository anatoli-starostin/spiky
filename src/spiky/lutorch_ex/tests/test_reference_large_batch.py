"""The reference Manifesto cartridges train at the large benchmark geometry (h16 tph64 nap8 d48, 32,768 vectors).

ManifestoHardLUT's surrogate used to gather both cells' rows, [B, G, tph, d_out] (~6 GiB each at this size), plus their
difference and its weighted copy: ~27 GiB of temporaries, out of memory on a 32 GB card. It now reads them through one
embedding_bag (cartridges/_fused_ops.fused_surrogate_read). This test keeps that from silently coming back: one
training step, peak memory above the pre-step allocation bounded by PEAK_BOUND_GIB.
"""
import pytest
import torch

from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT, ManifestoSoftLUT

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

LARGE = dict(h_in=16, h_out=16, tph=64, nap=8, d_in=48, d_out=48)
B = 32768
#: Measured train peaks at this size (RTX 5090), both classes, both uncertainty functions: 5.09 GiB float32, 5.23 GiB bfloat16. 8 GiB
#: leaves headroom for allocator variation and is far below the ~27 GiB the gather form needed.
PEAK_BOUND_GIB = 8.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize("uncertainty_func", ["rational", "softsign"])
@pytest.mark.parametrize("cls", [ManifestoHardLUT, ManifestoSoftLUT], ids=["hard", "soft"])
def test_reference_trains_at_the_large_geometry(cls, uncertainty_func, dtype):
    torch._dynamo.reset()
    m = cls(LUTSpec(**LARGE), seed=1, uncertainty_func=uncertainty_func).cuda().to(dtype).train()
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(B, 16, 48, device="cuda", generator=g).to(dtype)
    go = torch.randn(B, 16, 48, device="cuda", generator=g).to(dtype)
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    m(x).backward(go)
    torch.cuda.synchronize()
    peak = (torch.cuda.max_memory_allocated() - base) / 2 ** 30
    print(f"\n{cls.__name__}({uncertainty_func}) {dtype}: train peak {peak:.2f} GiB above the pre-step allocation")
    assert m.weights.grad is not None and torch.isfinite(m.weights.grad).all()
    assert peak < PEAK_BOUND_GIB, f"train peak {peak:.2f} GiB >= {PEAK_BOUND_GIB} GiB"
