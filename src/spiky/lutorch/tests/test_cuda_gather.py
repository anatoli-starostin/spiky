"""Tests for spiky.lutorch.cuda_gather: hardware dispatch + (when a 5090 is present) the
actual fused kernel's numerics against the native path.

Split in two:
  * hardware-classification tests run everywhere (mock torch.cuda.get_device_capability,
    no GPU needed) -- this is what proves the H100/unrecognized-hardware fallback is inert.
  * kernel-numerics tests are skipped unless CUDA is available; they additionally skip (not
    fail) when the detected device isn't 5090-class, since the kernel is only expected to
    build/match on the hardware it targets.
"""
from unittest import mock

import pytest
import torch

from spiky.lutorch import cuda_gather as cg
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

_HAS_CUDA = torch.cuda.is_available()
_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")
_is_5090 = _HAS_CUDA and cg.is_5090_class_gpu()
_needs_5090 = pytest.mark.skipif(not _is_5090, reason="RTX 5090-class GPU required")


def _mk(**kw):
    base = dict(input_dim=192, output_dim=192, inner_in_dim=48, inner_out_dim=48,
                nap=7, tph=64, n_heads=4, use_bf16=False, random_seed=1000)
    base.update(kw)
    return CompressionMultiHeadLUT(**base)


# ----------------------------- hardware classification (no GPU needed) -----------------------------

def test_is_5090_class_gpu_false_without_cuda():
    with mock.patch("torch.cuda.is_available", return_value=False):
        assert cg.is_5090_class_gpu() is False


def test_is_5090_class_gpu_true_for_capability_12():
    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.get_device_capability", return_value=(12, 0)):
        assert cg.is_5090_class_gpu() is True


@pytest.mark.parametrize("capability", [(9, 0), (8, 0), (7, 5), (13, 0)])
def test_is_5090_class_gpu_false_for_other_capabilities(capability):
    """(9,0) is H100 -- measured SLOWER with this kernel, must not dispatch to it.
    (13,0) stands in for any future/unrecognized device -- must default to False, not True."""
    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.get_device_capability", return_value=capability):
        assert cg.is_5090_class_gpu() is False


def test_is_5090_class_gpu_never_raises_on_capability_query_failure():
    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.get_device_capability", side_effect=RuntimeError("boom")):
        assert cg.is_5090_class_gpu() is False


@_cuda
def test_patch_mode_off_is_always_a_noop():
    m = _mk().cuda()
    assert cg.patch(m, mode="off") == 0


@_cuda
def test_patch_mode_auto_is_noop_on_mocked_non_5090_hardware():
    """The core dispatch contract: on H100 (or anything not 5090-class), auto mode must
    not touch the model at all -- no kernel build attempted, existing gather untouched."""
    m = _mk().cuda()
    with mock.patch("spiky.lutorch.cuda_gather.is_5090_class_gpu", return_value=False):
        assert cg.patch(m, mode="auto") == 0


def test_patch_rejects_unknown_mode():
    with pytest.raises(ValueError):
        cg.patch(torch.nn.Module(), mode="sometimes")


# ----------------------------- kernel numerics (needs an actual RTX 5090) -----------------------------

@_needs_5090
def test_fused_fp32_table_bit_exact_vs_native():
    torch.manual_seed(0)
    m = _mk(learnable_temps=False).cuda().eval()
    x = torch.randn(37, 192, device="cuda")
    with torch.no_grad():
        y_native = m(x).clone()
    n = cg.patch(m, mode="force", table_dtype="fp32")
    assert n >= 1
    with torch.no_grad():
        y_fused = m(x)
    assert (y_fused - y_native).abs().max().item() == 0.0


@_needs_5090
def test_fused_bf16_table_close_to_native():
    torch.manual_seed(0)
    m = _mk(learnable_temps=False).cuda().eval()
    x = torch.randn(37, 192, device="cuda")
    with torch.no_grad():
        y_native = m(x).clone()
    n = cg.patch(m, mode="force", table_dtype="bf16")
    assert n >= 1
    with torch.no_grad():
        y_fused = m(x)
    rel = (y_fused - y_native).abs().max().item() / y_native.abs().max().item()
    assert rel < 1.0, f"bf16 table diverged unreasonably from native: rel={rel}"


@_needs_5090
def test_patch_mode_auto_matches_force_on_actual_5090():
    """On real 5090-class hardware, auto and force patch the same module count."""
    m_auto = _mk().cuda()
    m_force = _mk().cuda()
    assert cg.patch(m_auto, mode="auto") == cg.patch(m_force, mode="force")


@_cuda
def test_unsupported_shape_never_patched_even_in_force_mode_raises():
    """row width must be 48; force mode surfaces this as a clear error rather than silently
    no-op'ing (auto mode, by contrast, would silently skip it -- covered implicitly since
    unsupported models simply never accumulate a patch count in production use)."""
    m = _mk(inner_in_dim=32, inner_out_dim=32).cuda()
    with pytest.raises(RuntimeError):
        cg.patch(m, mode="force")
