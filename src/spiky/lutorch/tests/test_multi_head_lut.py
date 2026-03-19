"""Tests for MultiHeadLut module."""
import warnings
from contextlib import contextmanager

import pytest
import torch
import torch.nn as nn

try:
    from spiky.lut_fused.LUTLayer import LUTLayer, SynapseMeta
    _HAS_SPIKY_CUDA = True
except Exception:
    LUTLayer = None       # type: ignore[assignment]
    SynapseMeta = None    # type: ignore[assignment]
    _HAS_SPIKY_CUDA = False

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import UncertaintyMode
from spiky.lutorch import anchor_pairs_lookup as anchor_pairs_lookup_mod
from spiky.lutorch import l_projection as l_projection_mod

SEED = 42


@contextmanager
def _use_lutorch_custom_cuda_kernels(enabled: bool):
    prev_lookup = anchor_pairs_lookup_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS
    prev_proj = l_projection_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS
    anchor_pairs_lookup_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = enabled
    l_projection_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = enabled
    try:
        yield
    finally:
        anchor_pairs_lookup_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = prev_lookup
        l_projection_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = prev_proj


def test_multi_head_lut_simple(device):
    """
    Simple forward test for MultiHeadLut with n_heads=1.
    When spiky_cuda is available, verifies exact output match against LUTLayer.
    Otherwise runs as a finite-output smoke test.
    """
    torch.manual_seed(SEED)

    B, n_inputs, nap, n_det, n_out = 4, 100, 4, 10, 20
    x = torch.randn(B, n_inputs, device=device)

    mhl = MultiHeadLut(
        input_dim=n_inputs, n_heads=1, n_outputs=n_out,
        n_anchor_pairs=nap, tables_per_head=n_det,
        random_seed=SEED, device=device,
    ).eval()

    if not _HAS_SPIKY_CUDA:
        warnings.warn(
            "spiky_cuda (required by LUTLayer baseline) is not available; "
            "running MultiHeadLut simple test as a finite-output smoke test only.",
            UserWarning, stacklevel=2,
        )
        with torch.no_grad():
            out = mhl(x)
        assert out.shape == (B, 1, n_out)
        assert torch.isfinite(out).all()
        return

    lut = LUTLayer(
        n_inputs=n_inputs, n_anchors_per_detector=nap,
        n_detectors=n_det, n_outputs=n_out,
        synapse_meta=SynapseMeta(initial_weight=1.0, initial_noise_level=-1.0),
        random_seed=SEED, device=device,
    ).eval()

    anchors = lut._export_anchors()
    mhl.lookup.anchor_pairs_a.data = anchors[:, :, 0].to(torch.long).clone()
    mhl.lookup.anchor_pairs_b.data = anchors[:, :, 1].to(torch.long).clone()
    mhl.projection.weights.data = lut.export_weights(inverse_order=False).clone()

    with torch.no_grad():
        lut_out = lut(x.unsqueeze(1)).squeeze(1)
        mhl_out = mhl(x).squeeze(1)

    assert mhl_out.shape == (B, n_out)
    assert torch.abs(lut_out - mhl_out).max().item() < 1e-5


def test_multi_head_lut_training(device):
    """
    Training-mode test for MultiHeadLut (n_heads=1).
    When spiky_cuda is available, verifies outputs and weights match LUTLayer
    across 100 SGD iterations. Otherwise runs as a finite-loss smoke test.
    """
    torch.manual_seed(SEED)

    B, n_inputs, nap, n_det, n_out, n_iter = 4, 100, 4, 10, 20, 100

    mhl = MultiHeadLut(
        input_dim=n_inputs, n_heads=1, n_outputs=n_out,
        n_anchor_pairs=nap, tables_per_head=n_det,
        random_seed=SEED, device=device,
    ).train()

    if not _HAS_SPIKY_CUDA:
        warnings.warn(
            "spiky_cuda (required by LUTLayer baseline) is not available; "
            "running MultiHeadLut training test as a finite-loss smoke test only.",
            UserWarning, stacklevel=2,
        )
        opt = torch.optim.SGD(mhl.parameters(), lr=0.01)
        for i in range(min(n_iter, 50)):
            x = torch.randn(B, n_inputs, device=device)
            out = mhl(x)
            assert torch.isfinite(out).all(), f"iter {i}: non-finite output"
            loss = ((out - torch.randn_like(out)) ** 2).mean()
            assert torch.isfinite(loss), f"iter {i}: non-finite loss"
            opt.zero_grad(); loss.backward(); opt.step()
        return

    lut = LUTLayer(
        n_inputs=n_inputs, n_anchors_per_detector=nap,
        n_detectors=n_det, n_outputs=n_out,
        synapse_meta=SynapseMeta(initial_weight=1.0, initial_noise_level=-1.0),
        random_seed=SEED, device=device,
    ).train()

    anchors = lut._export_anchors()
    mhl.lookup.anchor_pairs_a.data = anchors[:, :, 0].to(torch.long).clone()
    mhl.lookup.anchor_pairs_b.data = anchors[:, :, 1].to(torch.long).clone()
    mhl.projection.weights.data = lut.export_weights(inverse_order=False).clone()

    opt_lut = torch.optim.SGD(lut.parameters(), lr=0.01)
    opt_mhl = torch.optim.SGD(mhl.parameters(), lr=0.01)

    for i in range(n_iter):
        x = torch.randn(B, n_inputs, device=device)
        target = torch.randn(B, n_out, device=device)

        lut_out = lut(x.unsqueeze(1)).squeeze(1)
        mhl_out = mhl(x).squeeze(1)

        assert torch.abs(lut_out - mhl_out).max().item() < 1e-5, \
            f"iter {i}: output mismatch"

        (((lut_out - target) ** 2).mean()).backward()
        opt_lut.step(); opt_lut.zero_grad()

        (((mhl_out - target) ** 2).mean()).backward()
        opt_mhl.step(); opt_mhl.zero_grad()

        lut_weights = lut.export_weights(inverse_order=False)
        assert torch.abs(mhl.projection.weights.data - lut_weights).max().item() < 1e-5, \
            f"iter {i}: weight mismatch"
        mhl.projection.weights.data = lut_weights.clone()


@pytest.mark.parametrize("n_alternatives", [2, 3, 4])
def test_multi_head_lut_smooth_simple(device, n_alternatives):
    """
    Smoke test for MultiHeadLut in smooth mode.
    On CUDA, verifies parity between custom kernels on/off.
    On CPU, verifies finite outputs in eval and train mode.
    """
    torch.manual_seed(SEED)

    B, n_inputs, n_heads, nap, n_det, n_out = 4, 100, 2, 4, 3, 20
    x = torch.randn(B, n_inputs, device=device)

    def make():
        return MultiHeadLut(
            input_dim=n_inputs, n_heads=n_heads, n_outputs=n_out,
            n_anchor_pairs=nap, tables_per_head=n_det,
            random_seed=SEED, device=device,
            n_alternatives=n_alternatives, smooth_mode=True,
            uncertainty_mode=UncertaintyMode.INVERSE_QUADRATIC,
        )

    if device != "cuda":
        m = make()
        m.eval()
        with torch.no_grad():
            out = m(x)
        assert out.shape == (B, n_heads, n_out)
        assert torch.isfinite(out).all()
        m.train()
        assert torch.isfinite(m(x)).all()
        return

    m_custom = make()
    m_ref = make()
    m_ref.load_state_dict(m_custom.state_dict())

    m_custom.eval(); m_ref.eval()
    with torch.no_grad():
        with _use_lutorch_custom_cuda_kernels(True):
            out_custom = m_custom(x)
        with _use_lutorch_custom_cuda_kernels(False):
            out_ref = m_ref(x)
    torch.testing.assert_close(out_custom, out_ref, atol=1e-5, rtol=1e-4)

    m_custom.train(); m_ref.train()
    with _use_lutorch_custom_cuda_kernels(True):
        out_custom = m_custom(x)
    with _use_lutorch_custom_cuda_kernels(False):
        out_ref = m_ref(x)
    torch.testing.assert_close(out_custom, out_ref, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("n_alternatives", [2, 3, 4])
def test_multi_head_lut_smooth_training(device, n_alternatives):
    """
    Training test for MultiHeadLut in smooth mode.
    On CUDA, runs two parallel models (custom kernels on/off) and checks outputs,
    losses, and parameters match at every iteration.
    On CPU, checks loss stays finite for 100 iterations.
    """
    torch.manual_seed(SEED)

    B, n_inputs, n_heads, nap, n_det, n_out, n_iter = 4, 100, 2, 4, 3, 20, 100

    def make():
        return MultiHeadLut(
            input_dim=n_inputs, n_heads=n_heads, n_outputs=n_out,
            n_anchor_pairs=nap, tables_per_head=n_det,
            random_seed=SEED, device=device,
            n_alternatives=n_alternatives, smooth_mode=True,
            uncertainty_mode=UncertaintyMode.INVERSE_QUADRATIC,
        )

    if device != "cuda":
        m = make().train()
        opt = torch.optim.SGD(m.parameters(), lr=0.1)
        for i in range(n_iter):
            x = torch.randn(B, n_inputs, device=device)
            loss = ((m(x) - torch.randn(B, n_heads, n_out)) ** 2).mean()
            assert torch.isfinite(loss), f"iter {i}: non-finite loss"
            opt.zero_grad(); loss.backward(); opt.step()
            for name, p in m.named_parameters():
                assert torch.isfinite(p).all(), f"iter {i}: non-finite param {name}"
        return

    m_custom = make().train()
    m_ref = make().train()
    m_ref.load_state_dict(m_custom.state_dict())
    opt_c = torch.optim.SGD(m_custom.parameters(), lr=0.1)
    opt_r = torch.optim.SGD(m_ref.parameters(), lr=0.1)

    for i in range(n_iter):
        x = torch.randn(B, n_inputs, device=device)
        target = torch.randn(B, n_heads, n_out, device=device)

        with _use_lutorch_custom_cuda_kernels(True):
            out_c = m_custom(x)
            loss_c = ((out_c - target) ** 2).mean()
            opt_c.zero_grad(); loss_c.backward(); opt_c.step()

        with _use_lutorch_custom_cuda_kernels(False):
            out_r = m_ref(x)
            loss_r = ((out_r - target) ** 2).mean()
            opt_r.zero_grad(); loss_r.backward(); opt_r.step()

        torch.testing.assert_close(out_c, out_r, atol=2e-4, rtol=2e-4,
                                   msg=f"iter {i}: output mismatch")
        torch.testing.assert_close(loss_c, loss_r, atol=2e-5, rtol=2e-4,
                                   msg=f"iter {i}: loss mismatch")
        for (nc, pc), (nr, pr) in zip(m_custom.named_parameters(), m_ref.named_parameters()):
            assert nc == nr
            torch.testing.assert_close(pc, pr, atol=5e-4, rtol=5e-4,
                                       msg=f"iter {i}: param mismatch for {nc}")
