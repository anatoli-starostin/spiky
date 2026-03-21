"""Tests for WTA, ProjectionWTA, and Conv2DWTA modules."""
import pytest
import torch

from spiky.lutorch.multi_head_wta import WTA, ProjectionWTA, Conv2DWTA
from spiky.lutorch.multi_head_lut import UnfoldConfiguration
from spiky.lutorch.lut_helpers import UncertaintyMode

SEED = 42


@pytest.mark.parametrize("n_alternatives", [1, 3])
@pytest.mark.parametrize("smooth_mode", [False, True])
def test_wta_modules_smoke(device, smooth_mode, n_alternatives):
    """
    Smoke test for all three WTA modules (WTA, ProjectionWTA, Conv2DWTA).
    Checks output shapes in train and eval, finiteness, and gradient flow.
    """
    torch.manual_seed(SEED)
    B = 3
    dev = torch.device(device)
    kwargs = dict(n_alternatives=n_alternatives, smooth_mode=smooth_mode,
                  random_seed=SEED, device=dev)

    # --- WTA: [B, n_channels, n_inputs] -> [B, n_channels, n_outputs] ---
    n_channels, n_inputs, n_outputs = 6, 16, 8
    wta = WTA(n_channels=n_channels, n_inputs=n_inputs, n_outputs=n_outputs, **kwargs)
    x_wta = torch.randn(B, n_channels, n_inputs, device=device)

    wta.train()
    out = wta(x_wta)
    assert out.shape == (B, n_channels, n_outputs) and torch.isfinite(out).all()
    out.sum().backward()
    assert wta.projection.weights.grad.norm() > 0

    wta.eval()
    with torch.no_grad():
        out = wta(x_wta)
    assert out.shape == (B, n_channels, n_outputs) and torch.isfinite(out).all()

    # --- ProjectionWTA: [B, H, W] -> [B, H_p, W_p, n_outputs] ---
    H, W, n_out_proj = 8, 8, 4
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=2)
    H_p, W_p = cfg.output_spatial_shape()
    pwta = ProjectionWTA(cfg, n_outputs=n_out_proj, **kwargs)
    x_proj = torch.randn(B, H, W, device=device)

    pwta.train()
    out = pwta(x_proj)
    assert out.shape == (B, H_p, W_p, n_out_proj) and torch.isfinite(out).all()
    out.sum().backward()
    assert pwta.wta.projection.weights.grad.norm() > 0

    pwta.eval()
    with torch.no_grad():
        out = pwta(x_proj)
    assert out.shape == (B, H_p, W_p, n_out_proj) and torch.isfinite(out).all()

    # --- Conv2DWTA: [B, C, H, W] -> [B, out_channels, H_p, W_p] ---
    C, out_channels = 4, 16
    cfg2 = UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=2)
    H_p2, W_p2 = cfg2.output_spatial_shape()
    cwta = Conv2DWTA(cfg2, in_channels=C, out_channels=out_channels, **kwargs)
    x_conv = torch.randn(B, C, H, W, device=device)

    cwta.train()
    out = cwta(x_conv)
    assert out.shape == (B, out_channels, H_p2, W_p2) and torch.isfinite(out).all()
    out.sum().backward()
    assert cwta.wta.projection.weights.grad.norm() > 0

    cwta.eval()
    with torch.no_grad():
        out = cwta(x_conv)
    assert out.shape == (B, out_channels, H_p2, W_p2) and torch.isfinite(out).all()


# ---------------------------------------------------------------------------
# WTA
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_alternatives", [1, 3])
@pytest.mark.parametrize("smooth_mode", [False, True])
def test_wta_output_shape(device, n_alternatives, smooth_mode):
    """Output shape is [B, n_channels, n_outputs] in both train and eval."""
    torch.manual_seed(SEED)
    B, n_channels, n_inputs, n_outputs = 4, 6, 16, 8

    m = WTA(
        n_channels=n_channels,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_alternatives=n_alternatives,
        smooth_mode=smooth_mode,
        random_seed=SEED,
        device=torch.device(device),
    )
    x = torch.randn(B, n_channels, n_inputs, device=device)

    m.train()
    out = m(x)
    assert out.shape == (B, n_channels, n_outputs)
    assert torch.isfinite(out).all()

    m.eval()
    with torch.no_grad():
        out = m(x)
    assert out.shape == (B, n_channels, n_outputs)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("smooth_mode", [False, True])
def test_wta_backward(device, smooth_mode):
    """Gradients flow to projection weights in both smooth and non-smooth modes."""
    torch.manual_seed(SEED)
    B, n_channels, n_inputs, n_outputs = 3, 4, 12, 6

    m = WTA(
        n_channels=n_channels,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_alternatives=2,
        smooth_mode=smooth_mode,
        random_seed=SEED,
        device=torch.device(device),
    ).train()

    m(torch.randn(B, n_channels, n_inputs, device=device)).sum().backward()

    assert m.projection.weights.grad is not None
    assert m.projection.weights.grad.norm() > 0


def test_wta_training_stays_finite(device):
    """Loss and parameters stay finite over 50 SGD iterations."""
    torch.manual_seed(SEED)
    B, n_channels, n_inputs, n_outputs = 4, 6, 16, 8

    m = WTA(
        n_channels=n_channels,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_alternatives=3,
        smooth_mode=True,
        random_seed=SEED,
        device=torch.device(device),
    ).train()
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    for i in range(50):
        x = torch.randn(B, n_channels, n_inputs, device=device)
        loss = ((m(x) - torch.randn(B, n_channels, n_outputs, device=device)) ** 2).mean()
        assert torch.isfinite(loss), f"iter {i}: non-finite loss"
        opt.zero_grad()
        loss.backward()
        opt.step()
        for name, p in m.named_parameters():
            assert torch.isfinite(p).all(), f"iter {i}: non-finite param {name}"


@pytest.mark.parametrize("uncertainty_mode", [
    UncertaintyMode.INVERSE_L1,
    UncertaintyMode.INVERSE_QUADRATIC,
])
def test_wta_uncertainty_modes(device, uncertainty_mode):
    """Both uncertainty modes produce finite outputs and gradients."""
    torch.manual_seed(SEED)
    B, n_channels, n_inputs, n_outputs = 2, 4, 10, 5

    m = WTA(
        n_channels=n_channels,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        n_alternatives=2,
        smooth_mode=True,
        uncertainty_mode=uncertainty_mode,
        device=torch.device(device),
    ).train()

    out = m(torch.randn(B, n_channels, n_inputs, device=device))
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert m.projection.weights.grad.norm() > 0


# ---------------------------------------------------------------------------
# ProjectionWTA
# ---------------------------------------------------------------------------

def test_projection_wta_output_shape(device):
    """Output shape is [B, H_p, W_p, n_outputs] in both train and eval."""
    torch.manual_seed(SEED)
    B, H, W, n_outputs = 3, 8, 8, 4
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=1)
    H_p, W_p = cfg.output_spatial_shape()  # 6, 6

    m = ProjectionWTA(
        cfg,
        n_outputs=n_outputs,
        n_alternatives=2,
        smooth_mode=True,
        random_seed=SEED,
        device=torch.device(device),
    )
    x = torch.randn(B, H, W, device=device)

    m.train()
    out = m(x)
    assert out.shape == (B, H_p, W_p, n_outputs)
    assert torch.isfinite(out).all()
    out.sum().backward()

    m.eval()
    with torch.no_grad():
        out = m(x)
    assert out.shape == (B, H_p, W_p, n_outputs)
    assert torch.isfinite(out).all()


def test_projection_wta_stride(device):
    """Output spatial shape follows unfold formula with stride > 1."""
    torch.manual_seed(SEED)
    B, H, W = 2, 16, 16
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=4, stride=2)
    H_p, W_p = cfg.output_spatial_shape()  # 7, 7

    m = ProjectionWTA(cfg, n_outputs=8, n_alternatives=1, device=torch.device(device)).eval()
    with torch.no_grad():
        out = m(torch.randn(B, H, W, device=device))
    assert out.shape == (B, H_p, W_p, 8)


def test_projection_wta_wrong_input_shape(device):
    """ValueError on wrong spatial input size."""
    cfg = UnfoldConfiguration(H=8, W=8, kernel_size=3, stride=1)
    m = ProjectionWTA(cfg, n_outputs=4, device=torch.device(device)).eval()
    with pytest.raises(ValueError, match="does not match ProjectionWTA"):
        m(torch.randn(2, 10, 10, device=device))


def test_projection_wta_fold_config(device):
    """With fold_config, output is [B, H_out, W_out] and gradients flow."""
    torch.manual_seed(SEED)
    B, H, W = 3, 8, 8
    # unfold: (8, k=4, s=2) → (8+0-4)//2+1 = 3 → 3×3 = 9 patches
    # fold:   (8, k=4, s=2) → same 3×3 = 9 patches, output grid 8×8
    unfold_cfg = UnfoldConfiguration(H=H, W=W, kernel_size=4, stride=2)
    fold_cfg = UnfoldConfiguration(H=H, W=W, kernel_size=4, stride=2)
    assert unfold_cfg.output_spatial_shape() == fold_cfg.output_spatial_shape()

    m = ProjectionWTA(
        unfold_cfg,
        n_outputs=4,
        fold_config=fold_cfg,
        n_alternatives=2,
        smooth_mode=True,
        random_seed=SEED,
        device=torch.device(device),
    )
    x = torch.randn(B, H, W, device=device)

    m.train()
    out = m(x)
    assert out.shape == (B, fold_cfg.H, fold_cfg.W)
    assert torch.isfinite(out).all()
    out.sum().backward()

    m.eval()
    with torch.no_grad():
        out = m(x)
    assert out.shape == (B, fold_cfg.H, fold_cfg.W)
    assert torch.isfinite(out).all()


def test_projection_wta_backward(device):
    """Gradients flow through ProjectionWTA."""
    torch.manual_seed(SEED)
    B, H, W = 2, 8, 8
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=1)
    m = ProjectionWTA(
        cfg, n_outputs=4, n_alternatives=2, smooth_mode=True,
        device=torch.device(device),
    ).train()

    m(torch.randn(B, H, W, device=device)).sum().backward()
    assert m.wta.projection.weights.grad.norm() > 0


# ---------------------------------------------------------------------------
# Conv2DWTA
# ---------------------------------------------------------------------------

def test_conv2d_wta_output_shape(device):
    """Output shape is [B, out_channels, H_p, W_p] in both train and eval."""
    torch.manual_seed(SEED)
    B, C, H, W, out_channels = 3, 4, 8, 8, 16
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=2)
    H_p, W_p = cfg.output_spatial_shape()  # 3, 3

    m = Conv2DWTA(
        cfg,
        in_channels=C,
        out_channels=out_channels,
        n_alternatives=2,
        smooth_mode=True,
        random_seed=SEED,
        device=torch.device(device),
    )
    x = torch.randn(B, C, H, W, device=device)

    m.train()
    out = m(x)
    assert out.shape == (B, out_channels, H_p, W_p)
    assert torch.isfinite(out).all()
    out.sum().backward()

    m.eval()
    with torch.no_grad():
        out = m(x)
    assert out.shape == (B, out_channels, H_p, W_p)
    assert torch.isfinite(out).all()


def test_conv2d_wta_with_padding(device):
    """Conv2DWTA handles padding correctly (same-size output)."""
    torch.manual_seed(SEED)
    B, C, H, W = 2, 3, 8, 8
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=1, padding=1)
    H_p, W_p = cfg.output_spatial_shape()  # 8, 8

    m = Conv2DWTA(
        cfg, in_channels=C, out_channels=4,
        n_alternatives=1, device=torch.device(device),
    ).eval()
    with torch.no_grad():
        out = m(torch.randn(B, C, H, W, device=device))
    assert out.shape == (B, 4, H_p, W_p)


def test_conv2d_wta_wrong_channels(device):
    """ValueError on wrong number of input channels."""
    cfg = UnfoldConfiguration(H=8, W=8, kernel_size=3, stride=1)
    m = Conv2DWTA(cfg, in_channels=3, out_channels=4, device=torch.device(device)).eval()
    with pytest.raises(ValueError, match="in_channels"):
        m(torch.randn(2, 5, 8, 8, device=device))


def test_conv2d_wta_backward(device):
    """Gradients flow through Conv2DWTA."""
    torch.manual_seed(SEED)
    B, C, H, W = 2, 3, 8, 8
    cfg = UnfoldConfiguration(H=H, W=W, kernel_size=3, stride=2)
    m = Conv2DWTA(
        cfg, in_channels=C, out_channels=8,
        n_alternatives=3, smooth_mode=True,
        device=torch.device(device),
    ).train()

    m(torch.randn(B, C, H, W, device=device)).sum().backward()
    assert m.wta.projection.weights.grad.norm() > 0
