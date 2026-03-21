"""Tests for WTALookup module."""
import contextlib
import pytest
import torch

from spiky.lutorch.wta_lookup import WTALookup
import spiky.lutorch.wta_lookup as wta_lookup_mod
from spiky.lutorch.lut_helpers import UncertaintyMode


@contextlib.contextmanager
def _use_wta_custom_cuda_kernels(enabled: bool):
    prev = wta_lookup_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS
    wta_lookup_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = enabled
    try:
        yield
    finally:
        wta_lookup_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = prev

SEED = 42


def test_wta_lookup_output_shapes(device):
    torch.manual_seed(SEED)
    B, C, N, nalt = 4, 8, 32, 3
    x = torch.randn(B, C, N, device=device)

    wta = WTALookup(n_inputs=N, n_alternatives=nalt).to(device).train()
    lookup_indices, lookup_alt_indices, lookup_alt_deltas, grad_c, grad_alt_c = wta(x)

    assert lookup_indices.shape == (B, C)
    assert lookup_alt_indices.shape == (B, C, nalt)
    assert lookup_alt_deltas.shape == (B, C, nalt)
    assert grad_c.shape == (B, C)
    assert grad_alt_c.shape == (B, C, nalt)

    assert lookup_indices.dtype == torch.long
    assert lookup_alt_indices.dtype == torch.long
    assert lookup_alt_deltas.dtype == x.dtype


def test_wta_lookup_winner_is_argmax(device):
    """lookup_indices must equal argmax over the N dimension."""
    torch.manual_seed(SEED)
    B, C, N = 3, 5, 16
    x = torch.randn(B, C, N, device=device)

    wta = WTALookup(n_inputs=N, n_alternatives=2).to(device).eval()
    lookup_indices, _, _ = wta(x)

    expected = x.argmax(dim=-1)
    assert (lookup_indices == expected).all()


def test_wta_lookup_alt_deltas_nonneg(device):
    """winner_val - alt_val must be >= 0 for all alternatives."""
    torch.manual_seed(SEED)
    B, C, N, nalt = 4, 6, 20, 4
    x = torch.randn(B, C, N, device=device)

    wta = WTALookup(n_inputs=N, n_alternatives=nalt).to(device).train()
    _, _, lookup_alt_deltas, _, _ = wta(x)

    assert (lookup_alt_deltas >= 0).all()


def test_wta_lookup_alt_indices_are_topk(device):
    """Alt indices must be top-k runner-ups (not the winner itself)."""
    torch.manual_seed(SEED)
    B, C, N, nalt = 2, 4, 12, 3
    x = torch.randn(B, C, N, device=device)

    wta = WTALookup(n_inputs=N, n_alternatives=nalt).to(device).eval()
    lookup_indices, lookup_alt_indices, _ = wta(x)

    # Winner must not appear in alternatives
    winner_in_alts = (lookup_alt_indices == lookup_indices.unsqueeze(-1)).any(dim=-1)
    assert not winner_in_alts.any()

    # Gathered alt values must be <= winner values
    winner_vals = x.gather(2, lookup_indices.unsqueeze(-1))           # [B, C, 1]
    alt_vals = x.gather(2, lookup_alt_indices)                        # [B, C, nalt]
    assert (winner_vals >= alt_vals).all()


def test_wta_lookup_gradient_flows(device):
    """Gradient carriers must route non-zero gradients back to x."""
    torch.manual_seed(SEED)
    B, C, N, nalt = 3, 4, 16, 3
    x = torch.randn(B, C, N, device=device, requires_grad=True)

    wta = WTALookup(n_inputs=N, n_alternatives=nalt).to(device).train()
    _, _, _, grad_c, grad_alt_c = wta(x)

    # grad_diff = grad_main - grad_alt must be non-zero for gradient to flow.
    # Backprop only through grad_c (grad_main=1, grad_alt=0 → grad_diff=1).
    loss = grad_c.sum()
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == x.shape
    assert x.grad.abs().sum() > 0, "Expected non-zero gradients on x"


def test_wta_lookup_gradient_only_at_winner_and_alts(device):
    """Gradient must be non-zero only at the winner and alt positions."""
    torch.manual_seed(SEED)
    B, C, N, nalt = 2, 3, 10, 2
    x = torch.randn(B, C, N, device=device, requires_grad=True)

    wta = WTALookup(n_inputs=N, n_alternatives=nalt).to(device).train()
    lookup_indices, lookup_alt_indices, _, grad_c, grad_alt_c = wta(x)

    loss = grad_c.sum() + grad_alt_c.sum()
    loss.backward()

    grad = x.grad                                                   # [B, C, N]

    # Build mask of positions that should receive gradient
    active = torch.zeros(B, C, N, dtype=torch.bool, device=device)
    active.scatter_(2, lookup_indices.unsqueeze(-1), True)           # winner
    active.scatter_(2, lookup_alt_indices, True)                     # alternatives

    # No gradient must flow to non-active positions
    assert grad[~active].abs().sum() == 0


@pytest.mark.parametrize("uncertainty_mode", [
    UncertaintyMode.INVERSE_L1,
    UncertaintyMode.INVERSE_QUADRATIC,
])
def test_wta_lookup_uncertainty_modes(device, uncertainty_mode):
    """Both uncertainty modes must produce finite, non-zero gradients."""
    torch.manual_seed(SEED)
    B, C, N, nalt = 3, 4, 16, 2
    x = torch.randn(B, C, N, device=device, requires_grad=True)

    wta = WTALookup(n_inputs=N, n_alternatives=nalt, uncertainty_mode=uncertainty_mode).to(device).train()
    _, _, _, grad_c, grad_alt_c = wta(x)

    # Backprop only through grad_c so grad_diff = grad_main - grad_alt = 1 - 0 = 1.
    loss = grad_c.sum()
    loss.backward()

    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_wta_lookup_eval_shapes(device):
    torch.manual_seed(SEED)
    B, C, N, nalt = 4, 6, 24, 3
    x = torch.randn(B, C, N, device=device)

    wta = WTALookup(n_inputs=N, n_alternatives=nalt).to(device).eval()
    lookup_indices, lookup_alt_indices, lookup_alt_deltas = wta(x)

    assert lookup_indices.shape == (B, C)
    assert lookup_alt_indices.shape == (B, C, nalt)
    assert lookup_alt_deltas.shape == (B, C, nalt)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alternatives", [1, 2, 3])
def test_wta_lookup_cuda_matches_fallback_eval(n_alternatives):
    """Native CUDA forward (eval) must exactly match the PyTorch fallback."""
    torch.manual_seed(SEED)
    B, C, N = 5, 7, 20
    x = torch.randn(B, C, N, device="cuda")

    wta = WTALookup(n_inputs=N, n_alternatives=n_alternatives).cuda().eval()
    with _use_wta_custom_cuda_kernels(True):
        wi_cuda, ai_cuda, ad_cuda = wta(x)
    with _use_wta_custom_cuda_kernels(False):
        wi_ref, ai_ref, ad_ref = wta(x)

    assert (wi_cuda == wi_ref).all(), "winner indices mismatch"
    assert (ai_cuda == ai_ref).all(), "alt indices mismatch"
    torch.testing.assert_close(ad_cuda, ad_ref)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alternatives", [1, 2, 3])
def test_wta_lookup_cuda_matches_fallback_train(n_alternatives):
    """Native CUDA forward+backward (train) must match the PyTorch fallback."""
    torch.manual_seed(SEED)
    B, C, N = 4, 6, 16
    x_cuda = torch.randn(B, C, N, device="cuda", requires_grad=True)
    x_ref  = x_cuda.detach().clone().requires_grad_(True)

    wta = WTALookup(n_inputs=N, n_alternatives=n_alternatives).cuda().train()

    with _use_wta_custom_cuda_kernels(True):
        wi_c, ai_c, ad_c, gc_c, gac_c = wta(x_cuda)
    with _use_wta_custom_cuda_kernels(False):
        wi_r, ai_r, ad_r, gc_r, gac_r = wta(x_ref)

    assert (wi_c == wi_r).all(), "winner indices mismatch"
    assert (ai_c == ai_r).all(), "alt indices mismatch"
    torch.testing.assert_close(ad_c, ad_r)

    (gc_c.sum()).backward()
    (gc_r.sum()).backward()
    torch.testing.assert_close(x_cuda.grad, x_ref.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("n_alternatives", [1, 2, 3])
def test_wta_lookup_cuda_strided_input(n_alternatives):
    """Native CUDA kernels must handle non-contiguous (strided) input correctly."""
    torch.manual_seed(SEED)
    B, C, N = 3, 5, 16
    # Create a strided view by transposing a [C, B, N] tensor → [B, C, N] non-contiguous
    x_base = torch.randn(C, B, N, device="cuda")
    x_strided = x_base.permute(1, 0, 2)  # [B, C, N], non-contiguous
    assert not x_strided.is_contiguous()

    wta = WTALookup(n_inputs=N, n_alternatives=n_alternatives).cuda().eval()
    with _use_wta_custom_cuda_kernels(True):
        wi_cuda, ai_cuda, ad_cuda = wta(x_strided)
    with _use_wta_custom_cuda_kernels(False):
        wi_ref, ai_ref, ad_ref = wta(x_strided)

    assert (wi_cuda == wi_ref).all(), "winner indices mismatch on strided input"
    assert (ai_cuda == ai_ref).all(), "alt indices mismatch on strided input"
    torch.testing.assert_close(ad_cuda, ad_ref)
