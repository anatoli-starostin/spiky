"""Tests for WTALookup module."""
import pytest
import torch

from spiky.lutorch.wta_lookup import WTALookup
from spiky.lutorch.lut_helpers import UncertaintyMode

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
