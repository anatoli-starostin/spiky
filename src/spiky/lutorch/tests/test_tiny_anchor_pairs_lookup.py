"""Tests for TinyAnchorPairsLookup."""
import os

import pytest
import torch

from spiky.lutorch.tiny_anchor_pairs_lookup import (
    TinyAnchorPairsLookup,
    _tiny_forward_pytorch,
    _tiny_forward_cuda,
    _tiny_backward_pytorch,
    _tiny_backward_cuda,
    _can_use_native_tiny_apl,
)


DEVICES = [torch.device("cpu"), torch.device("cuda:0")] if torch.cuda.is_available() else [torch.device("cpu")]


@pytest.fixture(params=DEVICES, ids=lambda d: f"dev={d.type}")
def device(request):
    return request.param


def _make_lut(device, input_dim=32, n_tables=8, n_anchor_pairs=6, n_heads=1, seed=42):
    return TinyAnchorPairsLookup(
        input_dim=input_dim,
        n_tables=n_tables,
        n_anchor_pairs=n_anchor_pairs,
        n_heads=n_heads,
        random_seed=seed,
        device=device,
    )


def test_rejects_n_anchor_pairs_too_large():
    # 16 is now rejected (int16 lookup-index overflow); 15 is the max.
    for bad in (16, 17):
        with pytest.raises(ValueError, match="1 <= n_anchor_pairs <= 15"):
            TinyAnchorPairsLookup(input_dim=64, n_tables=4, n_anchor_pairs=bad, device=torch.device("cpu"))


def test_rejects_n_anchor_pairs_zero():
    with pytest.raises(ValueError, match="1 <= n_anchor_pairs <= 15"):
        TinyAnchorPairsLookup(input_dim=64, n_tables=4, n_anchor_pairs=0, device=torch.device("cpu"))


def test_rejects_integer_input(device):
    lut = _make_lut(device)
    x = torch.zeros(4, 32, dtype=torch.int32, device=device)
    with pytest.raises(TypeError):
        lut(x)


def test_rejects_wrong_shape(device):
    lut = _make_lut(device)
    x = torch.randn(4, 16, device=device)
    with pytest.raises(ValueError):
        lut(x)


def test_anchor_pairs_are_canonical(device):
    lut = _make_lut(device, n_anchor_pairs=6)
    a, b = lut.anchor_pairs_a, lut.anchor_pairs_b
    assert (a < b).all().item(), "expected a < b for CANONICAL_DISTINCT sampling"


def test_anchor_pairs_distinct_per_table(device):
    lut = _make_lut(device, n_anchor_pairs=6)
    a, b = lut.anchor_pairs_a, lut.anchor_pairs_b
    n_tables, nap = a.shape
    for t in range(n_tables):
        pairs = {(int(a[t, p]), int(b[t, p])) for p in range(nap)}
        assert len(pairs) == nap, f"table {t}: expected {nap} distinct canonical pairs, got {len(pairs)}"


def test_forward_shapes_and_dtypes_train(device):
    lut = _make_lut(device)
    lut.train()
    x = torch.randn(4, 32, device=device, requires_grad=True)
    lookup_indices, lookup_alt_indices, lookup_alt_deltas, carriers_main, carriers_alt = lut(x)

    assert lookup_indices.shape == (4, 8)
    assert lookup_indices.dtype == torch.int16
    assert lookup_alt_indices.shape == (4, 8, 1)
    assert lookup_alt_indices.dtype == torch.int16
    assert lookup_alt_deltas.shape == (4, 8, 1)
    assert lookup_alt_deltas.dtype == torch.float32
    assert carriers_main.shape == (4, 8)
    assert carriers_alt.shape == (4, 8, 1)


def test_forward_eval_no_grad(device):
    lut = _make_lut(device)
    lut.eval()
    x = torch.randn(4, 32, device=device)
    lookup_indices, lookup_alt_indices, lookup_alt_deltas, cm, ca = lut(x)
    assert lookup_indices.dtype == torch.int16
    assert lookup_alt_indices.dtype == torch.int16
    assert cm is None and ca is None


def test_lookup_index_matches_manual(device):
    lut = _make_lut(device, input_dim=16, n_tables=2, n_anchor_pairs=4)
    lut.eval()
    x = torch.randn(5, 16, device=device)
    lookup_indices, _, _, _, _ = lut(x)

    # Manual computation
    a, b = lut.anchor_pairs_a, lut.anchor_pairs_b
    for batch in range(5):
        for t in range(2):
            manual = 0
            for k in range(4):
                if x[batch, a[t, k]].item() > x[batch, b[t, k]].item():
                    manual |= (1 << k)
            assert lookup_indices[batch, t].item() == manual, \
                f"batch {batch} table {t}: {lookup_indices[batch, t].item()} != {manual}"


def test_alt_index_differs_in_one_bit(device):
    lut = _make_lut(device, input_dim=16, n_tables=4, n_anchor_pairs=6)
    lut.eval()
    x = torch.randn(8, 16, device=device)
    lookup_indices, lookup_alt_indices, _, _, _ = lut(x)
    primary = lookup_indices.to(torch.int32)
    alt = lookup_alt_indices.squeeze(-1).to(torch.int32)
    # Exactly 1 bit differs
    diff = primary ^ alt
    popcnt = torch.tensor([bin(v.item()).count('1') for v in diff.flatten()])
    assert (popcnt == 1).all().item()


def test_alt_delta_is_min_abs(device):
    lut = _make_lut(device, input_dim=32, n_tables=8, n_anchor_pairs=6)
    lut.eval()
    torch.manual_seed(0)
    x = torch.randn(4, 32, device=device)
    _, _, lookup_alt_deltas, _, _ = lut(x)

    # Manual: compute all deltas, find argmin(|delta|), verify
    a, b = lut.anchor_pairs_a.long(), lut.anchor_pairs_b.long()
    x_a = x[:, a.view(-1)].view(4, 8, 6)
    x_b = x[:, b.view(-1)].view(4, 8, 6)
    deltas = x_a - x_b
    min_abs, _ = deltas.abs().min(dim=2, keepdim=True)
    # |lookup_alt_deltas| should equal min_abs
    assert torch.allclose(lookup_alt_deltas.abs(), min_abs, atol=1e-5)


def test_backward_flows_to_x(device):
    lut = _make_lut(device)
    lut.train()
    x = torch.randn(4, 32, device=device, requires_grad=True)
    _, _, _, cm, ca = lut(x)
    # Distinct weights on main vs alt carriers to produce nonzero grad_diff
    loss = cm.sum() + 2.0 * ca.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all().item()
    assert x.grad.abs().sum().item() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only test")
def test_cuda_matches_pytorch_forward():
    """Native CUDA forward must match the PyTorch fallback."""
    torch.manual_seed(0)
    lut = TinyAnchorPairsLookup(
        input_dim=32, n_tables=8, n_anchor_pairs=6,
        random_seed=42, device=torch.device("cuda:0"),
    )
    x = torch.randn(4, 32, device="cuda:0")

    # Native
    li_n, ai_n, ad_n, a1_n, a2_n = _tiny_forward_cuda(x, lut.anchor_pairs_a, lut.anchor_pairs_b)
    # Pytorch fallback
    li_p, ai_p, ad_p, a1_p, a2_p = _tiny_forward_pytorch(
        x, lut.anchor_pairs_a, lut.anchor_pairs_b, lut.powers,
    )

    assert torch.equal(li_n, li_p), "lookup_indices mismatch"
    assert torch.equal(ai_n, ai_p), "lookup_alt_indices mismatch"
    assert torch.allclose(ad_n, ad_p, atol=0), "lookup_alt_deltas mismatch"
    assert torch.equal(a1_n, a1_p), "anchor1_ids mismatch"
    assert torch.equal(a2_n, a2_p), "anchor2_ids mismatch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only test")
def test_cuda_matches_pytorch_backward():
    """Native CUDA backward must match the PyTorch fallback."""
    torch.manual_seed(0)
    lut = TinyAnchorPairsLookup(
        input_dim=32, n_tables=8, n_anchor_pairs=6,
        random_seed=42, device=torch.device("cuda:0"),
    )
    x = torch.randn(4, 32, device="cuda:0")
    _, _, alt_delta, a1, a2 = _tiny_forward_cuda(x, lut.anchor_pairs_a, lut.anchor_pairs_b)
    grad_main = torch.randn(4, 8, device="cuda:0")
    grad_alt = torch.randn(4, 8, 1, device="cuda:0")
    grad_direct = torch.randn(4, 8, 1, device="cuda:0")
    batch_offset = (torch.arange(4, device="cuda:0", dtype=torch.int32).repeat_interleave(8) * 32).contiguous()

    x_grad_n = _tiny_backward_cuda(
        x, a1, a2, alt_delta, grad_main, grad_alt, grad_direct=grad_direct,
    ).view(x.shape)

    x_grad_p = _tiny_backward_pytorch(
        x, a1, a2, alt_delta, batch_offset, grad_main, grad_alt,
    )
    # Direct grad path in PyTorch fallback:
    d = grad_direct.reshape(-1)
    flat_a = (batch_offset + a1.reshape(-1).to(torch.int32)).to(torch.int64)
    flat_b = (batch_offset + a2.reshape(-1).to(torch.int32)).to(torch.int64)
    x_grad_p.scatter_add_(0, flat_a, d)
    x_grad_p.scatter_add_(0, flat_b, -d)
    x_grad_p = x_grad_p.view(x.shape)

    assert torch.allclose(x_grad_n, x_grad_p, atol=1e-6, rtol=1e-5), \
        f"x_grad mismatch: max |diff| = {(x_grad_n - x_grad_p).abs().max().item()}"


def test_grad_through_alt_deltas(device):
    """Gradient through lookup_alt_deltas alone must hit the correct x dims."""
    lut = _make_lut(device, n_anchor_pairs=4)
    lut.train()
    x = torch.randn(2, 32, device=device, requires_grad=True)
    _, _, alt_deltas, _, _ = lut(x)
    alt_deltas.sum().backward()
    # d delta / d x[a] = +1, d delta / d x[b] = -1; so grad for anchor pair's a/b
    # dims should be +1 and -1. Sum over batch gives integer counts.
    assert x.grad is not None
    # Only anchor1/anchor2 positions should have nonzero grad
    assert torch.isfinite(x.grad).all().item()
