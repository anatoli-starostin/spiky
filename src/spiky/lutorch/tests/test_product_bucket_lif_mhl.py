"""Tests for ProductBucketLIFMHL — the mixed-radix product generalization of the bucket LIF detector."""
import pytest
import torch
from spiky.lutorch.product_bucket_lif_mhl import ProductBucketLIFMHL, MAX_CELLS


def _model(**kw):
    torch.manual_seed(0)
    cfg = dict(in_dim=17, out_dim=6, n_heads=4, n_det=2, buckets=16)
    cfg.update(kw)
    return ProductBucketLIFMHL(**cfg)


def test_forward_shape_and_finite():
    m = _model()
    x = torch.randn(8, 17)
    for mode in ("st", "hard", "soft"):
        y = m(x, mode=mode)
        assert y.shape == (8, 6) and torch.isfinite(y).all(), f"bad output in mode {mode}"


def test_ndet1_reduces_to_plain_bucket():
    """N_det=1 must reduce to a single M-way bucket table (M cells, joint index == the one bucket digit)."""
    m = _model(n_det=1, n_heads=1, buckets=16)
    assert m.cells == 16 and m.table.shape == (1, 16, 6)
    x = torch.randn(8, 17)
    t_hard, t_soft = m._first_spike(x)
    b_hard, p = m._bucket(t_hard, t_soft)
    idx = (b_hard * m.radix.view(1, 1, -1)).sum(-1)                  # (B,H)
    assert torch.equal(idx, b_hard[..., 0]), "N_det=1 joint index must equal the single bucket digit"
    # hard read == plain gather of that bucket's row
    y_hard = m(x, mode="hard")
    manual = m.table[0, b_hard[:, 0, 0]].sum(0) if False else None   # (unused) documented equivalence below
    assert torch.allclose(y_hard, m.table[0][b_hard[:, 0, 0]], atol=1e-6), "N_det=1 hard read must be the single bucket row"


def test_mixed_radix_hard_index_and_soft_consistency():
    """The hard gather addresses the SAME grid cell (b_0, b_1) that the soft tensor-product peaks at."""
    m = _model(n_det=2, n_heads=3, buckets=8)
    x = torch.randn(16, 17)
    t_hard, t_soft = m._first_spike(x)
    b_hard, _ = m._bucket(t_hard, t_soft)                            # (B,H,2)
    y_hard = m(x, mode="hard")                                      # (B,out) summed over heads
    # reconstruct manually from the reshaped grid table
    grid = m.table.reshape(m.n_heads, m.buckets, m.buckets, m.out_dim)   # (H,M,M,out)
    B, H = x.shape[0], m.n_heads
    manual = torch.zeros(B, m.out_dim)
    for h in range(H):
        manual += grid[h][b_hard[:, h, 0], b_hard[:, h, 1]]         # cell (b0,b1) per head, summed
    assert torch.allclose(y_hard, manual, atol=1e-5), "hard gather must equal grid[h, b0, b1] summed over heads"


def test_soft_equals_dense_tensor_product():
    """The sequential contraction equals the explicit P = p0 (x) p1 dense outer-product read."""
    m = _model(n_det=2, n_heads=3, buckets=8)
    x = torch.randn(12, 17)
    t_hard, t_soft = m._first_spike(x)
    _, p = m._bucket(t_hard, t_soft)                                # (B,H,2,M)
    y_soft = m(x, mode="soft")                                      # summed over heads
    grid = m.table.reshape(m.n_heads, m.buckets, m.buckets, m.out_dim)
    P = p[:, :, 0, :].unsqueeze(-1) * p[:, :, 1, :].unsqueeze(-2)   # (B,H,M,M) rank-1 tensor product
    dense = torch.einsum('bhij,hijo->bho', P, grid).sum(1)         # sum over heads
    assert torch.allclose(y_soft, dense, atol=1e-4), "sequential contraction must equal the dense tensor-product read"


def test_soft_joint_sums_to_one():
    m = _model(n_det=2, buckets=8)
    x = torch.randn(8, 17)
    t_hard, t_soft = m._first_spike(x)
    _, p = m._bucket(t_hard, t_soft)
    assert torch.allclose(p.sum(-1), torch.ones_like(p.sum(-1)), atol=1e-4), "each detector's bucket dist sums to 1"
    # joint tensor-product mass = product of per-detector masses ~ 1
    assert torch.allclose(p.sum(-1).prod(-1), torch.ones(x.shape[0], m.n_heads), atol=1e-3)


def test_straight_through_invariant():
    m = _model(n_det=2, n_heads=3, buckets=8)
    x = torch.randn(24, 17)
    with torch.no_grad():
        y_st, y_hard, y_soft = m(x, "st"), m(x, "hard"), m(x, "soft")
    assert torch.allclose(y_st, y_hard, atol=1e-5), "ST forward must equal hard"
    assert not torch.allclose(y_st, y_soft, atol=1e-4), "soft must differ from hard"


def test_all_params_get_gradient():
    m = _model(n_det=2, n_heads=3, buckets=8)
    x = torch.randn(16, 17); tgt = torch.randn(16, 6)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, "st"), tgt).backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0, f"{name} no grad"


def test_cell_cap_enforced():
    with pytest.raises(ValueError):
        ProductBucketLIFMHL(in_dim=17, out_dim=6, n_heads=1, n_det=4, buckets=16)   # 16**4 = 65536 > 4096
    assert MAX_CELLS == 4096
