"""Tests for LIFMultiHeadLUT — the unified LIF multi-head LUT generalizing Bucket and Product."""
import pytest
import torch
from spiky.lutorch.lif_multi_head_lut import LIFMultiHeadLUT, MAX_CELLS
from spiky.lutorch.bucket_lif_detectors_mhl import BucketLIFDetectorsMHL
from spiky.lutorch.product_bucket_lif_mhl import ProductBucketLIFMHL


def _m(**kw):
    torch.manual_seed(0)
    cfg = dict(input_dim=17, n_heads=2, n_outputs=6, tables_per_head=4, n_det=1, n_buckets=16)
    cfg.update(kw)
    return LIFMultiHeadLUT(**cfg)


def test_forward_shape_and_finite():
    for nd in (1, 2, 3):
        m = _m(n_det=nd, n_buckets=(16 if nd == 1 else (4 if nd == 3 else 8)))
        x = torch.randn(8, 17)
        for mode in ("st", "hard", "soft"):
            y = m(x, mode=mode)
            assert y.shape == (8, 2, 6) and torch.isfinite(y).all(), f"nd={nd} mode={mode}"


def test_straight_through_invariant():
    m = _m(n_det=2, n_buckets=8)
    x = torch.randn(24, 17)
    with torch.no_grad():
        y_st, y_hard, y_soft = m(x, mode="st"), m(x, mode="hard"), m(x, mode="soft")
    assert torch.allclose(y_st, y_hard, atol=1e-5) and not torch.allclose(y_st, y_soft, atol=1e-4)


def test_reduces_to_bucket_wrapper_exactly():
    """BucketLIFDetectorsMHL is a faithful n_det=1 view of the unified engine (byte-exact, same seed)."""
    x = torch.randn(8, 17)
    torch.manual_seed(1); u = LIFMultiHeadLUT(input_dim=17, n_heads=2, n_outputs=6, tables_per_head=16, n_det=1, n_buckets=16)
    torch.manual_seed(1); b = BucketLIFDetectorsMHL(input_dim=17, n_heads=2, n_outputs=6, tables_per_head=16, n_buckets=16)
    assert isinstance(b, LIFMultiHeadLUT)
    for mode in ("st", "hard", "soft"):
        assert torch.allclose(u(x, mode=mode), b(x, mode=mode), atol=0), f"bucket wrapper mismatch [{mode}]"
    # n_det=1 param shapes match the original Bucket layout (no n_det axis)
    assert u.w_raw.shape == (32, 17) and u.boundaries.shape == (32, 15)


def test_reduces_to_product_wrapper_exactly():
    """ProductBucketLIFMHL == unified(n_heads=H, tph=1, n_det) with heads summed (byte-exact, same seed)."""
    x = torch.randn(8, 17)
    torch.manual_seed(2); u = LIFMultiHeadLUT(input_dim=17, n_heads=8, n_outputs=6, tables_per_head=1, n_det=3, n_buckets=2)
    torch.manual_seed(2); p = ProductBucketLIFMHL(in_dim=17, out_dim=6, n_heads=8, n_det=3, buckets=2)
    assert isinstance(p, LIFMultiHeadLUT)
    for mode in ("st", "hard", "soft"):
        assert torch.allclose(u(x, mode=mode).sum(dim=1), p(x, mode=mode), atol=0), f"product wrapper mismatch [{mode}]"


def test_mixed_radix_and_tensor_product_soft():
    m = _m(n_heads=3, tables_per_head=1, n_det=2, n_buckets=8)
    x = torch.randn(16, 17)
    t_hard, t_soft = m._first_spike(x)
    b_hard, p = m._bucket(t_hard, t_soft)                     # (B,T,2), (B,T,2,M)
    # hard gather == reshaped grid cell (b0, b1)
    grid = m.table.reshape(m.n_tables, m.n_buckets, m.n_buckets, m.n_outputs)
    manual = torch.stack([grid[t][b_hard[:, t, 0], b_hard[:, t, 1]] for t in range(m.n_tables)], dim=1)  # (B,T,O)
    assert torch.allclose(m._hard_read(b_hard), manual, atol=1e-5)
    # soft contraction == dense tensor product
    P = p[:, :, 0, :].unsqueeze(-1) * p[:, :, 1, :].unsqueeze(-2)
    dense = torch.einsum('btij,tijo->bto', P, grid)
    assert torch.allclose(m._soft_read(p, detach=False), dense, atol=1e-4)


def test_all_params_incl_temperatures_get_gradient():
    m = _m(n_det=2, n_buckets=8, tables_per_head=3)
    x = torch.randn(16, 17); tgt = torch.randn(16, 2, 6)
    m.zero_grad(set_to_none=True)
    torch.nn.functional.mse_loss(m(x, mode="st"), tgt).backward()
    for name, pp in m.named_parameters():
        assert pp.grad is not None and torch.isfinite(pp.grad).all() and pp.grad.abs().sum() > 0, f"{name} no grad"
    # per-table soft temperatures are trainable and per-table shaped (init exp(0)=1.0 -> Bucket-parity)
    assert m.log_T_cross.shape == (m.n_tables,) and m.log_T_bkt.shape == (m.n_tables,)
    assert m.log_T_cross.grad.abs().sum() > 0 and m.log_T_bkt.grad.abs().sum() > 0


def test_table_init_and_cell_cap():
    T = 2 * 4
    tab = torch.randn(T, 16, 6)
    m = LIFMultiHeadLUT(input_dim=17, n_heads=2, n_outputs=6, tables_per_head=4, n_det=1, n_buckets=16, table_init=tab)
    assert torch.equal(m.table.detach(), tab)
    with pytest.raises(ValueError):
        LIFMultiHeadLUT(input_dim=17, n_heads=1, n_outputs=6, n_det=4, n_buckets=16)   # 16**4 > 4096
    assert MAX_CELLS == 4096
