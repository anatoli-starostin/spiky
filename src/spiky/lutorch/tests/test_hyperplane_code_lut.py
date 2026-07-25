"""CPU tests for HyperplaneCodeLUT (the code-scoring hyperplane unembedder)."""
import torch

from spiky.lutorch.hyperplane_code_lut import HyperplaneCodeLUT


def test_forward_shape_and_param_count():
    E, nap, T = 32, 4, 3
    V = 1 << nap
    m = HyperplaneCodeLUT(E, nap, T, V, random_seed=0)
    y = m(torch.randn(5, E))
    assert y.shape == (5, V)
    n = sum(p.numel() for p in m.parameters())
    assert n == T * nap * (E + 1) + T * V   # hyperplanes + per-code scalars


def test_n_outputs_must_equal_2_pow_nap():
    try:
        HyperplaneCodeLUT(16, nap=4, n_tables=2, n_outputs=17)
    except ValueError:
        return
    raise AssertionError("expected ValueError for n_outputs != 2^nap")


def test_code_matrix_msb_first_pm1():
    E, nap, T = 8, 5, 2
    V = 1 << nap
    m = HyperplaneCodeLUT(E, nap, T, V, random_seed=0)
    B = m.code_matrix
    assert B.shape == (V, nap)
    assert set(B.unique().tolist()) <= {-1.0, 1.0}
    for k in (0, 1, 7, V - 1, 13):
        for i in range(nap):
            exp = 1.0 if (k >> (nap - 1 - i)) & 1 else -1.0  # MSB-first
            assert B[k, i].item() == exp


def test_near_uniform_logits_with_tiny_gate():
    E, nap, T = 32, 6, 4
    V = 1 << nap
    m = HyperplaneCodeLUT(E, nap, T, V, w_cell_init=1e-5, random_seed=1)
    y = m(torch.randn(8, E))
    # tiny per-code gate -> logits ~constant across the vocab -> ~uniform softmax
    assert y.std(dim=-1).max().item() < 1e-2


def test_backprops_to_all_params():
    E, nap, T = 16, 4, 2
    V = 1 << nap
    m = HyperplaneCodeLUT(E, nap, T, V, random_seed=2)
    y = m(torch.randn(4, E, requires_grad=True))
    y.sum().backward()
    for name, p in m.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
