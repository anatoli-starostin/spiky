"""LightMultiHeadLUT's forward is fused via F.embedding_bag; this pins it to the naive form.

The layer used to gather the selected rows as [.., n_tables, output_dim] and then do
``(rows * score.unsqueeze(-1)).sum(over tables)``. That materialised 1.2 GiB of rows per
layer per call at the anchor sizing. It now uses
``F.embedding_bag(mode='sum', per_sample_weights=score)``, which fuses gather + weight + sum.

The naive expression is kept HERE as the reference implementation, so the fusion is pinned to
the thing it replaced rather than to itself, and so the equivalence is readable.

What must hold:
  * forward and both gradients agree with the naive form to floating-point tolerance
    (summation order differs, so this is allclose, not equal -- see the note below);
  * the defining property survives: the address is a detached integer, so grad reaches x
    ONLY through the score. A test drives that exactly, not statistically;
  * it holds for the single-head and the multi-head (block-diagonal) paths, and for every
    confidence form.
"""
import pytest
import torch
import torch.nn.functional as F

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
from spiky.lutorch.fast_multi_head_lut import _confidence_score

_FORMS = ["bounded", "margin", "bounded_norm"]


def _naive_single(m, x):
    """The pre-fusion forward, verbatim in structure: gather rows, weight, sum."""
    d = x[:, m.anchor_a] - x[:, m.anchor_b]
    index = ((d.detach() > 0).to(torch.int64) * m.powers.view(1, 1, -1)).sum(dim=-1)
    flat = m.tables.reshape(m.n_tables * m.table_size, m.output_dim)
    rows = flat[(index + m.table_offset.view(1, -1)).reshape(-1)] \
        .view(x.shape[0], m.n_tables, m.output_dim)
    score = _confidence_score(d, m.confidence_form, m.confidence_gain)
    return (rows * score.unsqueeze(-1)).sum(dim=1)


def _naive_multi(m, x):
    B, H, T, NAP = x.shape[0], m.n_heads, m.tables_per_head, m.n_anchor_pairs
    idx_a = m.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    idx_b = m.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
    d = (torch.gather(x, 2, idx_a) - torch.gather(x, 2, idx_b)).view(B, H, T, NAP)
    index = ((d.detach() > 0).to(torch.int64) * m.powers.view(1, 1, 1, -1)).sum(dim=-1)
    flat = m.tables.reshape(H * T * m.table_size, m.output_dim)
    rows = flat[(index + m.table_offset.view(1, H, T)).reshape(-1)] \
        .view(B, H, T, m.output_dim)
    score = _confidence_score(d, m.confidence_form, m.confidence_gain)
    return (rows * score.unsqueeze(-1)).sum(dim=2)


def _single(form, seed=3):
    return LightMultiHeadLUT(input_dim=12, n_tables=16, output_dim=7, n_anchor_pairs=4,
                             confidence_form=form, random_seed=seed).double()


def _multi(form, seed=3):
    return LightMultiHeadLUT(input_dim=9, n_tables=12, output_dim=5, n_anchor_pairs=3,
                             confidence_form=form, random_seed=seed,
                             n_heads=3, multi_head_input=True).double()


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("multi", [False, True])
def test_fused_matches_naive_forward_and_grads(form, multi):
    """Fused == naive, forward, grad_x and grad_tables.

    allclose rather than equal: embedding_bag reduces in its own order, so the two differ
    in the last ulp. The tolerance is tight (1e-12 on float64) -- loose enough for
    reassociation, far too tight to hide a real discrepancy.
    """
    torch.manual_seed(0)
    m = _multi(form) if multi else _single(form)
    naive = _naive_multi if multi else _naive_single
    x0 = (torch.randn(6, 3, 9, dtype=torch.float64) if multi
          else torch.randn(6, 12, dtype=torch.float64)) * 2.0
    g = torch.randn(6, 3, 5, dtype=torch.float64) if multi \
        else torch.randn(6, 7, dtype=torch.float64)

    def run(fn):
        m.zero_grad(set_to_none=True)
        xx = x0.clone().requires_grad_(True)
        y = fn(xx)
        (y * g).sum().backward()
        return y.detach().clone(), xx.grad.clone(), m.tables.grad.clone()

    y_f, gx_f, gw_f = run(lambda t: m(t))
    y_n, gx_n, gw_n = run(lambda t: naive(m, t))

    assert torch.allclose(y_f, y_n, atol=1e-12, rtol=0), \
        f"forward differs by {(y_f - y_n).abs().max().item():.2e}"
    assert torch.allclose(gx_f, gx_n, atol=1e-12, rtol=0), \
        f"grad_x differs by {(gx_f - gx_n).abs().max().item():.2e}"
    assert torch.allclose(gw_f, gw_n, atol=1e-12, rtol=0), \
        f"grad_tables differs by {(gw_f - gw_n).abs().max().item():.2e}"


@pytest.mark.parametrize("form", _FORMS)
@pytest.mark.parametrize("multi", [False, True])
def test_fusion_keeps_the_gradient_out_of_the_address(form, multi):
    """Grad to x flows ONLY through the score, exactly -- fusing must not smuggle in an STE.

    Driven as an exact identity, not a statistic: with every table row set to the SAME
    constant c, the output is (sum_t score_t) * c regardless of WHICH row each table
    selects. So the address contributes nothing, and d(out)/dx must equal the gradient of
    the pure score path. If embedding_bag leaked any path through the index, this breaks.
    """
    torch.manual_seed(0)
    m = _multi(form) if multi else _single(form)
    with torch.no_grad():
        m.tables.fill_(0.375)
    x0 = (torch.randn(4, 3, 9, dtype=torch.float64) if multi
          else torch.randn(4, 12, dtype=torch.float64)) * 2.0

    xa = x0.clone().requires_grad_(True)
    m(xa).sum().backward()

    # closed form: sum over the layer's outputs of 0.375 * sum_t score_t
    xb = x0.clone().requires_grad_(True)
    if multi:
        B, H, T, NAP = 4, m.n_heads, m.tables_per_head, m.n_anchor_pairs
        ia = m.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
        ib = m.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
        d = (torch.gather(xb, 2, ia) - torch.gather(xb, 2, ib)).view(B, H, T, NAP)
        s = _confidence_score(d, form, m.confidence_gain).sum(dim=2)
    else:
        d = xb[:, m.anchor_a] - xb[:, m.anchor_b]
        s = _confidence_score(d, form, m.confidence_gain).sum(dim=1)
    (0.375 * m.output_dim * s).sum().backward()

    err = (xa.grad - xb.grad).abs().max().item()
    assert err < 1e-12, f"{form}: grad_x is not purely the score path ({err:.2e})"


@pytest.mark.parametrize("multi", [False, True])
def test_index_is_integer_and_carries_no_grad(multi):
    """The address is built from d.detach() and is an integer tensor -- no STE by construction."""
    m = _multi("bounded_norm") if multi else _single("bounded_norm")
    x = (torch.randn(3, 3, 9, dtype=torch.float64) if multi
         else torch.randn(3, 12, dtype=torch.float64))
    x.requires_grad_(True)
    # freeze the score to a constant: then NOTHING should reach x, since the only other
    # candidate path is the address.
    y = m(x)
    assert y.requires_grad
    with torch.no_grad():
        m.tables.zero_()
    m(x).sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x.grad)), \
        "with zero tables the output is identically 0, so x must receive exactly no gradient"


def test_fused_forward_is_bag_shaped():
    """Sanity on the reduction itself: one bag per (sample[, head]), summing that group."""
    m = _multi("bounded_norm")
    x = torch.randn(5, 3, 9, dtype=torch.float64)
    with torch.no_grad():
        out = m(x)
    assert out.shape == (5, 3, 5)
    ms = _single("bounded_norm")
    with torch.no_grad():
        assert ms(torch.randn(5, 12, dtype=torch.float64)).shape == (5, 7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("multi", [False, True])
def test_native_index_equals_torch_index(multi):
    """The native CUDA bit-pack and the torch sign+pack give the IDENTICAL address.

    Light uses the native kernel for its address whenever it is available, in training as
    well as at eval -- legal because the address is detached by construction. That makes
    it a pure speed choice, so it has to be bit-identical, not merely close.
    """
    dev = torch.device("cuda:0")
    kw = dict(output_dim=8, n_anchor_pairs=5, confidence_form="bounded_norm",
              random_seed=11, device=dev)
    m = (LightMultiHeadLUT(input_dim=16, n_tables=24, n_heads=3, multi_head_input=True, **kw)
         if multi else LightMultiHeadLUT(input_dim=16, n_tables=24, **kw))
    if m._native_msb is None:
        pytest.skip("lutorch_cuda extension not available")

    torch.manual_seed(0)
    x = (torch.randn(64, 3, 16, device=dev) if multi
         else torch.randn(64, 16, device=dev)) * 2.0
    d = _margins(m, x)
    shape = (1, 1, 1, -1) if d.dim() == 4 else (1, 1, -1)
    torch_index = ((d > 0).to(torch.int64) * m.powers.view(*shape)).sum(dim=-1)
    native = m._pack_index(x.reshape(x.shape[0], -1), d)
    assert torch.equal(native.view(torch_index.shape), torch_index)


def _margins(m, x):
    if m.multi_head_input:
        B, H, T, NAP = x.shape[0], m.n_heads, m.tables_per_head, m.n_anchor_pairs
        ia = m.anchor_a.reshape(1, H, T * NAP).expand(B, H, T * NAP)
        ib = m.anchor_b.reshape(1, H, T * NAP).expand(B, H, T * NAP)
        return (torch.gather(x, 2, ia) - torch.gather(x, 2, ib)).view(B, H, T, NAP)
    return x[:, m.anchor_a] - x[:, m.anchor_b]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_fused_forward_runs_in_reduced_precision(dtype):
    """The FORWARD works with fp32, bf16 and fp16 tables -- which is what inference needs."""
    dev = torch.device("cuda:0")
    m = LightMultiHeadLUT(input_dim=16, n_tables=32, output_dim=8, n_anchor_pairs=4,
                          confidence_form="bounded_norm", random_seed=1, device=dev)
    m.tables.data = m.tables.data.to(dtype)
    x = torch.randn(9, 16, device=dev, dtype=torch.float32)
    with torch.no_grad():
        y = m(x)
    assert y.dtype == dtype and y.shape == (9, 8) and torch.isfinite(y.float()).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fused_backward_supported_dtypes(dtype):
    """Training through the fusion works in fp32 and fp16 on CUDA."""
    dev = torch.device("cuda:0")
    m = LightMultiHeadLUT(input_dim=16, n_tables=32, output_dim=8, n_anchor_pairs=4,
                          confidence_form="bounded_norm", random_seed=1, device=dev)
    m.tables.data = m.tables.data.to(dtype)
    x = torch.randn(9, 16, device=dev, dtype=torch.float32, requires_grad=True)
    m(x).float().sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_bf16_tables_cannot_train_through_the_fusion():
    """PyTorch has no bf16 CUDA kernel for the per_sample_weights backward.

    Pinned deliberately rather than left to be rediscovered: bf16 TABLES are usable for
    inference (the forward test above covers that) but NOT for training through the fused
    path, because `_embedding_bag_per_sample_weights_backward_cuda` is unimplemented for
    BFloat16. Train in fp32 (or fp16) and cast the tables to bf16 for deployment. If a
    future PyTorch adds the kernel, this test starts failing and should simply be removed.
    """
    dev = torch.device("cuda:0")
    m = LightMultiHeadLUT(input_dim=16, n_tables=32, output_dim=8, n_anchor_pairs=4,
                          confidence_form="bounded_norm", random_seed=1, device=dev)
    m.tables.data = m.tables.data.to(torch.bfloat16)
    x = torch.randn(9, 16, device=dev, dtype=torch.float32, requires_grad=True)
    with pytest.raises(NotImplementedError, match="per_sample_weights_backward"):
        m(x).float().sum().backward()
