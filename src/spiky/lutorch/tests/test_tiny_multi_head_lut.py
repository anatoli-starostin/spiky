"""Tests for the published TinyMultiHeadLut primitive.

Covers:
  - Forward shape and dtype across forward_mode in {"hard", "hybrid_smooth"}
    and weight_dtype in {fp32, bf16}.
  - Forward determinism (same inputs -> same outputs).
  - eval (no_grad) forward bit-identical to the train forward.
  - Runtime flip of forward_mode (used by the soft -> hard fine-tune recipe).
  - Argument validation: bad forward_mode, n_anchor_pairs out of range,
    unsupported AnchorSamplingPolicy.
  - Backward populates x.grad and weights.grad; learnable_temps controls
    whether log_T_*.grad is populated.
  - weights.grad dtype matches weights.dtype regardless of autocast,
    so the optimiser sees fp32 grads when weights are fp32 master copies.
  - Anchor sampling invariants: a != b within each table; same seed ->
    same anchor pairs; different seeds -> different.
  - Numerical sanity for the soft surrogate backward: x.grad agrees with
    a finite-difference estimate of the "full soft mode" reference output
    that the surrogate is the analytical gradient of.
  - hybrid_smooth collapses to hard mode when all |d| are large
    (u = sigma(-Delta/T_sel) ~ 0).
"""
from typing import Optional

import pytest
import torch

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


_HAS_CUDA = torch.cuda.is_available()
_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")


def _device() -> torch.device:
    return torch.device("cuda:0")


def _make(
    *,
    forward_mode: str = "hard",
    input_dim: int = 64,
    n_heads: int = 4,
    n_outputs: int = 8,
    n_anchor_pairs: int = 4,
    tables_per_head: int = 4,
    weight_dtype: torch.dtype = torch.float32,
    use_bf16: bool = True,
    learnable_temps: bool = True,
    anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
    soft_score_temp: float = 0.5,
    select_temp: float = 0.5,
    random_seed: int = 0,
) -> TinyMultiHeadLut:
    return TinyMultiHeadLut(
        input_dim=input_dim,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs,
        tables_per_head=tables_per_head,
        forward_mode=forward_mode,
        weight_dtype=weight_dtype,
        use_bf16=use_bf16,
        anchor_sampling_policy=anchor_sampling_policy,
        soft_score_temp=soft_score_temp,
        select_temp=select_temp,
        learnable_temps=learnable_temps,
        random_seed=random_seed,
        device=_device(),
    )


# =============================================================================
# Forward shape / dtype / determinism
# =============================================================================

@_cuda
@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_forward_shape_and_dtype(forward_mode, weight_dtype):
    """Output is [B, n_heads, n_outputs] in weight_dtype."""
    m = _make(forward_mode=forward_mode, weight_dtype=weight_dtype)
    x = torch.randn(7, 64, device=_device(), dtype=torch.float32)
    y = m(x)
    assert y.shape == (7, 4, 8), f"got {y.shape}"
    assert y.dtype == weight_dtype, f"output {y.dtype} != weight_dtype {weight_dtype}"


@_cuda
@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_forward_determinism(forward_mode):
    """Same input + module -> bit-identical output."""
    m = _make(forward_mode=forward_mode, random_seed=42)
    x = torch.randn(4, 64, device=_device(), dtype=torch.float32)
    y1 = m(x)
    y2 = m(x)
    assert torch.equal(y1, y2)


@_cuda
@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_eval_matches_train_forward(forward_mode):
    """no_grad eval is bit-identical to the train (autograd-enabled) forward."""
    m = _make(forward_mode=forward_mode, random_seed=3)
    x = torch.randn(6, 64, device=_device(), dtype=torch.float32)
    with torch.no_grad():
        y_eval = m(x)
    x_train = x.detach().requires_grad_(True)
    y_train = m(x_train)
    assert torch.equal(y_eval, y_train.detach())


@_cuda
def test_hard_and_hybrid_smooth_produce_different_outputs():
    """The two forward modes are not the same function: at NAP that gives a
    non-trivial soft mass, hybrid_smooth blends the alt row with weight u>0."""
    m = _make(forward_mode="hard", random_seed=5)
    x = torch.randn(8, 64, device=_device(), dtype=torch.float32) * 0.01
    y_hard = m(x)
    m.forward_mode = "hybrid_smooth"
    y_hybrid = m(x)
    assert not torch.equal(y_hard, y_hybrid)


@_cuda
def test_forward_mode_can_flip_at_runtime():
    """forward_mode is an instance attribute and can be flipped without
    rebuilding the module (the soft -> hard finetune recipe relies on this)."""
    m = _make(forward_mode="hybrid_smooth", random_seed=11)
    x = torch.randn(3, 64, device=_device(), dtype=torch.float32)
    y0 = m(x)
    m.forward_mode = "hard"
    y1 = m(x)
    m.forward_mode = "hybrid_smooth"
    y2 = m(x)
    assert torch.equal(y0, y2)
    assert not torch.equal(y0, y1)


# =============================================================================
# Argument validation
# =============================================================================

@_cuda
def test_invalid_forward_mode_raises():
    with pytest.raises(ValueError, match="forward_mode"):
        TinyMultiHeadLut(
            input_dim=16, n_heads=1, n_outputs=4, n_anchor_pairs=4,
            tables_per_head=1, forward_mode="ste",
            device=_device(),
        )


@_cuda
@pytest.mark.parametrize("bad_nap", [0, 16, -1])
def test_n_anchor_pairs_out_of_range_raises(bad_nap):
    with pytest.raises(ValueError, match="n_anchor_pairs"):
        TinyMultiHeadLut(
            input_dim=64, n_heads=1, n_outputs=4, n_anchor_pairs=bad_nap,
            tables_per_head=1, device=_device(),
        )


@_cuda
def test_unsupported_anchor_sampling_policy_raises():
    """The publish branch supports only CANONICAL_* policies; BALANCED is
    intentionally rejected here (it stays in lut_helpers for main-branch
    consumers but isn't part of TinyMultiHeadLut's contract)."""
    with pytest.raises(ValueError, match="CANONICAL"):
        TinyMultiHeadLut(
            input_dim=64, n_heads=1, n_outputs=4, n_anchor_pairs=4,
            tables_per_head=1, device=_device(),
            anchor_sampling_policy=AnchorSamplingPolicy.BALANCED,
        )


# =============================================================================
# Anchor sampling invariants
# =============================================================================

@_cuda
@pytest.mark.parametrize(
    "policy",
    [
        AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        AnchorSamplingPolicy.CANONICAL_DISTINCT,
    ],
)
def test_anchor_pairs_distinct_within_table(policy):
    """For every table, no pair (a_i, b_i) has a_i == b_i."""
    m = _make(
        input_dim=128, n_heads=2, n_outputs=8, n_anchor_pairs=6,
        tables_per_head=8, anchor_sampling_policy=policy,
        random_seed=0,
    )
    a = m.soft_anchor_a_long
    b = m.soft_anchor_b_long
    assert a.shape == b.shape
    assert not (a == b).any(), "found at least one (a, b) pair with a == b"


@_cuda
def test_anchor_pairs_seed_determinism():
    """Same random_seed -> same anchor pairs; different seeds -> different."""
    m0 = _make(random_seed=0)
    m0_again = _make(random_seed=0)
    m1 = _make(random_seed=1)
    assert torch.equal(m0.soft_anchor_a_long, m0_again.soft_anchor_a_long)
    assert torch.equal(m0.soft_anchor_b_long, m0_again.soft_anchor_b_long)
    assert not torch.equal(m0.soft_anchor_a_long, m1.soft_anchor_a_long)


# =============================================================================
# Backward: grad flow and dtypes
# =============================================================================

@_cuda
@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_backward_grads_flow_to_x_and_weights(forward_mode, weight_dtype):
    """grad_x and grad_w must be allocated, right shape, and non-zero on
    inputs that have nonzero pairwise differences."""
    m = _make(forward_mode=forward_mode, weight_dtype=weight_dtype)
    x = torch.randn(8, 64, device=_device(), dtype=torch.float32, requires_grad=True)
    y = m(x)
    y.float().sum().backward()
    assert x.grad is not None
    assert m.weights.grad is not None
    assert x.grad.shape == x.shape
    assert m.weights.grad.shape == m.weights.shape
    assert (x.grad.abs() > 0).any()
    assert (m.weights.grad.abs() > 0).any()


@_cuda
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_weight_grad_dtype_matches_weights_dtype(weight_dtype):
    """The autograd same-dtype-as-param invariant: grad_w.dtype == weights.dtype.
    Important for fp32-master-weight training where backward internally
    accumulates in bf16 and must cast back at the autograd boundary."""
    m = _make(weight_dtype=weight_dtype)
    x = torch.randn(4, 64, device=_device(), dtype=torch.float32, requires_grad=True)
    y = m(x); y.float().sum().backward()
    assert m.weights.grad.dtype == weight_dtype


@_cuda
@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_learnable_temps_receive_gradients(forward_mode):
    """With learnable_temps=True, both log temperatures are nn.Parameters
    and accumulate gradient through the soft surrogate."""
    m = _make(forward_mode=forward_mode, learnable_temps=True)
    assert isinstance(m.log_soft_score_temp, torch.nn.Parameter)
    assert isinstance(m.log_select_temp, torch.nn.Parameter)
    x = torch.randn(4, 64, device=_device(), dtype=torch.float32)
    y = m(x); y.float().sum().backward()
    assert m.log_soft_score_temp.grad is not None
    assert m.log_select_temp.grad is not None
    # Each temperature gradient should be a finite scalar.
    assert m.log_soft_score_temp.grad.shape == ()
    assert m.log_select_temp.grad.shape == ()
    assert torch.isfinite(m.log_soft_score_temp.grad).all()
    assert torch.isfinite(m.log_select_temp.grad).all()


@_cuda
def test_frozen_temps_are_buffers_with_no_grad():
    """learnable_temps=False -> log_T_* are registered buffers, not
    parameters, and don't request grad."""
    m = _make(learnable_temps=False)
    assert not isinstance(m.log_soft_score_temp, torch.nn.Parameter)
    assert not isinstance(m.log_select_temp, torch.nn.Parameter)
    assert not m.log_soft_score_temp.requires_grad
    assert not m.log_select_temp.requires_grad


# =============================================================================
# hybrid_smooth -> hard collapse property
# =============================================================================

@_cuda
def test_hybrid_smooth_collapses_to_hard_for_confident_inputs():
    """When all anchor-pair differences are large (high confidence), the
    blend weight u = sigmoid(-Delta/T_sel) approaches 0 (Delta -> 2,
    sigmoid(-2/T_sel) is small) and hybrid_smooth output -> hard output.
    """
    m = _make(forward_mode="hard", random_seed=7, select_temp=0.5,
              soft_score_temp=0.5, learnable_temps=False)
    # x with very large magnitudes -> all |d_j| >> T_soft -> Delta -> 2
    # -> u = sigmoid(-Delta/T_sel) -> sigmoid(-4) ~= 0.018.
    # The resulting hybrid_smooth output is (1-u)*W[main] + u*W[alt] ~ W[main].
    x = torch.randn(6, 64, device=_device(), dtype=torch.float32) * 1000.0
    with torch.no_grad():
        m.forward_mode = "hard"
        y_hard = m(x)
        m.forward_mode = "hybrid_smooth"
        y_hybrid = m(x)
    # u = sigmoid(-Delta/T_sel) with Delta ~ 2 and T_sel = 0.5 gives u ~ 0.02,
    # so y_hybrid - y_hard = u * (W[alt] - W[main]) summed over tph tables.
    # The max relative diff lives in a few-times-u envelope around the
    # typical magnitude of y_hard.
    max_diff = (y_hybrid - y_hard).abs().max().item()
    typical = y_hard.abs().mean().item() + 1e-8
    assert max_diff < 0.2 * typical, (
        f"hybrid_smooth should ~= hard for large |d| inputs "
        f"(got max_diff {max_diff:.3e}, typical |y_hard| {typical:.3e})"
    )


# =============================================================================
# Soft-surrogate backward: numerical sanity vs the full-soft forward
# =============================================================================

def _full_soft_forward(x, weights, anchor_a_long, anchor_b_long,
                       n_heads, tph, T_soft, T_sel):
    """Reference 'full soft mode' forward used only by the tests.

    The published soft surrogate backward is the analytical gradient of this
    forward: y = sum_k softmax(ts/T_sel)_k * W[t, k, :], summed across tph
    tables per head. We don't ship it because it's K times more expensive
    than the published 'hard' / 'hybrid_smooth' forwards, but it's the right
    ground-truth target for the surrogate's grad_x.
    """
    B, _ = x.shape
    n_tables, NAP = anchor_a_long.shape
    K = weights.shape[1]
    n_outputs = weights.shape[2]
    d = x[:, anchor_a_long] - x[:, anchor_b_long]                # [B, n_tables, NAP]
    abs_d = d.abs()
    denom = T_soft + abs_d
    p = d.sign() * abs_d / denom                                 # [B, n_tables, NAP]
    # ts(k) = sum_i p_i * chi_i(k) with chi_i(k) = +/-1 (MSB-first bit pattern).
    bits = ((torch.arange(K, device=x.device).unsqueeze(0)
             >> torch.arange(NAP - 1, -1, -1, device=x.device).unsqueeze(1)) & 1)
    bit_matrix = (bits.float() - 0.5) * 2.0                      # [NAP, K]
    ts = torch.einsum("btp,pk->btk", p, bit_matrix)
    sel_soft = torch.softmax(ts / T_sel, dim=-1)                 # [B, n_tables, K]
    # y[b, h, o] = sum_{t in head h} sum_k sel_soft[b, t, k] W[t, k, o]
    blended = torch.einsum("btk,tko->bto", sel_soft, weights)    # [B, n_tables, n_outputs]
    return blended.view(B, n_heads, tph, n_outputs).sum(dim=2)


@_cuda
def test_grad_x_matches_full_soft_finite_difference():
    """The soft surrogate backward should approximate the gradient of the
    full-soft reference forward.

    With "hard" forward + soft backward, the same x gets different forward
    paths (hard picks one row) but the SAME backward formula. So we compare:

      our backward's x.grad   vs   d/dx of [full-soft forward](x)

    estimated via central differences. Tolerance is generous because the
    paths differ at higher order in T_sel / T_soft.
    """
    torch.manual_seed(0)
    dev = _device()
    # Small shape so finite differences are tractable.
    B, NAP, n_heads, tph, n_outputs, input_dim = 2, 4, 2, 2, 4, 12
    K = 1 << NAP
    n_tables = n_heads * tph
    m = _make(
        forward_mode="hard",
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=NAP, tables_per_head=tph,
        weight_dtype=torch.float32, use_bf16=False,
        learnable_temps=False,
        soft_score_temp=1.0, select_temp=1.0,
        random_seed=0,
    )
    # x scaled so neither tail of sign(d) is sharply saturating.
    x = (torch.randn(B, input_dim, device=dev, dtype=torch.float32) * 0.3
         ).requires_grad_(True)

    # Backward via our autograd Function. Use a fixed reduction (loss = y.sum())
    # so the finite-difference comparison has a 1-vector upstream gradient.
    y = m(x)
    y.sum().backward()
    grad_analytical = x.grad.detach().clone()

    # Finite-difference reference: gradient of [full_soft_forward(x).sum()] w.r.t. x.
    eps = 1e-2
    with torch.no_grad():
        weights = m.weights.detach()
        a = m.soft_anchor_a_long
        b = m.soft_anchor_b_long
        T_soft = m.log_soft_score_temp.exp().item()
        T_sel  = m.log_select_temp.exp().item()
        grad_fd = torch.zeros_like(x)
        for bi in range(B):
            for di in range(input_dim):
                xp = x.detach().clone()
                xm = x.detach().clone()
                xp[bi, di] += eps
                xm[bi, di] -= eps
                yp = _full_soft_forward(xp, weights, a, b, n_heads, tph, T_soft, T_sel).sum()
                ym = _full_soft_forward(xm, weights, a, b, n_heads, tph, T_soft, T_sel).sum()
                grad_fd[bi, di] = (yp - ym) / (2 * eps)

    # The shapes match, the direction agrees (positive cosine similarity), and
    # the magnitudes are within an order of magnitude. We don't expect exact
    # equality: forward differs (hard one row vs full-soft over K rows).
    cos = torch.nn.functional.cosine_similarity(
        grad_analytical.flatten().unsqueeze(0),
        grad_fd.flatten().unsqueeze(0),
    ).item()
    assert cos > 0.9, (
        f"grad_x direction disagrees with full-soft FD reference (cos={cos:.3f})"
    )
    ratio = grad_analytical.norm() / (grad_fd.norm() + 1e-12)
    assert 0.2 < ratio.item() < 5.0, (
        f"grad_x magnitude differs by >5x from FD reference (ratio={ratio.item():.3f})"
    )


# =============================================================================
# Smoke: a couple of forward+backward steps without NaN/inf
# =============================================================================

@_cuda
@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.bfloat16])
def test_train_step_no_nan(forward_mode, weight_dtype):
    """A few SGD steps under bf16 autocast leave all parameters finite."""
    m = _make(forward_mode=forward_mode, weight_dtype=weight_dtype, random_seed=0)
    opt = torch.optim.SGD(m.parameters(), lr=1e-3)
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        x = torch.randn(16, 64, device=_device(), dtype=torch.float32)
        y = m(x)
        loss = y.float().pow(2).mean()
        loss.backward()
        opt.step()
    for n, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite values in {n} after training"
