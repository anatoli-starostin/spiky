"""Tests for HyperplaneMultiHeadLUT (learned-hyperplane generalization of
FastMultiHeadLut).

Most tests are device-agnostic and run on CPU when no GPU is present (real
GPU-free CI coverage) — they build tensors on `_device()` (CUDA if available,
else CPU) and the module uses eager bodies on CPU. Only the genuinely
GPU-specific cases are CUDA-gated:
  - bf16 *storage* cases (`weight_dtype=torch.bfloat16`): CPU bf16 autocast is a
    no-op, so the fp32-store / bf16-compute contract can't be exercised and the
    output-dtype assertions would mismatch. Gated per-param via `_WEIGHT_DTYPES`.

The parity-vs-FastMultiHeadLut tests run under `_eager_mode()`: FastMultiHeadLut
unconditionally wraps forward/backward in torch.compile, whose inductor path
misbehaves on CPU / Python 3.14, so we disable dynamo for those tests (numerically
equivalent, and it still passes on GPU).

Covers:
  - gradcheck on the soft backward for w, b, x (double precision, small shapes).
  - Index-packing correctness (MSB-first).
  - Parity with FastMultiHeadLut under anchor-pairs-equivalent init (fp32).
  - Both forward modes + hard <-> hybrid_smooth runtime flip.
  - dtype coverage: fp32 (CPU+GPU) and bf16 storage (GPU-only), CPU fallback.
  - Frozen-hyperplane gradient gating.
  - A tiny end-to-end train step; argument validation.
"""
import contextlib
from typing import Optional

import pytest
import torch

from spiky.lutorch.hyperplane_multi_head_lut import (
    HyperplaneMultiHeadLUT,
    _hyperplane_project,
)
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


_HAS_CUDA = torch.cuda.is_available()
_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA required")

# weight_dtype coverage: fp32 runs everywhere; bf16 *storage* is GPU-only (CPU
# bf16 autocast is a no-op, so the fp32-store/bf16-compute path and its
# output-dtype contract can't be faithfully exercised on CPU).
_WEIGHT_DTYPES = [torch.float32, pytest.param(torch.bfloat16, marks=_cuda)]


def _device() -> torch.device:
    return torch.device("cuda:0") if _HAS_CUDA else torch.device("cpu")


@contextlib.contextmanager
def _eager_mode():
    """Force eager execution (disable TorchDynamo/Inductor) for the block.

    FastMultiHeadLut wraps its forward/backward in torch.compile unconditionally;
    the inductor path misbehaves on CPU / Python 3.14. Disabling dynamo makes
    those calls run eager so the parity tests give real CPU coverage. Harmless on
    GPU — compile is only an optimization and eager is numerically equivalent
    (if anything, more exact for the bit-parity comparison).
    """
    try:
        import torch._dynamo as dynamo
    except Exception:
        yield
        return
    prev = dynamo.config.disable
    dynamo.config.disable = True
    try:
        yield
    finally:
        dynamo.config.disable = prev


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
    hyperplane_init: str = "anchor_pairs",
    learnable_temps: bool = True,
    anchor_sampling_policy: Optional[AnchorSamplingPolicy] = None,
    soft_score_temp: float = 0.5,
    select_temp: float = 0.5,
    random_seed: int = 0,
    device: Optional[torch.device] = None,
) -> HyperplaneMultiHeadLUT:
    return HyperplaneMultiHeadLUT(
        input_dim=input_dim,
        n_heads=n_heads,
        n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs,
        tables_per_head=tables_per_head,
        forward_mode=forward_mode,
        weight_dtype=weight_dtype,
        use_bf16=use_bf16,
        hyperplane_init=hyperplane_init,
        anchor_sampling_policy=anchor_sampling_policy,
        soft_score_temp=soft_score_temp,
        select_temp=select_temp,
        learnable_temps=learnable_temps,
        random_seed=random_seed,
        device=device or _device(),
    )


def _make_fast(**kwargs) -> FastMultiHeadLut:
    """A FastMultiHeadLut with the same architectural args (for parity)."""
    return FastMultiHeadLut(
        input_dim=kwargs.get("input_dim", 64),
        n_heads=kwargs.get("n_heads", 4),
        n_outputs=kwargs.get("n_outputs", 8),
        n_anchor_pairs=kwargs.get("n_anchor_pairs", 4),
        tables_per_head=kwargs.get("tables_per_head", 4),
        forward_mode=kwargs.get("forward_mode", "hard"),
        weight_dtype=kwargs.get("weight_dtype", torch.float32),
        use_bf16=kwargs.get("use_bf16", False),
        anchor_sampling_policy=kwargs.get("anchor_sampling_policy"),
        soft_score_temp=kwargs.get("soft_score_temp", 0.5),
        select_temp=kwargs.get("select_temp", 0.5),
        learnable_temps=kwargs.get("learnable_temps", True),
        random_seed=kwargs.get("random_seed", 0),
        device=kwargs.get("device") or _device(),
    )


# =============================================================================
# Forward shape / dtype / determinism
# =============================================================================

@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
@pytest.mark.parametrize("weight_dtype", _WEIGHT_DTYPES)
def test_forward_shape_and_dtype(forward_mode, weight_dtype):
    m = _make(forward_mode=forward_mode, weight_dtype=weight_dtype)
    x = torch.randn(7, 64, device=_device(), dtype=torch.float32)
    y = m(x)
    assert y.shape == (7, 4, 8), f"got {y.shape}"
    assert y.dtype == weight_dtype


@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_forward_determinism(forward_mode):
    m = _make(forward_mode=forward_mode, random_seed=42)
    x = torch.randn(4, 64, device=_device(), dtype=torch.float32)
    assert torch.equal(m(x), m(x))


@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_eval_matches_train_forward(forward_mode):
    m = _make(forward_mode=forward_mode, random_seed=3)
    x = torch.randn(6, 64, device=_device(), dtype=torch.float32)
    with torch.no_grad():
        y_eval = m(x)
    x_train = x.detach().requires_grad_(True)
    y_train = m(x_train)
    assert torch.equal(y_eval, y_train.detach())


def test_forward_mode_can_flip_at_runtime():
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
# Index-packing correctness (MSB-first, identical to FastMultiHeadLut)
# =============================================================================

def test_index_packing_msb_first():
    """The packed row index equals sum_i bit_i * 2^(NAP-1-i), with
    bit_i = 1[<w_i, x> + b_i > 0], MSB-first."""
    NAP = 5
    m = _make(n_anchor_pairs=NAP, n_heads=2, tables_per_head=3,
              hyperplane_init="random", random_seed=1)
    x = torch.randn(9, 64, device=_device(), dtype=torch.float32)
    a = _hyperplane_project(x, m.hyperplane_weight, m.hyperplane_bias)  # [B,T,NAP]
    bits = (a > 0).to(torch.int64)
    powers = (1 << torch.arange(NAP - 1, -1, -1, device=_device(), dtype=torch.int64))
    expected = (bits * powers.view(1, 1, -1)).sum(dim=-1)
    # Reference recomputation from scratch (independent of soft_powers buffer).
    manual = torch.zeros_like(expected)
    for i in range(NAP):
        manual = manual + bits[..., i] * (1 << (NAP - 1 - i))
    assert torch.equal(expected, manual)
    # And matches the module's own MSB powers buffer.
    assert torch.equal(m.soft_powers, powers)


# =============================================================================
# Parity with FastMultiHeadLut under anchor-pairs-equivalent init
# =============================================================================

def _sync_lut_weights(m_hyp, m_fast):
    """Copy LUT table weights fast->hyp so only the front-end differs."""
    with torch.no_grad():
        m_hyp.weights.copy_(m_fast.weights.to(m_hyp.weights.dtype))


@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_parity_with_fast_forward(forward_mode):
    """Anchor-pairs-equivalent init reproduces FastMultiHeadLut's forward
    (both hard and hybrid_smooth) to fp32 tolerance."""
    args = dict(forward_mode=forward_mode, input_dim=48, n_heads=3,
                n_outputs=8, n_anchor_pairs=4, tables_per_head=4,
                weight_dtype=torch.float32, use_bf16=False,
                learnable_temps=False, random_seed=7)
    # PIN (do not relax): parity is only bit-exact in fp32 with autocast OFF.
    # Under bf16 the front-end matmul can flip a sign bit at a decision boundary
    # (a discrete table-row change, not a tolerance diff), so the two modules can
    # legitimately pick different rows and this test would fail spuriously.
    assert args["weight_dtype"] == torch.float32 and args["use_bf16"] is False
    # Eager mode: FastMultiHeadLut forces torch.compile, whose inductor path
    # misbehaves on CPU / Python 3.14 (see _eager_mode).
    with _eager_mode():
        m_hyp = _make(hyperplane_init="anchor_pairs", **args)
        m_fast = _make_fast(**args)
        # Same seed -> same anchor pairs and same LUT-weight init already; sync to
        # be robust to any init-order differences.
        assert torch.equal(m_hyp.soft_anchor_a_long, m_fast.soft_anchor_a_long)
        assert torch.equal(m_hyp.soft_anchor_b_long, m_fast.soft_anchor_b_long)
        _sync_lut_weights(m_hyp, m_fast)

        x = torch.randn(16, 48, device=_device(), dtype=torch.float32)
        with torch.no_grad():
            y_hyp = m_hyp(x)
            y_fast = m_fast(x)
    assert torch.allclose(y_hyp, y_fast, atol=1e-5, rtol=1e-4), (
        f"max abs diff {(y_hyp - y_fast).abs().max().item():.3e}"
    )


def test_parity_with_fast_gradients():
    """Under anchor-pairs init, x / LUT-weight / temperature grads match
    FastMultiHeadLut (the hyperplane front-end reduces exactly to the
    anchor-pair scatter when w_i = e_p1 - e_p2, b_i = 0)."""
    args = dict(forward_mode="hard", input_dim=48, n_heads=3,
                n_outputs=8, n_anchor_pairs=4, tables_per_head=4,
                weight_dtype=torch.float32, use_bf16=False,
                learnable_temps=True, random_seed=7)
    # PIN (do not relax): grad parity is only bit-exact in fp32 / autocast OFF
    # (see forward-parity test) — a bf16 sign flip picks a different row and
    # produces a legitimately different gradient.
    assert args["weight_dtype"] == torch.float32 and args["use_bf16"] is False
    with _eager_mode():
        m_hyp = _make(hyperplane_init="anchor_pairs", **args)
        m_fast = _make_fast(**args)
        _sync_lut_weights(m_hyp, m_fast)

        x0 = torch.randn(16, 48, device=_device(), dtype=torch.float32)
        xh = x0.clone().requires_grad_(True)
        xf = x0.clone().requires_grad_(True)

        yh = m_hyp(xh); yh.float().pow(2).sum().backward()
        yf = m_fast(xf); yf.float().pow(2).sum().backward()

    assert torch.allclose(xh.grad, xf.grad, atol=1e-4, rtol=1e-3), (
        f"grad_x max diff {(xh.grad - xf.grad).abs().max().item():.3e}"
    )
    assert torch.allclose(m_hyp.weights.grad, m_fast.weights.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(
        m_hyp.log_soft_score_temp.grad, m_fast.log_soft_score_temp.grad, atol=1e-4, rtol=1e-3)
    assert torch.allclose(
        m_hyp.log_select_temp.grad, m_fast.log_select_temp.grad, atol=1e-4, rtol=1e-3)


# =============================================================================
# gradcheck on the soft backward for x, w, b (double precision)
# =============================================================================

def test_gradcheck_soft_backward_x_w_b():
    """Numerical vs analytic gradient of the full-soft surrogate for x, w, b."""
    torch.manual_seed(0)
    dev = _device()
    input_dim, n_heads, tph, n_outputs, NAP = 6, 2, 2, 3, 3
    m = HyperplaneMultiHeadLUT(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=NAP, tables_per_head=tph,
        weight_dtype=torch.float64, use_bf16=False,
        hyperplane_init="random", learnable_temps=False,
        soft_score_temp=1.0, select_temp=1.0, random_seed=0, device=dev,
    )
    # Double-precision params; hold LUT weights and temps constant, check x/w/b.
    m.double()
    x = (torch.randn(4, input_dim, device=dev, dtype=torch.float64) * 0.5
         ).requires_grad_(True)
    m.hyperplane_weight.requires_grad_(True)
    m.hyperplane_bias.requires_grad_(True)
    m.weights.requires_grad_(False)

    from spiky.lutorch.hyperplane_multi_head_lut import _HyperplaneMHLutFullSoft

    def fn(x_in, w_in, b_in):
        return _HyperplaneMHLutFullSoft.apply(
            x_in, m.weights.detach(), w_in, b_in,
            m.log_soft_score_temp, m.log_select_temp,
            m.soft_bit_matrix, m.soft_powers,
            m.n_heads, m.tables_per_head, m.table_dim, False,
        )

    assert torch.autograd.gradcheck(
        fn, (x, m.hyperplane_weight, m.hyperplane_bias),
        eps=1e-6, atol=1e-5, rtol=1e-3, nondet_tol=1e-10,
    )


# =============================================================================
# Backward: grad flow and dtypes
# =============================================================================

@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
@pytest.mark.parametrize("weight_dtype", _WEIGHT_DTYPES)
def test_backward_grads_flow_to_all_params(forward_mode, weight_dtype):
    m = _make(forward_mode=forward_mode, weight_dtype=weight_dtype,
              hyperplane_init="random", learnable_temps=True)
    x = torch.randn(8, 64, device=_device(), dtype=torch.float32, requires_grad=True)
    y = m(x)
    y.float().sum().backward()
    for name, t in [("x", x), ("weights", m.weights),
                    ("hyperplane_weight", m.hyperplane_weight),
                    ("hyperplane_bias", m.hyperplane_bias)]:
        g = x.grad if name == "x" else getattr(m, name).grad
        assert g is not None, f"{name}.grad is None"
        assert (g.abs() > 0).any(), f"{name}.grad all zero"
    assert m.log_soft_score_temp.grad is not None
    assert m.log_select_temp.grad is not None


@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_frozen_hyperplanes_skip_affine_grads(forward_mode):
    """When w/b are frozen (requires_grad=False), their grads stay None (the
    gated GEMMs are skipped) while x and LUT weights still receive grads, and
    the forward output is unchanged from the all-trainable case."""
    m = _make(forward_mode=forward_mode, hyperplane_init="random",
              learnable_temps=True, weight_dtype=torch.float32, random_seed=0)
    x0 = torch.randn(8, 64, device=_device(), dtype=torch.float32)

    # Reference forward with everything trainable.
    y_ref = m(x0.clone().requires_grad_(True))

    m.hyperplane_weight.requires_grad_(False)
    m.hyperplane_bias.requires_grad_(False)
    x = x0.clone().requires_grad_(True)
    y = m(x)
    assert torch.equal(y, y_ref)  # freezing doesn't change forward
    y.float().sum().backward()

    assert m.hyperplane_weight.grad is None, "frozen w should get no grad"
    assert m.hyperplane_bias.grad is None, "frozen b should get no grad"
    assert x.grad is not None and (x.grad.abs() > 0).any()
    assert m.weights.grad is not None and (m.weights.grad.abs() > 0).any()


@pytest.mark.parametrize("weight_dtype", _WEIGHT_DTYPES)
def test_grad_dtypes_match_param_dtypes(weight_dtype):
    m = _make(weight_dtype=weight_dtype, hyperplane_init="random")
    x = torch.randn(4, 64, device=_device(), dtype=torch.float32, requires_grad=True)
    y = m(x); y.float().sum().backward()
    assert m.weights.grad.dtype == weight_dtype
    assert m.hyperplane_weight.grad.dtype == weight_dtype
    assert m.hyperplane_bias.grad.dtype == weight_dtype


# =============================================================================
# Tiny end-to-end train step
# =============================================================================

@pytest.mark.parametrize("forward_mode", ["hard", "hybrid_smooth"])
def test_overfit_small_batch(forward_mode):
    """A few Adam steps on a fixed batch reduce the loss and every trainable
    tensor (w, b, temps, LUT weights) receives nonzero grad."""
    dev = _device()
    m = _make(forward_mode=forward_mode, hyperplane_init="random",
              n_outputs=8, learnable_temps=True, weight_dtype=torch.float32,
              random_seed=0)
    opt = torch.optim.Adam(m.parameters(), lr=5e-3)
    x = torch.randn(32, 64, device=dev, dtype=torch.float32)
    target = torch.randn(32, m.n_heads, m.n_outputs, device=dev, dtype=torch.float32)

    losses = []
    grads_seen = {n: False for n, _ in m.named_parameters()}
    for _ in range(40):
        opt.zero_grad(set_to_none=True)
        y = m(x).float()
        loss = (y - target).pow(2).mean()
        loss.backward()
        for n, p in m.named_parameters():
            if p.grad is not None and (p.grad.abs() > 0).any():
                grads_seen[n] = True
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0] * 0.9, (
        f"loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
    )
    for n, seen in grads_seen.items():
        assert seen, f"parameter {n} never received a nonzero gradient"
    for n, p in m.named_parameters():
        assert torch.isfinite(p).all(), f"non-finite in {n}"


# =============================================================================
# CPU fallback path
# =============================================================================

def test_cpu_forward_backward():
    """Runs on CPU (eager bodies, use_bf16=False): forward shape + grads flow."""
    dev = torch.device("cpu")
    m = HyperplaneMultiHeadLUT(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=3,
        tables_per_head=2, forward_mode="hard", weight_dtype=torch.float32,
        use_bf16=False, hyperplane_init="random", learnable_temps=True,
        random_seed=0, device=dev,
    )
    x = torch.randn(5, 32, requires_grad=True)
    y = m(x)
    assert y.shape == (5, 2, 4)
    y.sum().backward()
    assert (m.hyperplane_weight.grad.abs() > 0).any()
    assert (m.hyperplane_bias.grad.abs() > 0).any()


# =============================================================================
# Argument validation
# =============================================================================

def test_invalid_forward_mode_raises():
    with pytest.raises(ValueError, match="forward_mode"):
        HyperplaneMultiHeadLUT(
            input_dim=16, n_heads=1, n_outputs=4, n_anchor_pairs=4,
            tables_per_head=1, forward_mode="ste", device=_device(),
        )


def test_invalid_hyperplane_init_raises():
    with pytest.raises(ValueError, match="hyperplane_init"):
        HyperplaneMultiHeadLUT(
            input_dim=16, n_heads=1, n_outputs=4, n_anchor_pairs=4,
            tables_per_head=1, hyperplane_init="svd", device=_device(),
        )


@pytest.mark.parametrize("bad_nap", [0, 16, -1])
def test_n_anchor_pairs_out_of_range_raises(bad_nap):
    with pytest.raises(ValueError, match="n_anchor_pairs"):
        HyperplaneMultiHeadLUT(
            input_dim=64, n_heads=1, n_outputs=4, n_anchor_pairs=bad_nap,
            tables_per_head=1, device=_device(),
        )
