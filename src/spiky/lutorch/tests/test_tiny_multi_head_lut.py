"""Tests for TinyMultiHeadLut and TinyMultiHeadLutOptimizer.

Covers:
- Basic shape correctness across (fp32, fp16, bf16) input × (fp32, fp16, bf16) weights.
- Native CUDA path acceptance for all float dtypes (fp16, bf16, fp32, fp64).
- Forward determinism (same x, same weights, same output).
- Backward gradient flow to inputs (anchor STE) and weights.
- Optimizer step changes weights and Adam state.
- Optimizer state stored in user-chosen dtype (default bf16).
"""
import pytest
import torch

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.tiny_multi_head_lut_optimizer import TinyMultiHeadLutOptimizer
from spiky.lutorch.tiny_anchor_pairs_lookup import _can_use_native_tiny_apl
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.soft_multi_head_lut import SoftMultiHeadLUT


DEVICES = [torch.device("cuda:0")] if torch.cuda.is_available() else []


def _has_cuda():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("weight_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_shape_and_dtype(input_dtype, weight_dtype):
    """Output shape matches [B, n_heads, n_outputs] and dtype matches weights."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=64, n_heads=4, n_outputs=16, n_anchor_pairs=6,
        tables_per_head=8, weight_dtype=weight_dtype,
        random_seed=0, device=dev,
    )
    x = torch.randn(8, 64, device=dev, dtype=input_dtype)
    y = m(x)
    assert y.shape == (8, 4, 16), f"got {y.shape}"
    assert y.dtype == weight_dtype, f"output dtype {y.dtype} != weight_dtype {weight_dtype}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_native_cuda_path_accepts_all_float_dtypes():
    """After the AT_DISPATCH_FLOATING_TYPES_AND2 patch, fp16 and bf16 inputs
    take the native CUDA path (instead of the slow PyTorch fallback)."""
    dev = torch.device("cuda:0")
    for dt in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        x = torch.randn(2, 32, device=dev, dtype=dt)
        assert _can_use_native_tiny_apl(x), f"native path rejected dtype={dt}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_forward_determinism(input_dtype):
    """Same input, same weights → same output (exact equality)."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.bfloat16,
        random_seed=42, device=dev,
    )
    x = torch.randn(4, 32, device=dev, dtype=input_dtype)
    y1 = m(x)
    y2 = m(x)
    assert torch.equal(y1, y2)


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_backward_grad_flows_to_input_and_weights(input_dtype):
    """Backward through TinyMultiHeadLut produces gradients on both x and weights."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.bfloat16,
        random_seed=1, device=dev,
    )
    x = torch.randn(4, 32, device=dev, dtype=input_dtype, requires_grad=True)
    y = m(x)
    loss = y.float().sum()
    loss.backward()
    assert x.grad is not None
    assert m.weights.grad is not None
    assert x.grad.shape == x.shape
    assert m.weights.grad.shape == m.weights.shape
    # x.grad should be non-trivially nonzero for at least some entries.
    assert (x.grad.abs() > 0).any()
    assert (m.weights.grad.abs() > 0).any()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_native_path_matches_pytorch_path_within_tolerance():
    """For fp32 input, native CUDA forward matches the PyTorch reference."""
    dev = torch.device("cuda:0")
    # Build two identical modules; for the second, force the PyTorch path.
    import spiky.lutorch.tiny_anchor_pairs_lookup as tapl_mod
    saved = tapl_mod._USE_TINY_APL_CUSTOM_CUDA
    try:
        m = TinyMultiHeadLut(
            input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
            tables_per_head=4, weight_dtype=torch.float32,
            random_seed=7, device=dev,
        )
        x = torch.randn(4, 32, device=dev, dtype=torch.float32)
        # Native path
        tapl_mod._USE_TINY_APL_CUSTOM_CUDA = True
        y_native = m(x)
        # PyTorch path
        tapl_mod._USE_TINY_APL_CUSTOM_CUDA = False
        y_py = m(x)
    finally:
        tapl_mod._USE_TINY_APL_CUSTOM_CUDA = saved
    # Outputs should match (deterministic gather, deterministic anchor sampling).
    assert torch.equal(y_native, y_py), "native and PyTorch forwards diverged"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_optimizer_step_changes_weights(state_dtype):
    """Optimizer step actually moves weights and updates Adam state."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.bfloat16,
        random_seed=2, device=dev,
    )
    opt = TinyMultiHeadLutOptimizer([m], lr=1e-2, state_dtype=state_dtype)
    # Sanity: state allocated in requested dtype.
    assert opt._states[0]['m'].dtype == state_dtype
    assert opt._states[0]['v'].dtype == state_dtype

    weights_before = m.weights.detach().clone()
    x = torch.randn(8, 32, device=dev, dtype=torch.bfloat16)
    y = m(x)
    loss = y.float().sum()
    loss.backward()
    opt.step()
    weights_after = m.weights.detach()
    # Weights should have moved.
    assert not torch.equal(weights_before, weights_after)
    # Adam state should be nonzero.
    assert (opt._states[0]['m'].float().abs() > 0).any()
    assert (opt._states[0]['v'].float().abs() > 0).any()
    opt.zero_grad()
    assert m.weights.grad is None  # set_to_none default


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
def test_native_fused_bwd_matches_pytorch_fallback(dtype):
    """The native fused tiny_mhlut_backward_na1 must produce the same
    grad_weights and x.grad as the PyTorch reference path."""
    import spiky.lutorch.tiny_multi_head_lut as tmhlut_mod
    dev = torch.device("cuda:0")
    saved = tmhlut_mod._USE_TINY_MHLUT_NATIVE_BWD
    try:
        m1 = TinyMultiHeadLut(
            input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
            tables_per_head=4, weight_dtype=dtype, random_seed=11, device=dev,
        )
        m2 = TinyMultiHeadLut(
            input_dim=32, n_heads=2, n_outputs=8, n_anchor_pairs=5,
            tables_per_head=4, weight_dtype=dtype, random_seed=11, device=dev,
        )
        m2.load_state_dict(m1.state_dict())
        x = torch.randn(6, 32, device=dev, dtype=dtype, requires_grad=True)
        x_ref = x.detach().clone().requires_grad_(True)

        # Native path
        tmhlut_mod._USE_TINY_MHLUT_NATIVE_BWD = True
        y1 = m1(x); y1.float().sum().backward()
        # Fallback path
        tmhlut_mod._USE_TINY_MHLUT_NATIVE_BWD = False
        y2 = m2(x_ref); y2.float().sum().backward()
    finally:
        tmhlut_mod._USE_TINY_MHLUT_NATIVE_BWD = saved

    # Forwards must be identical (same weights/indices/algorithm).
    assert torch.equal(y1, y2)
    # Gradients must match within the dtype's precision.
    rtol, atol = (1e-5, 1e-6) if dtype == torch.float32 else (5e-2, 5e-2)
    assert torch.allclose(m1.weights.grad.float(), m2.weights.grad.float(), rtol=rtol, atol=atol), \
        f"weights.grad mismatch (max diff {(m1.weights.grad - m2.weights.grad).abs().max().item()})"
    assert torch.allclose(x.grad.float(), x_ref.grad.float(), rtol=rtol, atol=atol), \
        f"x.grad mismatch (max diff {(x.grad - x_ref.grad).abs().max().item()})"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_stochastic_rounding_is_unbiased(dtype):
    """Stochastic-rounded fp32 → {bf16, fp16} averages back to original fp32 value.

    Probes a sub-ULP value (one that round-to-nearest would map deterministically
    to one neighbour). Stochastic rounding should split between the two
    neighbours such that E[round(x)] ≈ x.
    """
    from spiky.lutorch.tiny_multi_head_lut_optimizer import (
        _stochastic_round_fp32_to_bf16, _stochastic_round_fp32_to_fp16,
    )
    sr_fn = _stochastic_round_fp32_to_bf16 if dtype == torch.bfloat16 else _stochastic_round_fp32_to_fp16
    dev = torch.device("cuda:0")
    target = 0.1
    x = torch.full((1024 * 1024,), target, dtype=torch.float32, device=dev)
    rounded = sr_fn(x)
    assert rounded.dtype == dtype
    mean = rounded.float().mean().item()
    # ULP at 0.1: bf16 ≈ 7.8e-4, fp16 ≈ 9.8e-5. 5σ over 1M samples covers both.
    assert abs(mean - target) < 5e-5, f"{dtype} SR biased: mean={mean} vs {target}"
    unique = torch.unique(rounded)
    assert unique.numel() >= 2, f"{dtype} SR produced only one value: {unique}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_optimizer_stochastic_rounding_breaks_subulp_floor(dtype):
    """With stochastic_rounding=True, repeated sub-ULP updates must accumulate.
    With stochastic_rounding=False (default RNE), they get rounded away."""
    from spiky.lutorch.tiny_multi_head_lut_optimizer import TinyMultiHeadLutOptimizer
    dev = torch.device("cuda:0")
    m_sr = TinyMultiHeadLut(input_dim=16, n_heads=1, n_outputs=4, n_anchor_pairs=4,
                            tables_per_head=2, weight_dtype=dtype,
                            random_seed=0, device=dev)
    m_rne = TinyMultiHeadLut(input_dim=16, n_heads=1, n_outputs=4, n_anchor_pairs=4,
                             tables_per_head=2, weight_dtype=dtype,
                             random_seed=0, device=dev)
    m_rne.load_state_dict(m_sr.state_dict())
    initial = m_sr.weights.detach().clone()
    # Tiny lr → per-step update is well below ULP at the weight magnitude.
    opt_sr  = TinyMultiHeadLutOptimizer([m_sr],  lr=1e-7, stochastic_rounding=True)
    opt_rne = TinyMultiHeadLutOptimizer([m_rne], lr=1e-7, stochastic_rounding=False)
    # Apply N tiny gradient steps, all in the same direction.
    grad = torch.full_like(m_sr.weights, 1.0).float()
    for _ in range(200):
        m_sr.weights.grad  = grad.to(dtype).clone()
        m_rne.weights.grad = grad.to(dtype).clone()
        opt_sr.step()
        opt_rne.step()
    delta_sr  = (m_sr.weights  - initial).float().abs().mean().item()
    delta_rne = (m_rne.weights - initial).float().abs().mean().item()
    # Stochastic rounding accumulates ~ N*lr*|grad| ≈ 2e-5 regardless of dtype.
    # RNE: rounds most steps back; how much depends on ULP. bf16 ULP at the
    # weight magnitude is ~8× larger than fp16's, so RNE preserves much more
    # in fp16 than in bf16. The sturdy claim is "SR matches expectation and
    # never undershoots RNE" — that's what we assert here.
    expected = 200 * 1e-7  # ≈ 2e-5
    assert abs(delta_sr - expected) / expected < 0.5, \
        f"{dtype} SR accumulation off: Δ_sr={delta_sr} (expected ~{expected})"
    assert delta_sr >= delta_rne, \
        f"{dtype} SR fell behind RNE: Δ_sr={delta_sr}, Δ_rne={delta_rne}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_constraint_n_anchor_pairs_15():
    """n_anchor_pairs > 15 is rejected (int16 limit)."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="1 <= n_anchor_pairs <= 15"):
        TinyMultiHeadLut(input_dim=64, n_heads=2, n_outputs=8,
                        n_anchor_pairs=16, tables_per_head=4, device=dev)


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_constraint_input_dim_32767():
    """input_dim > 32767 is rejected (int16 anchor index limit)."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="input_dim <= 32767"):
        TinyMultiHeadLut(input_dim=40000, n_heads=2, n_outputs=8,
                        n_anchor_pairs=6, tables_per_head=4, device=dev)


# ============================================================================
# sparse_scatter mode
# ============================================================================

def _build_sparse(dev, **overrides):
    """Build a TinyMultiHeadLut in sparse_scatter mode with sensible defaults."""
    kwargs = dict(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=6,
        tables_per_head=8, weight_dtype=torch.float32,
        random_seed=0, device=dev,
        sparse_scatter_n_outputs=16, sparse_scatter_seed=7,
    )
    kwargs.update(overrides)
    return TinyMultiHeadLut(**kwargs)


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_output_shape():
    """When sparse_scatter_n_outputs is set, forward returns
    [B, n_heads, sparse_scatter_n_outputs] (not [B, n_heads, n_outputs])."""
    dev = torch.device("cuda:0")
    m = _build_sparse(dev, n_heads=3, n_outputs=8, tables_per_head=64,
                      sparse_scatter_n_outputs=128)
    x = torch.randn(5, 32, device=dev)
    y = m(x)
    assert y.shape == (5, 3, 128), f"got {y.shape}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_indices_shape_and_range():
    """scatter_indices buffer has shape [n_heads, tables_per_head, n_outputs]
    and every row is a sample-without-replacement of [0, sparse_scatter_n_outputs)."""
    dev = torch.device("cuda:0")
    m = _build_sparse(dev, n_heads=4, n_outputs=3, tables_per_head=10,
                      sparse_scatter_n_outputs=32)
    assert m.scatter_indices.shape == (4, 10, 3)
    assert m.scatter_indices.min().item() >= 0
    assert m.scatter_indices.max().item() < 32
    # No duplicates within each (head, table) row.
    for h in range(4):
        for t in range(10):
            row = m.scatter_indices[h, t]
            assert row.unique().numel() == row.numel(), \
                f"duplicate scatter index at head={h} table={t}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_weights_shape():
    """Per-table weights only depend on n_sparse (= n_outputs in sparse mode),
    not on the wider sparse_scatter_n_outputs — that's the whole point."""
    dev = torch.device("cuda:0")
    n_heads, tph, n_sparse = 2, 16, 8
    m = _build_sparse(dev, n_heads=n_heads, n_outputs=n_sparse,
                      n_anchor_pairs=5, tables_per_head=tph,
                      sparse_scatter_n_outputs=256)
    expected = (n_heads * tph, 1 << 5, n_sparse)
    assert m.weights.shape == expected, f"got {m.weights.shape}, expected {expected}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_validates_n_outputs_le_scatter():
    """sparse_scatter_n_outputs must be >= n_outputs (each table contributes
    n_outputs distinct slots out of sparse_scatter_n_outputs)."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="sparse_scatter_n_outputs"):
        TinyMultiHeadLut(
            input_dim=16, n_heads=1, n_outputs=8, n_anchor_pairs=4,
            tables_per_head=4, weight_dtype=torch.float32,
            random_seed=0, device=dev,
            sparse_scatter_n_outputs=4,  # < n_outputs = 8
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_seed_determinism():
    """Same sparse_scatter_seed → same scatter_indices."""
    dev = torch.device("cuda:0")
    a = _build_sparse(dev, sparse_scatter_seed=123)
    b = _build_sparse(dev, sparse_scatter_seed=123)
    c = _build_sparse(dev, sparse_scatter_seed=124)
    assert torch.equal(a.scatter_indices, b.scatter_indices)
    assert not torch.equal(a.scatter_indices, c.scatter_indices)


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_matches_manual_scatter():
    """Forward output equals: gather per-table → scatter_add through the
    stored scatter_indices into a zero-init [B, H, sparse_n_out] tensor."""
    dev = torch.device("cuda:0")
    m = _build_sparse(dev)
    x = torch.randn(3, 32, device=dev)

    y = m(x)

    # Manual reference: replay forward path explicitly.
    with torch.no_grad():
        (lookup_indices, _, _, _, _) = m.lookup(x)
        lookup_indices = lookup_indices.to(torch.int64)
        B = x.shape[0]
        n_lt = m.n_lookup_tables
        table_ix = torch.arange(n_lt, device=dev).view(1, -1).expand(B, -1)
        per_table = m.weights[table_ix, lookup_indices]                 # [B, n_lt, n_outputs]
        per_table = per_table.view(B, m.n_heads, m.tables_per_head, m.n_outputs)
        ref = torch.zeros(B, m.n_heads, m.sparse_scatter_n_outputs,
                          dtype=m.weights.dtype, device=dev)
        idx = m.scatter_indices.unsqueeze(0).expand(B, -1, -1, -1)
        ref.scatter_add_(2, idx.reshape(B, m.n_heads, -1),
                         per_table.reshape(B, m.n_heads, -1))

    assert torch.allclose(y, ref, atol=1e-5, rtol=1e-5), \
        f"max diff: {(y - ref).abs().max().item()}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_eval_path_matches_train():
    """no_grad eval path (no carriers) gives the same numbers as the autograd
    training path for the same inputs and weights."""
    dev = torch.device("cuda:0")
    m = _build_sparse(dev)
    x = torch.randn(4, 32, device=dev)
    m.train()
    y_train = m(x)
    m.eval()
    with torch.no_grad():
        y_eval = m(x)
    assert torch.allclose(y_train, y_eval, atol=1e-5, rtol=1e-5), \
        f"train/eval mismatch: max diff {(y_train - y_eval).abs().max().item()}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_backward_grads_flow():
    """Backward populates grad on weights and on x (via STE carriers)."""
    dev = torch.device("cuda:0")
    m = _build_sparse(dev, n_heads=2, n_outputs=4, tables_per_head=8,
                      sparse_scatter_n_outputs=16)
    x = torch.randn(3, 32, device=dev, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert m.weights.grad is not None
    assert m.weights.grad.abs().sum().item() > 0
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_grad_consistent_with_manual():
    """grad_weights from autograd matches a manual recomputation: gather +
    scatter, then backprop a known upstream gradient through the same ops."""
    dev = torch.device("cuda:0")
    m = _build_sparse(dev)
    x = torch.randn(2, 32, device=dev)
    y = m(x)
    g = torch.randn_like(y)
    y.backward(g)
    grad_weights_auto = m.weights.grad.detach().clone()

    # Manual: replay forward as differentiable torch ops with a cloned weight
    # that requires grad, then backprop g.
    m.weights.grad = None
    w_ref = m.weights.detach().clone().requires_grad_(True)
    with torch.no_grad():
        (lookup_indices, _, _, _, _) = m.lookup(x)
        lookup_indices = lookup_indices.to(torch.int64)
    B = x.shape[0]
    n_lt = m.n_lookup_tables
    table_ix = torch.arange(n_lt, device=dev).view(1, -1).expand(B, -1)
    per_table = w_ref[table_ix, lookup_indices].view(
        B, m.n_heads, m.tables_per_head, m.n_outputs,
    )
    out = torch.zeros(B, m.n_heads, m.sparse_scatter_n_outputs,
                      dtype=w_ref.dtype, device=dev)
    idx = m.scatter_indices.unsqueeze(0).expand(B, -1, -1, -1)
    out.scatter_add_(2, idx.reshape(B, m.n_heads, -1),
                     per_table.reshape(B, m.n_heads, -1))
    out.backward(g)
    grad_weights_manual = w_ref.grad

    assert torch.allclose(grad_weights_auto, grad_weights_manual,
                          atol=1e-5, rtol=1e-5), \
        f"grad_weights mismatch: max diff " \
        f"{(grad_weights_auto - grad_weights_manual).abs().max().item()}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("T,S,N", [
    (10, 1, 4),       # S=1, divisible
    (10, 1, 7),       # S=1, non-divisible
    (10, 3, 4),       # S>1, non-divisible
    (384, 8, 64),     # divisible, exact balance
    (2048, 8, 256),   # divisible, exact balance
    (100, 5, 32),     # non-divisible, T*S=500, /32=15.625
])
def test_balanced_coverage_indices_helper(T, S, N):
    """`_balanced_coverage_indices` returns a [T, S] long tensor where every
    row has S distinct values in [0, N) and per-slot counts are within {floor,
    ceil} of T*S/N."""
    from spiky.lutorch.tiny_multi_head_lut import _balanced_coverage_indices
    gen = torch.Generator().manual_seed(123)
    out = _balanced_coverage_indices(n_tables=T, n_per_row=S, n_slots=N, generator=gen)
    assert out.shape == (T, S), f"shape {out.shape} vs expected ({T}, {S})"
    assert out.min().item() >= 0 and out.max().item() < N
    # All rows distinct
    for t in range(T):
        assert out[t].unique().numel() == S, f"row {t} has duplicates"
    # Balanced counts
    counts = torch.bincount(out.reshape(-1), minlength=N)
    base = (T * S) // N
    extra = (T * S) - base * N
    expected_max = base + (1 if extra > 0 else 0)
    assert counts.min().item() == base, \
        f"min count {counts.min().item()} != floor(T*S/N)={base}"
    assert counts.max().item() == expected_max, \
        f"max count {counts.max().item()} != ceil(T*S/N)={expected_max}"
    assert counts.sum().item() == T * S
    # The number of slots that get the extra +1 equals `extra`.
    if extra > 0:
        assert (counts == base + 1).sum().item() == extra
        assert (counts == base).sum().item() == N - extra


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_balanced_mode_distribution():
    """sparse_scatter_balanced=True (default) yields exactly-balanced
    per-slot counts when T*S is divisible by N."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=64, n_heads=2, n_outputs=8, n_anchor_pairs=6,
        tables_per_head=128, weight_dtype=torch.float32,
        random_seed=0, device=dev,
        sparse_scatter_n_outputs=64, sparse_scatter_seed=7,
        # default sparse_scatter_balanced=True
    )
    # T=128, S=8, N=64 → T*S/N = 16 exactly per head
    for h in range(2):
        counts = torch.bincount(m.scatter_indices[h].reshape(-1), minlength=64)
        assert counts.min().item() == counts.max().item() == 16, \
            f"head {h}: min={counts.min()} max={counts.max()}, expected exactly 16"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_balanced_off_uses_iid():
    """sparse_scatter_balanced=False falls back to i.i.d. randperm sampling,
    which has multinomial variance — counts will NOT be exactly equal."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=64, n_heads=1, n_outputs=8, n_anchor_pairs=6,
        tables_per_head=128, weight_dtype=torch.float32,
        random_seed=0, device=dev,
        sparse_scatter_n_outputs=64, sparse_scatter_seed=7,
        sparse_scatter_balanced=False,
    )
    counts = torch.bincount(m.scatter_indices[0].reshape(-1), minlength=64)
    # i.i.d.: with T*S=1024 over N=64, mean=16, std≈sqrt(16)=4. Almost
    # surely some spread (min < max). Probability of perfect balance is
    # vanishingly small.
    assert counts.min().item() < counts.max().item(), \
        "i.i.d. sampling should produce variance; got perfectly balanced output"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_balanced_seed_determinism():
    """Same seed -> same balanced indices."""
    dev = torch.device("cuda:0")
    a = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=6,
        tables_per_head=16, weight_dtype=torch.float32,
        random_seed=0, device=dev,
        sparse_scatter_n_outputs=32, sparse_scatter_seed=42,
    )
    b = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=6,
        tables_per_head=16, weight_dtype=torch.float32,
        random_seed=0, device=dev,
        sparse_scatter_n_outputs=32, sparse_scatter_seed=42,
    )
    assert torch.equal(a.scatter_indices, b.scatter_indices)


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_sparse_scatter_collision_accumulates():
    """If two table rows map to the same output slot, the slot accumulates
    the sum of both contributions (scatter_add semantics)."""
    dev = torch.device("cuda:0")
    m = _build_sparse(dev, n_heads=1, n_outputs=2, tables_per_head=4,
                      sparse_scatter_n_outputs=6)
    # Force a collision: every row writes to slots [0, 1].
    with torch.no_grad():
        m.scatter_indices.fill_(0)
        m.scatter_indices[..., 1] = 1
    x = torch.randn(2, 32, device=dev)
    y = m(x)

    with torch.no_grad():
        (lookup_indices, _, _, _, _) = m.lookup(x)
        lookup_indices = lookup_indices.to(torch.int64)
        B = x.shape[0]
        n_lt = m.n_lookup_tables
        table_ix = torch.arange(n_lt, device=dev).view(1, -1).expand(B, -1)
        per_table = m.weights[table_ix, lookup_indices].view(
            B, m.n_heads, m.tables_per_head, m.n_outputs,
        )
        # All values fall into slots 0 and 1; slots 2..5 stay zero.
        expected_0 = per_table[..., 0].sum(dim=2)
        expected_1 = per_table[..., 1].sum(dim=2)
    assert torch.allclose(y[..., 0], expected_0, atol=1e-5, rtol=1e-5)
    assert torch.allclose(y[..., 1], expected_1, atol=1e-5, rtol=1e-5)
    assert torch.allclose(y[..., 2:], torch.zeros_like(y[..., 2:]),
                          atol=1e-6, rtol=1e-6)


# ============================================================================
# max_anchor_distance: per-table local window constraint
# ============================================================================

def _per_table_widths(m: TinyMultiHeadLut) -> torch.Tensor:
    a = m.lookup.anchor_pairs_a.long()
    b = m.lookup.anchor_pairs_b.long()
    combined = torch.cat([a, b], dim=1)              # [n_tables, 2*nap]
    return combined.max(dim=1).values - combined.min(dim=1).values


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("K", [4, 8, 15])
def test_max_anchor_distance_constraint_holds(K):
    """Every table's combined anchor set has max-min span <= K."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=128, n_heads=4, n_outputs=8, n_anchor_pairs=4,
        tables_per_head=32, weight_dtype=torch.float32, random_seed=42, device=dev,
        max_anchor_distance=K,
    )
    widths = _per_table_widths(m)
    assert widths.max().item() <= K, \
        f"some table exceeds max_anchor_distance: max width={widths.max().item()}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_max_anchor_distance_per_pair_also_holds():
    """Per-pair |a - b| <= K is automatically implied by per-table span <= K."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=64, n_heads=2, n_outputs=4, n_anchor_pairs=6,
        tables_per_head=16, weight_dtype=torch.float32, random_seed=0, device=dev,
        max_anchor_distance=8,
    )
    a = m.lookup.anchor_pairs_a.long()
    b = m.lookup.anchor_pairs_b.long()
    diffs = (b - a).abs()
    assert diffs.max().item() <= 8


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_max_anchor_distance_starts_linspace_spreads_across_input():
    """Default 'linspace' window starts cover the whole valid start range
    [0, input_dim - K - 1] approximately uniformly."""
    dev = torch.device("cuda:0")
    input_dim, K = 128, 7
    m = TinyMultiHeadLut(
        input_dim=input_dim, n_heads=1, n_outputs=4, n_anchor_pairs=4,
        tables_per_head=64, weight_dtype=torch.float32, random_seed=0, device=dev,
        max_anchor_distance=K,
    )
    a = m.lookup.anchor_pairs_a.long()
    b = m.lookup.anchor_pairs_b.long()
    starts = torch.minimum(a.min(dim=1).values, b.min(dim=1).values)
    assert starts.min().item() == 0
    assert starts.max().item() == input_dim - K - 1


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_max_anchor_distance_window_too_small_raises():
    """If (K+1)·K/2 < n_anchor_pairs, the within-window pair pool is too
    small and init must raise."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="local_window: window_size"):
        # K=3 → window=4 → only 6 pairs available. nap=8 won't fit.
        TinyMultiHeadLut(
            input_dim=64, n_heads=1, n_outputs=4, n_anchor_pairs=8,
            tables_per_head=8, weight_dtype=torch.float32, random_seed=0, device=dev,
            max_anchor_distance=3,
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_max_anchor_distance_window_too_big_raises():
    """K+1 > input_dim is rejected."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="max_distance\\+1=33 > input_dim=32"):
        TinyMultiHeadLut(
            input_dim=32, n_heads=1, n_outputs=4, n_anchor_pairs=4,
            tables_per_head=8, weight_dtype=torch.float32, random_seed=0, device=dev,
            max_anchor_distance=32,
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_max_anchor_distance_mutually_exclusive_with_partition_sets():
    """max_anchor_distance + partition_sets is rejected."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="mutually exclusive"):
        TinyMultiHeadLut(
            input_dim=32, n_heads=1, n_outputs=4, n_anchor_pairs=4,
            tables_per_head=8, weight_dtype=torch.float32, random_seed=0, device=dev,
            max_anchor_distance=8,
            partition_sets=[set(range(0, 16)), set(range(16, 32))],
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_max_anchor_distance_random_starts_seed_determinism():
    """Same random_seed + starts_mode='random' → identical anchors."""
    dev = torch.device("cuda:0")
    kwargs = dict(
        input_dim=128, n_heads=2, n_outputs=4, n_anchor_pairs=4,
        tables_per_head=16, weight_dtype=torch.float32, random_seed=7, device=dev,
        max_anchor_distance=10, local_window_starts="random",
    )
    a = TinyMultiHeadLut(**kwargs)
    b = TinyMultiHeadLut(**kwargs)
    assert torch.equal(a.lookup.anchor_pairs_a, b.lookup.anchor_pairs_a)
    assert torch.equal(a.lookup.anchor_pairs_b, b.lookup.anchor_pairs_b)


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_max_anchor_distance_forward_backward():
    """Local-window TinyMHLut still trains (forward + backward populate
    grads on weights and on x via STE carriers)."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=64, n_heads=2, n_outputs=4, n_anchor_pairs=4,
        tables_per_head=16, weight_dtype=torch.float32, random_seed=0, device=dev,
        max_anchor_distance=8,
    )
    x = torch.randn(8, 64, device=dev, requires_grad=True)
    y = m(x)
    y.sum().backward()
    assert m.weights.grad is not None
    assert m.weights.grad.abs().sum().item() > 0
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_max_anchor_distance_input_coverage_uniform_for_linspace():
    """Linspace start mode + many tables ⇒ each input dim is covered by at
    least some tables (no cold spots)."""
    dev = torch.device("cuda:0")
    input_dim, K = 64, 7
    m = TinyMultiHeadLut(
        input_dim=input_dim, n_heads=1, n_outputs=4, n_anchor_pairs=4,
        tables_per_head=128, weight_dtype=torch.float32, random_seed=0, device=dev,
        max_anchor_distance=K,
    )
    # Count how many tables touch each input dim (via either a or b indices).
    a = m.lookup.anchor_pairs_a.long()
    b = m.lookup.anchor_pairs_b.long()
    coverage = torch.zeros(input_dim, dtype=torch.long)
    for t in range(a.shape[0]):
        for v in torch.cat([a[t], b[t]]).unique():
            coverage[v] += 1
    assert (coverage > 0).all(), \
        f"some input dim is uncovered: {torch.nonzero(coverage == 0).flatten().tolist()}"


# ============================================================================
# aligned_local_scatter: per-table aligned input + scatter windows
# ============================================================================

@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_aligned_local_scatter_constraint():
    """Per (head, table), all lookup anchors AND all scatter destinations
    lie within the same width-(K+1) window starting at linspace start_t."""
    from spiky.lutorch.lut_helpers import local_window_starts as _starts_fn
    dev = torch.device("cuda:0")
    H, tph, K, n_outputs, n_anchor_pairs = 2, 32, 15, 8, 6
    input_dim = 64
    m = TinyMultiHeadLut(
        input_dim=input_dim, n_heads=H, n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs, tables_per_head=tph,
        weight_dtype=torch.float32, random_seed=0, device=dev,
        sparse_scatter_n_outputs=input_dim, sparse_scatter_seed=7,
        max_anchor_distance=K,
        aligned_local_scatter=True,
    )
    starts = _starts_fn(n_tables=H * tph, input_dim=input_dim,
                       max_distance=K, n_heads=H, starts_mode="linspace")
    starts = starts.view(H, tph)
    ap_a = m.lookup.anchor_pairs_a.long().view(H, tph, n_anchor_pairs)
    ap_b = m.lookup.anchor_pairs_b.long().view(H, tph, n_anchor_pairs)
    sc = m.scatter_indices  # [H, tph, n_outputs]
    for h in range(H):
        for t in range(tph):
            s = starts[h, t].item()
            lo, hi = s, s + K
            assert (ap_a[h, t] >= lo).all() and (ap_a[h, t] <= hi).all()
            assert (ap_b[h, t] >= lo).all() and (ap_b[h, t] <= hi).all()
            assert (sc[h, t] >= lo).all() and (sc[h, t] <= hi).all()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_aligned_local_scatter_distinct_within_table():
    """Each (head, table)'s n_outputs scatter destinations are distinct."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=64, n_heads=2, n_outputs=8, n_anchor_pairs=4,
        tables_per_head=16, weight_dtype=torch.float32, device=dev,
        sparse_scatter_n_outputs=64, sparse_scatter_seed=7,
        max_anchor_distance=15, aligned_local_scatter=True,
    )
    sc = m.scatter_indices.view(-1, 8)
    for t in range(sc.shape[0]):
        assert sc[t].unique().numel() == 8, f"table {t} has duplicates"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_aligned_local_scatter_requires_max_anchor_distance():
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="aligned_local_scatter requires max_anchor_distance"):
        TinyMultiHeadLut(
            input_dim=32, n_heads=1, n_outputs=8, n_anchor_pairs=4,
            tables_per_head=8, weight_dtype=torch.float32, device=dev,
            sparse_scatter_n_outputs=32, aligned_local_scatter=True,
            # missing max_anchor_distance
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_aligned_local_scatter_requires_input_eq_output():
    """input_dim must equal sparse_scatter_n_outputs for alignment."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="aligned_local_scatter requires input_dim"):
        TinyMultiHeadLut(
            input_dim=32, n_heads=1, n_outputs=4, n_anchor_pairs=4,
            tables_per_head=8, weight_dtype=torch.float32, device=dev,
            sparse_scatter_n_outputs=64, max_anchor_distance=8,
            aligned_local_scatter=True,
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_aligned_local_scatter_requires_linspace_starts():
    """aligned_local_scatter currently only supports linspace starts (so the
    lookup and scatter recompute the same deterministic start positions)."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="local_window_starts='linspace'"):
        TinyMultiHeadLut(
            input_dim=64, n_heads=1, n_outputs=4, n_anchor_pairs=4,
            tables_per_head=8, weight_dtype=torch.float32, device=dev,
            sparse_scatter_n_outputs=64, max_anchor_distance=8,
            aligned_local_scatter=True,
            local_window_starts="random",
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_aligned_local_scatter_n_outputs_capped_by_window():
    """n_outputs must be <= K+1 (you can't sample n_outputs distinct slots
    from a window of width K+1 if n_outputs > K+1)."""
    dev = torch.device("cuda:0")
    with pytest.raises(ValueError, match="aligned_local_scatter: n_outputs"):
        TinyMultiHeadLut(
            input_dim=64, n_heads=1, n_outputs=10, n_anchor_pairs=4,
            tables_per_head=8, weight_dtype=torch.float32, device=dev,
            sparse_scatter_n_outputs=64, max_anchor_distance=8,  # window = 9 < 10
            aligned_local_scatter=True,
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_aligned_local_scatter_forward_backward():
    """Forward + backward populate grads on weights and inputs."""
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=64, n_heads=2, n_outputs=8, n_anchor_pairs=6,
        tables_per_head=16, weight_dtype=torch.float32, random_seed=0, device=dev,
        sparse_scatter_n_outputs=64, sparse_scatter_seed=7,
        max_anchor_distance=15, aligned_local_scatter=True,
    )
    x = torch.randn(4, 64, device=dev, requires_grad=True)
    y = m(x)
    assert y.shape == (4, 2, 64)
    y.sum().backward()
    assert m.weights.grad.abs().sum().item() > 0
    assert x.grad.abs().sum().item() > 0


# =====================================================================
# soft backward_mode tests — TinyMultiHeadLut(backward_mode='soft')
# =====================================================================
# Drop-in for SoftMultiHeadLUT(hard=True, learnable_temps=True) but with
# TinyMHLut-style fast forward (sign-pack + embedding_bag) and pure-PyTorch
# + @torch.compile soft backward. Covers:
#   - construction guards (sparse_scatter / partition_sets rejected)
#   - parameter registration (learnable vs fixed temps)
#   - forward output equality with SoftMHLut(hard=True) at fp32
#   - gradient parity with SoftMHLut(hard=True) at fp32
#   - argmax_noise_eps: forward changes stochastically, grads stay finite
#   - learnable temperatures receive non-zero gradients
#   - argmax invariant: signbit-pack == argmax(einsum(p, bit_matrix)) at fp32

def _soft_mhlut_match_tiny(tiny, use_bf16=False):
    """Build a SoftMHLut whose weights and anchor pairs match a TinyMHLut(soft)
    instance — bypasses sampler-RNG differences so the equivalence test
    isolates the algorithm, not the init."""
    soft = SoftMultiHeadLUT(
        input_dim=tiny.input_dim, n_heads=tiny.n_heads, n_outputs=tiny.n_outputs,
        n_anchor_pairs=tiny.n_anchor_pairs, tables_per_head=tiny.tables_per_head,
        soft_score_temp=float(tiny.log_soft_score_temp.detach().exp()),
        select_temp=float(tiny.log_select_temp.detach().exp()),
        gumbel=False, hard=True, learnable_temps=True,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        weight_dtype=tiny.weights.dtype, device=tiny.weights.device,
        use_bf16=use_bf16, compile_forward=False,
    ).to(tiny.weights.device)
    with torch.no_grad():
        soft.weights.copy_(tiny.weights)
        soft.anchor_pairs_a.copy_(tiny.lookup.anchor_pairs_a)
        soft.anchor_pairs_b.copy_(tiny.lookup.anchor_pairs_b)
    return soft


def test_soft_backward_mode_invalid_string():
    dev = torch.device("cuda:0" if _has_cuda() else "cpu")
    with pytest.raises(ValueError, match="backward_mode must be"):
        TinyMultiHeadLut(
            input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
            tables_per_head=4, weight_dtype=torch.float32, device=dev,
            backward_mode='not_a_mode',
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_soft_backward_mode_learnable_temps_are_parameters():
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.float32, device=dev,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True,
    )
    assert isinstance(m.log_soft_score_temp, torch.nn.Parameter)
    assert isinstance(m.log_select_temp, torch.nn.Parameter)
    assert m.log_soft_score_temp.requires_grad
    assert m.log_select_temp.requires_grad


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_soft_backward_mode_fixed_temps_are_buffers():
    dev = torch.device("cuda:0")
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.float32, device=dev,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=False,
    )
    assert not isinstance(m.log_soft_score_temp, torch.nn.Parameter)
    # Tensor `in` comparison is ambiguous; check by id presence in buffers.
    buffer_ids = {id(b) for _, b in m.named_buffers()}
    assert id(m.log_soft_score_temp) in buffer_ids
    assert id(m.log_select_temp) in buffer_ids


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_soft_backward_mode_forward_matches_softmhlut_fp32():
    """Forward of TinyMHLut(soft, no noise) equals SoftMHLut(hard=True) at fp32."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    tiny = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.float32, random_seed=0, device=dev,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True, use_bf16=False,
        argmax_noise_eps=0.0,
    ).to(dev)
    soft = _soft_mhlut_match_tiny(tiny, use_bf16=False)
    x = torch.randn(16, 32, device=dev)
    out_tiny = tiny(x.clone())
    out_soft = soft(x.clone())
    assert out_tiny.shape == out_soft.shape
    diff = (out_tiny - out_soft).abs().max().item()
    # Bit-exact: both gather the same row of the same weights at the same
    # argmax index. Any non-zero difference is a bug.
    assert diff < 1e-5, f"forward mismatch: {diff}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_soft_backward_mode_gradients_match_softmhlut_fp32():
    """All four gradients match SoftMHLut(hard=True) within fp32 noise."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    tiny = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.float32, random_seed=0, device=dev,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True, use_bf16=False,
        argmax_noise_eps=0.0,
    ).to(dev)
    soft = _soft_mhlut_match_tiny(tiny, use_bf16=False)
    x = torch.randn(16, 32, device=dev)
    target = torch.randn(16, 2, 4, device=dev)

    x1 = x.clone().requires_grad_(True)
    x2 = x.clone().requires_grad_(True)
    ((tiny(x1) - target).pow(2).sum()).backward()
    ((soft(x2) - target).pow(2).sum()).backward()

    # dL/dw: exact match expected (both implementations scatter at the same
    # picked row, deterministic anchor pairs and weights).
    dw = (tiny.weights.grad - soft.weights.grad).abs().max().item()
    assert dw < 1e-5, f"dL/dweights abs mismatch: {dw}"
    # dL/dx, dL/dlog_T_*: fp32 numeric noise (rel ~1e-6 to 1e-4).
    dx = (x1.grad - x2.grad).abs().max().item()
    ref_x = max(x1.grad.abs().max().item(), 1e-12)
    assert dx / ref_x < 1e-4, f"dL/dx rel mismatch: {dx} / {ref_x}"
    dts = (tiny.log_soft_score_temp.grad - soft.log_soft_score_temp.grad).abs().item()
    ref_ts = max(tiny.log_soft_score_temp.grad.abs().item(), 1e-12)
    assert dts / ref_ts < 1e-3, f"dL/dlog_T_soft rel mismatch: {dts} / {ref_ts}"
    dtx = (tiny.log_select_temp.grad - soft.log_select_temp.grad).abs().item()
    ref_tx = max(tiny.log_select_temp.grad.abs().item(), 1e-12)
    assert dtx / ref_tx < 1e-3, f"dL/dlog_T_sel rel mismatch: {dtx} / {ref_tx}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_soft_backward_mode_noise_changes_forward_but_gradients_stay_finite():
    """argmax_noise_eps>0 → forward is stochastic across calls (random
    bit-flips at low-confidence positions); gradients still flow and are
    finite."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    common = dict(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.float32, random_seed=0, device=dev,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True, use_bf16=False,
    )
    m_no_noise = TinyMultiHeadLut(argmax_noise_eps=0.0, **common).to(dev)
    m_noise    = TinyMultiHeadLut(argmax_noise_eps=0.5, **common).to(dev)
    # Force identical state so any forward difference is from noise alone.
    with torch.no_grad():
        m_noise.weights.copy_(m_no_noise.weights)
        m_noise.lookup.anchor_pairs_a.copy_(m_no_noise.lookup.anchor_pairs_a)
        m_noise.lookup.anchor_pairs_b.copy_(m_no_noise.lookup.anchor_pairs_b)
        m_noise.soft_anchor_a_long.copy_(m_no_noise.soft_anchor_a_long)
        m_noise.soft_anchor_b_long.copy_(m_no_noise.soft_anchor_b_long)
    x = torch.randn(64, 32, device=dev)
    out_a = m_noise(x)
    out_b = m_noise(x)
    # Two calls under noise → different random bit flips → different outputs.
    assert (out_a - out_b).abs().max().item() > 0.0
    # eps=0 → deterministic across calls.
    out_c = m_no_noise(x)
    out_d = m_no_noise(x)
    assert (out_c - out_d).abs().max().item() == 0.0
    # Backward flows through noise path; gradients are finite.
    loss = m_noise(x.clone().requires_grad_(True)).sum()
    loss.backward()
    assert torch.isfinite(m_noise.weights.grad).all()
    assert torch.isfinite(m_noise.log_soft_score_temp.grad).all()
    assert torch.isfinite(m_noise.log_select_temp.grad).all()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_soft_backward_mode_temperature_gradients_nontrivial():
    """Learnable T_soft, T_sel parameters get non-zero gradients."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
        tables_per_head=4, weight_dtype=torch.float32, random_seed=0, device=dev,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True, use_bf16=False,
    ).to(dev)
    x = torch.randn(16, 32, device=dev)
    target = torch.randn(16, 2, 4, device=dev)
    loss = (m(x) - target).pow(2).sum()
    loss.backward()
    assert m.log_soft_score_temp.grad is not None
    assert m.log_select_temp.grad is not None
    assert m.log_soft_score_temp.grad.abs().item() > 0.0
    assert m.log_select_temp.grad.abs().item() > 0.0


# =====================================================================
# soft + sparse_scatter combination.
# =====================================================================

def _build_soft_sparse(dev, *, einsum_bf16_forward=False, **overrides):
    """TinyMultiHeadLut(backward_mode='soft', sparse_scatter_n_outputs=...)."""
    kwargs = dict(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
        tables_per_head=8, weight_dtype=torch.float32,
        random_seed=0, device=dev,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True,
        use_bf16=einsum_bf16_forward,        # einsum path requires use_bf16=True
        argmax_noise_eps=0.0,
        einsum_bf16_forward=einsum_bf16_forward,
        sparse_scatter_n_outputs=16, sparse_scatter_seed=7,
    )
    kwargs.update(overrides)
    return TinyMultiHeadLut(**kwargs)


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("einsum_bf16_forward", [False, True])
def test_soft_sparse_scatter_output_shape(einsum_bf16_forward):
    """soft + sparse_scatter returns [B, H, sparse_scatter_n_outputs]."""
    dev = torch.device("cuda:0")
    m = _build_soft_sparse(
        dev, n_heads=3, n_outputs=4, tables_per_head=6,
        sparse_scatter_n_outputs=20, einsum_bf16_forward=einsum_bf16_forward,
    )
    x = torch.randn(5, 32, device=dev)
    y = m(x)
    assert y.shape == (5, 3, 20), f"got {y.shape}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("einsum_bf16_forward", [False, True])
def test_soft_sparse_scatter_matches_manual_scatter(einsum_bf16_forward):
    """soft+sparse forward output equals: per-table soft gather (== signbit
    argmax at fp32) → scatter_add through scatter_indices into [B, H, sparse_no].

    For the einsum bf16 path we use fp32 weights + a wider tolerance: index can
    still flip on ties from bf16 rounding, but the gather→scatter structure is
    the same."""
    dev = torch.device("cuda:0")
    m = _build_soft_sparse(dev, einsum_bf16_forward=einsum_bf16_forward)
    x = torch.randn(3, 32, device=dev)

    # Same forward twice — both paths are deterministic (no Bernoulli noise,
    # einsum path uses fixed inputs).
    y = m(x)

    # Manual reference: replay forward path explicitly through SoftMHLut-style
    # per-table gather. At fp32 (no noise, no bf16), index = signbit-pack of
    # (x[a] - x[b]), which is exactly what tiny computes too.
    with torch.no_grad():
        anchor_a = m.soft_anchor_a_long
        anchor_b = m.soft_anchor_b_long
        d = x[:, anchor_a] - x[:, anchor_b]               # [B, n_tables, NAP]
        if einsum_bf16_forward:
            # Replay under the same bf16 autocast to match tiny's index.
            T_soft = m.log_soft_score_temp.detach().exp()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                p = d / (T_soft + d.abs())
                ts = torch.einsum("btp,pk->btk", p, m.soft_bit_matrix.to(p.dtype))
                index = ts.argmax(dim=-1).to(torch.int64)
        else:
            bits = (d > 0).to(torch.int64)
            powers = m.soft_powers.view(1, 1, -1)
            index = (bits * powers).sum(dim=-1)            # [B, n_tables]
        B = x.shape[0]
        n_lt = m.n_lookup_tables
        table_ix = torch.arange(n_lt, device=dev).view(1, -1).expand(B, -1)
        per_table = m.weights[table_ix, index]             # [B, n_lt, n_outputs]
        per_table = per_table.view(B, m.n_heads, m.tables_per_head, m.n_outputs)
        ref = torch.zeros(B, m.n_heads, m.sparse_scatter_n_outputs,
                          dtype=m.weights.dtype, device=dev)
        idx = m.scatter_indices.unsqueeze(0).expand(B, -1, -1, -1)
        ref.scatter_add_(2, idx.reshape(B, m.n_heads, -1),
                         per_table.reshape(B, m.n_heads, -1))

    diff = (y - ref).abs().max().item()
    # fp32 path: exact (single deterministic gather + scatter_add).
    # bf16-einsum path: also exact when ties don't drift — but we keep the
    # tolerance loose because index can change on ties from bf16 rounding.
    tol = 1e-5 if not einsum_bf16_forward else 1e-3
    assert diff < tol, f"forward mismatch: {diff}"


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize("einsum_bf16_forward", [False, True])
def test_soft_sparse_scatter_backward_grads_flow(einsum_bf16_forward):
    """soft+sparse backward populates grad on weights, x, and temperatures."""
    dev = torch.device("cuda:0")
    m = _build_soft_sparse(dev, einsum_bf16_forward=einsum_bf16_forward)
    x = torch.randn(3, 32, device=dev, requires_grad=True)
    target = torch.randn(3, m.n_heads, m.sparse_scatter_n_outputs, device=dev)
    loss = (m(x) - target).pow(2).sum()
    loss.backward()
    assert m.weights.grad is not None
    assert m.weights.grad.abs().sum().item() > 0
    assert x.grad is not None
    assert x.grad.abs().sum().item() > 0
    assert m.log_soft_score_temp.grad is not None
    assert m.log_select_temp.grad is not None
    assert torch.isfinite(m.weights.grad).all()
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_soft_sparse_scatter_grad_weights_match_manual_scatter():
    """Cross-check grad_weights against manual gather+scatter recomputation.

    The dense soft backward broadcasts grad_out over tph; here, scatter_indices
    routes specific output slots per (head, table), so grad_pt per table must
    GATHER from grad_out at the table's scatter destinations. Verifies the
    backward path handles this correctly (no silent broadcast)."""
    dev = torch.device("cuda:0")
    m = _build_soft_sparse(dev)
    x = torch.randn(2, 32, device=dev)
    g = torch.randn(2, m.n_heads, m.sparse_scatter_n_outputs, device=dev)
    y = m(x)
    y.backward(g)
    grad_weights_auto = m.weights.grad.detach().clone()
    grad_log_T_soft_auto = m.log_soft_score_temp.grad.detach().clone()
    grad_log_T_sel_auto = m.log_select_temp.grad.detach().clone()
    m.weights.grad = None
    m.log_soft_score_temp.grad = None
    m.log_select_temp.grad = None

    # Build a reference module that uses dense soft (no sparse_scatter) on
    # the SAME weights and anchor pairs. We construct the equivalent dense
    # grad_out from g via the inverse of scatter_add (gather at scatter_idx),
    # then compare grad_weights.
    with torch.no_grad():
        anchor_a = m.soft_anchor_a_long
        anchor_b = m.soft_anchor_b_long
        d = x[:, anchor_a] - x[:, anchor_b]
        bits = (d > 0).to(torch.int64)
        powers = m.soft_powers.view(1, 1, -1)
        index = (bits * powers).sum(dim=-1)                 # [B, n_tables]
        # grad_pt[b, t, :] = g[b, head(t), scatter_indices[head(t), local_t, :]]
        B = x.shape[0]
        H, T_per_h, S = m.scatter_indices.shape             # S == n_outputs
        # idx_per_head shaped to gather from g along its last dim
        idx = m.scatter_indices.unsqueeze(0).expand(B, -1, -1, -1)  # [B, H, T, S]
        # gather: dim=2 on g [B, H, sparse_no]; need per-(table, output) gather
        # do it per-head with advanced indexing
        grad_pt = torch.empty(B, H, T_per_h, S, device=dev, dtype=g.dtype)
        for h in range(H):
            grad_pt[:, h] = g[:, h, :].gather(1, idx[:, h, :, :].reshape(B, -1)).reshape(B, T_per_h, S)
        # Manual grad_weights: scatter grad_pt onto (table, index) entries.
        n_lt = m.n_lookup_tables
        table_dim = m.table_dim
        n_outputs = m.n_outputs
        flat_offset = torch.arange(n_lt, device=dev) * table_dim
        flat_idx = (index + flat_offset[None, :]).reshape(-1)
        grad_w_flat = torch.zeros(n_lt * table_dim, n_outputs,
                                  dtype=m.weights.dtype, device=dev)
        grad_w_flat.index_add_(
            0, flat_idx,
            grad_pt.reshape(B * n_lt, n_outputs).to(m.weights.dtype),
        )
        grad_weights_manual = grad_w_flat.view(n_lt, table_dim, n_outputs)

    diff = (grad_weights_auto - grad_weights_manual).abs().max().item()
    assert diff < 1e-5, f"grad_weights mismatch (manual): {diff}"
    # Temperature grads should be finite & nontrivial.
    assert torch.isfinite(grad_log_T_soft_auto).all()
    assert torch.isfinite(grad_log_T_sel_auto).all()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_soft_sparse_scatter_forward_matches_dense_when_identity_scatter():
    """When sparse_scatter is configured so that every (head, table) writes its
    n_outputs values into the SAME n_outputs slots of a width-n_outputs target
    (identity scatter accumulating across tables, since sparse_no == n_outputs),
    the result matches the dense embedding_bag-reduced soft forward. Confirms
    the per-table forward + scatter_add path is consistent with the dense
    forward."""
    dev = torch.device("cuda:0")
    n_heads, n_outputs, tph = 2, 4, 8
    # sparse_scatter_n_outputs == n_outputs collapses scatter to identity per
    # table (each table writes into slots [0..n_outputs-1]).
    m_sparse = _build_soft_sparse(
        dev, n_heads=n_heads, n_outputs=n_outputs, tables_per_head=tph,
        sparse_scatter_n_outputs=n_outputs,
    )
    # Reference dense module with identical anchor pairs and weights.
    m_dense = TinyMultiHeadLut(
        input_dim=32, n_heads=n_heads, n_outputs=n_outputs, n_anchor_pairs=5,
        tables_per_head=tph, weight_dtype=torch.float32,
        random_seed=0, device=dev,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True, use_bf16=False, argmax_noise_eps=0.0,
    ).to(dev)
    with torch.no_grad():
        m_dense.weights.copy_(m_sparse.weights)
        m_dense.soft_anchor_a_long.copy_(m_sparse.soft_anchor_a_long)
        m_dense.soft_anchor_b_long.copy_(m_sparse.soft_anchor_b_long)
        m_dense.lookup.anchor_pairs_a.copy_(m_sparse.lookup.anchor_pairs_a)
        m_dense.lookup.anchor_pairs_b.copy_(m_sparse.lookup.anchor_pairs_b)
        m_dense.log_soft_score_temp.copy_(m_sparse.log_soft_score_temp)
        m_dense.log_select_temp.copy_(m_sparse.log_select_temp)
    x = torch.randn(4, 32, device=dev)
    y_sparse = m_sparse(x)
    y_dense = m_dense(x)
    # m_sparse has scatter_indices into width-n_outputs; for an identity
    # scatter, each table writes its n_outputs values into a permutation of
    # those slots — the SUM (over tables) of permuted vectors generally
    # differs from the dense sum (which has no permutation). So we compare
    # SUMS over the output axis: the total mass should match (each table
    # contributes its full n_outputs sum into the sparse output).
    s_sparse = y_sparse.sum(dim=-1)
    s_dense  = y_dense.sum(dim=-1)
    diff = (s_sparse - s_dense).abs().max().item()
    assert diff < 1e-4, f"sum mismatch: {diff}"


# =====================================================================
# Cross-module: SoftMultiHeadLUT bit_matrix convention invariant.
# =====================================================================
@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_softmhlut_argmax_equals_signbit_pack_at_fp32():
    """At fp32, argmax of `ts = einsum(p, bit_matrix)` equals the MSB-first
    sign-bit-pack of d, because sign(p) = sign(d) and bit_matrix is ±1. This
    is the invariant TinyMHLut(soft) exploits to skip the
    einsum in forward. Test guards against any future change to bit_matrix
    or sign-pack convention.
    """
    dev = torch.device("cuda:0")
    from spiky.lutorch.tiny_multi_head_lut import (
        _soft_bit_matrix_msb, _msb_powers,
    )
    for NAP in (4, 6, 8):
        K = 1 << NAP
        bm = _soft_bit_matrix_msb(NAP, dev, dtype=torch.float32)
        powers = _msb_powers(NAP, dev)
        torch.manual_seed(NAP)
        d = torch.randn(1024, NAP, device=dev)
        p = d / (0.5 + d.abs())
        ts = torch.einsum("bp,pk->bk", p, bm)
        idx_einsum = ts.argmax(dim=-1)
        bits = (d > 0).to(torch.int64)
        idx_bitpack = (bits * powers).sum(dim=-1)
        assert torch.equal(idx_einsum, idx_bitpack), \
            f"NAP={NAP}: argmax mismatch ({(idx_einsum != idx_bitpack).sum().item()}/{d.shape[0]})"


# =====================================================================
# einsum_bf16_forward branch — SoftMHLut(hard=True)-style forward via
# bf16 einsum + argmax instead of fp32 sign-bit-pack
# =====================================================================
@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_einsum_bf16_forward_rejects_ste_mode():
    """einsum_bf16_forward only makes sense with soft backward."""
    with pytest.raises(ValueError, match="backward_mode='soft'"):
        TinyMultiHeadLut(
            input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
            tables_per_head=4, random_seed=0,
            backward_mode='ste', einsum_bf16_forward=True,
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_einsum_bf16_forward_rejects_use_bf16_false():
    """einsum_bf16_forward requires bf16 autocast — the rounding is the noise."""
    with pytest.raises(ValueError, match="use_bf16=True"):
        TinyMultiHeadLut(
            input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=5,
            tables_per_head=4, random_seed=0,
            backward_mode='soft', use_bf16=False, einsum_bf16_forward=True,
        )


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_einsum_bf16_forward_matches_signbit_pack_at_fp32_when_no_noise():
    """When `use_bf16=True` is OFF the einsum and sign-pack would be bit-equal
    (`test_softmhlut_argmax_equals_signbit_pack_at_fp32`). With bf16 autocast
    they can differ at low-confidence rows — that's the regularization. Here
    we just sanity-check the new path produces finite, well-shaped output and
    gradients flow."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=6,
        tables_per_head=4, weight_dtype=torch.float32, random_seed=0, device=dev,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True, use_bf16=True, einsum_bf16_forward=True,
        argmax_noise_eps=0.0,
    ).to(dev)
    x = torch.randn(16, 32, device=dev, requires_grad=True)
    out = m(x)
    assert out.shape == (16, 2, 4)
    assert torch.isfinite(out).all()
    out.sum().backward()
    assert torch.isfinite(m.weights.grad).all()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(m.log_soft_score_temp.grad).all()
    assert torch.isfinite(m.log_select_temp.grad).all()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_einsum_bf16_forward_is_deterministic_across_calls():
    """Unlike `argmax_noise_eps>0` (stochastic Bernoulli flips), the bf16
    einsum path is deterministic: same input → same bf16 rounding → same
    index → same output."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    m = TinyMultiHeadLut(
        input_dim=32, n_heads=2, n_outputs=4, n_anchor_pairs=6,
        tables_per_head=4, weight_dtype=torch.float32, random_seed=0, device=dev,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True, use_bf16=True, einsum_bf16_forward=True,
    ).to(dev)
    x = torch.randn(64, 32, device=dev)
    out_a = m(x)
    out_b = m(x)
    assert (out_a - out_b).abs().max().item() == 0.0


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
def test_einsum_bf16_forward_can_differ_from_signbit_path_under_bf16():
    """The whole point: under bf16 autocast, the einsum-argmax path picks a
    different row at some low-confidence positions vs the fp32 sign-bit-pack.
    With many tables and large input, the disagreement is observable."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    common = dict(
        input_dim=64, n_heads=4, n_outputs=8, n_anchor_pairs=8,
        tables_per_head=32, weight_dtype=torch.float32, random_seed=0, device=dev,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
        backward_mode='soft', soft_score_temp=0.5, select_temp=0.5,
        learnable_temps=True, use_bf16=True, argmax_noise_eps=0.0,
    )
    m_signpack = TinyMultiHeadLut(einsum_bf16_forward=False, **common).to(dev)
    m_einsum   = TinyMultiHeadLut(einsum_bf16_forward=True,  **common).to(dev)
    # Force identical state so any output difference is from the index path.
    with torch.no_grad():
        m_einsum.weights.copy_(m_signpack.weights)
        m_einsum.lookup.anchor_pairs_a.copy_(m_signpack.lookup.anchor_pairs_a)
        m_einsum.lookup.anchor_pairs_b.copy_(m_signpack.lookup.anchor_pairs_b)
        m_einsum.soft_anchor_a_long.copy_(m_signpack.soft_anchor_a_long)
        m_einsum.soft_anchor_b_long.copy_(m_signpack.soft_anchor_b_long)
    x = torch.randn(128, 64, device=dev)
    out_signpack = m_signpack(x)
    out_einsum   = m_einsum(x)
    # The two paths should produce non-trivially different outputs because
    # bf16 rounding flips the argmax at some low-confidence rows.
    diff = (out_signpack - out_einsum).abs().max().item()
    assert diff > 0.0, "bf16 einsum and fp32 sign-pack should disagree at some rows"
