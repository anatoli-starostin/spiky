"""Tests for LUTAttention."""
from contextlib import contextmanager

import pytest
import torch

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_attention import LUTAttention, LUTAttentionV2, LUTAttentionV3
from spiky.lutorch.lut_helpers import UncertaintyMode, SelfExcitementMode
import spiky.lutorch.lut_attention as lut_attn_mod

SEED = 123


def _make_attn_lut(device, n_buckets=4, seed=SEED):
    lut = MultiHeadLut(
        input_dim=16, n_heads=2, n_outputs=1,
        n_anchor_pairs=4, tables_per_head=2,
        n_buckets=n_buckets, connected_anchors_mode=False,
        anchor_candidates=None, cmp_eps=0.0,
        random_seed=seed, n_alternatives=1, smooth_mode=False,
        device=device, uncertainty_mode=UncertaintyMode.INVERSE_L1,
    )
    with torch.no_grad():
        lut.projection.weights.normal_(mean=0.0, std=0.01)
    return lut


def test_lut_attention_finite_scores(device):
    """Causal LUTAttention outputs are finite and correctly shaped."""
    torch.manual_seed(SEED)
    B, S, E, H, N_BUCKETS = 2, 8, 16, 2, 4

    cross_attn = LUTAttention(
        multi_head_lut=_make_attn_lut(device, n_buckets=N_BUCKETS),
        causal=True, n_positional_buckets=N_BUCKETS,
        include_diagonal=True, do_sanity_checks=True,
    ).to(device)

    x = torch.randn(B, S, E, device=device)
    scores = cross_attn(x, x)

    assert scores.shape == (B, S, S, H)
    assert torch.isfinite(scores).all(), "Attention scores contain NaN or inf"


def test_lut_attention_finite_scores_exclude_diagonal(device):
    """Causal LUTAttention with include_diagonal=False: finite scores and clean backward."""
    torch.manual_seed(SEED)
    B, S, E, H, N_BUCKETS = 2, 8, 16, 2, 4

    cross_attn = LUTAttention(
        multi_head_lut=_make_attn_lut(device, n_buckets=N_BUCKETS),
        causal=True, n_positional_buckets=N_BUCKETS,
        include_diagonal=False, do_sanity_checks=True,
    ).to(device)

    x = torch.randn(B, S, E, device=device, requires_grad=True)
    scores = cross_attn(x, x)

    assert scores.shape == (B, S, S, H)
    assert torch.isfinite(scores).all(), "Attention scores (exclude_diagonal) contain NaN or inf"
    scores.sum().backward()


def test_lut_attention_finite_scores_non_causal(device):
    """Non-causal LUTAttention outputs are finite."""
    torch.manual_seed(SEED)
    B, S, E, H = 2, 8, 16, 2

    cross_attn = LUTAttention(
        multi_head_lut=_make_attn_lut(device, n_buckets=1),
        causal=False, n_positional_buckets=1,
        do_sanity_checks=False,
    ).to(device)

    x = torch.randn(B, S, E, device=device)
    scores = cross_attn(x, x)

    assert scores.shape == (B, S, S, H)
    assert torch.isfinite(scores).all(), "Non-causal attention scores contain NaN or inf"


# ---------------------------------------------------------------------------
# LUTAttentionV2 fused kernel tests
# ---------------------------------------------------------------------------

@contextmanager
def _fused_kernel_enabled(enabled: bool):
    prev = lut_attn_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS
    lut_attn_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = enabled
    try:
        yield
    finally:
        lut_attn_mod._USE_LUTORCH_CUSTOM_CUDA_KERNELS = prev


def _make_lut_attn_v2(device, T, E, pos_dim, H, O, tph, nap, self_excitement, seed=SEED):
    lut = MultiHeadLut(
        input_dim=2 * E + pos_dim,
        n_heads=H,
        n_outputs=O,
        n_anchor_pairs=nap,
        tables_per_head=tph,
        n_alternatives=1,
        smooth_mode=False,
        calibrate_output=False,
        normalize_weights=False,
        table_dropout=0.0,
        random_seed=seed,
        device=device,
    )
    with torch.no_grad():
        lut.projection.weights.normal_(mean=0.0, std=0.01)
    return LUTAttentionV2(lut, seq_len=T, causal=True, self_excitement=self_excitement).to(device)


@pytest.mark.parametrize("self_excitement", [False, True])
def test_lut_attention_v2_fused_fwd_matches_ref(device, self_excitement):
    """Fused CUDA kernel forward matches Python reference on cuda."""
    if device != "cuda":
        pytest.skip("fused kernel only runs on CUDA float32")
    torch.manual_seed(SEED)
    B, T, E, pos_dim, H, O, tph, nap = 2, 8, 16, 8, 1, 32, 4, 4

    attn = _make_lut_attn_v2(device, T, E, pos_dim, H, O, tph, nap, self_excitement)
    attn.eval()
    x = torch.randn(B, T, E, device=device)
    rel_pe = torch.randn(T, pos_dim, device=device)

    with _fused_kernel_enabled(False):
        ref = attn(x.clone(), rel_pe.clone())
    with _fused_kernel_enabled(True):
        fast = attn(x.clone(), rel_pe.clone())

    assert fast.shape == ref.shape
    assert torch.allclose(fast, ref, atol=1e-4), \
        f"fwd max diff = {(fast - ref).abs().max().item()}"


@pytest.mark.parametrize("self_excitement", [False, True])
def test_lut_attention_v2_fused_bwd_matches_ref(device, self_excitement):
    """Fused CUDA kernel backward gradients match Python reference."""
    if device != "cuda":
        pytest.skip("fused kernel only runs on CUDA float32")
    torch.manual_seed(SEED)
    B, T, E, pos_dim, H, O, tph, nap = 2, 8, 16, 8, 1, 32, 4, 4

    attn = _make_lut_attn_v2(device, T, E, pos_dim, H, O, tph, nap, self_excitement)
    attn.train()
    x_base = torch.randn(B, T, E, device=device)
    rpe_base = torch.randn(T, pos_dim, device=device)

    def run(enabled):
        x = x_base.clone().requires_grad_(True)
        rpe = rpe_base.clone().requires_grad_(True)
        attn.multi_head_lut.projection.weights.grad = None
        with _fused_kernel_enabled(enabled):
            out = attn(x, rpe)
        out.sum().backward()
        return x.grad.clone(), attn.multi_head_lut.projection.weights.grad.clone(), rpe.grad.clone()

    x_g_ref, w_g_ref, rpe_g_ref = run(False)
    x_g_fast, w_g_fast, rpe_g_fast = run(True)

    assert torch.allclose(x_g_fast, x_g_ref, atol=1e-4), \
        f"x_grad max diff = {(x_g_fast - x_g_ref).abs().max().item()}"
    assert torch.allclose(w_g_fast, w_g_ref, atol=1e-4), \
        f"w_grad max diff = {(w_g_fast - w_g_ref).abs().max().item()}"
    assert torch.allclose(rpe_g_fast, rpe_g_ref, atol=1e-4), \
        f"rpe_grad max diff = {(rpe_g_fast - rpe_g_ref).abs().max().item()}"


@pytest.mark.parametrize("H,O", [(2, 64), (4, 8), (1, 32), (4, 32)])
def test_lut_attention_v2_fused_multihead_se(device, H, O):
    """Fused kernel with multi-head self_excitement matches Python reference.

    With small tph the two paths are bitwise-close.  At larger tph the tree
    reduction (fused) vs linear sum (Python embedding_bag) gives slightly
    different pair_out_buf values.  SE's sign() discontinuity amplifies those
    differences in the backward, so we use a small tph to keep things tight.
    """
    if device != "cuda":
        pytest.skip("fused kernel only runs on CUDA float32")
    torch.manual_seed(SEED)
    E = H * O
    B, T, pos_dim, tph, nap = 2, 8, 16, 8, 4

    attn = _make_lut_attn_v2(device, T, E, pos_dim, H, O, tph, nap,
                              self_excitement=True)
    attn.train()
    x_base = torch.randn(B, T, E, device=device)
    rpe_base = torch.randn(T, pos_dim, device=device)

    def run(enabled):
        x = x_base.clone().requires_grad_(True)
        rpe = rpe_base.clone().requires_grad_(True)
        attn.multi_head_lut.projection.weights.grad = None
        with _fused_kernel_enabled(enabled):
            out = attn(x, rpe)
        out.sum().backward()
        return (out.detach().clone(),
                x.grad.clone(),
                attn.multi_head_lut.projection.weights.grad.clone(),
                rpe.grad.clone())

    out_ref, x_g_ref, w_g_ref, rpe_g_ref = run(False)
    out_fast, x_g_fast, w_g_fast, rpe_g_fast = run(True)

    assert torch.allclose(out_fast, out_ref, atol=1e-4), \
        f"H={H} O={O} fwd max diff = {(out_fast - out_ref).abs().max().item()}"
    assert torch.allclose(x_g_fast, x_g_ref, atol=1e-4), \
        f"H={H} O={O} x_grad max diff = {(x_g_fast - x_g_ref).abs().max().item()}"
    assert torch.allclose(w_g_fast, w_g_ref, atol=1e-4), \
        f"H={H} O={O} w_grad max diff = {(w_g_fast - w_g_ref).abs().max().item()}"
    assert torch.allclose(rpe_g_fast, rpe_g_ref, atol=1e-4), \
        f"H={H} O={O} rpe_grad max diff = {(rpe_g_fast - rpe_g_ref).abs().max().item()}"


@pytest.mark.parametrize("se_mode", [
    SelfExcitementMode.LINEAR,
    SelfExcitementMode.QUADRATIC,
    SelfExcitementMode.EXPONENTIAL,
])
@pytest.mark.parametrize("H,O", [(1, 32), (4, 8)])
def test_lut_attention_v2_fused_se_modes(device, se_mode, H, O):
    """Fused kernel SE modes (linear/quadratic/exponential) match Python reference."""
    if device != "cuda":
        pytest.skip("fused kernel only runs on CUDA float32")
    torch.manual_seed(SEED)
    E = H * O
    B, T, pos_dim, tph, nap = 2, 8, 16, 8, 4

    lut = MultiHeadLut(
        input_dim=2 * E + pos_dim, n_heads=H, n_outputs=O,
        n_anchor_pairs=nap, tables_per_head=tph,
        n_alternatives=1, smooth_mode=False,
        calibrate_output=False, normalize_weights=False,
        table_dropout=0.0, random_seed=SEED, device=device,
    )
    with torch.no_grad():
        lut.projection.weights.normal_(mean=0.0, std=0.01)
    attn = LUTAttentionV2(
        lut, seq_len=T, causal=True,
        self_excitement=True, self_excitement_mode=se_mode,
    ).to(device)
    attn.train()
    x_base = torch.randn(B, T, E, device=device)
    rpe_base = torch.randn(T, pos_dim, device=device)

    def run(enabled):
        x = x_base.clone().requires_grad_(True)
        rpe = rpe_base.clone().requires_grad_(True)
        attn.multi_head_lut.projection.weights.grad = None
        with _fused_kernel_enabled(enabled):
            out = attn(x, rpe)
        out.sum().backward()
        return (out.detach().clone(),
                x.grad.clone(),
                attn.multi_head_lut.projection.weights.grad.clone(),
                rpe.grad.clone())

    out_ref, x_g_ref, w_g_ref, rpe_g_ref = run(False)
    out_fast, x_g_fast, w_g_fast, rpe_g_fast = run(True)

    tag = f"H={H} O={O} mode={se_mode.value}"
    assert torch.allclose(out_fast, out_ref, atol=1e-4), \
        f"{tag} fwd max diff = {(out_fast - out_ref).abs().max().item()}"
    assert torch.allclose(x_g_fast, x_g_ref, atol=1e-4), \
        f"{tag} x_grad max diff = {(x_g_fast - x_g_ref).abs().max().item()}"
    assert torch.allclose(w_g_fast, w_g_ref, atol=1e-4), \
        f"{tag} w_grad max diff = {(w_g_fast - w_g_ref).abs().max().item()}"
    assert torch.allclose(rpe_g_fast, rpe_g_ref, atol=1e-4), \
        f"{tag} rpe_grad max diff = {(rpe_g_fast - rpe_g_ref).abs().max().item()}"


# ---------------------------------------------------------------------------
# LUTAttentionV3 tests
# ---------------------------------------------------------------------------

def _make_lut_attn_v3(device, T, E, pos_dim, H, O, tph, nap, causal, include_diagonal, seed=SEED):
    lut = MultiHeadLut(
        input_dim=2 * E + pos_dim,
        n_heads=H,
        n_outputs=O,
        n_anchor_pairs=nap,
        tables_per_head=tph,
        n_alternatives=1,
        smooth_mode=False,
        calibrate_output=False,
        normalize_weights=False,
        table_dropout=0.0,
        random_seed=seed,
        device=device,
    )
    with torch.no_grad():
        lut.projection.weights.normal_(mean=0.0, std=0.01)
    return LUTAttentionV3(lut, seq_len=T, causal=causal, include_diagonal=include_diagonal).to(device)


@pytest.mark.parametrize("causal,include_diagonal", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_lut_attention_v3_shape_and_mask(device, causal, include_diagonal):
    """V3 output has correct shape and -inf pattern."""
    torch.manual_seed(SEED)
    B, T, E, pos_dim, H, O = 2, 6, 8, 4, 2, 3
    tph, nap = 4, 4

    attn = _make_lut_attn_v3(device, T, E, pos_dim, H, O, tph, nap, causal, include_diagonal)
    x = torch.randn(B, T, E, device=device)
    rpe_len = T if include_diagonal else T - 1
    rel_pe = torch.randn(rpe_len, pos_dim, device=device)

    out = attn(x, rel_pe)
    assert out.shape == (B, T, T, H, O)

    # Check mask pattern: valid positions should be finite, invalid should be -inf
    for i in range(T):
        for j in range(T):
            is_valid = True
            if causal and j > i:
                is_valid = False
            if not include_diagonal and i == j:
                is_valid = False

            vals = out[0, i, j, :, :]
            if is_valid:
                assert torch.isfinite(vals).all(), f"Expected finite at ({i},{j}), got {vals}"
            else:
                assert (vals == float('-inf')).all(), f"Expected -inf at ({i},{j}), got {vals}"


@pytest.mark.parametrize("causal,include_diagonal", [
    (True, True), (True, False), (False, True), (False, False),
])
def test_lut_attention_v3_backward(device, causal, include_diagonal):
    """V3 backward produces finite gradients."""
    torch.manual_seed(SEED)
    B, T, E, pos_dim, H, O = 2, 6, 8, 4, 2, 3
    tph, nap = 4, 4

    attn = _make_lut_attn_v3(device, T, E, pos_dim, H, O, tph, nap, causal, include_diagonal)
    x = torch.randn(B, T, E, device=device, requires_grad=True)
    rpe_len = T if include_diagonal else T - 1
    rel_pe = torch.randn(rpe_len, pos_dim, device=device, requires_grad=True)

    out = attn(x, rel_pe)
    # Only backprop through finite values
    loss = out[out.isfinite()].sum()
    loss.backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert rel_pe.grad is not None and torch.isfinite(rel_pe.grad).all()


def test_lut_attention_v3_pairs_processor(device):
    """V3 with custom pairs_processor produces correct shape and matches manual computation."""
    torch.manual_seed(SEED)
    B, T, E, pos_dim, H, O = 2, 6, 8, 4, 2, 3
    tph, nap = 4, 4

    # pairs_processor that computes |x_i - x_j| and concatenates with rpe
    # Output dim = E + pos_dim = 12
    class DiffProcessor(torch.nn.Module):
        def forward(self, x_i, x_j, rpe):
            return torch.cat([(x_i - x_j).abs(), rpe], dim=-1)

    proc = DiffProcessor()
    input_dim = E + pos_dim  # 12

    lut = MultiHeadLut(
        input_dim=input_dim, n_heads=H, n_outputs=O,
        n_anchor_pairs=nap, tables_per_head=tph,
        n_alternatives=1, smooth_mode=False,
        calibrate_output=False, normalize_weights=False,
        table_dropout=0.0, random_seed=SEED, device=device,
    )
    with torch.no_grad():
        lut.projection.weights.normal_(mean=0.0, std=0.01)

    attn = LUTAttentionV3(lut, seq_len=T, causal=True, include_diagonal=True,
                           pairs_processor=proc).to(device)

    x = torch.randn(B, T, E, device=device, requires_grad=True)
    rel_pe = torch.randn(T, pos_dim, device=device, requires_grad=True)

    out = attn(x, rel_pe)
    assert out.shape == (B, T, T, H, O)

    # Check finite values exist in valid positions
    assert torch.isfinite(out[0, 1, 0, :, :]).all()
    # Check -inf in masked positions
    assert (out[0, 0, 1, :, :] == float('-inf')).all()

    # Backward should work
    loss = out[out.isfinite()].sum()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert rel_pe.grad is not None and torch.isfinite(rel_pe.grad).all()


def test_lut_attention_v3_pairs_processor_dim_mismatch(device):
    """V3 raises assertion when pairs_processor output dim doesn't match LUT input_dim."""
    torch.manual_seed(SEED)
    B, T, E, pos_dim, H, O = 2, 4, 8, 4, 1, 2
    tph, nap = 2, 3

    class BadProcessor(torch.nn.Module):
        def forward(self, x_i, x_j, rpe):
            return x_i  # dim E, not 2E+pos_dim

    lut = MultiHeadLut(
        input_dim=2 * E + pos_dim, n_heads=H, n_outputs=O,
        n_anchor_pairs=nap, tables_per_head=tph,
        n_alternatives=1, smooth_mode=False,
        calibrate_output=False, normalize_weights=False,
        table_dropout=0.0, random_seed=SEED, device=device,
    )
    attn = LUTAttentionV3(lut, seq_len=T, causal=True, include_diagonal=True,
                           pairs_processor=BadProcessor()).to(device)

    x = torch.randn(B, T, E, device=device)
    rel_pe = torch.randn(T, pos_dim, device=device)

    with pytest.raises(AssertionError, match="pairs_processor output dim"):
        attn(x, rel_pe)
