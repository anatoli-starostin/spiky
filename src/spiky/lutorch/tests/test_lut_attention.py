"""Tests for LUTAttention."""
from contextlib import contextmanager

import pytest
import torch

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_attention import LUTAttention, LUTAttentionV2
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
