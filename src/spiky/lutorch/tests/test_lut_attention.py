"""Tests for LUTAttention."""
import torch

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_attention import LUTAttention
from spiky.lutorch.lut_helpers import UncertaintyMode

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
