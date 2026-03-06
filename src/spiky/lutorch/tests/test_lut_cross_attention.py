"""
Tests for LUTCrossAttention.
"""
import os

# Disable torch.compile in lutorch so tests run without compilation.
os.environ["SPIKY_LUTORCH_NO_COMPILE"] = "1"

import torch

from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_cross_attention import LUTCrossAttention
from spiky.lutorch.lut_helpers import UncertaintyMode


def test_lut_cross_attention_finite_scores(device, seed=None):
    """
    Basic sanity test for LUTCrossAttention:
    - small batch/sequence
    - tiny LUTs
    - verify all attention scores are finite (no NaN/inf)
    """
    if seed is not None:
        torch.manual_seed(seed)

    B = 2
    S = 8
    E = 16
    H = 2
    N_BUCKETS = 4

    # Small MultiHeadLut used as attention score generator: outputs shape [P, H, 1]
    attn_lut = MultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=1,
        n_anchor_pairs=4,
        tables_per_head=2,
        n_buckets=N_BUCKETS,
        connected_pairs=False,
        anchor_candidates=None,
        cmp_eps=0.0,
        random_seed=seed,
        n_alternatives=1,
        smooth_mode=False,
        device=device,
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
    )
    # Initialize attention LUT weights with small random values to avoid large logits
    with torch.no_grad():
        attn_lut.projection.weights.normal_(mean=0.0, std=0.01)

    cross_attn = LUTCrossAttention(
        multi_head_lut=attn_lut,
        causal=True,
        n_positional_buckets=N_BUCKETS,
        do_sanity_checks=True,
    ).to(device)

    # Random inputs
    x = torch.randn(B, S, E, device=device)

    scores = cross_attn(x, x)  # [B, S, S, H]

    assert scores.shape == (B, S, S, H)
    assert torch.isfinite(scores).all(), "Attention scores contain NaN or inf"

    print("✓ LUTCrossAttention finite-scores sanity test passed")
    return True


def test_lut_cross_attention_finite_scores_non_causal(device, seed=None):
    """
    Sanity test for LUTCrossAttention in non-causal mode (no positional buckets).
    Verifies all attention scores are finite.
    """
    if seed is not None:
        torch.manual_seed(seed)

    B = 2
    S = 8
    E = 16
    H = 2

    attn_lut = MultiHeadLut(
        input_dim=E,
        n_heads=H,
        n_outputs=1,
        n_anchor_pairs=4,
        tables_per_head=2,
        n_buckets=1,  # no positional buckets in this test
        connected_pairs=False,
        anchor_candidates=None,
        cmp_eps=0.0,
        random_seed=seed,
        n_alternatives=1,
        smooth_mode=False,
        device=device,
        uncertainty_mode=UncertaintyMode.INVERSE_L1,
    )
    with torch.no_grad():
        attn_lut.projection.weights.normal_(mean=0.0, std=0.01)

    cross_attn = LUTCrossAttention(
        multi_head_lut=attn_lut,
        causal=False,
        n_positional_buckets=1,
        do_sanity_checks=False,
    ).to(device)

    x = torch.randn(B, S, E, device=device)
    scores = cross_attn(x, x)

    assert scores.shape == (B, S, S, H)
    assert torch.isfinite(scores).all(), "Non-causal attention scores contain NaN or inf"

    print("✓ LUTCrossAttention non-causal finite-scores test passed")
    return True

def main():
    """
    Run LUTCrossAttention tests on available devices.
    """
    print("=" * 60)
    print("LUTCrossAttention TESTS")
    print("=" * 60)

    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")

    seed = 123

    for device in devices:
        print(f"\nTesting on {device}...")

        print("\n1. Finite-scores sanity test...")
        success = test_lut_cross_attention_finite_scores(device, seed=seed)
        if not success:
            print(f"❌ Test failed on {device}")
            return -1

        print("\n2. Non-causal finite-scores sanity test...")
        success = test_lut_cross_attention_finite_scores_non_causal(device, seed=seed)
        if not success:
            print(f"❌ Non-causal test failed on {device}")
            return -1

        print(f"\n✓ All LUTCrossAttention tests passed on {device}!")

    return 0


if __name__ == "__main__":
    exit(main())
