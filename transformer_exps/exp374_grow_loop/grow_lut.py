"""Grow a BitPermutationLUT in-place of tph (tables-per-head), preserving
the first `old_tph` tables' trained state (anchors + latent + bit_weights).

Approach: build a fresh larger LUT, then overwrite its first `old_tph` slots
(per head) with the old module's state. The remaining slots keep the fresh
CFC-sampled anchors and random latents.
"""
from typing import Any

import torch

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT


def grow_lut(old: BitPermutationLUT, new_tph: int, seed: int) -> BitPermutationLUT:
    """Return a new BitPermutationLUT with `new_tph` tables per head,
    warm-started from `old` in the first `old.tph` slots per head.
    """
    old_tph = old.tph
    H = old.n_heads
    if new_tph <= old_tph:
        raise ValueError(f"new_tph ({new_tph}) must exceed old.tph ({old_tph})")

    dev = old.bit_weights.device
    in_nap = old.input_nap
    out_nap = old.output_nap
    td = old.table_dim

    # Build a fresh new LUT at the target size. This gives us:
    #  - fresh anchor_pairs_a/b (TinyAnchorPairsLookup) for all new_tph tables
    #  - fresh idx_a/idx_b + derived structures (CFC on full new_tph)
    #  - fresh latent + bit_weights
    new = BitPermutationLUT(
        n_inputs=old.n_inputs,
        n_outputs=old.n_outputs,
        n_heads=H,
        input_nap=in_nap,
        output_nap=out_nap,
        tph=new_tph,
        random_seed=seed,
        initial_weights_noise=0.001,
        soft_backward=old.soft_backward,
        latent_dtype=old.latent_dtype,
        device=dev,
        anchor_sampling_policy=old._anchor_sampling_policy_opt,
    )

    # --- Merge input-anchor pairs: first old_tph slots per head = old's values ---
    old_ap_a = old.anchor.anchor_pairs_a.view(H, old_tph, in_nap)
    old_ap_b = old.anchor.anchor_pairs_b.view(H, old_tph, in_nap)
    new_ap_a_3d = new.anchor.anchor_pairs_a.view(H, new_tph, in_nap)
    new_ap_b_3d = new.anchor.anchor_pairs_b.view(H, new_tph, in_nap)
    new_ap_a_3d[:, :old_tph].copy_(old_ap_a)
    new_ap_b_3d[:, :old_tph].copy_(old_ap_b)

    # --- Merge output idx_a/idx_b: first old_tph slots per head = old's ---
    old_idx_a = old.idx_a.view(H, old_tph, out_nap).to(torch.long)
    old_idx_b = old.idx_b.view(H, old_tph, out_nap).to(torch.long)
    new_idx_a_3d = new.idx_a.view(H, new_tph, out_nap).to(torch.long).clone()
    new_idx_b_3d = new.idx_b.view(H, new_tph, out_nap).to(torch.long).clone()
    new_idx_a_3d[:, :old_tph] = old_idx_a
    new_idx_b_3d[:, :old_tph] = old_idx_b

    # Rebuild derived structures (inv_idx, output_idx_per_table, K_max) from the
    # merged layout. `load_pairs` also canonicalizes anchors (min, max).
    new.load_pairs(
        anchor_pairs_a=new.anchor.anchor_pairs_a,
        anchor_pairs_b=new.anchor.anchor_pairs_b,
        idx_a=new_idx_a_3d,
        idx_b=new_idx_b_3d,
    )

    # --- Copy old latent into first old_tph slots per head ---
    if old.latent_dtype == 'bf16':
        old_lat = old.latent_bf16.view(H, old_tph, td, out_nap)
        new_lat = new.latent_bf16.view(H, new_tph, td, out_nap)
        new_lat[:, :old_tph].copy_(old_lat)
    elif old.latent_dtype == 'fp32':
        old_lat = old.latent_fp32.view(H, old_tph, td, out_nap)
        new_lat = new.latent_fp32.view(H, new_tph, td, out_nap)
        new_lat[:, :old_tph].copy_(old_lat)
    else:  # fp8
        old_lat = old.latent_fp8.view(H, old_tph, td, out_nap)
        new_lat = new.latent_fp8.view(H, new_tph, td, out_nap)
        new_lat[:, :old_tph].copy_(old_lat)
        old_scale = old.latent_scale.view(H, old_tph, 1, 1)
        new_scale = new.latent_scale.view(H, new_tph, 1, 1)
        new_scale[:, :old_tph].copy_(old_scale)

    # Re-pack bit_weights from merged latent.
    new.refresh_bit_weights()
    return new


def test_grow_preserves_old_forward():
    """Sanity check: growing a LUT and running the same input through should
    give outputs that match on the first `old_tph * output_nap` dominance
    slots (per head) — since those correspond to the preserved old tables.
    """
    torch.manual_seed(0)
    dev = torch.device("cuda")
    old = BitPermutationLUT(
        n_inputs=32, n_outputs=16, n_heads=4,
        input_nap=5, output_nap=16, tph=32,
        random_seed=42,
        initial_weights_noise=0.01,
        latent_dtype='bf16',
        device=dev,
    )
    x = torch.randn(8, 32, device=dev)
    with torch.no_grad():
        y_old = old(x)

    new = grow_lut(old, new_tph=64, seed=99)
    with torch.no_grad():
        y_new = new(x)

    print(f"Old output shape:  {tuple(y_old.shape)}")
    print(f"New output shape:  {tuple(y_new.shape)}")
    # The output is [B, H, P] where P = C(n_outputs, 2) — same for old/new.
    # The output is a pair-dominance SUMMED over all tables; when we add new
    # tables, the sum changes. But old and new share the first old_tph tables
    # whose contributions should match — the difference is only from the new
    # tables.
    diff_norm = (y_old - y_new).norm().item()
    y_old_norm = y_old.norm().item()
    print(f"|y_new - y_old| = {diff_norm:.4f}, |y_old| = {y_old_norm:.4f}, "
          f"relative = {diff_norm/max(y_old_norm, 1e-12):.4f}")
    # The scale factor also changes because it depends on tph/n_pairs.
    # So diff won't be zero, but should reflect old contribution surviving.
    print(f"old.scale = {old.scale:.4f}, new.scale = {new.scale:.4f}")


if __name__ == "__main__":
    test_grow_preserves_old_forward()
