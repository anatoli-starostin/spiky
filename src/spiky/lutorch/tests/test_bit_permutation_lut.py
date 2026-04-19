"""Tests for BitPermutationLUT.

Two correctness tests:
  1. CUDA kernel vs a vectorized PyTorch reference that uses the exact same
     inputs (lookup_indices, decoded ±1 signs, inv_idx). Isolates the kernel.
  2. BitPermutationLUT vs a matched PermutationalLut (pair_mode='scrambled',
     soft_mode='ste', return_dominance=True, CANONICAL_DISTINCT policies),
     with identical anchor pairs, output pair indices, and ±1 weights.
"""
import math

import pytest
import torch

from spiky.lutorch.bit_permutation_lut import BitPermutationLUT
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
from spiky.lutorch.permutational_lut import PermutationalLut


CUDA_ONLY = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-only")


def _reference_bit_dominance(
    lookup_indices: torch.Tensor,      # int16 [B, n_heads*tph]
    bit_signs: torch.Tensor,            # float ±1  [n_heads*tph, table_dim, output_nap]
    idx_a: torch.Tensor,                # long [n_heads, tph, output_nap]
    idx_b: torch.Tensor,
    n_outputs: int,
    scale: float,
) -> torch.Tensor:
    """Pure-PyTorch reference of the bit dominance kernel (vectorized).

    CANONICAL_DISTINCT ⇒ idx_a < idx_b everywhere and per-slot sign is +1.
    """
    B, N = lookup_indices.shape
    n_heads, tph, output_nap = idx_a.shape
    assert N == n_heads * tph
    P = n_outputs * (n_outputs - 1) // 2

    # Canonical pair index per (h, t, k).
    pair_map = torch.full((n_outputs, n_outputs), -1, dtype=torch.long, device=idx_a.device)
    tri_i, tri_j = torch.triu_indices(n_outputs, n_outputs, offset=1)
    pair_map[tri_i, tri_j] = torch.arange(P, device=idx_a.device)
    pair_map[tri_j, tri_i] = torch.arange(P, device=idx_a.device)
    pair_idx = pair_map[idx_a, idx_b]  # [n_heads, tph, output_nap] long

    # Per-slot ±1 vote for every (b, table_global, k).
    #   entries[b, t] = lookup_indices[b, t]
    #   votes[b, t, k] = bit_signs[t, entries[b, t], k]
    entries_long = lookup_indices.long()  # [B, N]
    # Gather table slice per batch.
    # bit_signs: [N, table_dim, output_nap]
    gather_idx = entries_long.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, output_nap)
    votes = bit_signs.unsqueeze(0).expand(B, -1, -1, -1).gather(2, gather_idx).squeeze(2)
    # votes: [B, N, output_nap]

    votes_hp = votes.view(B, n_heads, tph * output_nap)
    idx_flat = pair_idx.reshape(n_heads, tph * output_nap)
    out = torch.zeros(B, n_heads, P, device=votes.device, dtype=votes.dtype)
    out.scatter_add_(2, idx_flat.unsqueeze(0).expand(B, -1, -1), votes_hp)
    return out * scale


@CUDA_ONLY
@pytest.mark.parametrize(
    "n_inputs,n_outputs,n_heads,input_nap,output_nap,tph,B",
    [
        (16,  8, 2, 4,  5, 4, 3),     # output_nap < 32, single block
        (16, 10, 2, 4,  6, 3, 5),
        (32, 12, 2, 5,  8, 6, 4),
        (32, 16, 4, 6, 12, 8, 2),
        (32, 20, 1, 6, 10, 5, 7),
        (32, 24, 2, 6, 33, 4, 2),     # output_nap > 32 → 2 blocks
    ],
)
def test_bit_kernel_matches_pytorch_reference(
    n_inputs, n_outputs, n_heads, input_nap, output_nap, tph, B
):
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    lut = BitPermutationLUT(
        n_inputs=n_inputs, n_outputs=n_outputs,
        n_heads=n_heads, input_nap=input_nap, output_nap=output_nap, tph=tph,
        random_seed=42, device=dev,
    )
    x = torch.randn(B, n_inputs, device=dev)
    out = lut(x)

    # Reference: decode bit weights → ±1 signs, reuse lookup_indices from anchor,
    # reuse idx_a/idx_b from BitPermLut (shape [H, tph, output_nap]).
    with torch.no_grad():
        lookup_indices, _, _, _, _ = lut.anchor(x)
        signs = lut.get_bit_weights_as_signs()  # [N, table_dim, output_nap]
        ref = _reference_bit_dominance(
            lookup_indices, signs, lut.idx_a, lut.idx_b, n_outputs, lut.scale,
        )

    assert out.shape == ref.shape
    # Exact equality expected: both sum identical ±1 counts * same float scale.
    diff = (out - ref).abs().max().item()
    assert diff < 1e-5, (
        f"BitPermLUT kernel vs PyTorch reference: max |diff| = {diff}; "
        f"shapes={out.shape}, scale={lut.scale}"
    )


@CUDA_ONLY
@pytest.mark.parametrize(
    "n_inputs,n_outputs,n_heads,input_nap,output_nap,tph,B",
    [
        (16,  8, 2, 4,  5, 4, 3),
        (32, 12, 2, 5,  8, 6, 4),
        (32, 16, 4, 6, 12, 4, 2),
    ],
)
def test_bit_matches_permutational_lut_with_sign_weights(
    n_inputs, n_outputs, n_heads, input_nap, output_nap, tph, B
):
    """BitPermLut ≡ PermLut(soft_mode='ste', return_dominance=True) when weights are ±1
    and both share the same anchor pairs + output pair assignments."""
    dev = torch.device("cuda:0")
    torch.manual_seed(0)
    bit_lut = BitPermutationLUT(
        n_inputs=n_inputs, n_outputs=n_outputs,
        n_heads=n_heads, input_nap=input_nap, output_nap=output_nap, tph=tph,
        random_seed=42, device=dev,
    )
    signs = bit_lut.get_bit_weights_as_signs()  # ±1 float [N, table_dim, output_nap]

    # Matched PermLut: same policies (CANONICAL_DISTINCT), STE hard vote, dominance return.
    perm_lut = PermutationalLut(
        n_inputs=n_inputs, n_outputs=n_outputs,
        input_nap=input_nap, output_nap=output_nap,
        n_heads=n_heads, tph=tph,
        pair_mode='scrambled', soft_mode='ste', return_dominance=True,
        scrambled_policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
        input_anchor_policy=AnchorSamplingPolicy.CANONICAL_DISTINCT,
        random_seed=42, device=dev,
    ).to(dev)
    # Anchor seeds already match (same random_seed path); output seeds differ
    # (BitPermLut uses +2_000_003, PermLut scrambled uses +1_000_003). Override
    # PermLut's output-pair buffers and derived dominance state to match BitPermLut.
    assert torch.equal(
        perm_lut.inner.lookup.anchor_pairs_a.to(torch.int64),
        bit_lut.anchor.anchor_pairs_a.to(torch.int64),
    )
    assert torch.equal(
        perm_lut.inner.lookup.anchor_pairs_b.to(torch.int64),
        bit_lut.anchor.anchor_pairs_b.to(torch.int64),
    )
    _rewire_perm_lut_outputs_to_bitlut(perm_lut, bit_lut, n_heads, tph, output_nap, n_outputs)

    # Inject ±1 weights: PermLut inner weights shape is [n_heads*tph, table_dim, output_nap]
    # — identical to signs — so we copy directly.
    assert perm_lut.inner.projection.weights.shape == signs.shape
    with torch.no_grad():
        perm_lut.inner.projection.weights.data.copy_(signs)

    x = torch.randn(B, n_inputs, device=dev)
    perm_lut.eval()
    bit_out = bit_lut(x)
    perm_out = perm_lut(x)
    assert bit_out.shape == perm_out.shape, f"{bit_out.shape} vs {perm_out.shape}"
    diff = (bit_out - perm_out).abs().max().item()
    # Both apply the same scale (0.5/sqrt(n_votes_per_pair)) over summed ±1 votes.
    assert diff < 1e-4, f"bit vs perm dominance max |diff| = {diff}"


def _rewire_perm_lut_outputs_to_bitlut(perm_lut, bit_lut, n_heads, tph, output_nap, n_outputs):
    """Overwrite `perm_lut`'s output-pair state (idx_a/b + dom buffers) so that
    it mirrors `bit_lut`'s output-pair assignments."""
    dev = perm_lut.idx_a.device
    TP = tph * output_nap
    idx_a_hk = bit_lut.idx_a.reshape(n_heads, TP).to(torch.long).to(dev)
    idx_b_hk = bit_lut.idx_b.reshape(n_heads, TP).to(torch.long).to(dev)
    perm_lut.idx_a = idx_a_hk.contiguous()
    perm_lut.idx_b = idx_b_hk.contiguous()

    # proj_matrix (only present for aggregation='matmul'): rebuild since we
    # changed idx_a/idx_b. If absent, skip.
    if hasattr(perm_lut, 'proj_matrix'):
        M = torch.zeros(n_heads, TP, n_outputs, device=dev)
        h_idx = torch.arange(n_heads, device=dev).view(n_heads, 1).expand(n_heads, TP)
        tp_idx = torch.arange(TP, device=dev).view(1, TP).expand(n_heads, TP)
        M[h_idx, tp_idx, idx_a_hk] = 1.0
        M[h_idx, tp_idx, idx_b_hk] = -1.0
        perm_lut.proj_matrix = M.contiguous()

    # Dominance buffers.
    P = n_outputs * (n_outputs - 1) // 2
    perm_lut._dom_n_pairs = P
    pair_map = torch.full((n_outputs, n_outputs), -1, dtype=torch.long)
    tri_i, tri_j = torch.triu_indices(n_outputs, n_outputs, offset=1)
    pair_map[tri_i, tri_j] = torch.arange(P)
    pair_map[tri_j, tri_i] = torch.arange(P)
    pair_idx = pair_map[idx_a_hk.cpu(), idx_b_hk.cpu()].to(dev).long()
    sign = torch.where(idx_a_hk < idx_b_hk, 1.0, -1.0).float().to(dev)
    perm_lut.dom_pair_idx = pair_idx.contiguous()
    perm_lut.dom_sign = sign.contiguous()

    # Rebuild inverse index + inv_sign.
    counts = torch.zeros(n_heads, P, dtype=torch.long)
    pair_idx_cpu = pair_idx.cpu()
    sign_cpu = sign.cpu()
    for h in range(n_heads):
        for slot in range(TP):
            counts[h, int(pair_idx_cpu[h, slot].item())] += 1
    K_max = int(counts.max().item())
    inv_idx_cpu = torch.full((n_heads, P, K_max), -1, dtype=torch.long)
    inv_sign_cpu = torch.zeros(n_heads, P, K_max)
    cursor = torch.zeros(n_heads, P, dtype=torch.long)
    for h in range(n_heads):
        for slot in range(TP):
            p = int(pair_idx_cpu[h, slot].item())
            k = int(cursor[h, p].item())
            inv_idx_cpu[h, p, k] = slot
            inv_sign_cpu[h, p, k] = float(sign_cpu[h, slot].item())
            cursor[h, p] += 1
    perm_lut.dom_inv_idx = inv_idx_cpu.to(dev).contiguous()
    perm_lut.dom_inv_sign = inv_sign_cpu.to(dev).float().contiguous()


@CUDA_ONLY
def test_forward_is_integer_sum_times_scale():
    """Every value in the output must be k * scale for some integer k in [-K_max, K_max]."""
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=2,
        input_nap=4, output_nap=5, tph=4,
        random_seed=7, device=dev,
    )
    x = torch.randn(4, 16, device=dev)
    out = lut(x)
    quotient = out / lut.scale
    nearest_int = quotient.round()
    assert torch.allclose(quotient, nearest_int, atol=1e-4), \
        f"Output not integer multiple of scale: max |frac| = {(quotient - nearest_int).abs().max().item()}"
    assert nearest_int.abs().max().item() <= lut.K_max


@CUDA_ONLY
def test_set_get_bit_weights_roundtrip():
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=16, n_outputs=12, n_heads=2,
        input_nap=4, output_nap=33, tph=3,        # straddle 32-bit block boundary
        random_seed=0, device=dev,
    )
    signs = torch.where(
        torch.randn(2 * 3, 16, 33, device=dev) > 0, 1.0, -1.0,
    )
    lut.set_bit_weights_from_signs(signs)
    round_tripped = lut.get_bit_weights_as_signs()
    assert torch.equal(signs, round_tripped)


@CUDA_ONLY
def test_forward_is_deterministic():
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=32, n_outputs=16, n_heads=4,
        input_nap=5, output_nap=8, tph=4,
        random_seed=3, device=dev,
    )
    x = torch.randn(8, 32, device=dev)
    o1 = lut(x)
    o2 = lut(x)
    assert torch.equal(o1, o2)


def _reference_carrier_grads(
    grad_out: torch.Tensor,            # [B, H, P]
    lookup_indices: torch.Tensor,      # int16 [B, N]
    lookup_alt_indices: torch.Tensor,  # int16 [B, N, 1]
    signs: torch.Tensor,                # ±1 float [N, table_dim, output_nap]
    pair_idx_per_slot: torch.Tensor,   # long [H, tph, output_nap]
    scale: float,
):
    B, H, _ = grad_out.shape
    _, tph, output_nap = pair_idx_per_slot.shape
    N = H * tph
    # Per-slot gradient = scale * grad_out[b, h, pair_of(t, k)].
    pair_idx_flat = pair_idx_per_slot.reshape(H, tph * output_nap)
    grad_per_slot = grad_out.gather(2, pair_idx_flat.unsqueeze(0).expand(B, -1, -1)) * scale
    grad_per_slot = grad_per_slot.reshape(B, N, output_nap)
    # Gather ±1 signs for entry_main / entry_alt.
    entries_main = lookup_indices.long()
    entries_alt = lookup_alt_indices.squeeze(-1).long()
    gm = entries_main.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, output_nap)
    ga = entries_alt.unsqueeze(-1).unsqueeze(-1).expand(B, N, 1, output_nap)
    signs_main = signs.unsqueeze(0).expand(B, -1, -1, -1).gather(2, gm).squeeze(2)
    signs_alt = signs.unsqueeze(0).expand(B, -1, -1, -1).gather(2, ga).squeeze(2)
    grad_main = (grad_per_slot * signs_main).sum(dim=-1)
    grad_alt = (grad_per_slot * signs_alt).sum(dim=-1, keepdim=True)
    return grad_main.contiguous(), grad_alt.contiguous()


@CUDA_ONLY
@pytest.mark.parametrize(
    "n_inputs,n_outputs,n_heads,input_nap,output_nap,tph,B",
    [
        (16,  8, 2, 4,  5, 4, 3),
        (32, 12, 2, 5,  8, 6, 4),
        (32, 16, 4, 6, 12, 4, 2),
        (32, 24, 2, 6, 33, 4, 2),   # > 32-bit block
    ],
)
def test_backward_kernel_matches_pytorch_reference(
    n_inputs, n_outputs, n_heads, input_nap, output_nap, tph, B
):
    """CUDA backward kernel: the (grad_main, grad_alt) carrier projection must
    match the vectorized PyTorch reference exactly."""
    dev = torch.device("cuda:0")
    from lutorch_cuda import get_lutorch_manager
    native = get_lutorch_manager()

    lut = BitPermutationLUT(
        n_inputs=n_inputs, n_outputs=n_outputs,
        n_heads=n_heads, input_nap=input_nap, output_nap=output_nap, tph=tph,
        random_seed=7, device=dev,
    )
    x = torch.randn(B, n_inputs, device=dev)
    lookup_indices, lookup_alt_indices, _, _, _ = lut.anchor(x)

    P = n_outputs * (n_outputs - 1) // 2
    grad_out = torch.randn(B, n_heads, P, device=dev)

    # Hard STE (default): uses ±1 from bit_weights.
    grad_main_n, grad_alt_n = native.bit_perm_lut_dom_gather_backward(
        grad_out.contiguous(), lookup_indices, lookup_alt_indices,
        lut.bit_weights, lut.pair_idx_per_slot,
        int(n_heads), int(tph), int(output_nap), int(P), float(lut.scale),
    )
    signs = lut.get_bit_weights_as_signs()
    grad_main_p, grad_alt_p = _reference_carrier_grads(
        grad_out, lookup_indices, lookup_alt_indices, signs,
        lut.pair_idx_per_slot.long(), lut.scale,
    )
    assert torch.allclose(grad_main_n, grad_main_p, atol=1e-5, rtol=1e-5), \
        f"grad_main max|diff| = {(grad_main_n - grad_main_p).abs().max().item()}"
    assert torch.allclose(grad_alt_n, grad_alt_p, atol=1e-5, rtol=1e-5), \
        f"grad_alt  max|diff| = {(grad_alt_n - grad_alt_p).abs().max().item()}"


@CUDA_ONLY
def test_backward_latent_kernel_matches_reference():
    """Opt-in STE-soft backward kernel: uses dequantized fp8 latent in [-1, 1]
    instead of ±1, and matches a PyTorch reference built the same way."""
    dev = torch.device("cuda:0")
    from lutorch_cuda import get_lutorch_manager
    native = get_lutorch_manager()

    lut = BitPermutationLUT(
        n_inputs=32, n_outputs=12, n_heads=2, input_nap=5, output_nap=8, tph=4,
        random_seed=11, device=dev,
    )
    B, n_outputs = 3, 12
    x = torch.randn(B, 32, device=dev)
    li, lai, _, _, _ = lut.anchor(x)
    P = n_outputs * (n_outputs - 1) // 2
    grad_out = torch.randn(B, lut.n_heads, P, device=dev)

    grad_main_n, grad_alt_n = native.bit_perm_lut_dom_gather_backward_latent(
        grad_out.contiguous(), li, lai, lut.latent_fp8, lut.latent_scale,
        lut.pair_idx_per_slot,
        int(lut.n_heads), int(lut.tph), int(lut.output_nap), int(P), float(lut.scale),
    )
    latent_f32 = lut.latent_fp8.to(torch.float32) / lut.latent_scale
    grad_main_p, grad_alt_p = _reference_carrier_grads(
        grad_out, li, lai, latent_f32, lut.pair_idx_per_slot.long(), lut.scale,
    )
    assert torch.allclose(grad_main_n, grad_main_p, atol=1e-5, rtol=1e-5)
    assert torch.allclose(grad_alt_n, grad_alt_p, atol=1e-5, rtol=1e-5)


@CUDA_ONLY
def test_backward_soft_flag_routes_to_latent_kernel():
    """BitPermutationLUT(soft_backward=True) routes backward through the latent
    kernel. We force latents to an intermediate magnitude (±0.5) so the soft
    path's gradient demonstrably differs from the hard (±1) path."""
    dev = torch.device("cuda:0")
    kwargs = dict(
        n_inputs=32, n_outputs=12, n_heads=2, input_nap=5, output_nap=8, tph=4,
        random_seed=0, device=dev,
    )
    from spiky.lutorch.bit_permutation_lut_optimizer import _to_fp8_per_table

    lut_hard = BitPermutationLUT(soft_backward=False, **kwargs)
    lut_soft = BitPermutationLUT(soft_backward=True, **kwargs)
    # Align both to the same bit pattern and force latents to ±0.5 (mid range).
    signs = lut_hard.get_bit_weights_as_signs()
    half_latent = (signs * 0.5).contiguous()
    fp8, scale = _to_fp8_per_table(half_latent)
    for lut in (lut_hard, lut_soft):
        lut.latent_fp8 = fp8.clone()
        lut.latent_scale = scale.clone()
        lut.set_bit_weights_from_signs(signs)
    assert torch.equal(lut_hard.bit_weights, lut_soft.bit_weights)

    x = torch.randn(4, 32, device=dev, requires_grad=True)
    grad_out = torch.randn(4, lut_hard.n_heads, lut_hard.n_pairs, device=dev)

    out_hard = lut_hard(x)
    out_hard.backward(grad_out)
    g_hard = x.grad.clone()
    x.grad = None

    out_soft = lut_soft(x)
    out_soft.backward(grad_out)
    g_soft = x.grad.clone()

    # Forward output matches (bits are identical).
    assert torch.equal(out_hard, out_soft)
    # Soft gradient is ~0.5 × hard gradient at this latent magnitude.
    assert not torch.allclose(g_hard, g_soft)
    ratio = (g_soft[g_hard.abs() > 1e-6] / g_hard[g_hard.abs() > 1e-6])
    assert (ratio.abs() - 0.5).abs().max().item() < 0.2, \
        f"soft/hard ratio not near 0.5: {ratio.abs().mean().item():.3f}"


@CUDA_ONLY
def test_backward_x_grad_flows():
    """End-to-end: x.grad is finite, non-zero, and has no nn.Parameter gradients."""
    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=16, n_outputs=8, n_heads=2,
        input_nap=4, output_nap=5, tph=4,
        random_seed=0, device=dev,
    )
    assert len(list(lut.parameters())) == 0
    x = torch.randn(3, 16, device=dev, requires_grad=True)
    out = lut(x)
    out.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all().item()
    assert x.grad.abs().sum().item() > 0


@CUDA_ONLY
def test_backward_end_to_end_matches_manual_tiny_apl_route():
    """BitPermLut x.grad ≡ (reference grad_main/grad_alt) → TinyAPL PyTorch backward.
    Verifies the full pipeline against a reference reconstructed from primitives."""
    from spiky.lutorch.tiny_anchor_pairs_lookup import _tiny_backward_pytorch

    dev = torch.device("cuda:0")
    lut = BitPermutationLUT(
        n_inputs=32, n_outputs=12, n_heads=2,
        input_nap=5, output_nap=8, tph=4,
        random_seed=11, device=dev,
    )
    lut.train()
    B = 4
    x = torch.randn(B, 32, device=dev, requires_grad=True)
    out = lut(x)
    P = 12 * 11 // 2
    grad_out = torch.randn_like(out)
    out.backward(grad_out)

    # Manual reconstruction of x.grad.
    x2 = x.detach().clone()
    lookup_indices, lookup_alt_indices, alt_deltas, _, _ = lut.anchor(x2)
    a1 = lut.anchor.anchor_pairs_a[
        torch.arange(lut.n_heads * lut.tph, device=dev), 0
    ]  # placeholder; properly recomputed below
    # Default backward is hard ±1 from bit_weights.
    signs = lut.get_bit_weights_as_signs()
    grad_main, grad_alt = _reference_carrier_grads(
        grad_out, lookup_indices, lookup_alt_indices, signs,
        lut.pair_idx_per_slot.long(), lut.scale,
    )

    # Feed grad_main, grad_alt through TinyAnchorPairsLookup's PyTorch backward.
    # We need the actual anchor1/anchor2 ids (the pair that was chosen as alt).
    # Recompute via _tiny_forward_pytorch for reference.
    from spiky.lutorch.tiny_anchor_pairs_lookup import _tiny_forward_pytorch
    _, _, _, a1, a2 = _tiny_forward_pytorch(
        x2, lut.anchor.anchor_pairs_a, lut.anchor.anchor_pairs_b, lut.anchor.powers,
    )
    N = lut.n_heads * lut.tph
    batch_offset = (
        torch.arange(B, device=dev, dtype=torch.int32).repeat_interleave(N) * 32
    ).contiguous()
    x_grad_flat = _tiny_backward_pytorch(
        x2, a1, a2, alt_deltas, batch_offset, grad_main, grad_alt,
    )
    x_grad_expected = x_grad_flat.view(x2.shape)
    assert torch.allclose(x.grad, x_grad_expected, atol=1e-5, rtol=1e-5), \
        f"x.grad mismatch: max|diff| = {(x.grad - x_grad_expected).abs().max().item()}"
