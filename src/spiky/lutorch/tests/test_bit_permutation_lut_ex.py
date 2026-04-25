"""Tests for BitPermutationLUTEx (Part1 via fused kernel + Part2 WTA+scatter)."""
import pytest
import torch
import torch.nn as nn

from spiky.lutorch.bit_permutation_lut_ex import (
    BitPermutationLUTEx,
    BitPermutationLUTVoting,
    _build_entry_patterns,
    _build_routing_and_inv_idx,
)
from spiky.lutorch.bit_permutation_lut_optimizer import BitPermutationLUTOptimizer


CUDA_MARK = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make(**overrides):
    kw = dict(
        n_inputs=16, n_outputs=8, n_heads=2,
        input_nap=4, input_tph=8,
        voting_nap=16, output_nap=4, output_tph=6,
        random_seed=0, latent_dtype='bf16', soft_backward=True,
        device='cuda',
    )
    kw.update(overrides)
    return BitPermutationLUTEx(**kw).cuda()


# ---------- shape / structure ----------

@CUDA_MARK
def test_output_shape():
    lut = _make()
    x = torch.randn(4, 16, device='cuda')
    out = lut(x)
    assert out.shape == (4, 2, 28)   # P = 8*7/2 = 28
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


@CUDA_MARK
def test_voting_submodule_output_shape():
    """Part 1 alone produces [B, H, V_ex]."""
    lut = _make(output_nap=4, output_tph=6)
    V_ex = 6 * (1 << 4)                              # 96
    x = torch.randn(3, 16, device='cuda')
    votes = lut.voting(x)
    assert votes.shape == (3, 2, V_ex)


@CUDA_MARK
def test_entry_patterns_binary_encoding():
    pat = _build_entry_patterns(5, torch.device('cuda'))
    assert pat.shape == (32, 5)
    assert ((pat == 1) | (pat == -1)).all()
    assert (pat[0] == -1).all()                      # entry 0 → all 0 bits
    assert (pat[-1] == 1).all()                      # entry 31 → all 1 bits


@CUDA_MARK
def test_output_pair_indices_distinct_per_table():
    lut = _make(output_nap=5, output_tph=4)
    P = lut.n_pairs
    for h in range(lut.n_heads):
        for t in range(lut.output_tph):
            vals = lut.output_pair_indices[h, t]
            assert len(set(vals.tolist())) == lut.output_nap_out
            assert ((vals >= 0) & (vals < P)).all()


@CUDA_MARK
def test_inv_idx_routing_consistency():
    """For every (h, slot_idx) in inv_idx[v], output_idx_per_table at slot must equal v."""
    lut = _make()
    voting_nap = lut.voting.output_nap
    input_tph = lut.voting.tph
    for h in range(lut.n_heads):
        for v in range(lut.V_ex):
            for entry in lut.voting.inv_idx[h, v]:
                s = int(entry.item())
                if s < 0:
                    break
                t = s // voting_nap
                slot = s % voting_nap
                assert lut.voting.output_idx_per_table[h, t, slot].item() == v


# ---------- gradient flow ----------

@CUDA_MARK
def test_x_gradient_flows():
    """x-gradient flows via carriers from the fused kernel."""
    lut = _make()
    x = torch.randn(4, 16, device='cuda', requires_grad=True)
    out = lut(x)
    out.sum().backward()
    assert x.grad is not None and (x.grad.abs() > 0).any()


@CUDA_MARK
def test_optimizer_trains_latent_via_voting_submodule():
    """BitPermutationLUTOptimizer hooks `lut.voting` and trains the latent
    through the SAME projection path it uses for BitPermutationLUT — no code
    changes in the optimizer."""
    torch.manual_seed(0)
    lut = _make()
    init_latent = lut.voting.latent_bf16.detach().clone()
    init_bits = lut.voting.bit_weights.detach().clone()

    opt = BitPermutationLUTOptimizer([lut.voting], lr=1e-1)
    x = torch.randn(32, 16, device='cuda', requires_grad=True)
    for _ in range(10):
        opt.zero_grad()
        loss = lut(x).pow(2).mean()
        loss.backward()
        opt.step()
    opt.close()

    assert not torch.equal(lut.voting.latent_bf16, init_latent)
    # After enough lr * steps at small init_std, some bit_weights must flip.
    assert not torch.equal(lut.voting.bit_weights, init_bits)


@CUDA_MARK
def test_end_to_end_toy_training():
    """End-to-end: Ex + readout trained with BitPermutationLUTOptimizer on a
    non-overfittable task. Loss decreases meaningfully."""
    torch.manual_seed(7)
    lut = _make(
        n_inputs=8, n_outputs=6, n_heads=2,
        input_nap=3, input_tph=8,
        voting_nap=16, output_nap=3, output_tph=8,
    )
    readout = nn.Linear(lut.n_heads * lut.n_pairs, 1).cuda()

    opt_bit = BitPermutationLUTOptimizer([lut.voting], lr=1e-3)
    opt_adam = torch.optim.Adam(readout.parameters(), lr=1e-3)

    # Large dataset so readout can't memorise.
    x_data = torch.randn(2048, 8, device='cuda')
    w_fixed = torch.randn(8, 8, device='cuda')
    y_data = torch.tanh((x_data @ w_fixed).sum(dim=-1, keepdim=True))

    losses = []
    for _ in range(300):
        idx = torch.randint(0, 2048, (64,), device='cuda')
        xb, yb = x_data[idx], y_data[idx]
        opt_bit.zero_grad(); opt_adam.zero_grad()
        feat = lut(xb).reshape(64, -1)
        loss = ((readout(feat) - yb) ** 2).mean()
        loss.backward()
        opt_adam.step(); opt_bit.step()
        losses.append(loss.item())
    opt_bit.close()

    # Loss should drop substantially over the run.
    first_20 = sum(losses[:20]) / 20
    last_20 = sum(losses[-20:]) / 20
    assert last_20 < first_20 * 0.85, f"loss barely moved: {first_20:.4f} → {last_20:.4f}"


# ---------- WTA variants ----------

@CUDA_MARK
@pytest.mark.parametrize("n_alt", [1, 2, 3])
def test_wta_n_alternatives(n_alt):
    lut = _make(wta_n_alternatives=n_alt)
    x = torch.randn(3, 16, device='cuda', requires_grad=True)
    out = lut(x)
    out.sum().backward()
    assert x.grad is not None


# ---------- helpers ----------

@CUDA_MARK
def test_build_routing_and_inv_idx_round_trip():
    """For each slot that routes to v, inv_idx[h, v] contains its slot_idx."""
    H, tph, voting_nap, V = 3, 7, 5, 40
    pair_idx, inv_idx, K_max = _build_routing_and_inv_idx(
        H, tph, voting_nap, V, random_seed=42, device=torch.device('cuda'),
    )
    assert pair_idx.shape == (H, tph, voting_nap)
    assert inv_idx.shape == (H, V, K_max)
    # Reverse map must be consistent.
    for h in range(H):
        for t in range(tph):
            for slot in range(voting_nap):
                v = int(pair_idx[h, t, slot].item())
                slot_idx = t * voting_nap + slot
                assert slot_idx in inv_idx[h, v].tolist(), \
                    f"missing (t={t}, slot={slot}) → v={v}"
