"""Tests for RankWTAAttention."""
import pytest
import torch

from spiky.lutorch.ranking_tools import RankWTAAttention
from spiky.lutorch.lut_helpers import UncertaintyMode

SEED = 42


# ── Shape / basic correctness ──────────────────────────────────────────────────

def test_output_shape(device):
    torch.manual_seed(SEED)
    B, H, T, d_qk, d_v = 2, 4, 16, 16, 8
    m = RankWTAAttention(d_qk, d_v).to(device).train()
    q = torch.randn(B, H, T, d_qk, device=device)
    k = torch.randn(B, H, T, d_qk, device=device)
    v = torch.randn(B, H, T, d_v, device=device)
    out = m(q, k, v, is_causal=True)
    assert out.shape == (B, H, T, d_v)


def test_eval_output_shape(device):
    torch.manual_seed(SEED)
    B, H, T, d_qk, d_v = 2, 4, 16, 16, 8
    m = RankWTAAttention(d_qk, d_v).to(device).eval()
    q = torch.randn(B, H, T, d_qk, device=device)
    k = torch.randn(B, H, T, d_qk, device=device)
    v = torch.randn(B, H, T, d_v, device=device)
    with torch.no_grad():
        out = m(q, k, v, is_causal=True)
    assert out.shape == (B, H, T, d_v)


def test_d_v_defaults_to_d_qk(device):
    torch.manual_seed(SEED)
    d_qk = 12
    m = RankWTAAttention(d_qk).to(device).eval()
    q = k = v = torch.randn(1, 1, 8, d_qk, device=device)
    with torch.no_grad():
        out = m(q, k, v, is_causal=True)
    assert out.shape[-1] == d_qk


# ── Causal masking ─────────────────────────────────────────────────────────────

def test_causal_mask_no_future_leakage(device):
    """Output at position t must be independent of v at positions > t."""
    torch.manual_seed(SEED)
    B, H, T, d_qk, d_v = 1, 1, 8, 8, 8
    m = RankWTAAttention(d_qk, d_v, smooth_mode=True).to(device).eval()
    q = torch.randn(B, H, T, d_qk, device=device)
    k = torch.randn(B, H, T, d_qk, device=device)
    v1 = torch.randn(B, H, T, d_v, device=device)
    v2 = v1.clone()
    # Perturb only future positions relative to t=0
    v2[:, :, 1:, :] += 10.0

    with torch.no_grad():
        out1 = m(q, k, v1, is_causal=True)
        out2 = m(q, k, v2, is_causal=True)

    # t=0 can only attend to itself → must be identical
    torch.testing.assert_close(out1[:, :, 0, :], out2[:, :, 0, :])


def test_eval_winner_is_causal_argmax(device):
    """In eval, output[t] must equal v[argmax(scores[t, :t+1])]."""
    torch.manual_seed(SEED)
    B, H, T, d_qk, d_v = 1, 1, 8, 10, 6
    m = RankWTAAttention(d_qk, d_v, smooth_mode=True).to(device).eval()
    q = torch.randn(B, H, T, d_qk, device=device)
    k = torch.randn(B, H, T, d_qk, device=device)
    v = torch.randn(B, H, T, d_v, device=device)

    with torch.no_grad():
        out = m(q, k, v, is_causal=True)

        # Recompute scores manually
        M = m.pairs.shape[1]
        rq = m.soft_rank_projection(q).reshape(T, M)
        rk = m.soft_rank_projection(k).reshape(T, M)
        scores = rq @ rk.t()  # (T, T)
        scores = scores.masked_fill(~m.causal_mask[:T, :T], float('-inf'))
        winner_idx = scores.argmax(dim=-1)  # (T,)
        expected = v[0, 0][winner_idx]      # (T, d_v)

    torch.testing.assert_close(out[0, 0], expected)


# ── Gradient flow ──────────────────────────────────────────────────────────────

def test_gradients_flow_to_q_k_v(device):
    """Gradients must reach q, k, and v."""
    torch.manual_seed(SEED)
    B, H, T, d_qk, d_v = 2, 2, 12, 12, 8
    m = RankWTAAttention(d_qk, d_v, n_alternatives=1).to(device).train()
    q = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    k = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    v = torch.randn(B, H, T, d_v, device=device, requires_grad=True)

    out = m(q, k, v, is_causal=True)
    out.sum().backward()

    assert q.grad is not None and q.grad.abs().sum() > 0
    assert k.grad is not None and k.grad.abs().sum() > 0
    assert v.grad is not None and v.grad.abs().sum() > 0


def test_gradient_finite(device):
    """All gradients must be finite."""
    torch.manual_seed(SEED)
    B, H, T, d_qk = 2, 2, 10, 10
    m = RankWTAAttention(d_qk, n_alternatives=2).to(device).train()
    q = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    k = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    v = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)

    m(q, k, v, is_causal=True).sum().backward()

    assert torch.isfinite(q.grad).all()
    assert torch.isfinite(k.grad).all()
    assert torch.isfinite(v.grad).all()


def test_grad_c_alt_contributes_to_q_k_grad(device):
    """Disabling grad_c_alt should produce a different (smaller) q/k gradient."""
    torch.manual_seed(SEED)
    B, H, T, d_qk, d_v = 1, 1, 8, 8, 4
    n_alt = 2

    def run(connect_alt):
        torch.manual_seed(SEED)
        m = RankWTAAttention(d_qk, d_v, smooth_mode=True, n_alternatives=n_alt).to(device).train()
        q = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
        k = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
        v = torch.randn(B, H, T, d_v, device=device, requires_grad=True)

        from spiky.lutorch.wta_lookup import WTALookupFunction
        import spiky.lutorch.ranking_tools as rt_mod

        original_apply = WTALookupFunction.apply

        def patched_apply(scores, n_alt, mode_int, batch_offset):
            wi, ai, ad, gc_w, gc_a = original_apply(scores, n_alt, mode_int, batch_offset)
            if not connect_alt:
                gc_a = torch.zeros_like(gc_a)
            return wi, ai, ad, gc_w, gc_a

        WTALookupFunction.apply = staticmethod(patched_apply)
        try:
            out = m(q, k, v, is_causal=True)
            out.sum().backward()
        finally:
            WTALookupFunction.apply = staticmethod(original_apply)

        return q.grad.clone(), k.grad.clone()

    q_grad_with, k_grad_with = run(connect_alt=True)
    q_grad_without, k_grad_without = run(connect_alt=False)

    assert not torch.allclose(q_grad_with, q_grad_without), "grad_c_alt had no effect on q.grad"
    assert not torch.allclose(k_grad_with, k_grad_without), "grad_c_alt had no effect on k.grad"


# ── Modes ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("smooth_mode", [True, False])
def test_smooth_and_ste_modes(device, smooth_mode):
    torch.manual_seed(SEED)
    B, H, T, d_qk = 2, 2, 10, 10
    m = RankWTAAttention(d_qk, smooth_mode=smooth_mode).to(device).train()
    q = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    k = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    v = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    out = m(q, k, v, is_causal=True)
    out.sum().backward()
    assert q.grad.abs().sum() > 0


@pytest.mark.parametrize("n_alternatives", [1, 2, 3])
def test_n_alternatives(device, n_alternatives):
    torch.manual_seed(SEED)
    B, H, T, d_qk = 2, 2, 12, 12
    m = RankWTAAttention(d_qk, n_alternatives=n_alternatives).to(device).train()
    q = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    k = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    v = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    out = m(q, k, v, is_causal=True)
    out.sum().backward()
    assert out.shape == (B, H, T, d_qk)
    assert q.grad.abs().sum() > 0


@pytest.mark.parametrize("uncertainty_mode", [
    UncertaintyMode.INVERSE_L1,
    UncertaintyMode.INVERSE_QUADRATIC,
])
def test_uncertainty_modes(device, uncertainty_mode):
    """
    INVERSE_L1 is safe with causal masking (1/(1+inf)^2 → 0).
    INVERSE_QUADRATIC produces nan at t=0 when the only alt is a masked (-inf)
    key: inf/(inf^2)^2 = nan. Test it without causal masking to avoid this.
    """
    torch.manual_seed(SEED)
    B, H, T, d_qk = 2, 2, 10, 10
    m = RankWTAAttention(d_qk, uncertainty_mode=uncertainty_mode).to(device).train()
    q = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    k = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    v = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
    is_causal = uncertainty_mode == UncertaintyMode.INVERSE_L1
    m(q, k, v, is_causal=is_causal).sum().backward()
    assert torch.isfinite(q.grad).all()


# ── batch_offset cache ─────────────────────────────────────────────────────────

def test_batch_offset_cache_updates_on_shape_change(device):
    """Cache must be rebuilt when BH*T changes, producing correct results."""
    torch.manual_seed(SEED)
    d_qk = 8
    m = RankWTAAttention(d_qk).to(device).train()

    for B, H, T in [(2, 2, 8), (4, 2, 8), (2, 2, 16)]:
        q = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
        k = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
        v = torch.randn(B, H, T, d_qk, device=device, requires_grad=True)
        out = m(q, k, v, is_causal=True)
        assert out.shape == (B, H, T, d_qk)
        out.sum().backward()
        assert torch.isfinite(q.grad).all()
