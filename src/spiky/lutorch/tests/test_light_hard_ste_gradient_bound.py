"""Regression test for the LightMultiHeadLUT hard-forward STE gradient bound.

Fix for the Gen-3 hard-mode divergence (ablation rows 3.3 n=1 / 3.4 n=2): in `_hard_read`
the straight-through surrogate `f`'s per-table confidence score is rescaled by a DETACHED
per-token(-per-head) mean over the tables that reduce into each bag,

    score = score / (score.detach().mean(dim=-1, keepdim=True) + 1e-6)

so the gradient the STE injects into the input, the confidence scalars and tau can no longer
grow with activation magnitude (the learned_margin `sum_j|d_j|` term), while the RELATIVE
per-table confidence is preserved. The change is BACKWARD-ONLY: the forward value stays the
pure hard read `plain` (`plain + (f - f.detach())` cancels in value).

Asserts:
  (a) train forward value == eval value == the plain hard read (STE cancels), n=1 and n=2;
  (b) the input-gradient norm does NOT grow when activations are scaled up (1x/4x/16x) --
      the property the fix restores; without it the margin score amplifies it ~linearly;
  (c) beta and gamma still receive a finite, nonzero, bounded gradient (learning preserved).
"""
import pytest
import torch

torch._dynamo.config.suppress_errors = True

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

NAP, TPH, H, IN, OUT, B = 4, 3, 2, 8, 5, 6


def _make(n=2, dtype=torch.float64):
    m = LightMultiHeadLUT(
        input_dim=IN, n_tables=H * TPH, output_dim=OUT, n_anchor_pairs=NAP,
        confidence_form="learned_margin", learned_margin_init=(0.0, 2.0, 1.0),
        learned_margin_freeze_g=True, random_seed=0, initial_weights_noise=0.5,
        device=torch.device("cpu"), n_heads=H, multi_head_input=False,
        read_top_n=n, read_tau=0.5, read_tau_learnable=(n > 1), forward_mode="hard").to(dtype)
    m._compile_enabled = False   # eager, deterministic for the value check
    return m


def _x(seed=0, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, IN, generator=g, dtype=dtype)


@pytest.mark.parametrize("n", [1, 2])
def test_forward_value_is_the_plain_hard_read(n):
    """(a) backward-only fix: the forward value is unchanged (== eval == plain)."""
    m = _make(n=n)
    x = _x(0)
    m.eval()
    with torch.no_grad():
        eval_val = m(x)
    m.train()
    train_val = m(x.clone().requires_grad_(True))
    assert torch.allclose(train_val, eval_val, atol=1e-12, rtol=0)


@pytest.mark.parametrize("n", [1, 2])
def test_input_gradient_bounded_under_activation_scaling(n):
    """(b) the fix's purpose: input gradient must not explode as activations grow.

    sign(d) is preserved under positive scaling, so the hard address (and `plain`) is the
    same at every scale; the gradient flows only through the surrogate `f`. With the detached
    mean-rescale it stays bounded (empirically shrinks); the unfixed code grew it ~linearly."""
    m = _make(n=n)
    x0 = _x(1)

    def gnorm(scale):
        m.zero_grad(set_to_none=True)
        xx = (x0 * scale).clone().requires_grad_(True)
        m(xx).sum().backward()
        return xx.grad.norm().item()

    g1, g4, g16 = gnorm(1.0), gnorm(4.0), gnorm(16.0)
    assert g1 > 0.0
    # generous slack: the point is "does not explode", not the exact ratio.
    assert g4 <= 1.5 * g1 + 1e-9, f"input grad grew under 4x activations: {g1:.4g} -> {g4:.4g}"
    assert g16 <= 1.5 * g1 + 1e-9, f"input grad grew under 16x activations: {g1:.4g} -> {g16:.4g}"


def test_beta_gamma_still_get_bounded_gradient():
    """(c) learning preserved: beta/gamma get a finite, nonzero, bounded gradient."""
    m = _make(n=2)
    x = _x(2).requires_grad_(True)
    m(x).sum().backward()
    for name, t in (("log_beta", m.confidence_log_beta.grad),
                    ("log_gamma", m.confidence_log_gamma.grad)):
        assert t is not None and torch.isfinite(t).all(), f"{name} gradient missing/non-finite"
        assert float(t.abs()) > 0.0, f"{name} gradient is exactly zero (no learning signal)"
        assert float(t.abs()) < 1e3, f"{name} gradient is unbounded: {float(t.abs()):.4g}"
