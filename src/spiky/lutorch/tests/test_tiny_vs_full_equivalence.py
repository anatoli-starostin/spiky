"""End-to-end numerical equivalence between TinyMultiHeadLut and MultiHeadLut.

If both modules are numerically identical (anchors, gather kernel, backward
gradients), then training one vs the other from identical init weights must
produce identical losses up to floating-point noise. A consistent ~0.03 bpb
gap can't come from weight-init RNG drift alone — it would imply a
systematic difference in the gradients or forward computation.

This test pins both modules to the same anchors AND same weights, then
verifies:
  1. forward outputs bit-equivalent (already checked elsewhere).
  2. weights.grad bit-equivalent.
  3. x.grad bit-equivalent.
  4. After N SGD steps on identical synthetic data, final weights match.
"""
import pytest
import torch

from spiky.lutorch.tiny_multi_head_lut import TinyMultiHeadLut
from spiky.lutorch.multi_head_lut import MultiHeadLut
from spiky.lutorch.lut_helpers import AnchorSamplingPolicy


def _has_cuda():
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda(), reason="needs CUDA")
@pytest.mark.parametrize(
    "input_dim,n_heads,n_outputs,n_anchor_pairs,tables_per_head",
    [
        # exp061 layer-0 qk_joint slot
        (96, 6, 64, 6, 256),
        # exp061 layer-0 v_lut slot
        (96, 6, 16, 7, 256),
        # exp061 layer-0 out_proj slot (largest LUT)
        (96, 1, 96, 6, 2048),
    ],
)
def test_forward_grad_and_step_match(input_dim, n_heads, n_outputs, n_anchor_pairs, tables_per_head):
    dev = torch.device("cuda:0")
    common = dict(
        input_dim=input_dim, n_heads=n_heads, n_outputs=n_outputs,
        n_anchor_pairs=n_anchor_pairs, tables_per_head=tables_per_head,
        random_seed=42, device=dev,
        anchor_sampling_policy=AnchorSamplingPolicy.CANONICAL_FULL_COVERAGE,
    )
    m_tiny = TinyMultiHeadLut(**common, weight_dtype=torch.float32)
    m_full = MultiHeadLut(**common, n_alternatives=1, smooth_mode=False,
                          recompute_in_backward=True, initial_weights_noise=0.001)

    # Pin both modules to the same anchors AND weights.
    with torch.no_grad():
        m_full.lookup.anchor_pairs_a.copy_(m_tiny.lookup.anchor_pairs_a)
        m_full.lookup.anchor_pairs_b.copy_(m_tiny.lookup.anchor_pairs_b)
        m_full.projection.weights.copy_(m_tiny.weights)

    # Sanity: anchors are identical (Tiny stores int16, Full stores int64; cast
    # to int64 for comparison).
    assert torch.equal(
        m_tiny.lookup.anchor_pairs_a.to(torch.int64),
        m_full.lookup.anchor_pairs_a,
    )

    # Same input.
    torch.manual_seed(123)
    x_tiny = torch.randn(64, input_dim, device=dev, requires_grad=True)
    x_full = x_tiny.detach().clone().requires_grad_(True)

    # ----- (1) Forward -----
    y_tiny = m_tiny(x_tiny)
    y_full = m_full(x_full)
    fwd_max_diff = (y_tiny - y_full).abs().max().item()
    assert fwd_max_diff < 1e-5, f"forward mismatch: max diff {fwd_max_diff}"

    # ----- (2) Backward — synthetic upstream grad -----
    torch.manual_seed(456)
    g = torch.randn_like(y_tiny)
    y_tiny.backward(g)
    y_full.backward(g)

    wgrad_max = (m_tiny.weights.grad - m_full.projection.weights.grad).abs().max().item()
    xgrad_max = (x_tiny.grad - x_full.grad).abs().max().item()
    assert wgrad_max < 1e-5, f"weights.grad mismatch: max diff {wgrad_max}"
    assert xgrad_max < 1e-5, f"x.grad mismatch: max diff {xgrad_max}"

    # ----- (3) Multi-step SGD: do they diverge over training? -----
    # Reset grads + start from same init weights
    m_tiny.weights.grad = None
    m_full.projection.weights.grad = None
    with torch.no_grad():
        m_full.projection.weights.copy_(m_tiny.weights)

    opt_tiny = torch.optim.SGD([m_tiny.weights], lr=1e-2)
    opt_full = torch.optim.SGD([m_full.projection.weights], lr=1e-2)

    torch.manual_seed(789)
    xs = torch.randn(8, 64, input_dim, device=dev)
    targets = torch.randn(8, 64, n_heads, n_outputs, device=dev)

    for s in range(8):
        opt_tiny.zero_grad(); opt_full.zero_grad()
        yt = m_tiny(xs[s])
        yf = m_full(xs[s])
        lt = ((yt - targets[s]) ** 2).mean()
        lf = ((yf - targets[s]) ** 2).mean()
        lt.backward(); lf.backward()
        opt_tiny.step();  opt_full.step()

    final_wdiff = (m_tiny.weights - m_full.projection.weights).abs().max().item()
    final_ldiff = abs(lt.item() - lf.item())
    assert final_wdiff < 1e-5, f"after 8 SGD steps weights drift: {final_wdiff}"
    assert final_ldiff < 1e-5, f"after 8 SGD steps loss differs: {final_ldiff}"
