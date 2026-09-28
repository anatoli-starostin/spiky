"""Regression tests for LightMultiHeadLUT trainable_anchors mode (multi-head, pair).
Default off == unchanged; on -> hard argmax/argmin margins == baked-index gather; grads reach the
anchor logits + temperature; bake(drop_logits) leaves a pure gather eval path."""
import torch
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

H, TPH, NAP, EIN, EO = 4, 16, 8, 48, 48
NT = H * TPH


def _mk(trainable):
    return LightMultiHeadLUT(
        input_dim=EIN, n_tables=NT, output_dim=EO, n_anchor_pairs=NAP,
        confidence_form="learned_margin", learned_margin_freeze_g=True,
        n_heads=H, multi_head_input=True, read_top_n=2, read_tau=0.5,
        forward_mode="scored", random_seed=1, trainable_anchors=trainable, anchor_tau_init=1.0)


def test_default_off_unchanged():
    m = _mk(False).cuda().eval()
    assert not m.trainable_anchors
    assert not hasattr(m, "anchor_logits") or getattr(m, "anchor_logits", None) is None
    assert m.anchor_a.dtype in (torch.int64, torch.int32)
    x = torch.randn(8, H, EIN, device="cuda")
    with torch.no_grad():
        a, b = m(x), m(x)
    assert torch.allclose(a, b)          # deterministic in eval


def test_on_hard_equals_baked_gather_margins():
    m = _mk(True).cuda()
    # baked index buffers must equal argmax/argmin of the logits
    assert torch.equal(m.anchor_a, m.anchor_logits.argmax(-1))
    assert torch.equal(m.anchor_b, m.anchor_logits.argmin(-1))
    x = torch.randn(8, H, EIN, device="cuda")
    T = TPH
    # gather margins (the eval/inference path)
    ia = m.anchor_a.reshape(1, H, T * NAP).expand(8, H, T * NAP)
    ib = m.anchor_b.reshape(1, H, T * NAP).expand(8, H, T * NAP)
    d_gather = (torch.gather(x, 2, ia) - torch.gather(x, 2, ib)).view(8, H, T, NAP)
    # hard margins straight from the logits
    amax = m.anchor_logits.argmax(-1); amin = m.anchor_logits.argmin(-1)   # [H,T,NAP]
    d_hard = (x.gather(2, amax.reshape(1, H, T * NAP).expand(8, H, T * NAP))
              - x.gather(2, amin.reshape(1, H, T * NAP).expand(8, H, T * NAP))).view(8, H, T, NAP)
    assert torch.allclose(d_gather, d_hard)
    # STE forward VALUE (train path) == the hard/gather margins (value-preserving)
    m.train()
    d_ste = m._trainable_margins(x)
    assert torch.allclose(d_ste.detach(), d_hard, atol=1e-5)
    # and the full eval forward runs and is finite
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(m(x)).all()


def test_grads_reach_logits_and_tau():
    m = _mk(True).cuda().train()
    for mod in [m]:
        pass
    # give the value tables signal so the read (hence loss) depends on the selected cells
    with torch.no_grad():
        m.tables.normal_(0, 1.0)
    x = torch.randn(8, H, EIN, device="cuda", requires_grad=True)
    out = m(x)
    out.sum().backward()
    assert m.anchor_logits.grad is not None and torch.isfinite(m.anchor_logits.grad).all()
    assert m.anchor_logits.grad.abs().sum() > 0
    assert m.anchor_log_tau.grad is not None and torch.isfinite(m.anchor_log_tau.grad).all()


def test_bake_and_drop_logits():
    m = _mk(True).cuda()
    x = torch.randn(8, H, EIN, device="cuda")
    m.eval()
    with torch.no_grad():
        before = m(x)
    m.bake_anchors(drop_logits=True)
    assert m.anchor_logits is None and m.anchor_log_tau is None
    assert "anchor_a" in m.state_dict() and "anchor_b" in m.state_dict()
    with torch.no_grad():
        after = m(x)
    assert torch.allclose(before, after)      # baked gather path reproduces the eval output
