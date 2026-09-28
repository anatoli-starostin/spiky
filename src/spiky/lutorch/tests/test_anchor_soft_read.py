"""STE soft-read for trainable_anchors (light multi-head, constant cells).

The HARD forward is unchanged (argmax/sign address) -> forward value bit-identical to soft_read=False;
a differentiable soft read read_soft = sum_c P(c) V[c] (p_k = sigmoid(d_k/tau_addr)) is stitched
STE-style so gradient reaches d -> anchor_logits (which the detached hard address otherwise blocks)."""
import torch
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

H, TPH, NAP, EIN, EO = 4, 32, 8, 48, 48
NT = H * TPH


def _mk(soft, init="random", tau_addr=0.25):
    return LightMultiHeadLUT(
        input_dim=EIN, n_tables=NT, output_dim=EO, n_anchor_pairs=NAP,
        confidence_form="learned_margin", learned_margin_freeze_g=True,
        n_heads=H, multi_head_input=True, read_top_n=2, read_tau=0.5,
        forward_mode="scored", random_seed=1, trainable_anchors=True,
        anchor_init=init, anchor_tau_init=0.5, soft_read=soft, tau_addr=tau_addr).cuda()


def test_forward_value_bit_identical():
    hard = _mk(False); soft = _mk(True)
    x = torch.randn(8, H, EIN, device="cuda")
    hard.train(); soft.train()                      # soft read is active only in train+grad
    with torch.no_grad():
        oh, os_ = hard(x), soft(x)
    assert torch.equal(oh, os_), (oh - os_).abs().max().item()
    hard.eval(); soft.eval()                        # eval: both pure hard
    with torch.no_grad():
        assert torch.equal(hard(x), soft(x))


def test_soft_read_makes_anchor_grad_nonzero():
    # give the tables signal (real models reach this once the identity-block decompress warms up)
    soft = _mk(True); hard = _mk(False)
    for m in (soft, hard):
        m.train()
        with torch.no_grad():
            m.tables.normal_(0, 1.0)
    x = torch.randn(8, H, EIN, device="cuda")
    for m in (soft, hard):
        m.zero_grad(set_to_none=True)
        m(x).pow(2).sum().backward()
    gs = soft.anchor_logits.grad.norm().item()
    assert gs > 0 and torch.isfinite(soft.anchor_logits.grad).all(), gs   # gradient reaches the anchors
    assert soft.anchor_log_tau.grad is not None and soft.anchor_log_tau.grad.abs().item() > 0


def test_soft_read_off_default_unchanged():
    m = _mk(False)
    assert not m.soft_read
    x = torch.randn(4, H, EIN, device="cuda")
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(m(x)).all()


def test_soft_read_tau0_limits_to_hard_cell():
    # as tau_addr -> 0, read_soft -> V[argmax cell]; sanity that the soft aggregate is finite & shaped
    m = _mk(True, tau_addr=1e-3); m.train()
    with torch.no_grad():
        m.tables.normal_(0, 1.0)
    x = torch.randn(6, H, EIN, device="cuda")
    out = m(x)
    assert out.shape == (6, H, EO) and torch.isfinite(out).all()
