"""Step-0 equivalence for the trainable_anchors WARM start.

anchor_init="warm" seeds anchor_logits so the baked argmax/argmin pairs reproduce the fixed-path
BALANCED draw head-for-head. With identical seed the value tables are drawn from the same
per-head generator, so a warm trainable model must be BYTE-IDENTICAL to the fixed-anchor model at
construction (step 0): same anchor_a/anchor_b, same eval forward. anchor_movement() must read 0 at
init and rise once the baked pairs drift. "random" init must NOT match the fixed pairs."""
import torch
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

H, TPH, NAP, EIN, EO = 4, 128, 2, 48, 48
NT = H * TPH


def _mk(**kw):
    return LightMultiHeadLUT(
        input_dim=EIN, n_tables=NT, output_dim=EO, n_anchor_pairs=NAP,
        confidence_form="learned_margin", learned_margin_freeze_g=True,
        n_heads=H, multi_head_input=True, read_top_n=2, read_tau=0.5,
        forward_mode="scored", random_seed=1, **kw).cuda()


def test_warm_reproduces_fixed_pairs_and_forward():
    fixed = _mk(trainable_anchors=False).eval()
    warm = _mk(trainable_anchors=True, anchor_init="warm", anchor_tau_init=0.5,
               anchor_init_scale=8.0).eval()
    # 1) the baked pairs must equal the fixed BALANCED pairs, exactly
    assert torch.equal(warm.anchor_a, fixed.anchor_a)
    assert torch.equal(warm.anchor_b, fixed.anchor_b)
    # 2) argmax/argmin of the warm logits land on the fixed 'a'/'b' coords
    assert torch.equal(warm.anchor_logits.argmax(-1), fixed.anchor_a)
    assert torch.equal(warm.anchor_logits.argmin(-1), fixed.anchor_b)
    # 3) value tables identical (same per-head generator) -> full eval forward byte-identical
    assert torch.equal(warm.tables, fixed.tables)
    x = torch.randn(16, H, EIN, device="cuda")
    with torch.no_grad():
        of, ow = fixed(x), warm(x)
    assert torch.equal(of, ow), (of - ow).abs().max().item()


def test_movement_zero_at_init_and_rises_on_drift():
    warm = _mk(trainable_anchors=True, anchor_init="warm", anchor_tau_init=0.5).eval()
    assert warm.anchor_movement() == 0.0
    # force a couple of slots to a different pair, re-bake, movement must reflect it
    with torch.no_grad():
        warm.anchor_logits[0, 0, 0, :] = 0.0
        warm.anchor_logits[0, 0, 0, 7] = 5.0     # new argmax
        warm.anchor_logits[0, 0, 0, 3] = -5.0    # new argmin
    warm._refresh_anchor_idx()
    mv = warm.anchor_movement()
    assert mv > 0.0
    # exactly one of H*TPH*NAP slots changed
    assert abs(mv - 1.0 / (H * TPH * NAP)) < 1e-9, mv


def test_random_init_differs_from_fixed():
    fixed = _mk(trainable_anchors=False).eval()
    rnd = _mk(trainable_anchors=True, anchor_init="random", anchor_tau_init=0.5).eval()
    # random init should not coincide with the balanced fixed pairs (overwhelmingly)
    same = (torch.equal(rnd.anchor_a, fixed.anchor_a) and torch.equal(rnd.anchor_b, fixed.anchor_b))
    assert not same
