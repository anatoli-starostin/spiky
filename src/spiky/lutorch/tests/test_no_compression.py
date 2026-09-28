"""No-compression multi-head LUT: inner_in_dim=-1 with n_heads>1 on the light path makes every
head read the WHOLE input_dim embedding (broadcast) and own its own anchors, so trainable anchors
index the raw embedding directly (anchor_logits [..., input_dim]). Default (real compress) is
untouched. Soft-read STE and the baked-index eval path work in this mode."""
import torch
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT

E, H, TPH, NAP = 384, 4, 16, 8


def _comp(**kw):
    base = dict(input_dim=E, output_dim=E, inner_in_dim=-1, inner_out_dim=48, nap=NAP, tph=TPH,
                n_heads=H, lut_impl="light", forward_confidence=True, confidence_form="learned_margin",
                learned_margin_freeze_g=True, read_top_n=2, read_tau=0.5, random_seed=1,
                anchor_sampling_policy=None, device="cuda")
    base.update(kw)
    return CompressionMultiHeadLUT(**base).cuda()


def test_default_compression_untouched():
    # a REAL-compress light config: no_compress_multi_head must be False and forward deterministic
    m = _comp(inner_in_dim=48)
    assert not m.no_compress_multi_head
    m.eval()
    x = torch.randn(8, E, device="cuda")
    with torch.no_grad():
        assert torch.equal(m(x), m(x))


def test_no_compression_forward_and_anchor_shape():
    m = _comp(trainable_anchors=True, soft_read=True, tau_addr=0.25, anchor_init="random")
    assert m.no_compress_multi_head and m.eff_in == E
    # anchors index the raw 384-d embedding
    assert tuple(m.lut_light.anchor_logits.shape) == (H, TPH, NAP, E)
    x = torch.randn(8, E, device="cuda")
    m.eval()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (8, E) and torch.isfinite(y).all()


def _mk_lut(soft, init="random"):
    return LightMultiHeadLUT(
        input_dim=E, n_tables=H * TPH, output_dim=48, n_anchor_pairs=NAP,
        confidence_form="learned_margin", learned_margin_freeze_g=True, n_heads=H,
        multi_head_input=True, read_top_n=2, read_tau=0.5, forward_mode="scored", random_seed=1,
        trainable_anchors=True, anchor_init=init, anchor_tau_init=0.5, soft_read=soft, tau_addr=0.25).cuda()


def test_anchor_grad_reaches_384_logits():
    m = _mk_lut(soft=True); m.train()
    with torch.no_grad():
        m.tables.normal_(0, 1.0)               # tables carry signal (as after warmup in a real run)
    x = torch.randn(8, H, E, device="cuda")    # each head sees the full 384 embedding
    m(x).pow(2).sum().backward()
    g = m.anchor_logits.grad
    assert g is not None and tuple(g.shape) == (H, TPH, NAP, E)
    assert g.norm().item() > 0                  # gradient reaches the 384-d anchor logits


def test_baked_index_eval_reproduces_and_forward_value():
    m = _mk_lut(soft=True)
    with torch.no_grad():
        m.tables.normal_(0, 1.0)
    x = torch.randn(8, H, E, device="cuda")
    # STE forward VALUE (train) == the hard eval read
    m.train()
    with torch.no_grad():
        train_val = m(x)
    m.eval()
    with torch.no_grad():
        eval_val = m(x)
    assert torch.equal(train_val, eval_val)     # soft read is backward-only; value is the hard read
    # bake + drop logits -> pure gather path reproduces eval output
    before = eval_val
    m.bake_anchors(drop_logits=True)
    assert m.anchor_logits is None
    with torch.no_grad():
        after = m(x)
    assert torch.equal(before, after)


def test_fixed_anchors_no_compression():
    # Arm 1: fixed (non-trainable) anchors in no-compression mode
    m = _comp(trainable_anchors=False)
    assert m.no_compress_multi_head
    assert not getattr(m.lut_light, "trainable_anchors", False)
    assert tuple(m.lut_light.anchor_a.shape) == (H, TPH, NAP)   # per-head fixed pairs over 384
    x = torch.randn(6, E, device="cuda")
    m.eval()
    with torch.no_grad():
        assert torch.isfinite(m(x)).all()
