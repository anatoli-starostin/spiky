"""Tests for LightMultiHeadLUT's block-diagonal multi_head_input path.

Mirrors the style of test_light_multi_head_lut.py. The properties that matter:
  * shapes and per-head independence (no mixing between heads);
  * the defining Light property is preserved -- the routing index is detached, so the
    ONLY gradient to x is through the confidence score;
  * per-head anchors/tables are drawn with (random_seed + h), matching Fast, so head h
    of a multi-head layer is identical to a single-head layer seeded (random_seed + h);
  * the single-head path is untouched (flag-off equivalence);
  * at the anchor sizing, CompressionMHL's light path matches Fast's projections exactly.
"""
import pytest
import torch

from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT

H, T, NAP, DIN, DOUT = 4, 8, 5, 12, 7
SEED = 1234


def _mh(**kw):
    return LightMultiHeadLUT(input_dim=DIN, n_tables=H * T, output_dim=DOUT,
                             n_anchor_pairs=NAP, random_seed=SEED,
                             n_heads=H, multi_head_input=True, **kw)


def test_shapes_and_dtype():
    m = _mh()
    x = torch.randn(6, H, DIN)
    y = m(x)
    assert y.shape == (6, H, DOUT)
    assert y.dtype == x.dtype
    assert m.anchor_a.shape == (H, T, NAP)
    assert m.tables.shape == (H * T, 1 << NAP, DOUT)


def test_wrong_shape_rejected():
    m = _mh()
    with pytest.raises(ValueError):
        m(torch.randn(6, DIN))              # missing the head axis
    with pytest.raises(ValueError):
        m(torch.randn(6, H + 1, DIN))       # wrong head count


def test_n_tables_must_divide_by_heads():
    with pytest.raises(ValueError):
        LightMultiHeadLUT(input_dim=DIN, n_tables=H * T + 1, output_dim=DOUT,
                          n_anchor_pairs=NAP, n_heads=H, multi_head_input=True)


def test_heads_are_independent():
    """Perturbing head h's input slice must move only head h's output block."""
    m = _mh()
    x = torch.randn(3, H, DIN)
    y0 = m(x)
    x2 = x.clone()
    x2[:, 1] += 3.0
    y2 = m(x2)
    moved = [(y0[:, h] - y2[:, h]).abs().max().item() > 0 for h in range(H)]
    assert moved == [False, True, False, False]


def test_matches_single_head_layers_head_for_head():
    """Head h must equal a single-head layer seeded (random_seed + h) -- the Fast
    convention -- so the two implementations are initialisation-comparable."""
    m = _mh()
    x = torch.randn(5, H, DIN)
    y = m(x)
    for h in range(H):
        single = LightMultiHeadLUT(input_dim=DIN, n_tables=T, output_dim=DOUT,
                                   n_anchor_pairs=NAP, random_seed=SEED + h)
        assert torch.equal(single.anchor_a, m.anchor_a[h])
        assert torch.equal(single.anchor_b, m.anchor_b[h])
        assert torch.allclose(single.tables, m.tables[h * T:(h + 1) * T])
        assert torch.allclose(single(x[:, h]), y[:, h], atol=1e-6)


def test_routing_is_detached_grad_only_through_score():
    """The defining Light property must survive the multi-head rewrite: x receives
    gradient ONLY through the confidence score, never through the routing direction.

    Made exact rather than statistical. With every table row set to the same constant
    vector c, the output is (sum_t score_t) * c, so grad_x must equal the gradient of
    that closed form -- computed here from d with plain autograd, independent of the
    module. Any leakage through the index would break the match.
    """
    from spiky.lutorch.fast_multi_head_lut import _confidence_score
    m = _mh()
    with torch.no_grad():
        m.tables.copy_(torch.ones_like(m.tables))         # every row identical, c = 1
    x = torch.randn(4, H, DIN, dtype=torch.float64, requires_grad=True)
    m = m.double()
    m(x).sum().backward()
    got = x.grad.clone()

    # closed form: DOUT * sum over tables of score(|d|), with d built by plain indexing
    x2 = x.detach().clone().requires_grad_(True)
    idx_a = m.anchor_a.reshape(1, H, T * NAP).expand(4, H, T * NAP)
    idx_b = m.anchor_b.reshape(1, H, T * NAP).expand(4, H, T * NAP)
    d = (torch.gather(x2, 2, idx_a) - torch.gather(x2, 2, idx_b)).view(4, H, T, NAP)
    (DOUT * _confidence_score(d, m.confidence_form).sum()).backward()
    assert torch.allclose(got, x2.grad, atol=1e-12), (got - x2.grad).abs().max()


def test_no_temperature_parameters():
    m = _mh()
    assert not [n for n, _ in m.named_parameters() if "temp" in n]


def test_single_head_path_unchanged():
    """multi_head_input=False must be byte-identical to the pre-existing behaviour."""
    a = LightMultiHeadLUT(input_dim=DIN, n_tables=H * T, output_dim=DOUT,
                          n_anchor_pairs=NAP, random_seed=SEED)
    b = LightMultiHeadLUT(input_dim=DIN, n_tables=H * T, output_dim=DOUT,
                          n_anchor_pairs=NAP, random_seed=SEED,
                          n_heads=H, multi_head_input=False)
    assert torch.equal(a.anchor_a, b.anchor_a)
    assert torch.equal(a.tables, b.tables)
    x = torch.randn(4, DIN)
    assert torch.equal(a(x), b(x))


@pytest.mark.parametrize("form", ["bounded", "margin", "bounded_norm"])
def test_gradcheck_float64(form):
    torch.manual_seed(0)
    m = LightMultiHeadLUT(input_dim=6, n_tables=4, output_dim=3, n_anchor_pairs=3,
                          random_seed=7, n_heads=2, multi_head_input=True,
                          confidence_form=form).double()
    x = torch.randn(2, 2, 6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda t: m(t), (x,), eps=1e-6, atol=1e-7)


def test_compression_mhl_matches_fast_projections_at_anchor_sizing():
    """At the anchor sizing the light path must now carry EXACTLY Fast's projections
    and an identical table budget."""
    kw = dict(input_dim=384, output_dim=384, inner_in_dim=32, inner_out_dim=48,
              nap=8, tph=256, n_heads=4, joint_head_compression=False,
              random_seed=1000)
    fast = CompressionMultiHeadLUT(**kw)
    light = CompressionMultiHeadLUT(**kw, lut_impl="light")
    f = dict((n, p.numel()) for n, p in fast.named_parameters())
    g = dict((n, p.numel()) for n, p in light.named_parameters())
    assert g["compress.weight"] == f["compress.weight"] == 49152
    assert g["compress.bias"] == f["compress.bias"] == 128
    assert g["decompress.weight"] == f["decompress.weight"] == 73728
    assert g["decompress.bias"] == f["decompress.bias"] == 384
    assert g["lut_light.tables"] == f["lut_batched.weights"] == 12582912
    # the only remaining difference is Fast's two learnable temperature scalars
    assert sum(f.values()) - sum(g.values()) == 2
    x = torch.randn(3, 384)
    assert light(x).shape == (3, 384)


# --- bounded_norm wiring: the value must survive CompressionMHL and model_build ---

@pytest.mark.parametrize("impl", ["fast", "light"])
def test_bounded_norm_reaches_the_inner_layer(impl):
    """confidence_form="bounded_norm" is carried to the LUT for BOTH impls and gates."""
    kw = dict(input_dim=32, output_dim=32, inner_in_dim=8, inner_out_dim=8,
              nap=4, tph=2, n_heads=2, joint_head_compression=False, random_seed=3)
    m = CompressionMultiHeadLUT(**kw, lut_impl=impl, forward_confidence=True,
                                confidence_form="bounded_norm")
    assert m.confidence_form == "bounded_norm"
    inner = m.lut_light if impl == "light" else m.lut_batched
    assert inner.confidence_form == "bounded_norm"
    if impl == "fast":
        assert inner.forward_confidence is True
    # and it changes the output relative to the un-normalised form
    other = CompressionMultiHeadLUT(**kw, lut_impl=impl, forward_confidence=True,
                                    confidence_form="bounded")
    other.load_state_dict(m.state_dict())
    x = torch.randn(4, 32)
    with torch.no_grad():
        assert not torch.allclose(m(x), other(x)), "bounded_norm behaved like bounded"


def test_bounded_norm_flows_through_model_build_cfg():
    """`lut_confidence_form` in the experiment cfg reaches every block's FFN."""
    import sys
    from pathlib import Path
    tools = Path(__file__).resolve().parents[4] / "experiments/ffn_replacement/tools"
    if not tools.is_dir():
        pytest.skip("experiment tools not present in this checkout")
    sys.path.insert(0, str(tools))
    from model_build import MinimalBlock

    cfg = dict(ffn_type='compression', lut_inner_in_dim=8, lut_inner_out_dim=8,
               lut_n_anchor_pairs=4, lut_tables_per_head=2, lut_n_heads=2,
               lut_forward_confidence=True, lut_confidence_form='bounded_norm')
    blk = MinimalBlock(32, 2, 0, cfg)
    assert blk.ffn.confidence_form == 'bounded_norm'
    assert blk.ffn.lut_batched.forward_confidence is True
    assert blk.ffn.lut_batched.confidence_form == 'bounded_norm'


def test_exp_outputs_guard_still_holds_for_bounded_norm():
    """The forward_confidence + exp_outputs incompatibility is form-independent."""
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    with pytest.raises(ValueError, match="exp_outputs"):
        FastMultiHeadLut(input_dim=8, n_heads=1, n_outputs=4, n_anchor_pairs=2,
                         forward_confidence=True, confidence_form="bounded_norm",
                         exp_outputs=True, use_bf16=False)
