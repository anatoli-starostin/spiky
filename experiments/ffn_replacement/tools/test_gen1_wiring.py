"""Tests for the Gen-1 (MultiHeadLut) wiring into CompressionMultiHeadLUT / model_build (ablation rows 1.1 / 1.2):
module layout (block-diagonal anchors inside each head's slice), forward/backward, the guards, TV coverage, and the
train_fixed.py optimiser exempting Gen-1 tables from weight decay like every other LUT's."""
import ast
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                                   # tools/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))     # spiky
torch._dynamo.config.suppress_errors = True
from model_build import build_model  # noqa: E402
from spiky.lutorch.bh4_multi_head_lut import BH4MultiHeadLUT  # noqa: E402
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut  # noqa: E402
from spiky.lutorch.l_projection import LProjection  # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402
from spiky.lutorch.lut_helpers import UncertaintyMode  # noqa: E402
from spiky.lutorch.multi_head_lut import MultiHeadLut  # noqa: E402

_TOOLS = os.path.dirname(os.path.abspath(__file__))
H, IN, OUT, NAP, TPH, E = 2, 16, 16, 4, 4, 96


def _cfg(smooth=False, **kw):
    c = {"ffn_type": "compression", "depth": 2, "n_embd": E, "n_head": 4, "seq_len": 16, "tokenizer_vocab_size": 64,
         "tie_unembedder": False, "gamma": 0, "lut_impl": "gen1", "lut_gen1_smooth": smooth,
         "lut_inner_in_dim": IN, "lut_inner_out_dim": OUT, "lut_n_anchor_pairs": NAP, "lut_tables_per_head": TPH,
         "lut_n_heads": H, "lut_joint_head_compression": False, "lut_base_seed": 1000}
    c.update(kw)
    return c


@pytest.mark.parametrize("smooth", [False, True])
def test_gen1_block_layout_and_settings(smooth):
    m = build_model(_cfg(smooth), 64, device="cpu")
    for i, b in enumerate(m.blocks):
        lut = b.ffn.lut_gen1
        assert isinstance(lut, MultiHeadLut) and lut.smooth_mode is smooth and lut.n_alternatives == 1
        assert lut.uncertainty_mode == UncertaintyMode.INVERSE_L1 and lut.n_buckets == 1
        assert lut.projection.weights.shape == (H * TPH, 1 << NAP, OUT)
        a = lut.lookup.anchor_pairs_a.view(H, TPH, NAP)
        bb = lut.lookup.anchor_pairs_b.view(H, TPH, NAP)
        for h in range(H):                                   # head h routes only on its own slice [h*IN, (h+1)*IN)
            assert int(a[h].min()) >= h * IN and int(a[h].max()) < (h + 1) * IN
            assert int(bb[h].min()) >= h * IN and int(bb[h].max()) < (h + 1) * IN
        assert not any(isinstance(x, (FastMultiHeadLut, LightMultiHeadLUT, BH4MultiHeadLUT)) for x in b.modules())


@pytest.mark.parametrize("smooth", [False, True])
def test_gen1_forward_backward_cpu(smooth):
    torch.manual_seed(0)
    m = build_model(_cfg(smooth), 64, device="cpu")
    m.train()
    x = torch.randint(0, 64, (3, 16))
    loss = m(x, x)
    loss.backward()
    b = m.blocks[0]
    assert torch.isfinite(loss)
    for p in (b.ffn.compress.weight, b.ffn.decompress.weight, b.ffn.lut_gen1.projection.weights):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    m.eval()
    with torch.no_grad():
        assert m(x).shape == (3, 16, 64)


@pytest.mark.parametrize("bad", [
    dict(lut_joint_head_compression=True),
    dict(lut_inner_in_dim=-1),
    dict(lut_forward_confidence=True),
    dict(lut_light_forward_mode="hard"),
])
def test_gen1_guards(bad):
    with pytest.raises(ValueError):
        build_model(_cfg(**bad), 64, device="cpu")


def test_gen1_keys_refused_on_other_impls():
    with pytest.raises(ValueError, match="gen1"):
        build_model(_cfg(lut_impl="fast"), 64, device="cpu")


def test_gen1_tv_is_covered_and_guard_allows_it():
    m = build_model(_cfg(lut_cell_smoothness=10.0), 64, device="cpu")
    luts = [x for x in m.modules() if isinstance(x, MultiHeadLut)]
    assert m.lut_tv_modules() == luts and len(luts) == 2
    p = m.lut_tv_penalty()
    torch.testing.assert_close(p, torch.stack([x.cell_tv() for x in luts]).mean())
    assert m.lut_tv_by_layer() == pytest.approx([float(x.cell_tv()) for x in luts])


def _setup_optimizer_from(path):
    src = open(path).read()
    fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "setup_optimizer")
    ns = dict(torch=torch, FastMultiHeadLut=FastMultiHeadLut, LightMultiHeadLUT=LightMultiHeadLUT,
              BH4MultiHeadLUT=BH4MultiHeadLUT, LProjection=LProjection)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), path, "exec"), ns)
    return ns["setup_optimizer"]


def _non_gen1(impl):
    c = {k: v for k, v in _cfg().items() if not k.startswith("lut_gen1_")}
    c["lut_impl"] = impl
    return c


def test_gen1_default_init_is_multiheadlut_normal():
    m = build_model(_cfg(), 64, device="cpu")
    for i, b in enumerate(m.blocks):
        w = b.ffn.lut_gen1.projection.weights
        ref = torch.randn(w.shape, generator=torch.Generator().manual_seed(1000 + i)) * 1e-3
        assert torch.equal(w.detach(), ref)


def test_gen1_uniform_init_equals_fast_and_light_tables():
    g = build_model(_cfg(lut_gen1_weights_init="uniform"), 64, device="cpu")
    f = build_model(_non_gen1("fast"), 64, device="cpu")
    li = build_model(_non_gen1("light"), 64, device="cpu")
    for bg, bf, bl in zip(g.blocks, f.blocks, li.blocks):
        w = bg.ffn.lut_gen1.projection.weights.detach()
        assert torch.equal(w, bf.ffn.lut_batched.weights.detach())
        assert torch.equal(w, bl.ffn.lut_light.tables.detach())
        assert float(w.abs().max()) <= 1e-3
        torch.testing.assert_close(bg.ffn.lut_gen1.cell_tv(), bf.ffn.lut_batched.cell_tv())


def test_gen1_weights_init_key_guards():
    with pytest.raises(ValueError, match="gen1"):
        build_model({**_non_gen1("fast"), "lut_gen1_weights_init": "uniform"}, 64, device="cpu")
    with pytest.raises(ValueError, match="weights_init"):
        build_model(_cfg(lut_gen1_weights_init="xavier"), 64, device="cpu")


@pytest.mark.parametrize("tables_no_decay", [False, True])
def test_train_fixed_exempts_gen1_tables_from_weight_decay(tables_no_decay):
    setup = _setup_optimizer_from(os.path.join(_TOOLS, "..", "train_fixed.py"))
    m = build_model(_cfg(), 64, device="cpu")
    opt = setup(m, 3e-4, 0.1, tables_no_decay=tables_no_decay)
    nodecay = {id(p) for g in opt.param_groups if g["weight_decay"] == 0.0 for p in g["params"]}
    for x in m.modules():
        if isinstance(x, MultiHeadLut):
            assert id(x.projection.weights) in nodecay
