"""Tests for the model-level cell TV penalty (model_build): Gen-2 (FastMultiHeadLut) coverage, Light unchanged,
per-layer logging values, the build-time guard that refuses a lut_cell_smoothness the penalty cannot deliver, and
the train_fixed.py switch (read the key; backprop the penalty before clipping)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                                   # tools/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))     # spiky
torch._dynamo.config.suppress_errors = True
from model_build import build_model  # noqa: E402
from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut  # noqa: E402
from spiky.lutorch.light_multi_head_lut import LightMultiHeadLUT  # noqa: E402

_TOOLS = os.path.dirname(os.path.abspath(__file__))


def _cfg(impl="fast", lam=None, ffn_type="compression", **kw):
    c = {
        "ffn_type": ffn_type, "depth": 2, "n_embd": 96, "n_head": 4, "seq_len": 32,
        "tokenizer_vocab_size": 256, "tie_unembedder": False, "gamma": 0,
        "lut_impl": impl, "lut_inner_in_dim": 16, "lut_inner_out_dim": 16, "lut_n_anchor_pairs": 4,
        "lut_tables_per_head": 4, "lut_n_heads": 2, "lut_joint_head_compression": False,
        "lut_learnable_temps": True, "lut_base_seed": 1000,
    }
    if impl == "light":
        c.update({"lut_forward_confidence": True, "lut_confidence_form": "margin"})
    if lam is not None:
        c["lut_cell_smoothness"] = lam
    c.update(kw)
    return c


def _old_light_penalty(model):
    """The pre-change model_build.lut_tv_penalty, verbatim: LightMultiHeadLUT modules only."""
    ms = [m for m in model.modules() if isinstance(m, LightMultiHeadLUT)]
    if not ms:
        return torch.zeros((), device=model.get_device())
    return torch.stack([m.cell_tv() for m in ms]).mean()


@pytest.mark.parametrize("joint", [False, True])
def test_fast_model_penalty_covers_every_fast_lut_and_has_a_gradient(joint):
    torch.manual_seed(0)
    m = build_model(_cfg("fast", lut_joint_head_compression=joint), 256, device="cpu")
    luts = [x for x in m.modules() if isinstance(x, FastMultiHeadLut)]
    assert len(luts) == 2 and m.lut_tv_modules() == luts
    p = m.lut_tv_penalty()
    assert p.requires_grad and float(p) > 0.0
    torch.testing.assert_close(p, torch.stack([x.cell_tv() for x in luts]).mean())
    p.backward()
    assert all(x.weights.grad is not None and x.weights.grad.abs().sum() > 0 for x in luts)
    assert m.lut_tv_by_layer() == pytest.approx([float(x.cell_tv()) for x in luts])


def test_light_model_penalty_is_unchanged():
    torch.manual_seed(0)
    m = build_model(_cfg("light", lam=10.0), 256, device="cpu")
    new, old = m.lut_tv_penalty(), _old_light_penalty(m)
    assert torch.equal(new, old)
    tables = [x.tables for x in m.modules() if isinstance(x, LightMultiHeadLUT)]
    g_new = torch.autograd.grad(new, tables)
    g_old = torch.autograd.grad(old, tables)
    assert all(torch.equal(a, b) for a, b in zip(g_new, g_old))


def test_dense_model_penalty_is_zero_without_grad_when_tv_is_off():
    m = build_model(_cfg(ffn_type="dense"), 256, device="cpu")
    p = m.lut_tv_penalty()
    assert float(p) == 0.0 and not p.requires_grad and m.lut_tv_by_layer() == []


@pytest.mark.parametrize("cfg", [
    _cfg("bh4", lam=10.0),                       # BH4 tables are not covered
    _cfg(ffn_type="dense", lam=10.0),            # no LUT tables at all
])
def test_guard_refuses_tv_the_penalty_cannot_deliver(cfg):
    with pytest.raises(ValueError, match="cannot reach this model's tables"):
        build_model(cfg, 256, device="cpu")


def test_guard_refuses_negative_and_allows_supported_or_off():
    with pytest.raises(ValueError, match=">= 0"):
        build_model(_cfg("fast", lam=-1.0), 256, device="cpu")
    build_model(_cfg("fast", lam=10.0), 256, device="cpu")
    build_model(_cfg("light", lam=10.0), 256, device="cpu")
    build_model(_cfg("bh4", lam=0.0), 256, device="cpu")
    build_model(_cfg(ffn_type="dense"), 256, device="cpu")


def test_train_fixed_reads_the_switch_and_backprops_tv_before_clipping():
    src = open(os.path.join(_TOOLS, "..", "train_fixed.py")).read()
    assert "cfg.get('lut_cell_smoothness', 0.0)" in src
    i_tv = src.index("(LUT_TV_LAMBDA * model.lut_tv_penalty()).backward()")
    i_clip = src.index("clip_grad_norm_(model.parameters(), 1.0)")
    i_acc = src.index("(loss / grad_accum).backward()")
    assert i_acc < i_tv < i_clip
