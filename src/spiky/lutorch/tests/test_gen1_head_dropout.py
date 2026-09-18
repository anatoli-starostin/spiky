"""Regression test: gen1 (MultiHeadLut) whole-table head dropout wired via CompressionMultiHeadLUT.

CompressionMultiHeadLUT(lut_impl='gen1', head_dropout_rate=p) must pass p through to every
MultiHeadLut child as table_dropout (gen1 has no confidence score, so the LightMHL score-mask has
no analogue; the equivalent is dropping whole tables on the per-table LProjection output). The mask
is an inverted Bernoulli (survivors / (1-p)), train-only, off at eval; p=0.0 is byte-identical.
"""
import torch

from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from spiky.lutorch.multi_head_lut import MultiHeadLut


def _build(head_dropout_rate):
    torch.manual_seed(0)
    return CompressionMultiHeadLUT(
        input_dim=384, output_dim=384, inner_in_dim=48, inner_out_dim=48,
        nap=8, tph=128, n_heads=4, lut_impl="gen1", forward_mode="hard",
        head_dropout_rate=head_dropout_rate, random_seed=1,
    )


def test_gen1_head_dropout_wired_to_table_dropout():
    ffn = _build(0.2)
    mhls = [m for m in ffn.modules() if isinstance(m, MultiHeadLut)]
    assert mhls, "no MultiHeadLut built for gen1"
    assert all(abs(m.table_dropout - 0.2) < 1e-12 for m in mhls)


def test_gen1_head_dropout_off_by_default():
    ffn = _build(0.0)
    for m in ffn.modules():
        if isinstance(m, MultiHeadLut):
            assert m.table_dropout == 0.0


def test_gen1_head_dropout_train_stochastic_eval_deterministic():
    ffn = _build(0.2)
    # init projection weights are ~0, so give them signal to make the mask observable
    for m in ffn.modules():
        if isinstance(m, MultiHeadLut):
            with torch.no_grad():
                m.projection.weights.normal_(0.0, 1.0)
    x = torch.randn(32, 384)
    ffn.eval()
    with torch.no_grad():
        e1, e2 = ffn(x), ffn(x)
    assert torch.allclose(e1, e2), "eval must be deterministic (dropout off)"
    ffn.train()
    torch.manual_seed(10); t1 = ffn(x)
    torch.manual_seed(20); t2 = ffn(x)
    assert not torch.allclose(t1, t2), "train must be stochastic (dropout on)"


def test_gen1_head_dropout_unbiased():
    ffn = _build(0.2)
    for m in ffn.modules():
        if isinstance(m, MultiHeadLut):
            with torch.no_grad():
                m.projection.weights.normal_(0.0, 1.0)
    x = torch.randn(32, 384)
    ffn.eval()
    with torch.no_grad():
        ref = ffn(x)
    ffn.train()
    torch.manual_seed(0)
    acc = torch.zeros_like(ref)
    n = 600
    with torch.no_grad():
        for _ in range(n):
            acc += ffn(x)
    acc /= n
    rel = (acc - ref).abs().mean() / ref.abs().mean()
    assert rel < 0.05, f"inverted dropout should be unbiased, rel err {rel.item()}"
