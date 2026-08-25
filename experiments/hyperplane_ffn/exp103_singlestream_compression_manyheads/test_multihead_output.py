"""Focused tests for CompressionMultiHeadLUTMH (the external multihead-output subclass).

Run: sbox .venv/bin/python experiments/hyperplane_ffn/exp101_.../test_multihead_output.py
Asserts:
  (a) default path (multihead_output=False) is bit-for-bit identical to the stock
      CompressionMultiHeadLUT (same seed/config) — nothing existing changes;
  (b) multihead_output=True, inner_out_dim=-1, n_heads=6, output_dim=64 returns [N,6,64];
  (c) ValueError when multihead_output=True with inner_out_dim != -1 (decompress mixes heads);
  (d) sanity: the collapsed (stock) output equals the multihead output summed over heads.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from spiky.lutorch.compression_mhl import CompressionMultiHeadLUT
from mh_compression import CompressionMultiHeadLUTMH

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
CFG = dict(input_dim=384, output_dim=64, inner_in_dim=48, inner_out_dim=-1,
           nap=6, tph=32, n_heads=6, forward_mode="hard", use_bf16=True)


def _mk(cls, seed=123, **extra):
    torch.manual_seed(0)
    m = cls(**CFG, random_seed=seed, device=DEV, **extra).to(DEV).eval()
    return m


def test_default_unchanged():
    # Same seed/config -> the subclass with multihead_output=False must match stock exactly.
    stock = _mk(CompressionMultiHeadLUT, seed=777)
    sub = _mk(CompressionMultiHeadLUTMH, seed=777, multihead_output=False)
    sub.load_state_dict(stock.state_dict())
    x = torch.randn(16, 384, device=DEV)
    with torch.no_grad():
        a = stock(x); b = sub(x)
    assert a.shape == b.shape == (16, 64), (a.shape, b.shape)
    assert torch.equal(a, b), f"default path diverged: max|d|={(a-b).abs().max()}"
    print("(a) default path bit-for-bit identical  ✓  shape", tuple(a.shape))


def test_multihead_shape():
    stock = _mk(CompressionMultiHeadLUT, seed=42)
    mh = _mk(CompressionMultiHeadLUTMH, seed=42, multihead_output=True)
    mh.load_state_dict(stock.state_dict())
    x = torch.randn(16, 384, device=DEV)
    with torch.no_grad():
        y = mh(x)
    assert y.shape == (16, 6, 64), y.shape
    print("(b) multihead_output shape", tuple(y.shape), " ✓")
    # (d) collapsed stock == multihead summed over heads
    with torch.no_grad():
        collapsed = stock(x)
        summed = y.sum(dim=1)
    assert collapsed.shape == summed.shape == (16, 64)
    d = (collapsed - summed).abs().max().item()
    assert d < 1e-3, f"sum(heads) != stock collapsed: max|d|={d}"
    print(f"(d) stock == sum-over-heads of multihead  ✓  (max|d|={d:.2e})")


def test_valueerror_on_decompress():
    bad = dict(CFG); bad['inner_out_dim'] = 48   # has_decompress -> must raise
    try:
        CompressionMultiHeadLUTMH(**bad, random_seed=1, device=DEV, multihead_output=True)
    except ValueError as e:
        print("(c) ValueError on inner_out_dim!=-1  ✓ :", str(e)[:70])
        return
    raise AssertionError("expected ValueError for multihead_output with inner_out_dim!=-1")


if __name__ == '__main__':
    test_default_unchanged()
    test_multihead_shape()
    test_valueerror_on_decompress()
    print("\nALL TESTS PASSED")
