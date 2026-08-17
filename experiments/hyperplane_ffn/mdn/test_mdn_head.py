"""Shape/gradient smoke test for MDNHead (general block B) + LowRankLinearHead."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdn_head import MDNHead, LowRankLinearHead


def check_head(head, V, extra_loss=None):
    h = torch.randn(4, 384, requires_grad=True)
    logits = head(h)
    assert logits.shape == (4, V), logits.shape
    assert torch.isfinite(logits).all()
    assert head(torch.randn(2, 3, 384)).shape == (2, 3, V)
    tgt = torch.randint(0, V, (4,))
    loss = torch.nn.functional.cross_entropy(logits, tgt)
    if extra_loss is not None:
        loss = loss + extra_loss()
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"zero grad {name}"


def test():
    torch.manual_seed(0)
    d, V = 384, 1000
    for B in (3, 4, 5):
        for N, M in ((11, 1), (24, 1), (11, 8)):
            head = MDNHead(d, V, n_maps=N, n_mix=M, block=B)
            pc = MDNHead.param_count(d, V, N, M, B)
            actual = sum(p.numel() for p in head.parameters())
            assert pc['total'] == actual, (B, N, M, pc['total'], actual)
            check_head(head, V, head.decorrelation)
        print(f"MDN block={B} OK (param_count + shapes + grads for N in 11,24 / M in 1,8)")

    for r in (33, 120):
        lr = LowRankLinearHead(d, V, rank=r)
        pc = LowRankLinearHead.param_count(d, V, r)
        assert pc['total'] == sum(p.numel() for p in lr.parameters())
        check_head(lr, V)
        print(f"LowRankLinear r={r} OK (param_count + shapes + grads)")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test()
