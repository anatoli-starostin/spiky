"""Shape/gradient smoke test for MDNHead."""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mdn_head import MDNHead


def test():
    torch.manual_seed(0)
    d, V, N, M = 384, 1000, 11, 8
    head = MDNHead(d, V, n_maps=N, n_mix=M)

    # param count matches
    pc = MDNHead.param_count(d, V, N, M)
    actual = sum(p.numel() for p in head.parameters())
    assert pc['total'] == actual, (pc['total'], actual)
    print(f"param_count OK: total={actual:,} (X={pc['X']:,} P={pc['P']:,} b={pc['b']:,})")

    # forward [B,d] -> [B,V]
    h = torch.randn(4, d, requires_grad=True)
    logits = head(h)
    assert logits.shape == (4, V), logits.shape
    assert torch.isfinite(logits).all()
    # forward [B,T,d] -> [B,T,V]
    h2 = torch.randn(2, 3, d)
    assert head(h2).shape == (2, 3, V)
    print("shape OK: [B,d]->[B,V] and [B,T,d]->[B,T,V]")

    # gradient flows to all head params + decorrelation adds a finite grad
    tgt = torch.randint(0, V, (4,))
    loss = torch.nn.functional.cross_entropy(logits, tgt) + head.decorrelation()
    loss.backward()
    for name, p in head.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
        assert p.grad.abs().sum() > 0, f"zero grad for {name}"
    print(f"grad OK: loss={loss.item():.4f}, all params have finite nonzero grads")

    # softmax normalization sanity: logits finite, exp-sum positive
    probs = torch.softmax(head(h.detach()), dim=-1)
    assert torch.allclose(probs.sum(-1), torch.ones(4), atol=1e-4)
    print("softmax OK")

    # decorrelation is a non-negative scalar
    dec = head.decorrelation()
    assert dec.ndim == 0 and dec.item() >= 0
    print(f"decorrelation OK: {dec.item():.6f}")
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test()
