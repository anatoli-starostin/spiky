"""Lock in the optimized read paths: hard eval reads one cell; otherwise a fused pair read."""
import torch

from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT, ManifestoSoftLUT


def _instrument(cart):
    counts = {"read": 0, "pair": 0}
    orig_read, orig_pair = cart._read, cart._read_pair

    def read(idx, _o=orig_read):
        counts["read"] += 1
        return _o(idx)

    def pair(c, c_alt, _o=orig_pair):
        counts["pair"] += 1
        return _o(c, c_alt)

    cart._read = read
    cart._read_pair = pair
    return counts


def _spec():
    return LUTSpec(h_in=2, h_out=2, tph=2, nap=3, d_in=4, d_out=3)


def test_hard_eval_uses_single_read():
    cart = ManifestoHardLUT(_spec(), seed=0).eval()
    counts = _instrument(cart)
    with torch.no_grad():
        cart(torch.randn(4, cart.spec.h_in, cart.spec.d_in))
    assert counts == {"read": 1, "pair": 0}, counts  # eval shortcut: one gather, no alt


def test_hard_train_uses_fused_pair():
    cart = ManifestoHardLUT(_spec(), seed=0).train()
    counts = _instrument(cart)
    cart(torch.randn(4, cart.spec.h_in, cart.spec.d_in)).sum().backward()
    assert counts == {"read": 0, "pair": 1}, counts  # train: both cells, one fused gather


def test_soft_always_uses_fused_pair():
    for training in (False, True):
        cart = ManifestoSoftLUT(_spec(), seed=0)
        cart.train(training)
        counts = _instrument(cart)
        cart(torch.randn(4, cart.spec.h_in, cart.spec.d_in))
        assert counts == {"read": 0, "pair": 1}, (training, counts)


def test_compile_not_used_on_cpu():
    # torch.compile is GPU-only; a CPU forward must stay eager (self._compiled stays None).
    for cls in (ManifestoHardLUT, ManifestoSoftLUT):
        cart = cls(_spec(), seed=0)
        cart(torch.randn(2, cart.spec.h_in, cart.spec.d_in))  # CPU tensor
        assert cart._compiled is None
