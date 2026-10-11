"""Lock in the optimized read paths: hard eval reads one cell; hard train and soft eval read the pair in one gather;
soft train reads through embedding_bag."""
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


def test_soft_eval_uses_fused_pair_and_train_the_embedding_bag_blend():
    """Soft eval: both cells in one fused gather, then the blend. Soft train: one embedding_bag over both cells with
    the blend weights as per-sample weights (no [B, G, tph, d_out] gather at all)."""
    for training, want in ((False, {"read": 0, "pair": 1}), (True, {"read": 0, "pair": 0})):
        cart = ManifestoSoftLUT(_spec(), seed=0)
        cart.train(training)
        counts = _instrument(cart)
        cart(torch.randn(4, cart.spec.h_in, cart.spec.d_in))
        assert counts == want, (training, counts)


def test_compile_not_used_on_cpu():
    # torch.compile is CUDA-only and (on CUDA) eval-path-only; on CPU every forward stays
    # eager, so self._compiled is never built — in either eval or train mode.
    for cls in (ManifestoHardLUT, ManifestoSoftLUT):
        for training in (False, True):
            cart = cls(_spec(), seed=0)
            cart.train(training)
            cart(torch.randn(2, cart.spec.h_in, cart.spec.d_in))  # CPU tensor
            assert cart._compiled is None


# ---- unknown constructor keywords are rejected, not swallowed --------------------------------------------------------
import pytest as _pytest  # noqa: E402

import spiky.lutorch_ex as _lx  # noqa: E402
_FusedConfidenceLUT = _lx.FusedConfidenceLUT  # the public export (spec amendment E)  # noqa: E305
from spiky.lutorch_ex.tests._weights import softsign  # noqa: E402


@_pytest.mark.parametrize("cls", [
    _lx.ManifestoHardLUT, _lx.ManifestoSoftLUT, _lx.FusedManifestoHardLUT, _lx.FusedManifestoSoftLUT,
    softsign(_lx.ManifestoHardLUT), softsign(_lx.ManifestoSoftLUT), softsign(_lx.FusedManifestoHardLUT),
    softsign(_lx.FusedManifestoSoftLUT),
    _lx.ConfidenceLUT, _FusedConfidenceLUT, _lx.QuantisedConfidenceLUT,
])
def test_a_misspelt_keyword_raises_instead_of_being_ignored(cls):
    """A misspelt option (here table_dropout_rate) used to be swallowed by ManifestoLUT.__init__(**unused): the
    cartridge built fine with the option silently at its default (dropout off)."""
    spec = _lx.LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8)
    with _pytest.raises(TypeError, match=r"table_droput_rate"):
        cls(spec, seed=0, table_droput_rate=0.5)
    with _pytest.raises(TypeError, match=r"\['also_bogus', 'bogus'\]"):
        cls(spec, seed=0, bogus=1, also_bogus=2)


# ---- spec amendment B: the shared base has no runnable forward; no fused class inherits a reference forward ----------
from spiky.lutorch_ex.cartridges.confidence_base import ConfidenceBase  # noqa: E402
from spiky.lutorch_ex.cartridges.manifesto_base import ManifestoLUT  # noqa: E402
from spiky.lutorch_ex.cartridges.reference_base import ReferenceForward  # noqa: E402


@_pytest.mark.parametrize("base", [ManifestoLUT, ConfidenceBase])
def test_the_shared_bases_have_no_forward(base):
    spec = _lx.LUTSpec(h_in=2, h_out=2, tph=4, nap=3, d_in=8, d_out=8)
    class Bare(base):          # implements the abstract _forward_impl, inherits the base's forward
        def _forward_impl(self, x):
            return x

    with _pytest.raises(NotImplementedError, match="shared scaffolding base and has no forward"):
        Bare(spec, seed=0)(torch.randn(3, 2, 8))


def test_no_fused_class_inherits_a_reference_forward():
    for fused, ref in ((_lx.FusedManifestoHardLUT, _lx.ManifestoHardLUT), (_lx.FusedManifestoSoftLUT, _lx.ManifestoSoftLUT),
                       (_FusedConfidenceLUT, _lx.ConfidenceLUT)):
        assert not issubclass(fused, ref) and not issubclass(fused, ReferenceForward), fused.__name__
        assert issubclass(ref, ReferenceForward), ref.__name__


def test_every_fused_class_is_exported():
    """Spec amendment E: the class names are the public API, so every fused class is importable from lutorch_ex."""
    for name in ("FusedManifestoHardLUT", "FusedManifestoSoftLUT", "FusedConfidenceLUT"):
        assert name in _lx.__all__ and getattr(_lx, name).__name__ == name
