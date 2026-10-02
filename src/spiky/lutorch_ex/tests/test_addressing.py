"""Pin the MSB-first bit-packing convention."""
import torch

from spiky.lutorch_ex import LUTSpec, ManifestoHardLUT
from spiky.lutorch_ex.addressing import msb_first_powers


def test_msb_first_powers_values():
    # pair i -> 2**(nap-1-i): pair 0 is the most-significant bit, last pair is the LSB.
    assert msb_first_powers(4).tolist() == [8, 4, 2, 1]
    assert msb_first_powers(1).tolist() == [1]
    p = msb_first_powers(6)
    assert int(p[0]) == 2 ** 5 and int(p[-1]) == 1
    assert p.dtype == torch.long


def test_cartridge_packs_msb_first():
    spec = LUTSpec(h_in=1, h_out=1, tph=1, nap=5, d_in=8, d_out=2)
    cart = ManifestoHardLUT(spec, seed=0)
    assert cart.powers.tolist() == [16, 8, 4, 2, 1]
    assert int(cart.powers[0]) == 2 ** (spec.nap - 1), "pair 0 must map to the high bit"
    assert int(cart.powers[-1]) == 1, "the last pair must be the low bit"
