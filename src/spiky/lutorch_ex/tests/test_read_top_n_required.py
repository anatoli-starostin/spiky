"""read_top_n has no default on any confidence cartridge: every construction states the read width, and omitting it
raises TypeError naming the argument (construction only, so it runs on CPU without the extensions)."""
import pytest

from spiky.lutorch_ex import (ConfidenceLUT, FusedConfidenceLUT, FusedQuantisedConfidenceLUT, LUTSpec,
                              QuantisedConfidenceLUT)


@pytest.mark.parametrize("cls", [ConfidenceLUT, FusedConfidenceLUT, QuantisedConfidenceLUT,
                                 FusedQuantisedConfidenceLUT])
def test_omitting_read_top_n_raises(cls):
    spec = LUTSpec(h_in=4, h_out=4, tph=8, nap=4, d_in=16, d_out=16)
    with pytest.raises(TypeError, match="read_top_n"):
        cls(spec, seed=0)
