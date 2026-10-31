"""DominanceStackLUT -- per-head stack ``DominanceLUT (hard, 'same' output anchors)  ->  second LUT layer``.

A ``MultiHeadLUT`` ([B, h_in, d_in] -> [B, h_out, d_out]) so it plugs into ``ProjectionMHL`` unchanged.
Both layers are G-batched (the G = 8 heads are one dimension of each layer's tables, no Python loop over heads), the
heads stay independent through the stack: head g's d_in slice -> its own Dominance tables -> its own d_out vector
-> its own tables of the second layer. Layer 1 is ``DominanceLUTFast`` (Triton gather + dense Borda; numerically
tested against the reference ``DominanceLUT``), layer 2 is any fused Manifesto cartridge (``FusedManifestoHardLUT``
rational for the stack under test; also ``FusedManifestoHardLUT`` at a smaller tph for the depth control).

The second layer is a ``ManifestoLUT`` submodule, so ``cell_tv_penalty(model)`` (mean over the model's Manifesto
cartridges) sees it and its table dropout is its own; the Dominance layer has no regulariser.
"""
from __future__ import annotations

import torch

from spiky.lutorch_ex import LUTSpec
from spiky.lutorch_ex.lut_base import MultiHeadLUT

from dominance_fast import DominanceLUTFast


class LUTStack(MultiHeadLUT):
    """Generic ``first -> second`` stack of two G-batched layers (used for the depth control)."""

    def __init__(self, spec: LUTSpec, first: MultiHeadLUT, second: MultiHeadLUT):
        super().__init__(spec)
        s1, s2 = first.spec, second.spec
        if (s1.h_in, s1.d_in) != (spec.h_in, spec.d_in) or (s2.h_out, s2.d_out) != (spec.h_out, spec.d_out) \
                or (s1.h_out, s1.d_out) != (s2.h_in, s2.d_in):
            raise ValueError("stack layer shapes do not chain")
        self.first, self.second = first, second

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.first(x))


class DominanceStackLUT(MultiHeadLUT):
    def __init__(self, spec: LUTSpec, second: MultiHeadLUT, *, dom_seed: int = 0, dom_weight_init_std=None,
                 dom_nop=None, dom_vote: str = "delta", dom_normalize: str = "none"):
        super().__init__(spec)
        dom_spec = LUTSpec(h_in=spec.h_in, h_out=spec.h_out, tph=spec.tph, nap=spec.nap,
                           d_in=spec.d_in, d_out=spec.d_out, anchor_mode="pairs")
        self.dom = DominanceLUTFast(dom_spec, nop=dom_nop, output_anchors="same", vote=dom_vote,
                                    normalize=dom_normalize, seed=dom_seed, weight_init_std=dom_weight_init_std)
        s2 = second.spec
        if (s2.h_in, s2.d_in) != (spec.h_out, spec.d_out):
            raise ValueError(f"second layer input [{s2.h_in},{s2.d_in}] != dominance output [{spec.h_out},{spec.d_out}]")
        self.second = second

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.second(self.dom(x))
