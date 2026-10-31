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
from dominance_conf import DominanceConfFast


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
                 dom_nop=None, dom_vote: str = "delta", dom_normalize: str = "none",
                 post_norm: str = "none", dom_conf: bool = False, beta_init: float = 2.0, gamma_init: float = 1.0):
        super().__init__(spec)
        dom_spec = LUTSpec(h_in=spec.h_in, h_out=spec.h_out, tph=spec.tph, nap=spec.nap,
                           d_in=spec.d_in, d_out=spec.d_out, anchor_mode="pairs")
        dkw = dict(nop=dom_nop, output_anchors="same", vote=dom_vote, normalize=dom_normalize, seed=dom_seed,
                   weight_init_std=dom_weight_init_std)
        # dom_conf: layer 1 is the ConfidenceLUT(n=1)-scored Dominance (``DominanceConfFast``), rows scaled by the score s
        self.dom = (DominanceConfFast(dom_spec, beta_init=beta_init, gamma_init=gamma_init, **dkw) if dom_conf
                    else DominanceLUTFast(dom_spec, **dkw))
        s2 = second.spec
        if (s2.h_in, s2.d_in) != (spec.h_out, spec.d_out):
            raise ValueError(f"second layer input [{s2.h_in},{s2.d_in}] != dominance output [{spec.h_out},{spec.d_out}]")
        self.second = second
        # post_norm: "none" | "ln" (LayerNorm over the d_out coordinates, learnable affine, w=1 b=0) | "ln_noaffine".
        # Applied on [B, G, d_out] per token and head, between the Dominance layer and the second layer. The Dominance
        # output is exactly zero-sum per head, so the mean is already 0 and LN is a per-token rescale to unit variance.
        if post_norm not in ("none", "ln", "ln_noaffine"):
            raise ValueError(f"unknown post_norm {post_norm!r}")
        self.post_norm = post_norm
        self.norm = None if post_norm == "none" else torch.nn.LayerNorm(
            spec.d_out, elementwise_affine=(post_norm == "ln"))
        self.reset_hidden_std()

    def reset_hidden_std(self):
        self._hs_sum, self._hs_n = 0.0, 0
        if isinstance(self.dom, DominanceConfFast):
            self.dom.reset_score_stat()

    def mean_score(self) -> float:
        return self.dom.mean_score() if isinstance(self.dom, DominanceConfFast) else float("nan")

    def mean_hidden_std(self) -> float:
        """Mean over tokens and heads of the std (over the d_out coords) of the Dominance output, accumulated over
        eval-mode forwards since the last ``reset_hidden_std`` (before the LN)."""
        return self._hs_sum / max(self._hs_n, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.dom(x)
        if not self.training:
            with torch.no_grad():
                sd = h.detach().float().std(dim=-1, unbiased=False)
                self._hs_sum += float(sd.sum()); self._hs_n += sd.numel()
        if self.norm is not None:
            h = self.norm(h)
        return self.second(h)
