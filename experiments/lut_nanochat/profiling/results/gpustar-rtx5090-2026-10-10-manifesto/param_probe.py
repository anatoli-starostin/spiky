"""Instantiate ProjectionMHL(cartridge) 1536 -> r -> 1536 for each family at the Stage 2a README geometries and count
parameters, to check P = r*(2*1536 + 1 + tph*2^nap) + 1536 + scalars per family."""
import torch

import spiky.lutorch_ex as lx
from spiky.lutorch_ex.cartridges.fused_confidence import FusedConfidenceLUT
from spiky.lutorch_ex.lut_spec import LUTSpec


def formula(r, tph, nap):
    return r * (2 * 1536 + 1 + tph * 2 ** nap) + 1536


FAMS = {
    "ManifestoHardLUT": lambda s: lx.ManifestoHardLUT(s, seed=0),
    "FusedManifestoSoftLUT": lambda s: lx.FusedManifestoSoftLUT(s, seed=0),
    "ConfidenceLUT n=1": lambda s: lx.ConfidenceLUT(s, seed=0, read_top_n=1),
    "ConfidenceLUT n=2": lambda s: lx.ConfidenceLUT(s, seed=0, read_top_n=2),
    "FusedConfidenceLUT n=1": lambda s: FusedConfidenceLUT(s, seed=0, read_top_n=1),
    "QuantisedConfidenceLUT n=2": lambda s: lx.QuantisedConfidenceLUT(s, seed=0, read_top_n=2),
    "SoftSignSmoothLUT (Gen-2, ref)": lambda s: lx.SoftSignSmoothLUT(s, seed=0),
}
for h, d, tph, nap in ((8, 8, 64, 8), (8, 8, 16, 8)):
    r = h * d
    spec = LUTSpec(h_in=h, h_out=h, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode="pairs")
    print(f"h{h} d{d} tph{tph} nap{nap} (r={r}): formula without scalars {formula(r, tph, nap):,}")
    for name, mk in FAMS.items():
        m = lx.ProjectionMHL(mk(spec), d_model=1536)
        p = sum(t.numel() for t in m.parameters())
        print(f"   {name:31s} {p:,}  (scalars = {p - formula(r, tph, nap)})  x12 = {12 * p:,}")
