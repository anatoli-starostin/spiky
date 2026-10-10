"""Stage 2a geometry at h16 (task b89bb43e): which h16 / tph64 / nap8 points are constructible, and their parameter
counts, by instantiating ProjectionMHL(1536 -> r -> 1536) around Manifesto and Confidence n=1 / n=2."""
import spiky.lutorch_ex as lx
from spiky.lutorch_ex.lut_spec import LUTSpec

VE = 603_979_776
FAMS = {"Manifesto": lambda s: lx.ManifestoHardLUT(s, seed=0),
        "Confidence n=1": lambda s: lx.ConfidenceLUT(s, seed=0, read_top_n=1),
        "Confidence n=2": lambda s: lx.ConfidenceLUT(s, seed=0, read_top_n=2)}
POINTS = [(16, 4, 64, 8), (16, 4, 64, 6), (16, 8, 64, 8), (16, 48, 64, 8), (16, 4, 16, 6), (16, 8, 16, 8),
          (8, 8, 64, 8)]
for h, d, tph, nap in POINTS:
    r = h * d
    tag = f"h{h} d{d} tph{tph} nap{nap} (r={r})"
    try:
        spec = LUTSpec(h_in=h, h_out=h, tph=tph, nap=nap, d_in=d, d_out=d, anchor_mode="pairs")
        row = []
        for name, mk in FAMS.items():
            p = sum(t.numel() for t in lx.ProjectionMHL(mk(spec), d_model=1536).parameters())
            row.append(f"{name}: {p:,} / x12 {12 * p:,} ({VE / (12 * p):.1f}x)")
        print(f"{tag}: " + " | ".join(row))
    except Exception as e:
        print(f"{tag}: REFUSED -- {type(e).__name__}: {str(e).splitlines()[0][:150]}")
