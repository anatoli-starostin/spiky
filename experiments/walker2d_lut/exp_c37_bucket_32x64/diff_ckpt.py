"""Compare two actor checkpoints tensor by tensor. Used to localise where the optimised
trainer diverges from the baseline, if it does."""
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
bad = 0
for k in sorted(a.files):
    if a[k].dtype.kind not in "fc":
        continue
    d = float(np.abs(a[k] - b[k]).max())
    r = d / max(float(np.abs(a[k]).max()), 1e-30)
    tag = "EXACT" if d == 0.0 else "DIFF"
    bad += 0 if d == 0.0 else 1
    print(f"  {tag:<6} {k:<14} max|Δ| {d:.3e}   rel {r:.3e}")
print(f"\n  {bad} of {len([k for k in a.files if a[k].dtype.kind in 'fc'])} "
      f"float tensors differ")
