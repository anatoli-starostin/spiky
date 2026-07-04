---
name: lut-lr-sweep-tiny
description: LR sweep on LUT param group for tiny LUT-LM. lut_lr=1e-3 (3.3x adam_lr=3e-4) is the sweet spot; ~−0.01 bpb gain. 1e-2 reverts to baseline; 3e-3 mediocre.
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Tiny LUT-LM LR sweep on LUT param group (2026-05-15, exp353/354/355, killed at step 2400)

Forks of exp352 (noise=0 baseline = 1.6302) splitting the LUT param group's LR from the global `adam_lr=3e-4`. Same Adam betas/wd/cosine schedule on all groups; only the LUT-group `initial_lr` differs.

| exp | lut_lr | step 1000 | step 1400 | step 2400 | Δ vs exp352@2400 |
|---|---|---|---|---|---|
| baseline (exp352) | 3e-4  | 1.8929 | 1.8409 | 1.7673 | — |
| **exp353** | **1e-3** | 1.8801 | **1.8285** | **1.7575** | **−0.0098** |
| exp354 | 3e-3  | 1.8757 | 1.8299 | 1.7594 | −0.0079 |
| exp355 | 1e-2  | 1.8782 | 1.8338 | 1.7667 | −0.0006 |

**Pattern**: higher LR helped initially (1e-2 led through step 800–1000), then collapsed back to baseline by step 2400. 1e-3 (intermediate) emerged as best — neither over-shoots like 1e-2 nor leaves the early-phase gains on the table like baseline.

**How to apply**: use `lut_lr=1e-3` as the default for tiny LUT-LM forks (E=48-ish, NAP=6–8). Patch needed in train.py: split LUT group from `adam_lr`:

```python
_LUT_LR = cfg.get('lut_lr', cfg['adam_lr'])
adam_groups = [
    dict(params=lut_params, lr=_LUT_LR, ...),
    ...
]
```

**Scope**: only tested at bs=16, 43.1 M, 8K steps. The right `lut_lr` may scale differently at bigger bs (denser per-row gradients should support higher LR since variance is lower — could re-sweep at bs=48/96).

**Bigger insight** (user observation that ended the sweep): "LR tuning is shaping the existing gradient signal; the real question is what makes that signal good." The 0.01 bpb gain is modest compared to bs-scaling (−0.11 to −0.19). True next step is variance-reduction-style tricks (grad_accum simulation, slow-EMA, per-row gradient buffer, sparse-aware Adam) — see [[tiny-lut-batch-scaling]].
