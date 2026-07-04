---
name: exp405-bs64-qkv-only
description: "exp405 (bs=64, no v_lut, all NAP=6, 65M) = 1.4467 bpb. Sits between exp348 (bs=48 exp365-arch) and exp362 (bs=96 exp365-arch). Arch advantage of dropping v_lut + uniform NAP=6 GROWS with batch: +18 mb at bs=16, +30 mb at bs=64."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# exp405 — exp404 arch at bs=64 (2026-05-17)

## Setup
Same architecture as exp404 (no v_lut; qkv_lut NAP=6 tph=96 supplies q,k,v; out_proj NAP=6 tph=512; residual_lut NAP=6 tph=64). Only changes vs exp404:
- `device_batch_size` 16 → 64 (4× per-step tokens)
- `total_batch_size` 8192 → 32768 (grad_accum stays at 1)
- Drop `out_proj_multi_nap` wrapper (single-component multi-nap is equivalent to plain TinyMHLut at the same NAP/tph; just cleaner code path)

All other hyperparams identical to exp404.

## Result
- **exp405 final = 1.4467 bpb @ 64.9M params, 8K steps, 0.99h**

## Bracket comparison (closest existing batch-scaling refs, all bs varied at exp365 arch 43M)
| Run | bs | Arch | Params | Final bpb | vs exp405 |
|---|---|---|---|---|---|
| exp348 | 48 | exp365 (with v_lut) | 43M | 1.5234 | +77 mb |
| **exp405** | **64** | **no v_lut, NAP=6 all** | **65M** | **1.4467** | **—** |
| exp362 | 96 | exp365 (with v_lut) | 43M | 1.4296 | −17 mb |
| exp363 | 128 | exp365 | 43M | 1.4105 | −36 mb |
| exp364 | 192 | exp365 | 43M | 1.3769 | −70 mb |

Linear interpolation of bs=48 → bs=96 at bs=64 = ~1.4765. exp405 beats this by ~30 mb.

## Arch advantage (no-v_lut + NAP=6 uniform) scales with batch
- bs=16: exp404 (1.6033) vs exp365 (1.6215) = **+18 mb** advantage
- bs=64: exp405 (1.4467) vs interpolated exp365-arch (1.4765) = **+30 mb** advantage

The arch shift is buying MORE quality at larger batch sizes, suggesting the no-v_lut/NAP=6-uniform topology has better batch-scaling slope.

## Implications / How to apply
1. **exp405 arch is a new tiny-LUT-LM defaults candidate** for bs ≥ 64. Use NAP=6 uniformly across qkv/out_proj/residual, drop the dedicated v_lut, fold v into qkv_lut's last d_v outputs.
2. **Try exp405 arch at bs=128 / bs=192** — the arch advantage growth suggests a steeper batch-scaling slope. If the pattern holds, exp405 arch at bs=192 could approach or beat exp364's 1.3769 with the same compute.
3. **Don't backport this to small-batch regimes**: at bs=16 the advantage is only 18 mb and exp365 is still the simpler reference.

## Code changes (persistent)
None — exp405 used the existing TinyMHLut code path (no new module changes). All differences vs exp404 live in config.json and one block of train.py edits to drop the v_lut module.

## Reference: exp404 (bs=16 sibling)
- final = 1.6033 bpb @ 64.9M, 0.26h
- Tied with exp392 (1.6029 @ 43M) at bs=16 — see [project_qkv_only_no_vlut_neutral.md](project_qkv_only_no_vlut_neutral.md)
