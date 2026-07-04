---
name: multinap-out-proj-sota
description: "Multi-NAP out_proj (sum of 3 TinyMHLut(soft) sub-LUTs with NAPs 4, 6, 8 and tphs 128, 64, 96) achieves new bs=16 LUT-LM SOTA = 1.6097 bpb @ 42.47 M params — beats exp365 baseline by 12 millibits at 590K fewer params, no multiplications, single-row lookup per sub-LUT."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Multi-NAP out_proj — new bs=16 SOTA (2026-05-17, exp387)

User's idea: instead of fixed NAP per LUT module, use multiple TinyMHLut(soft) instances inside one module with DIFFERENT NAPs, summing their outputs. Targets the row-collapse pathology documented in exp382 (upper-layer out_proj at 1% touch_frac with NAP=8).

## Implementation

`nanochat_exps/exp387_multinap_out_proj/tiny_multi_nap_multi_head_lut.py` — wrapper module:
```python
class TinyMultiNapMultiHeadLut(nn.Module):
    def __init__(self, input_dim, n_heads, n_outputs, nap_tph_pairs, base_random_seed, **shared_kwargs):
        # nap_tph_pairs: list of (nap, tph) tuples
        self.luts = ModuleList([
            TinyMultiHeadLut(input_dim, n_heads, n_outputs, nap, tph,
                             random_seed=base_random_seed + 1000*i, **shared_kwargs)
            for i, (nap, tph) in enumerate(nap_tph_pairs)])
    def forward(self, x): return sum(lut(x) for lut in self.luts)
```

Each inner LUT keeps single-row lookup via sign-pack. No multiplications anywhere.

## Configuration (exp387)

Only out_proj modified. Per layer:
- NAP=4, tph=128 → 2,048 rows (very dense, ~512 tokens/row at bs=16 — always-active baseline)
- NAP=6, tph=64 → 4,096 rows (medium)
- NAP=8, tph=96 → 24,576 rows (specific patterns, shrunk from baseline tph=128)

Total per-layer out_proj params: 1.47M (vs baseline 1.57M = 93%). Total tables per layer: 288 (vs baseline 128). Total model: 42.47M (vs baseline 43.06M, −590K).

All other modules unchanged from exp365 (qkv_lut NAP=6 tph=16, v_lut NAP=8 tph=32, residual_lut NAP=6 tph=64).

## Results

| Step | exp387 | exp365 | Δ |
|------|--------|--------|---|
| 200  | 2.2962 | 2.2975 | −0.001 |
| 400  | 2.0522 | 2.0627 | −0.011 |
| 800  | 1.9086 | 1.9154 | −0.007 |
| 2000 | 1.7681 | 1.7804 | −0.012 |
| 4000 | 1.6801 | 1.6904 | −0.010 |
| 6000 | 1.6293 | 1.6418 | −0.013 |
| 7000 | 1.6170 | 1.6286 | −0.012 |
| 8000 | **1.6097** | 1.6215 | **−0.0118** |

Lead of −7 to −14 millibits sustained across all 40 evals. **No collapse-then-recovery dynamic** — the multi-NAP setup is just consistently better.

## Comparisons

| Model | val bpb | params | inference cost |
|-------|---------|--------|----------------|
| exp365 (TinyMHLut, NAP=8 single) | 1.6215 | 43.06 M | 1 row lookup / token / table |
| exp386 (MHLut, n_alt=3, smooth=True) | 1.6164 | 43.06 M | **3 lookups + multiplications** |
| **exp387 (multi-NAP, NAPs 4/6/8)** | **1.6097** | **42.47 M** | **3 lookups (3 separate tables), no mult** |

exp387 strictly dominates both:
- vs exp365: −0.0118 bpb, fewer params, comparable inference (3 small lookups vs 1 large).
- vs exp386: −0.0067 bpb, fewer params, **no multiplications** (preserves the matmul-free LUT design goal).

## Why this works

The NAP=4 sub-LUT has only 16 rows per table — at bs=16 (8192 tokens), each row gets ~512 hits on average, fully dense, no collapse possible. It provides a robust "baseline output" that the model can rely on even when the NAP=8 sub-LUT's specific-pattern routing collapses to 1% of rows. The two are *additive*, so the NAP=8 component contributes "specificity correction" on top of the dense NAP=4 baseline.

This is the first architectural change in the exp366–exp387 sweep that gives a clear win with NO inference cost penalty.

## How to apply

- **Use multi-NAP out_proj as default for new LUT-LM forks.** Worth re-running exp364 (bs=192 SOTA = 1.3769) and the bs=128 chain (exp363) with this swap.
- Configuration: `out_proj_multi_nap = [[4, 128], [6, 64], [8, 96]]` per layer.
- Module-scope follow-ups: try multi-NAP on qkv_lut, residual_lut, v_lut (they had less collapse, but might still benefit from coarse-pattern baseline).
- Anchor alignment is currently independent (each sub-LUT samples its own anchor pairs with different seed). Could try nested/prefix alignment (NAP=4 anchors = first 4 of NAP=8) for multiscale interpretation. Untested.
- Equal-params-per-NAP variant (tph_4=683 vs 128) is untested — would dense NAP=4 dominate? Worth a sweep.

## What it doesn't address

Still bs=16. Vanilla exp328 = 1.3882 @ 23.2M is 0.22 bpb ahead. The gap to the bs=192 LUT-LM SOTA (exp364 = 1.3769) is 0.23 bpb and is *primarily* gradient-quality (per exp367 grad_accum proof) — multi-NAP doesn't close that on its own. But it stacks cleanly with bs scaling.
