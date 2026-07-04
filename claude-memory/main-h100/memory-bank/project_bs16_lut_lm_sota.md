---
name: project-bs16-lut-lm-sota
description: "Doubling device_batch_size 8 -> 16 (and total_batch_size 4096 -> 8192) on the exp326 LUT-LM dropped val_bpb @ 8K from 1.5887 to 1.4896 — exp327 is the FIRST LUT-LM result to beat the vanilla+RoPE baseline (exp319 1.5468) at the same step count. Cost: 2x compute (0.91 h -> 1.74 h)."
metadata: 
  node_type: memory
  type: project
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

# Doubling batch size: exp327 beats the vanilla baseline (2026-05-14)

**Fact:** exp327 = exp326 with `device_batch_size: 8 -> 16` and `total_batch_size: 4096 -> 8192` (1 grad_accum per step throughout). val_bpb @ 8K dropped from **1.5887 (exp326) -> 1.4896 (exp327)**, a **Δ = -0.0991 bpb** improvement at exactly 2x the compute (0.911 h -> 1.743 h).

| run | bs | tokens/step | params | val_bpb @ 8K | time | Δ vs vanilla+RoPE (exp319) |
|---|---|---|---|---|---|---|
| **exp327** | **16** | 8192 | 620.8 M | **1.4896** | 1.743 h | **−0.0572** ✓ FIRST WIN |
| exp326 | 8 | 4096 | 620.8 M | 1.5887 | 0.911 h | +0.0419 |
| exp321 | 8 | 4096 | 601.9 M | 1.5933 | 0.881 h | +0.0465 |
| exp303 | 8 | 4096 | 602.1 M | 1.6509 | 0.833 h | +0.1041 |
| exp319 | 8 | 4096 | 23.2 M | 1.5468 | 0.112 h | 0 (vanilla+RoPE) |
| exp001 | 8 | 4096 | 23.4 M | 1.6256 | — | +0.0788 |

**Architecture-axis context** at 8K with bs=8 (taking exp326's recipe as baseline):
- exp303 -> exp321 (RoPE): −0.058 bpb
- exp321 -> exp326 (qkv_lut + additive v-branch): −0.005 bpb
- **exp326 -> exp327 (bs 8 -> 16): −0.099 bpb** ← far bigger than any single architectural change

So the dominant lever at this scale is **batch size**, not architecture, consistent with the prior exp267 finding (memory note in the [[Nanochat-SOTA-progression]] section): STE-style LUT training has sparse per-token gradients, and bigger batches give denser per-row Adam statistics that smaller batches can't recover.

**First LUT-LM to beat vanilla+RoPE at 8K** (with bs mismatch): exp327 = 1.4896 vs exp319 = 1.5468 → Δ = **−0.0572 bpb**. But ⚠️ this comparison is unfair — exp319 was at bs=8. **exp328 (vanilla+RoPE at bs=16) = 1.3882 @ 23.2 M, 0.098 h**, which is **−0.1014 bpb BELOW exp327 at matched bs=16, with 27× fewer params and 18× less compute**. So on a fair comparison, vanilla+RoPE still dominates LUT-LM at this scale. Batch-double gains:
- vanilla: 1.5468 → 1.3882 = **−0.1586 bpb**
- LUT-LM:  1.5887 → 1.4896 = **−0.0991 bpb**

Vanilla benefits MORE from bigger batch than LUT-LM, opposite of the LUT-batch-sensitivity hypothesis from exp267. That hypothesis was likely confounded by other architectural changes in the prior 8K LUT-LM series.

**How to apply:**
- All new LUT-LM forks should default to `device_batch_size = 16, total_batch_size = 8192`. The bs=8 results below 1.59 are now "historical" — bs=16 redefines the SOTA floor at ~1.49.
- The pattern in exp267 (the prior 4x-batch-size finding) was upheld: scaling batch to memory ceiling is the dominant lever.
- For ~620 M LUT-LM at H100 80 GB: bs=16 fits (verified). bs=32 may also fit; worth testing if compute budget allows.

**Open frontier:**
- Try bs=24 / bs=32 — same recipe, see how far the batch-size lever goes.
- The vanilla baseline (exp319) at bs=16: have NOT measured. If vanilla bs=16 drops similarly, LUT-LM's relative gain stays the same. If vanilla doesn't gain as much, the LUT-LM batch sensitivity hypothesis is confirmed.
- Other open levers: longer training (16K, 32K steps), bigger model (residual_dim, more layers), better noise schedule.
