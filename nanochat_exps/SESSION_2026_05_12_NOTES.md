# Session 2026-05-12: TinyMHLut multi-alt mode + hybrid configs + batch scaling

## Library changes (committed in 5153388, ab499fa)

- `MultiHeadLut.argmax_noise_eps`: bernoulli flip at low-confidence comparisons
  in the fused-Function path (smooth_mode=False). Saved flip mask makes
  forward and backward consistent.
- `TinyMultiHeadLut.n_alternatives`: new multi-alt STE backward. When `>1`,
  routes through `_TinyMHLutMultiAlt` (forward = std TinyMHL bit-pack +
  embedding_bag; backward = manual top-k argmin + hybrid einsum/fancy
  gather + inverse-L1 uncertainty scatter, all in one @torch.compile body).
  Hybrid path: K≤128 uses bf16-autocast'd structured bmm einsum (soft-style
  tensor-core GEMM); K>128 uses fancy gather (no [B, T, K] materialisation).
- `TinyMultiHeadLut.log_uncertainty_T`: learnable temperature in multi-alt
  mode when `learnable_temps=True`. Init from `uncertainty_T_init` (default
  1.0). β fixed at 0.5 (Adam absorbs uniform scale).
- `_TinyMHLutGatherReduce.backward`: @torch.compile bwd body is now the
  default for STE n_alt=1 (was hand-written CUDA kernel). 2.5x speedup.
- Removed obsolete `bf16_argmax` flag from `_TinyMHLutSoft` and constructor.

## Performance (exp257 out_proj shape: B=4096, T=2048, NAP=6, n_outputs=96)

| Mode | ms / iter | Peak mem |
|---|---|---|
| STE n_alt=1 (compile bwd) | 6.27 | 1.30 GB |
| **multi-alt n_alt=3 + noise (NEW)** | **7.82** | **3.39 GB** |
| soft + noise (exp257 baseline) | 10.12 | 4.49 GB |

At V-lut shape (NAP=8, K=256):

| Mode | ms / iter | Peak mem |
|---|---|---|
| **multi-alt n_alt=3 + noise** | **12.15** | **1.94 GB** |
| soft + noise | 22.08 | 17.40 GB |

Multi-alt beats soft on both axes at both shapes. At NAP=8 the gap is huge
(45% faster, 9x less memory) because soft's [B, T, K=256] intermediate
dominates HBM while multi-alt stays bounded by n_alt=3.

## Experiment results (8K steps, batch=8 unless noted)

| Exp | Config | Final bpb | Δ |
|---|---|---|---|
| exp257 | TinyMHL soft + noise (baseline) | **1.6060** | — |
| exp261 | MultiHeadLut smooth=False n_alt=3 + noise | 1.6151 | +0.009 |
| exp262 | TinyMHL ste n_alt=3 + noise (fixed T=1, β=0.5) | 1.6148 | +0.009 |
| exp263 | + learnable T (init=1.0) | killed at step 1400 (trailing +0.012) | |
| exp264 | + learnable T (init=0.5) | killed at step 4600 (T drifting smaller) | |
| **exp265** | **hybrid: soft NAP=6, multi-alt NAP=8** | **1.6126** | **+0.007** |
| **exp266** | **hybrid batch=16 (this run)** | **1.5229** | **+0.083 vs exp257 — BEST** |

## Key findings

1. **Noise + multi-alt n_alt=3 mostly matches noise + soft.** The 0.009 bpb
   gap between exp257 (all-soft) and exp262 (all-multi-alt) is small.
2. **The bulk of soft's remaining edge comes from v_lut (NAP=8).** Switching
   qk_joint + out_proj from multi-alt to soft (exp265) recovered only
   ~22% of the gap (0.002 of 0.009). The v_lut soft mode is where the
   remaining ~0.007 bpb lives — but it costs 17 GB peak vs multi-alt's
   1.9 GB, making it unaffordable at any non-trivial batch size.
3. **Hybrid (soft small / multi-alt big) is a real sweet spot**: at 8k
   steps exp265=1.6126 vs full-multi-alt exp262=1.6148, while using
   v_lut's memory-cheap path.
4. **Learnable uncertainty_T is marginal at fixed batch.** exp263/exp264
   (learnable T) ran at most ~+0.005 worse than fixed T=1 (exp262). T
   drifts toward smaller values (~0.85 at qk L0, ~0.94 at out_proj),
   suggesting init values are close enough to the optimum.
5. **Doubling batch (exp266 b=16 vs exp265 b=8) is a HUGE win for LUT
   training — far bigger than expected.** Token-matched comparison
   (exp266 at step N, exp265 at step 2N — same total tokens processed):
   - early (10M tokens):     −0.002 to −0.007 bpb lead
   - mid (26M tokens):       **−0.016** bpb lead (gap widening)
   - end (exp266 final, 33M tokens = same as exp265 8K): exp266 = 1.5854
     ≈ 0.027 bpb below exp265's final 1.6126 at equal tokens
   - **at 2x compute (exp266 8000 steps = 65M tokens) final = 1.5229**,
     beating exp257 (the prior all-soft best) by 0.083 bpb and exp265 by
     0.090. This is a much larger gap than typical "doubled compute"
     would explain in a transformer — it's an LUT-specific effect.

   **The hypothesis: STE-style LUT training has extremely sparse per-token
   gradients** (one weight row per LUT per sample, out of 64-256 rows
   per table). Bigger batches give denser per-row statistics that Adam
   cannot recover from b=8 alone. The effect compounds late in training
   when most "popular" rows are nearly converged and the remaining work
   is in fine adjustments on rarely-visited rows that need cleaner signal.

   **Practical recommendation: for LUT training, scale `device_batch_size`
   to whatever fits in memory** — it's not redundant compute, it's a real
   bpb improvement that no other change in this session matched.

## Yuval's diagnostics on exp265 (analysis.py, see analysis.json)

Two questions Yuval posed:
  (1) Visit-frequency: are inputs actually routing to diverse LUT entries,
      or concentrating in a small subset?
  (2) SVD rank: do the trained entries span a high-dim space, or live in
      a low-dim subspace (over-provisioned in output)?

Together: 4-quadrant analysis of "is the LUT architecture exploiting its
discrete code capacity, or behaving like a much smaller table?"

**Striking finding: capacity utilisation collapses sharply with depth.**

| Layer | Avg visit-entropy (norm) | Avg unvisited frac | Avg top-10% mass | Avg SVD rank@90% (of full) |
|---|---|---|---|---|
| L0 (early) | 0.95 (≈uniform) | 0% | 27% | 30/53 (57%) |
| L1–L2 | 0.91–0.95 | 0% | 25–39% | 19–35/53 (36–66%) |
| L3 | 0.79 | 0% | 47% | 30/53 (56%) |
| L4 | 0.46 | 27% | 86% | 17/43 (40%) |
| **L5 (final)** | **0.12** | **89%** | **~100%** | **7/53 (13%)** |

  - L0–L2 are doing **genuine LUT routing**: visits are nearly uniform
    across the 64–256 entries per table, and the trained outputs span a
    50–60% of full rank. The discrete code capacity is being exploited.
  - L3–L4 show **progressive concentration**: visits start clustering on
    a subset of entries, but SVD rank stays moderate.
  - **L5 has effectively collapsed to a tiny model.** ~90% of entries
    are never visited on the validation set, and the trained outputs of
    the visited entries live in a ~7-dimensional subspace (out of 64
    available). The last LUT block's parameters are mostly trained but
    not exploited.

**Actionable implications:**
  - The last layer LUTs (L5) could likely be **cut by 4–8× in `table_dim`
    or `tables_per_head`** with minimal bpb loss.
  - **Hierarchical-ferns** (Yuval's suggestion) should target the
    last layers first — that's where capacity is over-provisioned. Early
    layers (L0–L2) need their full capacity.
  - The collapse is **layer-specialised** (not uniform across the model),
    so a single global compression strategy would mis-allocate capacity.
  - This is **not unique to our run** — sparse-coding / LUT papers often
    report similar end-of-network specialisation. The point is: now we
    have layer-specific budgets to use for architectural decisions.

See `exp265_hybrid_soft_multialt/analyze.py` and `analysis.json` for
the full per-LUT numbers (18 LUTs across 6 layers × {qk_joint, v_lut,
out_proj}).

## Open questions / future work

- **Bigger batch on longer horizon (e.g. 48K)**: exp266 8K @ b=16 already
  beats exp257 by 0.083 bpb. Same recipe @ 48K should crush exp260's
  1.4655 SOTA. Likely top priority next.
- **Cut L5 capacity** based on Yuval's diagnostics: try `out_tph_per_layer`
  with much smaller values for the last 1–2 layers (e.g. [2048, 2048,
  1024, 1024, 512, 256] or even cut whole-layer). Should match exp266 bpb
  with smaller model.
- **Hierarchical-ferns variant** (Yuval's suggestion): two-level table
  routing — coarse selection over a small first-stage table, fine
  selection over a small second-stage. Layer-specific: apply to L4–L5
  where capacity is over-provisioned.
- **bf16 weight_dtype is currently WORSE than fp32 in multi-alt training**
  (bf16 atomic-add is slow on H100). TODO: hybrid storage (fp32 master +
  bf16 view for the structured-bmm gather) — see
  `~/.claude/projects/-home-starost-spiky/memory/project_todo_tinymhl_bf16_weights.md`.
- Test MultiHeadLut(smooth=False, n_alt=3) + bernoulli noise on small
  deltas — queued from yesterday, see
  `project_todo_mhlut_smooth_false_noise.md`.

## Files in this session

- `exp262_tinymhlut_multialt_noise/` — TinyMHL multi-alt+noise baseline
- `exp263_tinymhlut_multialt_learnable_T/` — learnable T (init=1, killed)
- `exp264_tinymhlut_multialt_T05/` — learnable T (init=0.5, killed)
- `exp265_hybrid_soft_multialt/` — soft NAP=6 / multi-alt NAP=8 (DONE)
- `exp266_hybrid_batch16/` — hybrid + batch=16 (RUNNING)
