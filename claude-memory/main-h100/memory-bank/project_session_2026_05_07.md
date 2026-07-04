---
name: 2026-05-07 nanochat session — distillation refinements & E=96 SOTA
description: New SOTA exp183=1.6284 / 302M (-37% vs prior). Distillation: smooth-mode +17% lever; out_dim is dominant Final-Linear lever. Steiner saturation (~25% pair coverage) confirmed. exp178/179 reductions with qk/v sensitivity findings.
type: project
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
# 2026-05-07 nanochat session

## Headline

**New nanochat SOTA: exp183 = val_bpb 1.6284** at 302M params, 8000 steps.
- Recipe: exp132's E=96 LUT topology + exp174's big unembedder (mult=8) + BitAttention + attn_scale_init=0.4
- Beats exp174 (1.6478 / 481M) by **−0.0194 bpb at 37% fewer params**
- Only 0.0028 bpb above the 23M vanilla baseline (1.6256) — gap to vanilla nearly closed

## Distillation framework results (continuation of 2026-05-06)

**Sparse-output Pareto frontier (out_dim>=in_dim required, 20 ep, hard mode):**
| Total | Best | KL @ 20 |
|---|---|---|
| 100M | MLP (cosine 1e-3) | 0.0072 |
| 185M | Sparse ×1 nap=10 tph=4096 n=32 out=1536 (no_resid) | 0.0144 |
| 159M | Sparse ×1 nap=10 tph=4096 n=32 out=768 (no_resid) | 0.0160 |
| 117M | Sparse ×1 nap=10 tph=4096 n=16 out=1536 | 0.0156 |
| 56M | Sparse ×1 nap=10 tph=2640 n=16 out=384 (resid) | 0.0194 |

**Smooth-mode SOTA (MultiHeadLut + smooth_mode=True + n_alternatives=1 + SparseScatter):**
| Variant | Total | KL @ 20 | KL @ 40 |
|---|---|---|---|
| MLP cosine 1e-3 | 100M | 0.0072 | 0.0045 |
| **Smooth ×1 nap=10 tph=2048 n=32 out=1536** ✨ | **117M** | **0.0131** | **0.0083** |
| (Smooth nap=8 tph=8192 n=8 out=1536) | 67M | 0.0149 | — |

Gap to MLP at convergence: ~1.8× (stable across 20→40 ep).

## Key levers (refined ranking)

In sparse-output regime (residual + Linear final preserved):
1. **Smooth-mode (n_alternatives=1)** — clean +17% improvement at fixed config (vs hard-mode). Largest single architectural lever found in distillation.
2. **out_dim** (Final Linear width) — non-trivially monotone:
   - 192 → 384: ~−25% KL
   - 384 → 768: ~−12%
   - 768 → 1536: ~−10%
   - Below in_dim=384 hits a cliff (33% worse).
3. **Per-table expressivity (nap)** — diminishing returns:
   - 6→8: ~−27%
   - 8→10: ~−18%
4. **Votes per output (`tph · n_sparse / out_dim`)** — saturates ~42 in wide-Linear regime.
5. **Pair coverage** — saturates ~25% per head (in both hard and smooth modes). Above 50% gives nothing extra at fixed votes.

## Pure-LUT chain experiments (negative result, but informative)

Replacing the unembedder Final Linear with a second LUT (LUT 1 → LUT 2 → vocab) hits a ~0.25 KL ceiling **even at 537M params** (single big LUT) and ~0.07 KL with bias + tph=16384. The `Linear(out_dim, vocab)` is **structurally important** — discrete LUT lookups can't approximate it efficiently for many-class projection.

**Conclusion:** LUTs are great for inner mixing; Linear(hidden, vocab) is irreplaceable for the vocab head.

## Transformer experiments (today)

### Reduction sweep starting from exp174 (qk_tph saturation)
| Variant | Params | val_bpb | Δ vs exp174 |
|---|---|---|---|
| exp174 | 481M | 1.6478 | — |
| exp177 (qk_tph=192) | 368M | 1.6510 | +0.003 |
| **exp178 (qk_tph=96)** | **311M** | **1.6537** | **+0.006** |
| exp179 (qk=v=96) | 264M | 1.6666 | +0.019 |
| exp180 (+ d_v=24) | 257M | killed (~+0.04) | — |
| exp181 (+ out sparse, nap=8) | 232M | killed (~+0.05) | — |

**Findings:**
- qk LUTs are massively over-parameterized (saturated 4–9× over Steiner). Halving costs ~+0.003 bpb.
- v LUTs are ~5× more sensitive per-saved-param than qk (representational).
- d_v reduction stacks badly with v reductions; out_proj sparse-scatter at this scale hurt.

### exp176 — smooth-LUT unembedder swap (failed direction)
- Replaced exp174's MLP unembedder with smooth-LUT distillation SOTA (nap=10 tph=2048 n=32 out=1536).
- Cost +0.044 bpb (1.6478 → 1.6914 at 497M) with V2D/D2V kept; +0.024 with V2D/D2V removed.
- **The Linear(hidden, vocab) is irreplaceable in transformer too** (matches distillation finding).

### exp182 — wider unembedder hidden (failed)
- exp178 + unembed_hidden 3072→4096: +33M params, no improvement. unembed_hidden=3072 was already at sweet spot for E=64.

### exp183 — NEW SOTA recipe (success)
**Architecture (val_bpb=1.6284 / 302M):**
```
E = 96, N_LAYERS = 6, n_heads = 6, d_qk = 32, d_v = 24

Per-block LUTBlock:
  x → LayerNorm
    qk_joint LUT (nap=6, tph=256) → qk_v2d → BitAttention(±1 dom, scale=attn_scale/√P)
    v LUT (nap=8, tph=256)
  → out_proj LUT (nap=6, tph=[2048,2048,1024,1024,1024,1024]) → V2D → D2V (rank-canon, LN)
  + residual

Concat all 6 layer outputs → 576-dim
unembedder: LN(576) → Linear(576, 4608) → GELU → Linear(4608, vocab)
```
- attn_scale_init=0.4 (matched exp174)
- BitAttention TC fwd + bf16 bwd

**Decomposition:**
- E=96 efficient LUT topology (from exp132): saves ~50% LUT params vs E=64
- Big unembedder mult=8: ~150M, contributes the −0.063 from exp132→exp183 just like it did exp154→exp174
- BitAttention: math-equivalent to SDPA, faster wall-clock
- Distillation framework predicted Final Linear width was the dominant lever — held in transformer training

### exp184 — split rank-canon (small regression)
- exp183 + remove V2D/D2V from residual stream, keep on unembedder path only.
- Used `unembed_d2v(normalise=False)` since unembedder's LN handles normalization.
- Implemented as single batched V2D→D2V on stacked layer outputs.
- Cost +0.003 bpb (1.6314 vs 1.6284). **Rank-only residual is a small but real lever.**

### exp185 — smaller unembedder (regression)
- exp183 with unembed_hidden 4608 → 2304 (mult 8 → 4).
- Cost +0.034 bpb (1.6620 vs 1.6284) at −77M params. unembed_hidden=4608 is necessary, not over-parameterized.

## Complete nanochat SOTA leaderboard (after today)

| Variant | Params | val_bpb |
|---|---|---|
| exp001 vanilla baseline | 23M | 1.6256 |
| **exp183 ✨ (E=96 + big unembed + BitAttn)** | **302M** | **1.6284** |
| exp184 (split canon) | 302M | 1.6314 |
| exp174 (E=64 + big unembed) | 481M | 1.6478 |
| exp177 (qk=192) | 368M | 1.6510 |
| exp178 (qk=96) | 311M | 1.6537 |
| exp185 (smaller unembed) | 225M | 1.6620 |
| exp179 (qk=v=96) | 264M | 1.6666 |
| exp132 (E=96 vanilla unembed) | 167M | 1.6943 |

## Distillation framework changes (this session)

`nanochat_exps/distill_unembedder/fit_lut_stack.py` extended with:
- `--block_n_sparse N` — sparse_scatter per block (each table writes N of in_dim slots)
- `--block_out_dim D` — override LUT scatter output dim (only with `--no_residual`)
- `--no_residual` — remove the residual `x = x + LUT(x)` add
- `--smooth_mode` — use MultiHeadLut(smooth_mode=True, n_alternatives=1) + SparseScatter wrapper class `SmoothLutSparse`
- `--final_lut`, `--final_lut_nap/tph/n_sparse/bias` — replace Final Linear with a sparse-scatter LUT (chain experiment)

`nanochat_exps/distill_unembedder/fit_lut.py` extended with:
- `--lr_schedule constant|cosine`, `--warmup_steps`

## Open directions

- **Steiner-style anchor sampling** (PER_HEAD_DISJOINT policy) — sketched but not implemented; redundant for exp178's regime since CANONICAL_FULL_COVERAGE already does per-head disjoint when slots_per_head ≤ P. Worth implementing for over-Steiner regimes.
- **Informativeness-weighted anchor pairs** — `Var(x_i - x_j)` from calibration data, weighted sampling. Untested.
- **Smooth-mode in transformer** — distillation showed +17% lever; transformer integration could try `MultiHeadLut(smooth_mode=True, n_alternatives=1)` for qk/v/out_proj LUTs.
- **Combine exp183 + exp178's qk_tph reduction** — predicted ~280M total at maybe 1.63 bpb. Would push Pareto further.
