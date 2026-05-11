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
| exp266 | hybrid batch=16 (this run) | RUNNING — at step 3200 = 1.6127 (already matched exp265's final) | |

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
5. **Doubling batch (exp266 b=16 vs exp265 b=8) shows late-game compounding
   gains.** Token-matched comparison (exp266 at step N, exp265 at step 2N):
   - early (10M tokens): −0.002 to −0.007 bpb lead from cleaner gradients
   - later (26M tokens): **−0.016 bpb lead** — gap widening
   The hypothesis: STE-style LUT training has extremely sparse per-token
   gradients (one weight row per LUT per sample), and bigger batches give
   denser per-row statistics that Adam can't recover from b=8 alone.
   This is more pronounced in late training when most rows are already
   nearly converged and fine adjustments need cleaner signal.

## Open questions / future work

- **bf16 weight_dtype is currently WORSE than fp32 in multi-alt training**
  (bf16 atomic-add is slow on H100). TODO: hybrid storage (fp32 master +
  bf16 view for the structured-bmm gather) — see
  `~/.claude/projects/-home-starost-spiky/memory/project_todo_tinymhl_bf16_weights.md`.
- Test MultiHeadLut(smooth=False, n_alt=3) + bernoulli noise on small
  deltas — queued for next session, see
  `project_todo_mhlut_smooth_false_noise.md`.
- exp266 final result (in progress at time of writing).

## Files in this session

- `exp262_tinymhlut_multialt_noise/` — TinyMHL multi-alt+noise baseline
- `exp263_tinymhlut_multialt_learnable_T/` — learnable T (init=1, killed)
- `exp264_tinymhlut_multialt_T05/` — learnable T (init=0.5, killed)
- `exp265_hybrid_soft_multialt/` — soft NAP=6 / multi-alt NAP=8 (DONE)
- `exp266_hybrid_batch16/` — hybrid + batch=16 (RUNNING)
