# nanochat_exps — Experiment Journal

LUT-based transformer experiments on the nanochat BPE corpus
(V=32768, ClimbMix, 512-token context, 4096 effective batch tokens).
All experiments train an autoregressive LM and report `val_bpb` (bits per byte).

## Standard hyperparameters (unless noted)

```
n_steps        = 8000        # most exps
context        = 512
device_bs      = 8
total_bs       = 4096
adam_lr        = 3e-4
weight_decay   = 0.1
warmup_frac    = 0.1
random_seed    = 42
canon_T        = 0.1
attn_scale     = 0.25 (learnable)
bit_lut_lr     = 1e-3
bit_lut_β2     = 0.95 (post exp020), 0.999 before
latent_dtype   = bf16
soft_backward  = true
```

## Top-line results (sorted by val_bpb)

| Exp | val_bpb | n_steps | E | Arch summary |
|---|---:|---:|---:|---|
| exp041_vanilla_48k | **1.3406** | 48000 | 384 | Vanilla MinimalGPT — non-LUT reference |
| exp001_minimal_gpt | 1.6256 | 8000 | 384 | Vanilla MinimalGPT 8K reference |
| exp039_dqk32_heads6_48k | 1.6598 | 48000 | 64 | LUT capacity test, 48K steps |
| exp051_e96_outnap100 | **1.7187** | 8000 | 96 | fp32 Q/K + dense out_proj (out_nap=100) |
| exp057_full_mhlut | **1.7389** | 8000 | 144 | All-fp32-MultiHeadLut — best at E=144 |
| exp018_dqk32_heads6 | 1.7404 | 16000 | 64 | Sep Q/K bit-LUT, β2=0.999 |
| exp050_e96_concat_full_tph | 1.7509 | 8000 | 96 | fp32 Q/K + bit V/out_proj |
| exp056_e144_dv24 | 1.7628 | 8000 | 144 | Bit-LUT @ E=144, d_v=24 |
| **exp054_qk_sparse_deep** | **1.7671** | 8000 | 96 | **Best 8K bit-only (sparse-deep Q/K)** |
| exp055_qk_nap128 | 1.7698 | 8000 | 96 | Best 8K bit-only at lean budget |
| exp052_e96_qk_bml_joint | 1.7758 | 8000 | 96 | Joint bit Q/K, exp042 hyperparams |
| exp042_qk_multiheadlut_joint | 1.7852 | 8000 | 64 | First joint fp32 Q/K MHLut |
| exp053_e96_balanced | 1.7947 | 8000 | 96 | Q/K votes/pair=16.5 (too low) |
| exp046_qk_mhlut_dom_residual | 1.8778 | 8000 | 64 | Dominance residual stream A/B |
| exp044_dom_residual_raw | 1.8804 | 8000 | 64 | Raw dominance residual |
| exp045_e96_full_mhlut | 1.8747 | 8000 | 96 | Earlier all-MHLut attempt |

8K LUT SOTA: **exp057 = 1.7389** (fp32 MHLut), **exp054 = 1.7671** (best bit-only).

## Architectural progression (chronological story)

### exp001-exp017 — Foundational LUT baselines
- **exp001 (1.6256)**: vanilla MinimalGPT — non-LUT reference baseline.
- **exp002-exp008**: introduced LUT primitives (BitPermutationLUT for V, out_proj). E=64, sum-mode pos.
- **exp007 (1.8356)**: separate Q/K BitPermLUTs at d_qk=32, n_heads=4, 8K — early LUT-only baseline.
- **exp008/009 (1.8483/1.8174)**: joint Q/K via BitMultiHeadLUT(n_outputs=2*P_qk=992). exp009 wins.
- **exp010-exp013**: explored Q/K hybrid Linear (1.9253), out_proj variants.
- **exp014-exp017**: MLP unembedder, partition_sets, capacity scaling.

### exp018-exp026 — First serious LUT-only SOTA
- **exp018 (1.7404 @ 16K)**: separate Q/K, d_qk=32, n_heads=6, β2=0.999 — first sub-1.74 LUT result.
- **exp019**: flat lr ablation (didn't beat exp018).
- **exp020 (1.8286 @ 8K)**: β2=0.999 → 0.95. Established β2=0.95 as standard.
- **exp021-exp023**: joint Q/K with partition_sets. **exp023 (1.8167)** = best 8K LUT-only at the time.
- **exp024-exp026**: E=72 / d_qk=24/32 line; didn't improve over E=64 baseline.

### exp027-exp032 — Failed init + MultiBit tries
- **exp027/028**: saturated init + weight decay (failed, stalled at 2.38).
- **exp029-exp032**: MultiBitPermutationLUT K=4 with various init_std and tph configs. None beat K=1.

### exp033-exp038 — Regularization + batch experiments
- **exp033/034**: contrastive decorrelation (λ=1e-3 broken; λ=1e-5 negligible).
- **exp035**: fp32 PermutationalLut (didn't outperform bit-LUT at 8K; limited by training horizon).
- **exp036/037/038**: batch size ablations. exp037 (bs=1) plateaued; exp038 (ramp) didn't help.

### exp039-exp041 — Capacity scaling
- **exp039 (1.6598 @ 48K)**: exp018 architecture × 48K steps. Showed LUT capacity is real with enough training.
- **exp040**: bit_lut_lr 1e-3 → 2e-3 (worse — 1e-3 is well-tuned).
- **exp041 (1.3406 @ 48K)**: vanilla 48K reference. Vanilla still ~0.32 ahead at scale.

### exp042 — Joint fp32 Q/K MultiHeadLut breakthrough
- **exp042 (1.7852 @ 8K, E=64)**: replaced separate Q/K bit-LUTs with single joint MultiHeadLut producing 2·d_qk per head, V2D'd into Q/K dominance. **First serious dominance-residual-free fp32 Q/K** — 0.030 better than exp018-style. New 8K LUT SOTA.

### exp043-exp048 — Dominance residual stream experiments (failed)
- **exp043-044**: residual stream as P-dim dominance, with/without canon at residual add. Both worse than concat unembedder.
- **exp046**: clean A/B vs exp042 with dominance residual. **+0.093 worse** than exp042 → concat-of-layer-outputs unembedder carries significant info.
- **exp047**: D2V with normalise=False (killed early, no gain).
- **exp048-049**: E=32 ablations. E=32 hurts ~0.05 vs E=64.

### exp050-exp051 — E=96 scaling at concat unembedder
- **exp050 (1.7511)**: exp042 at E=96 with full tph schedule. **New 8K LUT SOTA.** E=96 wins decisively over E=64 even with same Q/K capacity.
- **exp051 (1.7187)**: + dense out_proj (out_nap=100). Best 8K LUT result. Confirms more votes per pair help (with diminishing returns).

### exp052-exp054 — Bit Q/K replacements
- **exp052 v2 (1.7758)**: bit BitMultiHeadLUT for joint Q/K (in_nap=5, out_nap=32, tph=1024 after fix). 60× compression in Q/K weights vs fp32.
- **exp053 (1.7947)**: lean balanced bit-LUT — Q/K votes/pair too low (16.5).
- **exp054 (1.7671)**: sparse-deep Q/K (in_nap=8, tph=128, out_nap=256). 22.5% input pair coverage but rich per-table capacity. **Best 8K bit-only.**

### exp055 — Lean balanced bit-LUT winner
- **exp055 (1.7698)**: minimum-sufficient-capacity recipe. Q/K nap=6/tph=256/out_nap=128 (votes/pair=33), V nap=7/out_nap=10/tph=256, out_proj front-loaded. **144M total bits = 18 MB** (vs exp054's 474M, 3.3× smaller). Validates "Q/K votes/pair ≥ 32 is essential, everything else can be lean."

### exp056-exp057 — E=144 scaling
- **exp056 (1.7628)**: exp055 architecture at E=144, d_v=24. **Best bit-LUT at this scale.** Shows E and d_v scale well at fixed bit budget despite V/out_proj votes/pair dropping.
- **exp057 (1.7389)**: full fp32 MHLut at E=144, d_v=24. **New 8K SOTA.** ~3× faster training than bit-LUT (better-optimized PyTorch kernels).

### exp058 — Fused QKV (failed)
- v1-v5: tried single fused BitMultiHeadLUT producing Q+K+V together. All variants underperformed exp055 by ~0.10 due to forced shared anchor partition + pos_emb leak into V. The "additive Q/K from V's LUT" v5 had the best warmup but couldn't hold — fused architecture is intrinsically weaker than separate Q/K + V.

### exp059 (running) — Progressive LUT growth
- Schedule [(0, 0.25), (2K, 0.25), (4K, 0.5), (6K, 1.0)] adds new bit-LUTs to all three projection types. Joint training (no freezing). FLOPS-matched to exp055 fixed run; ends with 2× capacity.

### exp060 (queued) — fp32 MHLut at E=96
- exp057-style architecture at E=96 (matching exp055's geometry). Tests fp32 MHLut at the lean E=96 setup.

## Key architectural insights (collected)

1. **E width matters more than per-pair vote density.** exp056 vs exp055 (E=144 vs 96 at fixed bit budget): +0.007 by widening E. Going E=32 hurts ~0.05.

2. **Q/K votes/pair ≥ 32 is essential.** Below ~16 hurts noticeably (exp053). Above ~33 is diminishing returns.

3. **V can be dramatically over-served (137 votes/pair) without hurt — and dramatically under-served (9.3 in exp056) only marginally hurts at E=144.** V is the most forgiving of the three projections.

4. **Sparse input-pair coverage works.** exp054 at 22.5% per-head Q/K coverage beats exp052 v2 at 112% — *fewer, richer tables* is better than *many lean tables*.

5. **Concat-of-layer-outputs unembedder ≫ dominance residual stream** at the 8K scale. The concat carries 0.03-0.09 bpb of useful information that the residual loses.

6. **Joint fp32 Q/K MHLut is much better than separate fp32 Q/K** (exp042 vs predecessors). Aligned input lookup.

7. **Bit-LUT can replace fp32 MHLut with ~0.01-0.02 bpb cost and 44× storage compression.** exp055 vs exp042 at same E=64-vs-bigger.

8. **Bit-LUT training is ~3× slower per step than fp32 MHLut** due to:
   - Larger output dims (P-dim dominance vs E-dim Euclidean) ~10-70× more memory traffic
   - Suboptimal custom CUDA kernels (weight_grad_kernel at 0.7% HBM bandwidth)

9. **Fused QKV with shared anchor partition is intrinsically worse than separate Q/K + V** — the shared partition can't simultaneously serve all three projections optimally; pos_emb also leaks into V.

10. **bit_lut_lr=1e-3 with β2=0.95 is well-tuned**. Higher lr (exp040) doesn't help.
