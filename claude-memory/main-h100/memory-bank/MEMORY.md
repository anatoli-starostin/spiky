# Spiky/LUTorch Project Memory

## Environment
- Repo: /home/starost/spiky
- Python venv: .venv (always use `.venv/bin/python`)
- GPU: NVIDIA H100 80GB HBM3, device `cuda:0` (only one GPU)
- Package installed as editable: `pip install -e .`

## Project structure
- LUTorch library: `src/spiky/lutorch/`
  - `Conv2DLut`, `MultiHeadLut`, `UnfoldConfiguration` in `multi_head_lut.py`
  - Phase 2 constraint: `n_alternatives=3, smooth_mode=True` everywhere
- Experiments: `experiments/` — full journal in `experiments/experiments.md`

## CIFAR-10 Experiments — Overall SOTA Progression
| Milestone | Exp | Val Acc | Params |
|-----------|-----|---------|--------|
| All-LUT reference | exp49 | 74.30% | 1.35M |
| All-LUT SOTA | exp85 | 85.92% | 3.03M |
| All-LUT late-ds | exp106 | 86.05% | 1.78M |
| Hybrid Conv2d+LUT ref | exp144 | 88.00% | 2.34M |
| Hybrid + LUT@32 | exp149 | **88.63%** | 2.48M |
| AlexNet baseline 50ep | exp145 | 89.69% | 2.68M |
| AlexNet baseline 100ep | exp146 | 90.22% | 2.68M |

## Batch 31 — Hybrid Conv2d+LUT (exp144–155, 50ep, batch=64)

**Reference architecture** (exp144, `experiments/reference_lut_model.py`):
```
Conv2d(3→32)+BN → LUT@16×16(k=4,s=2) → LUT@8×8(k=4,s=2) → LUT@8×8(k=3,s=1) → LUT@4×4(k=4,s=2)
CH_CONV=32, CH=24, NH=4, NAP=5, tph=ic*k²*12//(NAP*2)
classifier: Flatten → Linear(CH*16, 1024) → BN+ReLU+Drop(0.5) → Linear(1024, 10)
```

| Exp | Change | Val Acc | Params | Delta |
|-----|--------|---------|--------|-------|
| exp144 | Reference (50ep) | 88.00% | 2.34M | — |
| exp147 | Reference (100ep) | 88.34% | 2.34M | +0.34 |
| **exp149** | **+LUT@32 stride-1** | **88.63%** | **2.48M** | **+0.63** |
| exp154 | Flat CH=32 | 88.23% | 2.78M | +0.23 |
| exp155 | connected=True | 87.97% | 2.34M | −0.03 |
| exp150 | CH=28 | 87.86% | 3.01M | −0.14 |
| exp148 | CH_CONV=64 | 87.69% | 2.81M | −0.31 |
| exp152 | k=5 at c2 | 87.53% | 2.60M | −0.47 |
| exp153 | nap=6 at c3 | 87.35% | 2.81M | −0.65 |
| exp151 | nap=4 | 87.32% | 1.61M | −0.68 |

**Batch 31 findings**:
- **exp149 winner**: +LUT@32 stride-1 before first downsample → +0.63pp, fewer params. Late-ds helps hybrid too.
- CH=24 sweet spot; CH=28 worse despite more params; CH_CONV=32 better than 64
- connected=True negligible (−0.03pp); consistent with all-LUT findings
- nap=4 viable for tiny models (1.61M, only −0.68pp)
- Flat CH=32 (no bottleneck) ≈ reference; bottleneck expansion minimal benefit
- **Batch size**: always use batch=64 for hybrid experiments (tph values ~500+ OOM at batch=256)

## Key Architectural Findings (all experiments)

1. **Hybrid Conv2d+LUT** breaks through all-LUT ceiling: 88%+ vs ~86% for all-LUT
2. **Late-downsample topology** consistently +0.5–0.63pp: process at 32×32 before compressing
3. **tph formula** `ic * k² * 12 // (NAP*2)` — capacity-matched, works well
4. **nap=5 sweet spot**; nap=4 loses ~0.7pp but halves params; nap=6 marginal gains
5. **n_alternatives=3, smooth_mode=True**: always fixed (critical for gradients)
6. **connected_anchors_mode**: negligible effect at all scales tested
7. **tph scaling** beats depth and width at same param budget
8. **k=4,s=2 preferred** for downsampling; k=5 consistently hurts
9. Memory formula: `B × n_patches × tph × n_heads × 2^nap × 4 bytes`

## Batches 25–30 Key Results (all-LUT)
- **exp106** (late-ds, 50ep): 86.05% @ 1.78M — all-LUT SOTA
- **exp127** (late-ds+wide+depth, 20ep): 85.91% @ 3.41M — 20ep all-LUT SOTA
- tph=128 ceiling for all-LUT at batch=256 (32×32 stride-1 layers OOM at tph>128)
- Hybrid arch broke through: tph can exceed 256 at batch=64

## Experiments File
Full journal exp01–exp155 in `experiments/experiments.md`

## FastMultiHeadLUT(hard, dense_K) — deployment-quality SOTA @ E=384 NAP-bump
- See [exp764 tph-halved tiny Pareto](project_exp764_tph_halved_tiny_pareto.md) — **exp764 = exp760 with ALL tphs halved + eff bs doubled to 96 = 1.2116 hard @ 97.5M, 24K, 5.72h**. New tiny-deployment Pareto point: +6.8 mb vs exp760 at -44.6% params, ~1.8× lower inference HBM. Trajectory crosses exp760 at step ~14000 then settles +5-7 mb behind in late cosine. exp720 hardened (1.2909 @ 85M) strictly dominated.
- See [LUT wall leverboard](project_lut_wall_leverboard.md) — 200-step bench at exp760 scale: tphs ×0.5 = **-44.7% wall**, fwd hybrid_smooth→hard = **-37 to -39% wall** (this was the historical big win, NOT E shrink), E:192→96 + d_v:32→16 = only **-4.3% wall**. Don't tune wall via E/d_v — residual_lut + emb_resid_lut + unembedder dominate wall and are E-insensitive. For wall cuts: tph >> D >> NAP >> E/d_v. For capacity adds: reverse order.
- See [TinyMHLut hybrid_smooth hard-eval bug](project_tinymhl_hybrid_smooth_hard_eval_bug.md) — TinyMultiHeadLut's `soft_powers` is **MSB-first**; TAPL's hard STE path uses **LSB-first**. Naive `mod.backward_mode='ste'` reads wrong LUT rows → 2.93 bpb crash. **Fix**: bit-reverse-permute `weights` on axis 1 BEFORE flipping. exp720's true hard-deployable val is **1.2909** (not the soft 1.2052). FastMHL does NOT have this bug.
- See [exp735 v_lut NAP=7 SOTA](project_exp735_v_lut_nap7_sota.md) — **exp735 = exp731 with v_input_nap 6 -> 7 = 1.2138** hard-eval, new deployment SOTA, beats exp731 by 4 mb at +37.7M params (314.5M total). Recipe: same as exp731 but v_input_nap=7. exp734 confirmed residual+emb_resid NAP=7 adds nothing on top (tied exp733 @ 4K).
- See [exp731 FastMHL hard+dense_K SOTA](project_exp731_fastmhl_hard_densek_sota.md) — superseded by exp735. **exp731 (native hard fwd + K-row dense bwd) = 1.2178** hard-eval; was SOTA until exp735. Use `FastMultiHeadLUT(forward_mode='hard', backward_mode='dense_K')` for any deployment-bound LUT-LM run. ball still wins at NAP≥9 (memory).
- See [FastMHL hard/ball deployment SOTA](project_fastmhl_hard_ball_deployment_sota.md) — superseded by exp731. exp729 (hard+ball NAP+1) = 1.2360 hard-eval, was SOTA until exp731.
- See [exp730 hybrid_smooth+ball](project_fastmhl_hard_ball_deployment_sota.md) — top-2 fwd + ball NAP+1 bwd = soft 1.2115, hard 1.2810, gap +0.069. Soft training + hard inference produces ~+0.07 gap regardless of backward choice; native-hard training is the deployment path.
- See [hybrid_smooth wgrad compile ceiling](project_hybrid_smooth_wgrad_compile_ceiling.md) — `_hybrid_smooth_weight_grad` is at the torch.compile fusion ceiling (~17 ms at LUTGPT shapes); custom triton kernel attempted previously and failed. Don't retry PyTorch-level rewrites: einsum needs +6.5 GB, bf16 breaks numerics, buckets are slower, embedding_bag(per_sample_weights) regresses fwd.
- See [FastMHL hybrid_smooth fwd dispatch](project_fastmhl_hybrid_smooth_dispatch.md) — `_FastMHLutHybridSmooth.forward` dispatches on per-head **n_outputs >= 128**: bmm+sparse-S above, gather below. n_outputs (per-head N) controls tensor-core efficiency, not n_heads. Net −5.6% per LUTGPT step. Wrong "n_heads==1" criterion would miss wins of 12–118 ms on (n_heads>1, n_out>=128) shapes.
- See [FastMHL wgrad via sparse-S + bmm](project_fastmhl_wgrad_bmm.md) — `_soft_lut_bwd_body` / `_ball_lut_bwd_body` / `_ball_gather_lut_bwd_body` H phase dispatches on n_outputs >= 128 to bmm+sparse-S, replacing the atomic-add scatter. -16 to -41% bwd at n_out>=128 modules (out_proj, residual, qk), neutral at v_lut. **~10% faster end-to-end at exp731 (4.89 h → ~4.4 h)**. Trades 0.25 bf16 ULP (rel_rms 2.1e-3) in gw, safe under Lion (sign-based).

## Nanochat Transformer SOTA progression
- exp001 vanilla baseline: **1.6256 bpb** (23M params)
- exp154 LUT reference: 1.7105 (392M)
- exp174 (2026-05-06): 1.6478 (481M) — exp154 + big-unembedder MLP.
- exp183 (2026-05-07): 1.6284 bpb (302M) — E=96 + big unembedder + BitAttention.
- exp251 (2026-05-10): 1.6026 @ 8K — SoftMHLut(hard=True) + V2D learnable T + soft pipeline.
- exp257 v3 (2026-05-11): 1.6060 @ 8K — TinyMHLut(soft) drop-in + argmax_noise_eps=0.002 (within +0.003 of exp251). Validates noise-as-bf16-substitute, see `project_soft_lut_noise_regularization.md`.
- **exp260 (2026-05-11)**: 1.4655 bpb @ 48K — TinyMHLut(soft) + noise eps=0.002 + V2D learnable T, 358M, fork of exp257 to 48K. Prior 48K LUT-LM SOTA.
- **exp267 (current SOTA, 2026-05-12)**: **1.4504 bpb @ 8K** — hybrid (soft @ NAP=6, multi-alt @ NAP=8) + noise, device_batch_size=32 (4x exp257's b=8). Beats exp260's 48K SOTA by −0.015 bpb with **6× less compute**. Key insight: STE-style LUT training has extremely sparse per-token gradients; bigger batches give denser per-row Adam statistics that smaller batches can't recover. Scaling `device_batch_size` to memory ceiling is the dominant lever — much bigger than any soft-vs-multi-alt or learnable-T tweak.
- distill_unembedder framework at `nanochat_exps/distill_unembedder/` for offline LUT-vs-MLP head distillation tests; smooth-mode + sparse_scatter is the dominant architectural lever (~17% improvement).

## Bot Account
- See `reference_spikyclaudebot.md` — SSH key setup for pushing PRs as `spikyclaudebot`

## Telegram Bridge
- See `reference_telegram_bridge.md` — two-way Telegram chat for the active Claude Code session (scripts, hook, HTML formatting)
- See `feedback_telegram_no_tg_send.md` — don't call tg_send.sh; the Stop hook already mirrors normal replies to Telegram. Calling it duplicates the message and triggers a permission prompt.

## LUT Transformer Experiments
- Folder: `transformer_exps/`
- [LUT virtual bandwidth motivation](project_lut_bandwidth.md) — key evaluation metric
- [Transformer experiment summary (exp299–exp350+)](project_transformer_exp_summary.md) — BitPermLUT / PermutationalLut / MultiBitPermutationLUT results + distillation methodology
- [Permutational architecture](project_permutational_architecture.md) — no-residual design, backward ranking prediction, 32! capacity argument
- Baseline e2e result: **exp329 val=1.379 @ 25k**, `BitPermutationLUT` all modules (in=10, tph=2048, on=10 for out_proj). exp338 (+partition_sets) is 1.401.

### Distillation framework
- `transformer_exps/distill_exp338/` — per-layer (X_out_proj_input, Y_out_proj_output) pairs from exp338; candidate LUT stacks trained to match **pair-wise sign of Borda-projected E-dim output**.
- Key non-determinism findings are documented in `src/spiky/lutorch/determinism.py` and covered in the summary.

### Module primer
- `BitPermutationLUT` (1-bit), `PermutationalLut` (fp32, soft-rational), `MultiBitPermutationLUT` (K∈{2,4,8}, rational+midrise quantizer).
- `CANONICAL_FULL_COVERAGE` default pair-sampling policy; `partition_sets` for head-wise anchor restriction.

## Strategic Direction
- [PermutationalLut priority](project_permutational_direction.md) — fp8 training + 1-bit inference outweigh small accuracy gap
- [Matmul-free LM via LUTs and BitAttention](project_bitattention_matmulfree.md) — exp415 demonstrates end-to-end LM with no float matmuls; SDPA on ±1 dominance becomes popcount-based BitAttention
- [Soft-LUT noise injection](project_soft_lut_noise_regularization.md) — TinyMHLut(soft) needs `argmax_noise_eps≈0.002` to match SoftMHLut(hard=True); bf16 was doing this implicitly. Without noise: -0.013 bpb (exp252). With noise: matches within 0.003 (exp257, 2026-05-11).
- [TODO 2026-05-12](project_todo_mhlut_smooth_false_noise.md) — test MultiHeadLut(smooth=False, nap=3) + bernoulli noise on small deltas; queued by user 2026-05-11 to extend the noise-regularization insight to the standard WTA path.
- [TODO bf16 weights TinyMHL](project_todo_tinymhl_bf16_weights.md) — bf16 weight_dtype is currently SLOWER than fp32 in multi-alt STE training (bf16 atomic-add is slow, dtype-mismatch overhead). Don't use bf16 weights for training until hybrid storage / custom kernel exists.

## Launching Experiments
- See `feedback_launching_experiments.md` — use `python -u ... > stdout.log 2>&1 &` with `dangerouslyDisableSandbox: true`
- See `feedback_show_exp_description_before_launch.md` — always present full proposed config and wait for approval before starting a run
- See `feedback_new_folder_per_experiment.md` — every experiment fork gets a brand-new folder; never reuse the source dir unless explicitly told to
- See `feedback_long_horizon_fork_comparison.md` — never compare per-step bpb of a long-horizon fork (e.g. 48K) against its short-horizon source (e.g. 8K); warmup fraction skews everything. Compare against same-horizon peers or final only.
- See `feedback_monitor_reporting_format.md` — every Monitor-event reply during a run must include step, val metric, and delta vs reference; don't reply with just "Continuing."
- See `feedback_disk_space_check_before_exp.md` — run `df -h /` before every experiment launch; warn if root is low (only one writable disk, ~242 GB; `/mnt/cloud-metadata` is read-only).
- See `feedback_memos_not_tasks.md` — when the user dumps ideas labelled as "a memo / not actions", don't create TaskCreate entries; just acknowledge and let it go.

## Residual-LUT shape finding (2026-05-13, full 3-way sweep)
- See `project_residual_lut_prefers_more_tables.md` — at fixed residual_lut entry budget (131 072/slot, +151 M params), **wide-shape (tph=2048, NAP=6) beats deep-shape (tph=512, NAP=8) by ~0.013 bpb @ 8K**. exp303 (wide) = 1.6509 @ 602 M, new LUT-LM SOTA at 8K, beating exp300 baseline (1.6573 @ 451 M) by −0.0064 bpb. exp302 (deep) trailed exp300 by +0.005–0.008 bpb consistently. When tuning residual_lut, scale `residual_tph` not `residual_input_nap`.

## out_proj per-layer schedule (2026-05-13, full sweep exp304–exp308)
- See `project_outproj_per_layer_schedule.md` — out_proj **does NOT follow the wide-beats-deep rule**; it benefits from a **heavy-early per-layer tph schedule**. Optimal at `[2048, 2048, 1024, 1024, 1024, 1024]` (exp303). L0 boost worth −0.005 bpb @ +16.7 M; L1 boost worth −0.006 bpb @ +16.7 M; boosting L2+ is inert or negative; tapering L4-5 down to 512 is sharply harmful (+0.015 bpb). Uniform-2048 (exp305) and uniform-1024 (exp306) both ~+0.011 worse than exp303.

## qk_joint and v_lut shape sweep (2026-05-13, exp309–exp310)
- See `project_qkv_lut_shape_findings.md` — doubling `qk_tph` or `v_tph` (256 → 512 at fixed NAP, both +76 M params) is **net-negative** on top of exp303 SOTA. qk widening hurts +0.011 bpb; v widening hurts +0.005 bpb. The wide-beats-deep rule is **residual_lut-specific** — does NOT generalize to attention input LUTs. Keep `qk_tph=256, v_tph=256` as defaults. Param-spend ranking: residual_tph > out_proj L0/L1 boost ≫ v_tph > qk_tph > uniform out_proj changes.

## RoPE vs learned absolute pos_emb (2026-05-14, exp319)
- See `project_rope_vs_learned_pos_emb.md` — replacing `nn.Embedding(seq_len, n_embd)` (additive) with standard half-rotation RoPE on q,k before SDPA improves MinimalGPT @ 8K by **−0.0788 bpb** (exp001 1.6256 → **exp319 1.5468**, same depth/width/steps, ~0 param change). Use exp319 as the "vanilla RoPE" reference when comparing LUT-LM ablations.

## RoPE inside LUT-LM — new SOTA @ 8K (2026-05-14, exp321)
- See `project_rope_lut_lm.md` — RoPE applied on (q, k) post-q_norm/k_norm before SDPA inside LUT-LM beats exp303 by **−0.0576 bpb** (**exp321 1.5933** vs exp303 1.6509 @ 8K, both 602 M, same hyperparams otherwise). Biggest single architectural win in the exp30x series; bigger than wide-beats-deep, out_proj heavy-early, or noise-eps tuning. **Use RoPE for all new LUT-LM forks** — drop the additive per-layer pos_emb pattern.

## qkv_lut with additive v-branch — SOTA @ 8K bs=8 (2026-05-14, exp326)
- See `project_qkv_lut_plus_v.md` — Replace exp321's `qk_joint` (NAP=6, t=256, n_out=128) with a wider `qkv_lut` (NAP=6, t=256, **n_out=160**); route q,k from qkv_lut, and ADD qkv_lut's last d_v outputs to v_lut's output (v_lut unchanged at NAP=8). **exp326 = 1.5887 bpb @ 620.8 M (Δ=−0.0046 vs exp321)**. Key: keep dedicated NAP=8 v_lut, treat joint table's v-contribution as additive residual (shallow). Pure-joint replacements (exp322–exp325) all LOST to exp321.

## bs=16 — LUT-LM SOTA @ 8K but vanilla wins on fair compare (2026-05-14, exp327/328)
- See `project_bs16_lut_lm_sota.md` — Doubling batch on exp326 (1.5887 → exp327 1.4896, Δ=−0.0991) appeared to beat vanilla+RoPE bs=8 (exp319 1.5468). But **exp328 (vanilla+RoPE at bs=16) = 1.3882 @ 23.2 M, 0.098 h** — vanilla benefits MORE from bs=16 than LUT-LM (Δ=−0.1586 vs LUT's −0.0991). At matched bs=16, vanilla **beats LUT-LM by 0.1014 bpb** with 27× fewer params and 18× less compute. **The exp267 "LUT loves big batches" hypothesis is NOT confirmed here** — LUT-LM batch sensitivity is *lower*, not higher, than vanilla's. Use bs=16 by default still, but stop using "vanilla bs=8" as the reference; new vanilla baseline is exp328 = 1.3882.

## Soft backward beats ste at tiny LUT-LM scale (2026-05-15, exp351)
- See [project_soft_backward_beats_ste_tiny.md](project_soft_backward_beats_ste_tiny.md) — exp351 = exp340 fork with all 4 TinyMHLut modules switched soft→ste (n_alt=3, learnable_temps=True). At every step 200–4000 ste trails soft by +0.01–0.03 bpb, gap monotonically widening. Killed at step 4000; conclusion: `backward_mode='soft'` is the right default for tiny LUT-LM, even at NAP=8.

## Tiny LUT-LM: noise=0 beats noise=0.002 (2026-05-15, exp352)
- See [project_tiny_lut_noise_zero_better.md](project_tiny_lut_noise_zero_better.md) — exp352 (noise=0) = **1.6302** vs exp340 (noise=0.002) = 1.6366 → Δ=−0.0064. **Sign-flip vs the larger-scale finding** where noise=0.002 was needed; at tiny scale soft is self-regularizing enough. Use `argmax_noise_eps=0.0` for tiny LUT-LM (E=48-ish, 43 M-ish); keep 0.002 only for 300 M+ LUT-LM.

## Tiny LUT-LM LR sweep on LUT param group (2026-05-15, exp353/354/355)
- See [project_lut_lr_sweep_tiny.md](project_lut_lr_sweep_tiny.md) — split `lut_lr` from global `adam_lr=3e-4`. Sweet spot at **`lut_lr=1e-3`** (~−0.01 bpb at step 2400 vs exp352); 3e-3 mediocre; 1e-2 collapses back to baseline after early-phase lead. Modest gain compared to bs-scaling. Use `lut_lr=1e-3` going forward for tiny LUT-LM.

## LUT-group β1 ablation: momentum is load-bearing (2026-05-15, exp356)
- See [project_lut_beta1_ablation.md](project_lut_beta1_ablation.md) — β1=0 on LUT params with lut_lr=1e-3 is consistently +0.011–0.036 bpb worse than β1=0.9; even equivalent to β1=0.9 at 3.3× lower LR (the exp352 baseline). Momentum is doing real work despite the per-row sparsity. Keep β1=0.9; any Lookahead test would need to add value on top of, not replace, momentum.

## Lookahead on LUT params: mild regression (2026-05-15, exp357)
- See [project_lookahead_neutral_tiny.md](project_lookahead_neutral_tiny.md) — Lookahead(k=5, α=0.5) on LUT param group on top of AdamW(β1=0.9, lut_lr=1e-3) trails same-LR baseline by +0.001–0.008 bpb consistently. Weight-EMA on top of gradient-EMA double-smooths, kills useful Adam-driven motion. Don't use Lookahead with β1=0.9. The original Lookahead-style variance reduction we want for sparse LUT rows likely needs a different mechanism (per-row gradient buffer / sparse-aware Adam / grad_accum simulation).

## Hard-example mining (seq-level fwd 48 → bwd 16): regression (2026-05-15, exp358)
- See [project_hard_mining_tiny_lut.md](project_hard_mining_tiny_lut.md) — sequence-level hard mining (forward bs=48, backward on top-16 hardest) is consistently +0.014–0.026 bpb worse than no-mining at bs=16, and nowhere near closing the bs=48 gap. **Reason**: bs-scaling gains come from unbiased gradient averaging (better Monte Carlo estimate), NOT from picking harder examples. Hard mining samples the tail of the gradient distribution → distorts the direction. Conclusion of the optimizer/sampling sweep (exp353–358): cheap optimizer-side levers (LR, β1, Lookahead, hard-mining) all max out at ±0.01 bpb; to beat bs=16 by more requires actual bigger batch, true grad_accum, or sparse-aware Adam rewrite.

## Soft weight-gradient backward + inter-table contrastive: both dead ends (2026-05-15, exp360 + exp361)
- See [project_soft_wgrad_and_contrast_dead_ends.md](project_soft_wgrad_and_contrast_dead_ends.md) — Two LUT-specific algorithmic ideas both failed: (a) sel_soft-weighted weight gradient across all K rows (instead of hard index_add at chosen row) → mild regression (+0.007–0.010 bpb worse than baseline) due to reduced chosen-row update magnitude and noisy soft-tail Adam state. (b) Inter-table cosine contrastive loss at λ=0.01 was effectively dormant (tables start near-orthogonal with init std=0.001; contrast value 0.005 × λ=0.01 = 5e-5 vs main loss ~6) — indistinguishable from no-contrast. Whole optimizer-side / cheap LUT-side sweep (exp353–361) caps out at ±0.01 bpb; bs-scaling remains the dominant lever. Code mod: `_GLOBAL_SOFT_WEIGHT_GRAD` toggle + `_soft_lut_bwd_body_soft_w` in `src/spiky/lutorch/tiny_multi_head_lut.py` (off by default).

## Tiny LUT-LM SOTA @ 8K — exp364 = 1.3769 **BEATS vanilla** (2026-05-15)
- See [project_tiny_lut_sota_exp362.md](project_tiny_lut_sota_exp362.md) — Progression: exp362 (bs=96 + noise=0 + lut_lr=1e-3) = 1.4296 → exp363 (bs=128) = 1.4105 → **exp364 (bs=192) = 1.3769 @ 43.1 M, 2.00 h**. **First tiny-LUT-LM to beat vanilla bs=16 baseline** (exp328 = 1.3882 @ 23.2 M) — by −0.0113 bpb, with 1.86× params and ~20× more compute. bs scaling still steep (bs=128→192 = −0.0336); no saturation observed up to bs=192. **Use exp364 (bs=192) recipe as default base for further tiny-LUT-LM forks.**

## Tiny LUT-LM batch sweep — bs=16/48/96 @ exp340 shape (2026-05-15, exp340/348/349/350)
- exp340 (bs=16) = 1.6366 @ 43.1 M. exp348 (bs=48) = **1.5234** (−0.113). exp349 (bs=96) = **1.4478** (−0.189). Batch scaling not yet saturated at bs=96; bs=48→96 = −0.076 still steep. exp350 (bs=48 + out_tph/v_tph/qkv_tph all 2×) = 1.4848 @ 62.5 M — **batch scaling beats LUT widening per param** at this regime. Vanilla bs=16 exp328 = 1.3882 @ 23.2 M is the bandwidth/quality target. Use bs=96 for further tiny-LUT-LM forks.

## Grad-accum confirms bs=16+accum=8 == bs=128 direct (2026-05-16, exp367)
- See [project_grad_accum_reproduces_big_batch.md](project_grad_accum_reproduces_big_batch.md) — exp367 (bs=16 phys, accum=8) tracked exp363 (bs=128 phys, accum=1) within +0.002 to +0.011 bpb across all eval points 200–3600. **Full 0.21-bpb gap between exp365 (bs=16) and exp363 (bs=128) is pure gradient-quality** — no phys-batch matmul effect at LUT-LM scale. Any bs=16 optimiser trick has up to 0.21 bpb of recoverable headroom.

## Windowed-grad smoothing on LUT params — dead end (2026-05-16, exp368/369)
- See [project_windowed_grad_dead_end.md](project_windowed_grad_dead_end.md) — sliding-window mean of last W=8 micro-batch grads (`WindowedGradSmoother` ring buffer, LUT params only). **exp368 (W=8 + β1=0.9): −0.07 to −0.10 worse than exp365** at every step (double-smoothing on top of Adam EMA). **exp369 (W=8 + β1=0): matches exp365 in early phase but spikes and underperforms ~+0.09 once peak LR hits** (uniform window can't adaptively damp). Gradient-space variance reduction at fixed phys-batch is a closed direction; don't try W-variants. Only working bs=16 levers left: true grad_accum (8× wall-clock), sparse-aware Adam, or distillation.

## β1 is load-bearing for LUT params at any batch size (2026-05-16, exp370)
- See [project_beta1_load_bearing_at_bs128.md](project_beta1_load_bearing_at_bs128.md) — at bs=128, dropping β1 on LUT params (`lut_beta1=0.0`) costs +0.07 bpb during warmup and +0.12 at peak LR vs exp363. β1=0.9 is implicitly a sparse-aware integrator: integrates ~10 micro-batches of per-row gradient history. Row-sparsity persists at any practical batch size. Sparse-aware Adam should *augment* β1, not replace it. Up-knob tested in exp371 (β1=0.99 at bs=16).

## bs=16 nominal "win" — MultiHeadLut(n_alt=3, smooth) beats by 5 millibits at 3× inference cost (2026-05-17, exp386)
- See [project_mhlut_smooth_beats_tinymhlut.md](project_mhlut_smooth_beats_tinymhlut.md) — **exp386 = 1.6164 vs exp365 = 1.6215, Δ=−0.0051**. But: hardened inference (smooth=False or n_alt=1) destroys the model by +0.25 bpb. The smooth+3-alt mechanism is REQUIRED at inference, not training-only. Pays 3× LUT bandwidth + multiplications for 5 millibits. **Not adopted as default — exp365 (TinyMHLut soft, single-row, no mult) remains the practical bs=16 reference.** Open path: distill exp386 → TinyMHLut(soft) student.

## bs=16 LUT-LM SOTA — multi-NAP out_proj (2026-05-17, exp387)
- See [project_multinap_out_proj_sota.md](project_multinap_out_proj_sota.md) — **exp387 = 1.6097 bpb @ 42.47 M, bs=16, 8000 steps**. Δ=−0.0118 vs exp365 at 590K fewer params, no multiplications, 2.25× inference bandwidth (288 tables × 1 lookup vs 128 × 1). Recipe: out_proj = sum of 3 `TinyMHLut(soft)` sub-LUTs with `[(NAP=4, tph=128), (NAP=6, tph=64), (NAP=8, tph=96)]`. Strictly dominates exp386 (n_alt=3 smooth, multiplications). **Superseded by exp390** (NAP=4-only, larger gain at higher bandwidth) as the bs=16 quality SOTA. exp388 (multi-NAP on ALL modules) only beat baseline by 3 millibits — multi-NAP is a TARGETED fix, not a general one.

## bs=16 LUT-LM new SOTA + reframing — NAP=4-only out_proj (2026-05-17, exp390)
- See [project_nap4_only_out_proj_sota.md](project_nap4_only_out_proj_sota.md) — **exp390 = 1.6009 bpb @ 43.06 M (param-matched to baseline)**. Δ=−0.0206 vs exp365, −0.0088 vs exp387. Recipe: out_proj = single TinyMHLut(soft) with NAP=4, tph=2048 (16× more tables than baseline NAP=8 tph=128, each 16× smaller). Cost: **16× LUT bandwidth per token** vs baseline (2048 lookups vs 128). User insight from this run: the row-collapse pathology is really a **gradient-coverage at small batch** problem — NAP=8 needs ~256 tokens/row uniformly to train all rows, but bs=16+argmax doesn't deliver that. NAP=4's 16 rows × 512 tokens/row easily covers all rows → all rows train. Three options to recover NAP=8 inference at low bandwidth: (a) grad_accum at bs=16, (b) **distill from exp364 bs=192 SOTA (checkpoint at exp364_bs192/checkpoint.pt)**, (c) accept exp390's bandwidth cost. Distillation untested but most promising — could deliver exp390-class accuracy at exp365 bandwidth.

## NAP curriculum via additive merge — mechanism works, didn't beat baseline (2026-05-17, exp396-exp398)
- See [project_curriculum_nap_merge.md](project_curriculum_nap_merge.md) — Top-down anchor tree + additive weight merge (parent[bits_A, bits_B] = child_A + child_B) for stage transitions NAP=1→2→4→8. Mechanism works (function-preserving, smooth transitions). Adam state carry needs AVG merge (not sum). **Final result exp398 = 1.7684 bpb** (significantly above exp365 baseline 1.6215). Bottleneck: 87M bloated architecture (qkv/residual snap to NAP=8) + bs=16 + low late-stage LR.

## 2-stage NAP=4→8 curriculum BEATS exp365 (2026-05-17, exp399)
- exp399: 2-stage curriculum, NAP=4 (bs=8, 8000 steps) → NAP=8 (bs=16, 4000 steps), per-stage cosine LR with 10% warmup, avg Adam merge across stages. **Final: 1.6191 bpb @ 87M params** (Δ=−0.0024 vs exp365 baseline 1.6215). **First curriculum experiment to beat baseline.** Total compute = exp365 (128K bs-units). Caveat: 2× baseline params (bloated NAP=8 target). Doesn't beat exp387/exp390/exp391/exp392 (16-21 mb gap). Inference bandwidth = baseline (NAP=8, 1×) but at 2× params. Future: reduce target tph at NAP=8 to match exp365 param count exactly.

## Targeted out_proj-only curriculum BEATS baseline at MATCHED arch (2026-05-17, exp400)
- exp400: curriculum applied ONLY to out_proj (NAP=4→8); qkv/v/residual stay at exp365 baseline throughout (weights copied across merge boundary). 2 stages × 4000 steps each at bs=16. **Final: 1.6196 bpb @ 43.06M params — EXACT match to exp365 architecture, params, bandwidth, compute.** Δ=−0.0019 vs exp365 baseline. First curriculum to beat exp365 at MATCHED arch. Validates that out_proj IS the place where curriculum helps (matches the bandwidth-quality U-curve finding that out_proj is the collapse-prone module). Generalisable recipe: `curriculum_modules=["out_proj"]`, NAP=4→8 with shape-aware merge (only LUT params with shape mismatch get merged; everything else copied).

## exp419 — E-residual win generalises across archs, new bs=16 SOTA (2026-05-17)
- exp419 = exp406 arch (no v_lut, NAP=6 everywhere, 65M) + E-stream residual + ln_pre/ln_post, bs=16, 8K. **Final 1.5426** vs exp404 (same arch, no residual) = 1.6033 → **Δ=−60.7 mb**. Matches exp418's gain (−60.4 mb on exp365 arch) — E-residual is a robust architectural win across architectures. exp419 is the **new bs=16 LUT-LM SOTA** (1.5426 vs prior exp418=1.5611, exp390=1.6009). 64.88M params, 0.26h training.

## exp418 — E-stream residual + pre/post LN saves 60 mb arch win (2026-05-17)
- See [project_e_stream_residual_arch_win.md](project_e_stream_residual_arch_win.md) — exp365 design has NO E-stream residual: `x_lut_next = LN(out_proj(SDPA(...)))` REPLACES the carry. Routing entropy on exp364 ckpt showed L5 out_proj 97% dead → collapsed late layers. exp418 fixes this: `ln_pre` before qkv/v_lut, E-stream residual `x_lut + out_proj`, `ln_post` before residual_lut. **exp418 = 1.5611 vs exp365 = 1.6215 → Δ=−0.0604 @ matched compute/params**. Apply to ALL future LUT-LM forks; this should become the new baseline.

## exp416 — vanilla+RoPE bs=16 24K-step long-horizon baseline = 1.2530 (2026-05-17)
- exp416 = exp328 (vanilla MinimalGPT + RoPE, 23.2M, bs=16, 8K steps → 1.3882) extended to 24K steps. **Final 1.2530** (best 1.2527 @ step 23.6K), 0.293h, 196.6M tokens. −0.135 bpb vs exp328's 8K final — vanilla still scaling smoothly with more compute. **Beats every current LUT-LM at any horizon** (exp364 LUT-LM 1.3769 used 2.7× more tokens; exp414 1.4387 same). Use exp416 as the long-horizon vanilla reference for matched-compute LUT-LM comparisons. Warmup is fraction-of-n_steps so per-step early bpb is NOT comparable to exp328 — only final.

## exp415 — NAP=3→6 multi-module curriculum FAILED (2026-05-17)
- See [project_exp415_curriculum_nap3_to_6_negative.md](project_exp415_curriculum_nap3_to_6_negative.md) — 2-stage curriculum on exp414 arch (stage 0: NAP=3, 2× tph, bs=4, 4000 steps; stage 1: NAP=6, bs=32, 3500 steps no warmup). Killed mid-stage 1 — gap to plain exp413 at matched cumulative tokens settled at **+50 mb and not closing**. Forward IS preserved by merge (hard argmax decomposes cleanly when stage-1 anchors = concat(stage-0 A, B) anchors); backward isn't (K=64 softmax sharper than 2× K=8). Stage 0 budget too small + NAP=3 too restrictive + bs change between stages + no warmup at stage 1 — all working against the curriculum.

## E-vs-d_v attribution: d_v cuts cost 3× more per param than E cuts (2026-05-17, exp409)
- See [project_E_vs_dv_attribution.md](project_E_vs_dv_attribution.md) — exp409 (E=32, d_v=16, 39.39M) = 1.6144 vs exp407 (E=48, d_v=16) = 1.6038 → **E=48→32 alone costs +10.6 mb at −3.67M params (2.9 mb/M)**. Then d_v=16→8 on top (exp408) costs +22.8 mb more at −2.65M extra (8.6 mb/M, 3× worse rate). **Don't cut d_v below 16** — head width is the most expensive lever. E shrink is cheaper but neither is param-efficient vs trimming out_tph or residual_tph.

## exp408 — shrinking E=48→32 and d_v=16→8 costs +33 mb at bs=16 (2026-05-17)
- See [project_exp408_smaller_dims.md](project_exp408_smaller_dims.md) — exp407 with E=32 (was 48), d_v=8 (was 16). **exp408 = 1.6372 @ 36.74M vs exp407 = 1.6038 @ 43.06M — Δ=+0.0334 bpb at −14.7% params**. Quality dropped faster than param savings; bad bpb-per-param trade. Don't shrink E/d_v below exp407 defaults at bs=16. Gap was monotone +25 to +34 mb throughout middle and late phases — no convergence.

## exp407 — v_lut NAP=8 tph=32 → NAP=6 tph=128 trade is NEUTRAL at bs=16 (2026-05-17)
- See [project_v_lut_nap_trade_neutral.md](project_v_lut_nap_trade_neutral.md) — Cleaned-up fork of exp392. v_lut traded param-matched (NAP=8 tph=32 → NAP=6 tph=128 = same 786K/layer, 4× bandwidth). **exp407 = 1.6038 @ 43.06M vs exp392 = 1.6029 @ 43.06M — tied (Δ=+0.0009)**. wide-beats-deep does NOT generalize to v_lut; matches `project_qkv_lut_shape_findings` (wide-beats-deep is residual_lut-specific). Keep v_lut NAP=8 small-tph. Also: cleaned-up train.py / config layout is the new clean template (no STE plumbing, no multi-nap wrapper, no qk_* fallbacks, no pos_emb).

## exp405 — qkv-only arch at bs=64 = 1.4467 bpb, arch advantage GROWS with batch (2026-05-17)
- See [project_exp405_bs64_qkv_only.md](project_exp405_bs64_qkv_only.md) — exp404 arch (no v_lut, NAP=6 uniform, qkv_lut tph=96, out_proj tph=512, residual tph=64, 65M) at bs=64 (4× exp404 tokens/step). **Final 1.4467 @ 0.99h**. Sits between exp348 (bs=48, exp365 arch, 1.5234) and exp362 (bs=96, exp365 arch, 1.4296). Beats interpolated bs=64-on-exp365-arch by ~30 mb. Arch advantage was only +18 mb at bs=16 (exp404 vs exp365); at bs=64 it's +30 mb — the no-v_lut/NAP=6-uniform topology has steeper batch-scaling slope. Worth trying at bs=128/192.

## qkv-only fork (no v_lut) — neutral at bs=16 (2026-05-17, exp404)
- See [project_qkv_only_no_vlut_neutral.md](project_qkv_only_no_vlut_neutral.md) — Fork of exp392: v_lut REMOVED, qkv_lut widened NAP=6 tph=16→96 (q,k,v all from joint table). **exp404 = 1.6033 @ 64.9M vs exp392 = 1.6029 @ 43M — tied (Δ=+0.0004)**. 1.75× LUT params + 2× attention bandwidth bought zero quality. Dual-branch (small qkv + dedicated v_lut) > single wider joint at bs=16. Don't drop v_lut to "simplify" — keep the dual-branch design from exp326.

## Bandwidth-quality U-curve for out_proj NAP choice — knee at NAP=6 (2026-05-17, exp387–exp393)
- Full curve at bs=16 (param-matched 1.57M/layer except exp387 at 1.47M, exp393 at 0.44M):
  - **exp393 (NAP=1 tph=4560 all-pairs, 35.6× bw): 1.6348 (Δ=+0.013) ← WORSE than baseline**
  - exp365 (NAP=8 tph=128, 1× bw): 1.6215
  - exp387 (multi-NAP, 2.25× bw): 1.6097 (Δ=−0.012)
  - **exp392 (NAP=6 tph=512, 4× bw): 1.6029 (Δ=−0.019)** ← KNEE
  - exp391 (NAP=5 tph=1024, 8× bw): 1.6015 (Δ=−0.020)
  - exp390 (NAP=4 tph=2048, 16× bw): 1.6009 (Δ=−0.021)
- **Curve is U-shaped in NAP**: NAP=1 (too little per-table capacity → bad) and NAP=8 (gradient-coverage problem → bad), with sweet spot at NAP=4–6.
- **Marginal bandwidth gains**: 1×→2.25× = 5.3 mb/×; 2.25×→4× = 4.0 mb/×; 4×→16× flat (≈0). NAP=1 LOSES.
- **For bs=16 deployment: use exp392 recipe** (out_proj as single NAP=6 tph=512). 91% of max gain at 1/4 the bandwidth of exp390. Best quality/bandwidth tradeoff.
- **Gradient-coverage threshold ≈ 128 tokens/row uniform**: NAP=8 (32 tokens/row) insufficient; NAP=6 (128 tokens/row) sufficient. NAP=1 over-corrects, loses per-table capacity.
- **Hierarchical curriculum design** (user idea, untested): start at NAP=1 leaves (8× bandwidth, 6% of target params), schedule merges based on per-table load-balance saturation, end at NAP=8 root with exp365's bandwidth. Top-down pairing tree: 128 NAP=8 targets × 8 anchor pairs each = 1024 NAP=1 leaves. Each merge step adds capacity (params grow). Final inference cost = baseline 1× bandwidth.

## β1 plateau on LUT params at bs=16 (2026-05-16, exp371–374)
- See [project_lut_beta1_plateau.md](project_lut_beta1_plateau.md) — full β1 sweep at bs=16: **β1∈[0.80, 0.92] flat plateau** vs β1=0.9 (±0.005 noise), **β1=0.95 mild regression** (+0.005), **β1=0.99 catastrophic** warmup-lag (+0.30 bpb permanent deficit). β1=0.9 is the right default. Increasing β1 to "match bs=128's history" fails: blind exponential can't fake bs=128's uniform-sampling-per-row coverage. Real fix needs per-row visit counts — see SparseRowAdamW math in same file.

## soft_topk: math isolation of soft-vs-STE gap (2026-05-17, exp401-403)
- See [project_soft_topk_math_isolation.md](project_soft_topk_math_isolation.md) — new `backward_mode='soft_topk'` in TinyMultiHeadLut: softmax masked to chosen + top-K 1-bit-flip neighbors. exp401 (all-STE-n3 at bs=192) trailed soft by +0.05 bpb monotonically → STE math deficit is real and scaling-resistant. exp402 (soft_topk-3, 4 rows) = 1.6402 (Δ=+0.019 vs soft 1.6215). exp403 (soft_topk-NAP, 9 rows = Hamming-1 ball) = 1.6359 (Δ=+0.014). Conclusion: **soft attribution math = 60% of the gap (closed with just 4 rows), K row count saturates fast (Hamming-1 ball ≈ top-3), remaining +0.014 gap = Hamming-≥2 mass (6.2% of softmax mass)**. soft_topk in PyTorch currently SLOWER than soft (mask overhead) — quality-isolation tool only; production speed/memory win needs CUDA kernel.

## Soft-mixture forward beats hard-pick forward @ matched arch (2026-05-19, exp444)
- See [project_soft_forward_beats_hard_exp444.md](project_soft_forward_beats_hard_exp444.md) — swapping all 4 LUT modules from `TinyMHL(soft)` (hard-pick forward) to `SoftMultiHeadLUT(hard=False)` (genuine soft mixture forward) at the exp428 arch beats exp428 by **−16.2 mb** (1.4821 vs 1.4983 @ 89.4M, same effective batch 8192 tokens/step, 8K steps). Curve shape: soft LAGS through warmup, crosses over at step ~3.4K, lead widens monotonically to step 8K (no plateau). Cost: 5× wall-clock + O(K·n_outputs) per-token memory. **NOT deployable** — soft forward defeats the matmul-free goal; treat exp444 as an UPPER-BOUND reference only, NOT a default (do not "default to hard=False"). See [[hard-forward-is-the-goal]]. **Mechanism CORRECTED by exp445** (below): win is the soft FORWARD (functional), NOT the dense soft weight-gradient — earlier "every row gets a gradient" attribution was wrong.
- See [project_soft_wgrad_neutral_exp445.md](project_soft_wgrad_neutral_exp445.md) — **exp445 (2026-05-20): hard forward + soft weight-grad (all rows) = exp428 EXACTLY (1.498315 vs 1.498293, Δ=+0.00002 noise).** Soft weight gradient contributes nothing; entire exp444 −16.2 mb is the soft FORWARD. Tracked exp428 within noise at every eval, never exp444. Confirms exp360 at 89M. exp444's gain requires soft mixture AT INFERENCE. Also: `soft_topk` truncates X-grad support, not weight grad; valuable next axis is a top-K soft FORWARD (chosen + top-K neighbors) to get exp444's gain at K+1 lookups not 2^NAP.
- See [project_effective_lr_probe_exp446.md](project_effective_lr_probe_exp446.md) — **exp446 (2026-05-20): measured REAL Adam lr (nominal × m̂/(√v̂+ε)) per layer/module in exp428.** LUT modules realize only **~22%** of nominal `lut_lr=1e-3` → real peak lr ~2.2e-4 (the 1e-3 is mostly illusory; all 4 LUT modules converge to factor≈0.22). tok_emb most damped (0.15). LUT lr variance is **within-entry (~95%, across output channels), NOT between-row** — row-starvation signature weak (3-8%) and shrinks over training; uniform across layers at convergence. Corroborates exp445 (sparsity not the bottleneck). Files in `nanochat_exps/exp446_effective_lr_probe/`.
- See [project_lut_optimizer_sweep.md](project_lut_optimizer_sweep.md) — **LUT-optimizer sweep (2026-05-20): LION β=(0.9,0.95) BEATS AdamW at HALF optimizer memory — new bs=16 LUT-LM best.** Swapped only the LUT group's optimizer (exp428 arch). **exp453 LION(0.9,0.95)=1.4967 < AdamW exp428 1.4983 (−0.0016, below Adam at every eval 600→8000)** at ½ opt-state mem (1 momentum tensor vs m+v). β₂ sweet spot (β₁=0.9): 0.99 bad (~+0.05, stale), 0.95 beats Adam, 0.90≡Signum (1.5023, +0.004 ≈Adam). The two-timescale split is load-bearing: moderate-memory β₂ accumulates sparse-row signal while β₁ keeps direction fresh — beats both Signum (no mem) and Adam (√v̂ damping). Adam's per-coord √v̂ NOT the lever (Signum already ≈Adam). LION(β₁=β₂)≡Signum exactly (exp452 confirmed). lr for sign/SGD from grad scale not Adam (LUT grad RMS≈2.1e-3). SGD+mom lr0.1=1.5319 (+0.034, decay catch-up). Caveat: −0.0016 within single-eval noise but consistent trajectory; one seed/8K. Follow-up: 2nd seed + refine β₂ (0.93/0.97), then maybe adopt as default LUT optimizer. Code in exp453_lion_b2_0p95/train.py.

## Session 2026-05-21 (exp454–475) — norm characterization, scatter-specialization, convergence dead ends
Full writeup: `nanochat_exps/summary_2026-05-21.md`. Reference = exp453 (1.4967), the standing bs=16 LUT-LM best (unbeaten this session).
- See [project_lut_prenorm_magnitude_calibration.md](project_lut_prenorm_magnitude_calibration.md) — **Pre-LUT norm (ln_pre/ln_post) does ONLY per-token magnitude calibration; learnable affine (γ,β) is INERT.** Full 2×2 (std/RMS/MAD/mean-abs × center/no-center) all ≈ exp453; exp472 (LayerNorm no-affine) = exp453 EXACTLY. Divisor keeps |d|~temperature T (training-dynamics, sign-preserving). **mean-abs `x/(mean(|x|)+eps)` = cheapest drop-in (no params/sqrt/center) → adopt exp475 (1.4962, =exp453 within noise) as new baseline RECIPE** (cost win, not bpb SOTA). WD on LUT weights redundant (acts as effective-LR only).
- See [project_lut_scatter_specialization.md](project_lut_scatter_specialization.md) — **Output-scatter + proportional tph (many narrow output-specialized tables vs fewer wide, EQUAL params) lowers bpb.** exp464 out_proj −0.0024, exp465 qkv −0.0016, **exp466 both = 1.4937 (−0.0030, lowest bpb of session)**. Cost: ~2× LUT lookups → set aside as too heavy; exp453 preferred. Mechanism: prevents low-tph rank-1 collapse.
- See [project_lut_convergence_levers_dead_ends.md](project_lut_convergence_levers_dead_ends.md) — **bs=16 bottleneck = gradient quality, not capacity/conditioning/decoder.** Dead ends: Gauss-Newton optimizer (exp455, "denoised SGD", noise not conditioning is the limit), parallel main-effect/dominance branches (exp456-458, +0.027, base LUT not deficient), LUT(x)+Linear(x) (exp463, ~0, LUT not linearly deficient), frozen decoder (exp459, body convergence is the limit). CONFIRMED: lut_lr=2e-4 optimal (LION sweep exp467-469), β₂∈[0.93,0.95] plateau (exp454). Only untried info-adding lever: distillation.
- See [project_lut_entry_output_alignment.md](project_lut_entry_output_alignment.md) — **Probe `analysis_entry_output_corr/analyze.py`**: low-tph out_proj (exp365 tph=128) collapses to redundant rank-1 code (cos≈1); high-tph (exp454 tph=1024) stays distributed (cos≈0.15). Collapse is a selection phenomenon, not stored degeneracy → why more tables help out_proj.
- New library instrumentation (default off): `enable_visit_count_stash`/`get_visit_count` in `src/spiky/lutorch/tiny_multi_head_lut.py` (per-row visit counts for the GN optimizer).
- See [project_fair_vanilla_baseline_untied.md](project_fair_vanilla_baseline_untied.md) — **Fair vanilla reference for LUT-LM is UNTIED (exp476=1.4143), not tied (exp328=1.3882).** Untying vanilla emb↔head HURTS by +0.0261 (tying = strong small-model regularizer). But LUT-LM unembedder is structurally untied (tok_emb_E E=64 vs Linear D=384) → tied vanilla is an unfair target. **Real LUT-LM gap ≈ 0.082 (vs exp476), not ~0.108 (vs exp328).** Use exp476 as the apples-to-apples baseline.

## MatmulMultiHeadLut: softmax routing = exp444 win at 5.5x speed (2026-05-22, exp489-494)
- See [project_matmul_mhlut_softmax.md](project_matmul_mhlut_softmax.md) — new dense gated-matmul LUT (no STE, pure pytorch @torch.compile, weights on unembedder's AdamW). gate sweep bs=16 vs exp475 (1.4962): unit +0.23 (common-mode), signed +0.11, layernorm +0.15, hamming OOM (int64 index 77GB), **softmax 1.4806 (BEATS exp475, ≈exp444 1.4821)**. Insight: all gates were reinventing softmax = normalized exp-hamming-kernel (the one with Σ=1 + peaked). **exp493 = exp444-class quality @ 0.384h vs exp444's 2.118h (5.5× faster)** — exp444 slowness was impl overhead not softmax. Caveat: needs all-K soft inference (not matmul-free). exp494 (running): +temp_penalty λ·ΣT_sel to push T_sel down for HARDENABILITY (train-soft-deploy-hard argmax). BUG fixed: MatmulMHL forks need `lut_optimizer=None` guard in checkpoint save (empty lut_params → NameError; exp493 lost ckpt).

## Magnitude leakage: soft signs + softmax are an inseparable package (2026-05-23, exp500 + probe)
- See [project_magnitude_leakage_softmax_package.md](project_magnitude_leakage_softmax_package.md) — exp500 = exp493 (softmax MatmulMHL, 1.4806) + STE hard-sign forward (`hard_sign_ste`: `p=p+(sign(d)−p).detach()`, fwd=hard ±1, bwd=soft). **exp500 = 1.5437, +0.063 vs exp493 and +0.046 WORSE than clean hard argmax (exp428 1.4983 / exp475 1.4962)**. Two takeaways: (1) the whole soft apparatus buys only **~0.018 bpb** over honest hard forward; (2) soft signs + softmax are an **inseparable package** — hard signs + soft mixture is worse than EITHER clean extreme (softmax neighbor-mixing is noise without a continuous soft-sign coordinate; exp500's T_sel even ROSE to 0.51 → ~12% weight blurred onto Hamming-neighbor rows; fwd-hard/bwd-soft mismatch blocks T_sel→0). **Magnitude-leakage probe** (`exp494/probe_magnitude.py`, |p| on exp494 ckpt): mean|p|=0.42, **frac|p|>0.9 = 0.00 across ALL modules**, ~uniform on [0,0.7]; T_soft learned 0.47–0.78 never→0 ⇒ exp493 is a continuous rational net, **NOT a discrete LUT, NOT hardenable**. Honest matmul-free number stays exp475 1.4962. Hard-sign gate sweep (exp501-505) + hard-sign backward (exp506, ~+0.037) folded in: softmax beats every gate under hard signs too.

## Soft win is QKV-localizable + hybrid optimizer (2026-05-23, exp507-510)
- See [project_hybrid_qkv_soft_localization.md](project_hybrid_qkv_soft_localization.md) — **exp507**: fork exp475 (all argmax, 1.4962) with ONLY qkv swapped to the soft MatmulMHLut(softmax); v/out/residual stay argmax. Optimizer split by NAME (both have ndim-3 weights): qkv dense→AdamW wd=0.1, the 3 argmax modules→LION. **exp507 = 1.4797, matches/beats all-soft exp493 (1.4806)** by making just 1 of 4 modules soft (21M/89M params) — soft/magnitude-leakage win is **qkv-localizable**, hybrid is cheap to deploy (3 modules matmul-free). **exp508** (qk-only, no v_branch, v_lut tph 256→320, capacity-matched at 89.39M) = 1.4840, +0.0043 → soft magnitude-leaking v_branch worth ~0.004 over equal hard v-tables. **exp509/510**: replace qkv with `VectorToDominance(soft)->[LN]->Linear` (no LUT, emits q,k,v, v_lut removed) — trails exp507 by +0.090 (raw) / +0.055 (with LayerNorm + small-std init 0.005); the table's per-row lookup does real work a linear readout of dominance features can't match.

## Probabilistic-forward LUT selection — dead end (2026-05-22, exp484)
- See [project_prob_forward_dead_end.md](project_prob_forward_dead_end.md) — new `backward_mode='prob'` (sample 1 row from softmax(ts/T_sel) via multinomial during TRAIN, argmax at EVAL; hard index_add weight grad at sampled row; input grad from full softmax dist, index-independent). **exp484 = 1.5474 vs exp475 argmax 1.4962, Δ=+0.0512** — gap DEAD FLAT at +0.05 across all 40 evals (structural, not early-phase). Also 2.7× slower (multinomial not @torch.compile-able). Forward noise from sub-optimal-row sampling degrades the function; the cold-row-coverage hypothesis didn't pay off. Same family as other "improve grad coverage at fixed phys-batch" dead ends. Code left in place, off by default.

## LUT convergence bottleneck = per-parameter gradient sparsity (2026-05-22, measured)
- See [project_lut_convergence_bottleneck.md](project_lut_convergence_bottleneck.md) — WHY LUT-LM converges slower than vanilla at fixed bs=16 (exp475 1.4962 @89M vs exp476 1.4143 @36M, more params yet slower). **Root cause MEASURED:** a vanilla Linear weight gets gradient from all 8192 tokens/microbatch; a LUT weight-row only from the ~80-100 tokens whose hard-argmax selects it (~80x fewer samples/param) → LUT body gradient **2.2x noisier** (noise-to-signal, init) than vanilla body. out_proj worst (14% rows <16 tokens/step). NOT a coverage cliff (bs=16=8192 tokens, rows ~fully covered) — few SAMPLES per row. Structural cost of conditional routing vs dense matmul. Tool: `nanochat_exps/profile_lut_vs_vanilla/grad_snr.py` (gotcha: nanochat loader reuses one buffer, must .clone() batches).

## bs=48 new best + WHAT batch-scaling actually improves (2026-05-22, exp486 + analysis)
- See [project_bs48_what_improves.md](project_bs48_what_improves.md) — **exp486 (exp475 fork, bs 16→48, 3× tokens, 8K) = 1.3791 @ 89.4M**, Δ=−0.117 vs exp475, −0.035 BELOW fair vanilla exp476 (1.4143). Deep analysis (`analysis_exp475_vs_exp486/`): gain is (1) **NOT row coverage** (both cover all rows, 0 revived); (2) **NOT modular** — transplant of any single exp486 module into exp475 is catastrophic (−0.7 to −1.3), solutions holistically coadapted in different basins (rel_w_change≈1.4); (3) **in the LAST 2-3 layers** (logit-lens: early layers slightly WORSE, all gain at L5/L6: −0.086/−0.136); (4) **on RARE tokens** monotonically (rarest −1.16 bits, commonest −0.16); (5) position ~uniform. Interpretation: bs scaling feeds the HARD/RARE/DEEP gradient-undersampled paths. **Inverse-freq loss weighting TESTED+FAILED (exp487, +0.36 bpb): reweighting is zero-sum within a batch, can't replicate bs=48's ABSOLUTE more-samples-of-everything → bs gain is more gradient signal, not better allocation.** Late-layer LR×3 ALSO TESTED+FAILED (exp488, +0.012 bpb): non-zero-sum but still fails — late-layer deficit is gradient NOISE not under-movement, higher LR amplifies noise. **Final: bs gain is irreducible variance-reduction from more samples; no fixed-batch allocation/step-size trick recovers it (exp353–488). Need a lower-variance estimator (none found) or pay the tokens.**

## Scaled-hard forward (winner coeff) — dead end, diverges (2026-05-22, exp485)
- See [project_soft_winner_dead_end.md](project_soft_winner_dead_end.md) — new `backward_mode='soft_winner'` (out = softmax(ts/T_sel)[winner] * W[winner]; argmax pick scaled by winner's softmax confidence; single-row inference, deterministic train==eval; backward verified vs autograd ~1e-15). **exp485 DIVERGES** — gap vs exp475 widened monotonically (step 200 −0.001 → 1000 +0.204), val bpb rose, killed step 2000 (1.9491, +0.274). Root cause: init coeff≈1/K≈0.016 crushes output ~64×; learnable T_sel drifted UP (more diffuse → smaller coeff → runaway attenuation) instead of sharpening. Don't couple LUT output magnitude to selection confidence. Code left in place, off by default.

## Matmul-free LUT-LM shape rules + SOTA exp530=1.4731 (2026-05-24, exp511-532)
- See [project_matmul_free_lut_shape_rules.md](project_matmul_free_lut_shape_rules.md) — per-module nap-depth rule: **qk wants WIDE-SHALLOW (nap=4, beats soft qk, matmul-free); v_lut & residual_lut DECODERS want DEEP (nap=6)**, confirmed param-matched (exp532 nap4 vs exp530 nap6, −6mb). d_v helps monotonically (12/16/24 → 1.4943/1.4876/1.4812). **residual_lut tph 128→256 is the biggest lever (exp530=1.4731, −0.0094 over exp513, the clearest signal)**. Hard v-branch HURTS (+0.0076; v-branch only helps when soft/magnitude-leaking). Best=exp530 @108M. Caveat: gains largely capacity-bought, intermediate deltas within unmeasured ~5mb seed band; directions robust. Infra: `expandable_segments:True` halves footprint; soft E→V LUT=57GB (use ste).

## Tied LUT unembedder — invertible but not competitive as a head (2026-05-24, exp514-522)
- See [project_tied_lut_unembedder.md](project_tied_lut_unembedder.md) — replace Linear(D,V) with a matmul-free LUT head + identity/inverse reg. **Invertibility lever is COVERAGE K=tph·n_sparse/V, NOT LR** (K=4→100% self-decode at lr=1e-3; K=1 caps ~78%). Row=token head needs winner-take-all assignment STE (naive 2%, assignment 99.94%). But as an LM head it plateaus **~2.12 bpb vs Linear's 1.498** (sparse ~K-vote logits = weak output distribution); dual-stream worth ~0.12, aux speeds mid-run but same floor. Elegant + tiny + matmul-free but underperforms; not adopted. Tests in `workbooks/tied_*_inverse_test.py`.

## Residual-width SoTA: exp565=1.4513 @8K, D×tph super-additive (2026-05-25, exp562-565)
- See [project_lut_lm_residual_width_sota.md](project_lut_lm_residual_width_sota.md) — single-knob forks of exp513 (D=384/tph128/1.4825). **exp565 (D=512, residual_tph=512) = 1.4513 @ 175M, current matmul-free SoTA @8K.** D and residual_tph stack SUPER-additively (both = −0.0246 > sum of singles −0.0205). Residual *width* more param-efficient than *tables*. D-knee ~384 (D=128 exp563=1.5570); tph sharply diminishing (128→256 −0.0135/+25M, 256→512 −0.0066/+50M). Gains are param-bloat; off the real goal.

## LUT-LM per-token gap vs vanilla — crossover @step3000, late-slope deficit (2026-05-25)
- See [project_lut_lm_pertoken_gap_vs_vanilla.md](project_lut_lm_pertoken_gap_vs_vanilla.md) — **REAL goal = per-token convergence parity with vanilla** (not absolute bpb; longer training/more params are off-target). Gap inverts: LUT-LM BEATS vanilla for first ~3000 steps (−0.169 @200), crosses over ~3000, vanilla pulls away (+0.083 @8000, widening). Deficit = late-training slope (2nd-half descent vanilla −0.187 vs LUT −0.116). Target gradient resolution / per-row density, not tables/steps. Cosine-tail "gains-to-the-end" is shared with dense (~28% final-quarter for ALL); per-module weight_deltas show unembedder freezes fastest, LUT backbone most active late.

## Unembedder head explorations — all ≤ trivial dot (2026-05-25)
- See [project_unembedder_head_explorations.md](project_unembedder_head_explorations.md) — VQ VocabLUT (best matmul-free, near-Linear @8-26× compress), ScoreLUT/BitReadout/Kendall/MultiKendall all ≤ a **trivial tied dot** (exp560=1.6283 beats every rank head). The +0.13-0.15 gap to Linear = **residual narrowing (E=64)**, NOT the head. New modules: `vq_vocab_lut.py`, `score_lut_head.py`, `bit_readout_head.py`, `kendall_readout.py`, `multi_kendall_readout.py`, `mixture_bit_readout.py`, `trainable_anchors_multi_head_lut.py`.

## Big-head MLP at fixed D=384 — gradient-propagation hypothesis REFUTED (2026-05-26, exp566)
- See [project_big_head_mlp_falsified.md](project_big_head_mlp_falsified.md) — exp566 (exp513 backbone byte-for-byte + MLP head `Linear(384,1024)→GELU→Linear(1024,V)`, 33.95M head vs 12.6M, total 110.76M) = **1.4891 @ 8K, WORSE than exp513 1.4825 by +0.0066**. Strong early lead (−0.10 @200) decays monotonically; crosses behind exp513 at step ~5000. Late slope (3000→8000) exp566 −0.0205/k < exp513 −0.0234/k — NOT steepened. **D-sweep gain (exp562 −0.0111) was wider residual BUS** (residual_dim/residual_lut output/ln_final/head input dim all grew together), not head Jacobian. Head architecture is not the late-slope lever; gradient-resolution in LUT routing path is the next probe.

## E-bus widening: real but secondary lever (2026-05-26, exp567)
- See [project_e96_attention_bus.md](project_e96_attention_bus.md) — exp567 (exp513 + E 64→96, +13.6M, 103.02M) = **1.4768 @ 8K, BEATS exp513 1.4825 by −0.0057** (monotone direction across steps 3K-8K, not noise). Removes the out_proj 96→64 compression quirk (H*d_v=96, was crushing SDPA output every layer). Late slope (3000→8000) **~4% steeper** than exp513. Per-param efficiency: −0.00042 bpb/M, **~2.5× less efficient than D-widening** (exp562 −0.00106/M). **D-stream is the dominant capacity bus; E-stream is secondary**.

## qkv_lut trainable anchors at E=96 — REFUTED (2026-05-26, exp568)
- See [project_qkv_trainable_anchors_failed.md](project_qkv_trainable_anchors_failed.md) — exp568 = exp567 fork with ONLY qk_lut swapped TinyMHLut(random NAP=4) → `SoftAnchorPairMHLUT` (learnable anchor pairs), `anchor_tau` cosine 1.0→**0.001** over 0-3000 steps, then `.hard=True` snap. +7.1M anchor_logits (AdamW, lr=3e-4); LUT weights still Lion. **Final 1.4884 @ 110.10M, WORSE than exp567 by +0.0116 and even WORSE than exp513 E=64 by +0.006**. Snap was clean (no blip at boundary; τ_final=0.001 sharp enough). Brief −0.003 soft-mode lead at step 800 flipped to monotone widening +0.012 deficit post-snap. **Random NAP=4 anchors are near-optimal at this scale** (LUT only has 16 rows; any subset roughly equally informative for q,k) + soft-mode LUT weight co-adaptation mismatches the post-snap hard forward. **Closed direction at 100M scale**; could be re-tried at much larger NAP/E. Mechanism (SoftAnchorPair + τ-schedule + hard-snap) is sound and reusable.

## out_proj NAP=6 IS the sweet spot at mid-scale (2026-05-26, exp569+exp570 killed)
- See [project_nap6_out_proj_sweet_spot_at_E96.md](project_nap6_out_proj_sweet_spot_at_E96.md) — Two killed-early forks of exp567: **exp569** (out_proj multi-NAP `[(4,128),(6,64),(8,96)]`, exp387 recipe port: −20M params, −13.6 Mbits, BUT trailing 1.6086@3K vs 1.5986) and **exp570** (out_proj NAP=4/tph=2048, exp390 recipe port: −18.9M params, +18.8 Mbits, **+55% wall-clock**, trailing 1.6026@3K). **NAP=6 is the right shape for out_proj at exp567-class scale**; both deviations lost. Tiny-LUT-LM NAP=4 wins (exp390/exp387) DO NOT TRANSFER to mid-scale — at ~100M the row-collapse pathology that NAP=4 fixed at 43M doesn't bite, and NAP=6's deeper K=64 table becomes pure capacity win. Per-token wall-clock scales linearly with table count (tph) on H100 — sparse gathers don't get matmul-engine speedup. Closed direction; next levers must preserve NAP shapes (residual_tph, joint E×D, per-layer schedules, noise_eps, deeper L).

## Minimal-arch LUT-LM: no_resid + tied dot head (2026-05-26, exp571-575 arc)
- See [project_no_resid_tied_head_arc.md](project_no_resid_tied_head_arc.md) — Stripping exp567 (residual_lut + Linear unembedder REMOVED, tied dot on E-stream) trades ~0.10 bpb for ~3× bandwidth reduction. exp571 (raw dot, no LN-emb): trajectory tracks but slow warmup (small init logits). exp572/573 (LN-on-emb with γ=1.0): step 200 = 4.19 bpb = log(V)/bytes — softmax peaks CONFIDENTLY on RANDOM wrong vocab (NOT γ-collapse; just too-large logit scale at init). **exp574 = exp573 + ln_emb.γ init=0.1**: working recipe, projected final ~1.55 @ 71.57M/163 Mbits. exp575 (E=192, d_v=32, H·d_v=E invariant preserved): projected ~1.54 @ 124M/289 Mbits (39% vanilla bandwidth). Apples-to-apples reference for TIED LUT-LMs is **exp328 (1.3882)**, NOT exp476 (untied). LUTs are scale-invariant via MeanAbsNorm input, so tied tok_emb scale only affects head — fully decouplable.

## Head topology exhausted at E=96 LUT-LM (2026-05-26, exp582/583/584)
- See [project_head_arch_doesnt_stack.md](project_head_arch_doesnt_stack.md) — Dual-head variants exhaustively tested against exp567's single Linear(D, V) head: **exp582** (TIED dot E + Linear D + 1 final resid_lut) proj ~1.53. **exp583** (TIED dot E + Linear D + per-layer resid_lut) = **1.5200**. **exp584** (UNTIED Linear E + Linear D, E=96, full arch) = **1.4795** ≈ exp567's 1.4768 (statistical tie at +3M params and +100 Mbits head bandwidth). TIED dot fails to stack with Linear (+0.04-0.05 worse); UNTIED dual is a no-op (≈ ceiling). **At E=96 the LUT-LM final activation is saturated by a single Linear head — no useful "second direction" exists.** Closed direction at this scale. Levers to genuinely beat exp567 must target: wider D (exp562/565 path), wider E at full arch, LUT routing (gradient/coverage), or trunk sparsification — NOT head topology.

## NEW TINY LUT-LM SoTA: exp622 = 1.5741 (hybrid_smooth n_alt=6 + autograd consistent bwd) (2026-05-28)
- Full n_alt sweep at 43M (exp365 architecture, 8K, bs=16) with autograd self-consistent backward (input grad through abs_p→softmax→probs only; weight grad scatters at (n_alt+1) rows by softmax probs):
  - n_alt=1 killed @6K (worst, +0.07+)
  - n_alt=1 K-row surrogate (exp615): 1.5845
  - n_alt=2 (exp619): 1.5865 (worse than K-row — autograd needs ≥3 alts)
  - n_alt=3 (exp618): 1.5805
  - n_alt=4 (exp620): 1.5762
  - n_alt=5 (exp621): 1.5746
  - **n_alt=6 (exp622): 1.5741 — new SoTA, beats prior by 0.5 mb**
  - n_alt=NAP (exp616): 1.5769
- **Sweet spot at n_alt=5-6** (essentially tied), both beat n_alt=NAP. For NAP=8 modules (v_lut/out_proj), n_alt=NAP includes the 1-3 most-confident bit flips beyond the 5-6 genuinely uncertain ones — those add gradient noise. **n_alt=6 is cleanest**: matches NAP=6 modules (full ball for qkv/residual), gives top-6 for NAP=8 modules (drops the 2 noisiest). Constant memory/compute regardless of per-module NAP. Strongest candidate for big-scale deployment via custom autograd.Function (save indices+probs, recompute gathers in backward) to match exp611's memory profile.

## TINY LUT-LM SoTA: exp616 = 1.5769 (hybrid_smooth n_alt=NAP + autograd consistent bwd) (2026-05-28, superseded by exp621)
- Fork of exp615 (1.5845) with two changes: `hybrid_smooth_n_alt=-1` (= NAP per module, full Hamming-1 ball forward — main + all NAP single-bit-flip alternatives via (NAP+1)-way softmax) AND `hybrid_smooth_autograd=true` (plain autograd backward through compiled forward, NO soft K-row surrogate). Input gradient propagates only through smooth path (abs_p → softmax → probs → x); weight gradient scatters at (NAP+1) rows scaled by softmax probs — all self-consistent with the actual forward. **Final 1.5769 @ 43.06M, 8K bs=16. Beats exp615 (n_alt=1 + K-row bwd) by −7.6 mb; beats exp365 baseline by −44.6 mb; beats exp386 (MHLut smooth n_alt=3, prior tiny record) by −39.5 mb.** Confirms: full Hamming-1 ball + self-consistent backward is the cleanest mathematical combination, geometrically matched to the LUT routing structure. **Memory caveat**: at big scale (289M, exp611-class), autograd path saves all (NAP+1)·n_layers·n_modules gather tensors → OOM at bs=16 without grad_accum. To bring this to big scale, need custom autograd.Function with manual backward (save indices+probs, recompute gathers in backward). Open work.

## TINY LUT-LM SoTA at exp365 arch: exp615 = 1.5845 (hybrid_smooth n_alt=1 at 43M, bs=16) (2026-05-28, superseded by exp616)
- Fork of exp365 architecture (43.06M params, joint qkv+additive v, per-layer residual_lut) with all 4 LUT modules in `backward_mode='hybrid_smooth', hybrid_smooth_n_alt=1` (sigmoid-u top-2 softmax forward, soft K-row input grad, 2-row weight scatter). Same exact arch as exp365 + exp386 baselines. **Final 1.5845 @ 43.06M, 8K steps, bs=16. Beats exp365 (1.6215, soft) by −37.0 mb; beats exp386 (MHLut smooth n_alt=3, 1.6164) by −31.9 mb — prior tiny-scale architectural records.** The 289M-scale win (exp611, +24.7 mb vs exp603) transfers to 43M-scale at +37 mb. **hybrid_smooth is a general LUT-LM backward improvement, not scale-specific.** Still behind vanilla bs=16 (exp328=1.3882) at +0.196 — tiny LUT-LM gap persists; bs scaling (exp364 bs=192) closes it.

## NEW LUT-LM SoTA at bs=16: exp611 = 1.4048 (hybrid_smooth backward_mode) — BEATS untied vanilla (2026-05-28)
- See [project_readout_lut_capacity_sweep.md](project_readout_lut_capacity_sweep.md) — **exp611 = 1.4048 bpb @ 289M, 1.73h. Beats exp603 (1.4295, backward_mode=soft) by −0.0247 bpb** at matched architecture/bs/seed. Single config change: `backward_mode='hybrid_smooth'` (new mode added to `TinyMultiHeadLut`). Math: forward exactly approximates top-2 softmax over the two best LUT rows (main + Hamming-1 neighbor at argmin |d|), `out = (1−u)·W[main] + u·W[alt]` where `u = sigmoid(−Δts/T_sel)` and `Δts = 2·|d_min|/(T_soft + |d_min|)` — uses BOTH learnable temperatures. Backward: weight grad is 2-row scatter `(1−u)·grad_pt → main, u·grad_pt → alt` (chain rule of actual forward); input/temperature grads delegated to `_soft_lut_bwd_body` (soft K-row surrogate). Gap to tied vanilla (exp328=1.3882): +0.0166; **gap to untied vanilla (exp476=1.4143): −0.0095 (LUT-LM beats untied vanilla at matched bs=16 horizon for first time)**. Open: full chain-rule "self-consistent" backward (differentiate through u directly, NOT through soft K-row surrogate) — might work even better especially with n_alternatives=NAP (full Hamming-1 ball). Code: `src/spiky/lutorch/tiny_multi_head_lut.py` — `_hybrid_smooth_lut_fwd_body`, `_TinyMHLutHybridSmooth`, `_hybrid_smooth_weight_grad`. Run with `backward_mode='hybrid_smooth'` in TinyMultiHeadLut.

## NEW LUT-LM SoTA: exp608 = 1.2180 (bs=96 via grad_accum=6 on exp603) — BEATS vanilla by −170 mb (2026-05-28)
- See [project_exp608_bs96_lut_beats_vanilla.md](project_exp608_bs96_lut_beats_vanilla.md) — **exp608 = 1.2180 bpb @ 289.4M, 7.2h. Beats exp603 (1.4295, bs=16) by −0.2115 and vanilla exp328 (1.3882, bs=16) by −0.1702 at matched 8K horizon.** Single config change: `total_batch_size 8192 → 49152` (grad_accum 1 → 6, effective bs 16 → 96, phys bs unchanged at 16). All architecture/hyperparams from exp603 unchanged. Compute cost: 6× wall-clock (1.22h → 7.2h). Quality cost: nothing — pure win. **First LUT-LM in this branch to comprehensively beat vanilla.** Trajectory milestones: step 2000 already passes exp603's *final* SoTA; step 2400 passes vanilla's *final*; gap to vanilla widens to −170 mb by step 8000. Per-step convergence 3× faster than exp603. **bs scaling at this scale dwarfs architectural fine-tuning by 5×** — exp567→593 chain gained −43 mb across 30 experiments, this lever gave −212 mb in one. No saturation in trajectory (slope still −2 mb / 200 steps at step 7800), so bs=192 or bs=384 likely give further large gains. Open question: vanilla bs=96 result unknown; if vanilla also drops ~170 mb, the LUT-vs-vanilla gap is preserved at bs=96.

## LUT-LM SoTA: exp603 = 1.4295 (v_lut tph 320→512, NAP=6 unchanged) (2026-05-27, superseded by exp608)
- See [project_readout_lut_capacity_sweep.md](project_readout_lut_capacity_sweep.md) — **exp603 = 1.4295 @ 289.4M, 1.217h. Beats exp593 (1.4337) by −0.0042 bpb at +28.3M params, +14 Mbits trunk.** Single config change: v_lut tph 320 → 512 at fixed NAP=6. Pure capacity addition: atoms 122K → 196K (+60%), mixture depth 320 → 512 (+60%), per-row gradient coverage UNCHANGED at 128. exp602 (NAP=5 tph=512, halved K) failed +0.034 at step 1K — v_lut atoms ARE load-bearing; you can grow tph but can't shrink K. Trajectory: started warmup-tied with exp593, gradually closed gap through mid-phase, crossed over at step 3400, finished −4.2 mb ahead. Per-param efficiency −0.15 mb/M. Gap to tied vanilla (exp328): +0.041; gap to untied vanilla (exp476): +0.015. **Use exp603 as new baseline going forward.**

## LUT-LM SoTA: exp593 = 1.4337 (NAP=5 read_out, capacity sweep) (2026-05-27, superseded)
- See [project_readout_lut_capacity_sweep.md](project_readout_lut_capacity_sweep.md) — Decomposed exp588's read_out_lut into three independent capacity axes (atom count, atom width, mixture depth) and probed each. **Finding 1**: mixture depth dominates — 6× depth cut (exp589 multi-head, exp590 sparse-scatter) costs ~0.028 bpb regardless of topology. **Finding 2**: atom count is slack down to 24-49K at fixed depth — exp592 (NAP=4, 24K atoms, depth 1536) only +0.020. **Finding 3**: NAP=5 sweet spot — exp593 (32 rows/table, 49K atoms, depth 1536) **= 1.4337 @ 261.10M, 1.036h**. Ties exp588 final (Δ=−0.0007, ahead in every late-phase eval) at **half the read_out_lut params** (37.7M→18.87M). Mechanism: 2× per-row gradient coverage (256 vs 128 tokens/row/step at bs=16) → cleaner Lion sign updates, amplified by cosine LR decay.
- **Follow-on (exp594–exp601, 2026-05-27)**: NAP=5 trick **does NOT generalize** to out_proj or qkv_lut — all 8 attempts to extend the win to other modules failed. (a) **out_proj sweep (exp594–597)**: NAP=5, sparse-scatter, NAP=7/tph=512, NAP=5/tph=2048 all lost +5 to +18 mb. Out_proj output enters residual stream and compounds through 6 layers — any imprecision propagates; tph=1024/NAP=6 is the local optimum. (b) **qkv sweep (exp598–600)**: qk_lut tph doubled (+11 mid, killed); single-head qk H=1 (exp599, **+0.0115 final**) confirmed per-head routing diversity is required (LUT-LM can't reuse vanilla's "single W_q sliced into heads" trick); unified qkv H=1 multi-NAP (exp600, **+0.0295 final @ +104M**) collapses 6× routing entropy and fails despite mixture-depth uplift. (c) **Tied head (exp601)**: replace Linear(D, V) with x_resid @ ln_emb(tok_emb).T, γ_emb=0.05; lost +0.048 by step 3K (killed). **Lesson**: bandwidth-for-quality is cheap on read_out + v_lut only; out_proj/qkv/head are tight on every axis at this scale.

## LUT-LM SoTA chain (exp588): single read_out LUT @ end, kills per-layer residual_lut (2026-05-27)
- See [project_read_out_lut_topology.md](project_read_out_lut_topology.md) — exp588 forks exp587 with ONE single architectural change: remove per-layer residual_lut from all 6 LUTBlocks and replace with ONE `read_out_lut` at the end of the stack with **tph=1536 (6× exp587's per-layer 256)**. Identical params (~280M), identical trunk bandwidth (155 Mbits), identical compute. **Final = 1.4344 @ 279.97M, 1.027h. Beats exp587 (1.4471) by −0.0127 bpb.** Topology finding: the 6 distributed E→D injections summed into x_resid were *worse* than a single concentrated read-out from the final E-stream embedding. The per-layer residual_lut was redundant work, not depth-multiplying capacity. Full chain: exp567 1.4768 → exp585 1.4625 → exp586 1.4509 → exp587 1.4471 → exp588 1.4344 → **exp593 1.4337** (cumulative −43.1 mb). Gap to tied vanilla: +0.089 → **+0.0455**; gap to untied vanilla: +0.063 → **+0.0194**.
- See [project_e192_full_arch_sota.md](project_e192_full_arch_sota.md) — Full session SoTA chain on exp567 full-arch recipe (per-layer residual_lut + untied Linear(D=384, V), no topology variations): **exp567 1.4768 @103M → exp585 (E=192/d_v=32) 1.4625 @156M (−14.3) → exp586 (E=384/d_v=64) 1.4509 @261M (−11.6) → exp587 (residual_tph 128→256) 1.4471 @280M (−3.8).** Cumulative **−29.7 mb** absolute SoTA improvement. Gap to tied vanilla (exp328=1.3882) closed +0.089 → +0.0589; gap to untied vanilla (exp476=1.4143) closed +0.063 → **+0.0328**. Diminishing returns: per-param efficiency 0.275 → 0.110 → 0.20 mb/M across steps. Trunk bandwidth grew 72 → 96 → 146 → 155 Mbits (still 2.2× cheaper than vanilla's 340). Productive lever IS scaling exp567's exact recipe; topology variations (dual head tied/untied, minimal arch) all failed or were neutral. E-widening + residual_tph compound on full arch.
