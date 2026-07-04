# Why LUT-LM converges slower than vanilla — full investigation (2026-05-22)

Question: at the same batch size, LUT-LM converges to a worse loss than a vanilla
transformer despite having more parameters. Why? And what does batch-scaling
actually buy?

Models compared (all 6 layers, 6 heads, D=384, seq 512, vocab 32768, 8000 steps,
same cosine LR schedule):

| run | arch | bs | params | final bpb |
|-----|------|----|--------|-----------|
| exp476 | vanilla MinimalGPT+RoPE (untied) | 16 | 35.8M | **1.4143** |
| exp475 | LUT-LM (TinyMHLut soft, mean-abs norm, LION) | 16 | 89.4M | 1.4962 |
| exp486 | exp475 fork, batch only | 48 | 89.4M | **1.3791** |

At bs=16 vanilla beats LUT by **0.082 bpb with 2.5× fewer params**. 3× the batch
(exp486) flips it: LUT 1.3791 < vanilla 1.4143. So the gap is an *optimization*
problem (more tokens fixes it), not a capacity problem.

Harness (reusable for any checkpoint pair):
- `analysis_exp475_vs_exp486/model_def.py` — reconstruct either model from its
  checkpoint config (shared seed ⇒ identical anchors ⇒ row (table,r) = same
  bit-pattern in both ⇒ per-row weight comparison valid).
- `analyze.py` — selection/visit stats, weight stats, transplant, temps, attn.
- `probe.py` — logit-lens over depth, per-position & per-frequency loss.
- `transplant.py` — module/layer weight transplant ablation.
- `../profile_lut_vs_vanilla/grad_snr.py` — gradient density + noise-to-signal,
  LUT vs reconstructed vanilla.

> Dataloader gotcha: `tokenizing_distributed_data_loader_bos_bestfit` REUSES one
> buffer object per `next()`. Collecting `[next(loader) ...]` aliases the same
> data — `.clone()` each batch or you silently get zero gradient variance.

---

## Part A — What does batch-scaling (bs=48) actually improve? (exp475 vs exp486)

All on a fixed shared val batch (identical inputs to both).

1. **NOT row coverage.** On real data both models cover ~all LUT rows;
   `revived_rows = 0` (rows dead@bs16 that wake@bs48) for every module. A
   "bs=16 microbatch" is 16×512 = 8192 tokens, so 64-row tables get ~128
   selections/row — rows are covered. (A random-token dry-run showed false "dead
   rows piling up with depth"; rejected on real data.)

2. **NOT modular.** Transplant ablation: grafting any single exp486 module into
   exp475 is catastrophic (out_proj −1.28, qkv −0.67, unembedder −1.11; only the
   full swap recovers +0.14). The two solutions are holistically coadapted in
   different weight basins (rel_weight_change ≈ 1.4, rownorm_corr ≈ 0.05). No
   single module to target.

3. **Depth: the gain is in the last 2 layers.** Logit-lens (ln_final+unembedder
   on the cumulative residual after each layer), delta (486−475):
   L1 +0.075, L2 +0.075, L3 +0.048, L4 −0.021, L5 −0.086, L6 −0.136. Early
   layers are slightly *worse* in exp486; the whole advantage is built late.

4. **Tokens: the gain is on rare tokens, monotonically.** Per-frequency bits/token
   delta: rarest bucket −1.16 … commonest −0.16. Position is ~uniform (−0.63).

Interpretation: bs=48 feeds the **hard/rare/deep paths** that are gradient-
undersampled at bs=16.

---

## Part B — Can we get there with fewer tokens? Two interventions, both fail.

- **exp487 — inverse-frequency loss weighting** (upweight rare-token loss at bs=16,
  α=0.5). **+0.36 bpb worse**, stable, no crossover. Reweighting is **zero-sum**
  within a batch: rare tokens can only gain gradient at the expense of common
  tokens, which dominate bpb. bs=48 didn't reallocate — it gave more of everything.

- **exp488 — late-layer LR ×3** (last 2 layers' LUT params at 6e-4). **+0.012 bpb
  worse**, no crossover. Non-zero-sum (doesn't steal gradient) so it dodges the
  exp487 trap, but still fails: the late-layer deficit is gradient **noise**, not
  under-movement — a bigger step on a noisier gradient just amplifies the noise.

Conclusion: bs-scaling's gain is **irreducible gradient-variance reduction from more
samples**. Not reachable by reallocation (exp487) or step-size (exp488) at fixed batch.

---

## Part C — The bottleneck, measured: per-parameter gradient density

`grad_snr.py`, bs=16, gradient noise-to-signal E‖g−μ‖²/‖μ‖² over independent
microbatches (higher = noisier = slower SGD):

| state | LUT body | vanilla body | ratio |
|-------|----------|--------------|-------|
| init | 0.65 | 0.29 | **2.2× noisier** |
| trained optimum | 14.5 | 8.8 | 1.65× noisier |

(Both models' dense *heads* are similar — 0.70 vs 1.28 — confirming it's the body
that differs.)

Cause — gradient density (tokens contributing to each parameter per bs=16 step):

| LUT module | median tokens/row | p10 | rows <16 tokens | vanilla |
|------------|-------------------|-----|-----------------|---------|
| qkv/v/residual | ~100 | ~35 | ~2% | **8192/weight** |
| out_proj | 79 | 9 | **14%** | 8192/weight |

A vanilla `Linear` weight gets gradient from **all 8192** tokens; a LUT row from
only the **~80–100** that route to it — **~80× fewer samples per parameter** →
noisier per-parameter gradient → slower convergence. out_proj is worst (heavy
sparse tail), matching its known collapse-proneness.

The measured noise gap (2.2×) is smaller than the naive √80 ≈ 9× because tokens
routing to the same row have **correlated** gradients (routing groups similar
inputs), partially compensating for fewer samples.

This is the structural cost of conditional routing: LUT trades the dense matmul
(every parameter sees every token) for sparse table lookups (every parameter sees
only its routed slice), and the gradient inherits that sparsity.

---

## Part D — The crossover: vanilla adapts faster late (confirmed)

exp486 (LUT bs=48) vs exp476 (vanilla bs=16), per-step. The gap **peaks early then
collapses**:

| step | LUT bs48 | van bs16 | gap |
|------|----------|----------|-----|
| 1000 | 1.673 | 1.828 | +0.156 (peak) |
| 4000 | 1.444 | 1.525 | +0.081 |
| 8000 | 1.379 | 1.414 | +0.035 |

Late-stage slope (same schedule): 4k→8k LUT −0.0159/1k vs vanilla −0.0266/1k →
**vanilla descends 1.7× faster late**. exp486 still ends lower, but only on banked
early lead; extrapolated, vanilla crosses.

Mechanism — two optimization regimes:
- **Early = gradient-quantity-limited** (coarse moves). LUT bs=48's 3× tokens win.
- **Late = gradient-precision-limited** (fine refinement near the optimum). Per-
  parameter SNR rules. MEASURED: even at bs=48 each LUT param gets ~315 tokens/row
  vs vanilla bs=16's 8192/weight — **26–31× fewer samples per parameter** despite
  3× more total tokens. LUT's gradient is too noisy to refine finely → plateaus;
  vanilla's dense clean gradient keeps descending.

**You cannot out-batch a density deficit.** Batch buys *quantity* (tokens/row);
*density* is fixed at 1/K = 1/64 by the architecture. 3× batch chips 3× at a 64×
gap and never changes density.

---

## Bottom line & directions

The LUT convergence penalty = **per-parameter gradient sparsity**. Each parameter
is updated by ~1/K of the tokens, so its gradient is noisier; this slows
convergence (Part C) and especially late-stage fine refinement (Part D). Batch
scaling adds *quantity* and front-loads a lead, but never fixes *density*, so
vanilla out-adapts LUT late. Every fixed-batch trick fails because none add samples
per parameter (Part B).

To improve LUT *token-efficiency / late-stage adaptation* you must raise gradient
**density**, not batch:
- **Lower K** (e.g. NAP 6→4: K 64→16 = 4× tokens/row, ~315→~1260, closing most of
  the gap to vanilla's 8192) at the cost of expressivity per table; hold params
  fixed by adding more, smaller tables. The one lever that directly moves the
  measured tokens/row. **(proposed exp489, untested)**
- **Distillation** from a strong teacher — a smooth target delivers dense gradient
  to every parameter through the head, bypassing routing sparsity.
- Soft weight-grad does NOT help (exp445 neutral): a peaked softmax gives
  negligible gradient to non-winner rows, so it doesn't actually raise density.

Memory: `project_lut_convergence_bottleneck.md`, `project_bs48_what_improves.md`,
`project_prob_forward_dead_end.md`, `project_soft_winner_dead_end.md`.
