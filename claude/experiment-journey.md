# The experiment journey behind LUTGPT

*What was tried, the lineage of results, the durable mechanistic lessons, and the dead-ends
already falsified. The science/why is in [thesis.md](thesis.md); this is the "what we tried."
Distilled from ~90 project notes plus the experiment journals.*

> **Caveat on experiment ids.** Numbering is era-local and partly reused across eras. Trust
> the *lessons* below; verify any specific experiment number or figure against the historical
> record ([experiment-archive.md](experiment-archive.md)) and the LUTGPT research report
> before quoting it.

## Eras (the arc)

1. **Byte-level FineWeb** (V = 257, T = 32; metric val CE; vanilla 4.87M = 1.356).
   Brute-force LUT attention → discovered the NAP/tph tradeoff, full-coverage anchor
   sampling, concat positional encoding, the **V3 softmax** breakthrough (separate "what to
   attend to" from "how to combine"), then **Ranking Attention** (Kendall-τ ≡ cosine of ±1
   dominance signatures → SDPA) and **HyperLUT**. Best LUT within ~0.025 CE of vanilla but at
   26× params. The gap was diagnosed as **representational, not optimizational**.
2. **Nanochat BPE** (ClimbMix, V = 32,768, T = 512; metric val **bpb**). Port + scale. A big
   unembedder MLP was found dominant, then dropped in favor of the dual-stream design. RoPE
   resets the baseline.
3. **Dual-stream LUT-LM + RoPE** (the workhorse). E-stream rank backbone + D-stream Euclidean
   readout + linear unembedder; the E-stream **pre-norm identity skip** gave a discontinuous
   ~−60 mb jump; fused qkv; no FFN. Batch size emerges as *the* lever → the first decisive
   beats of vanilla.
4. **FastMHL deployment era** (16k–24k steps): shift to **native hard-forward** training
   (single lookup at inference — the real product goal), dense-K backward, kernel wgrad /
   dispatch tuning → the report's exp709 / 754 / 755 / 760 lineup (the headline table is in
   [thesis.md](thesis.md)).

## Durable mechanistic lessons (the load-bearing knowledge)

- **Gradient sparsity is THE bottleneck.** Each LUT row is updated by only ~tens of tokens
  per microbatch (dense vanilla weights see every token). This is fixed at 1/K by the
  architecture itself. Everything below follows from it.
- **Batch size is the dominant lever**, and it acts as pure gradient-variance reduction:
  `grad_accum` reproduces native big-batch training to within ±0.01 bpb. No cheap
  optimizer/LUT trick beats the baseline by more than ±0.01; only more *effective batch*
  (or distillation) moves the needle. Moderate **NAP ∈ {4,5,6}** are within 0.002 of each
  other (>6 needs a bigger batch). Coverage rule of thumb: keep ≥ ~200 tokens per row.
- **Two convergence regimes.** The LUT model is *more* token-efficient than vanilla early
  (~first 3k steps), then vanilla descends ~1.4–1.7× faster late. The real research target
  is the **late slope**.
- **Soft vs. hard forward.** A soft/blended forward is a *better function* (~−16 mb) but is
  not matmul-free and **not hardenable** (post-hoc hardening of a soft model adds ~+0.25).
  The win is the soft *forward* (representational); dense soft *weight-grad* coverage alone is
  exactly neutral. The answer: **train natively hard with a rich (dense-K) backward**;
  **hybrid_smooth** (exact top-2 softmax over the main row + its best Hamming-1 alternative)
  is the deployable middle that recovers most of the soft gap at 2 reads. Note
  `softmax = exp-Hamming kernel` — every non-normalized / non-peaked gate (relu, gelu,
  layernorm, signed) loses.
- **Optimizer: LION β = (0.9, 0.95)** matches AdamW at *half* the state (Adam's √v̂ magnitude
  normalization is not the lever on LUT tables); **β2 = 0.95 is the sweet spot** (LION's
  default 0.99 costs +0.04); effective lut_lr ≈ 2e-4.
- **RoPE ≫ learned positions**, and *more so* for a LUT-LM (a learned additive position must
  survive the qk argmax quantization; RoPE injects position *after* the LUT into smooth SDPA
  space, zero-param).
- **Pre-LUT MeanAbsNorm = per-token magnitude calibration** (keeps the pairwise differences
  in T_soft's regime); any positive scale is equivalent, mean-abs is cheapest, a learnable
  affine is inert.
- **Architecture wins that stuck:** the dual E/D stream; the **E-stream identity skip**
  (the biggest single architectural jump); fused **qkv_lut** + a shallow V-branch + a
  separate `v_lut`; **no FFN** (out_proj absorbs it); concentrating readout in ONE
  end-of-stack `read_out_lut` (per-layer D-injections were redundant); **per-head routing is
  structurally load-bearing** (collapsing to H = 1 costs mb); **don't mix LUT and dense in
  the same block** (be fully-LUT or fully-vanilla — LUT composition only emerges when all
  trunk modules are LUT). **Lever ranking:** batch ≫ tph > NAP > D-width > E/d_v; heads,
  anchor-learning, multi-NAP-at-scale, and dual-heads all lost. The **linear unembedder is
  irreplaceable** — pure-LUT vocab heads floor at ~2.12 bpb vs. ~1.5 (sparse K-vote logits
  are a weaker output distribution). Per-module NAP rule: qk wants wide-shallow (NAP ≈ 4),
  decoders (v / out / residual) want deeper (NAP ≈ 6), read_out NAP ≈ 5.

## Dead-ends / falsified (do NOT re-run without a genuinely new reason)

Probabilistic forward (sample a row); soft-winner forward (**diverges** — never couple
output magnitude to selection confidence); STE hard-sign + softmax (the soft apparatus is an
inseparable package); soft weight-grad (neutral); windowed-grad smoothing (Adam β1 already
does it); inter-table cosine contrastive (init is already orthogonal); **qkv trainable
anchors** (random anchors are already near-optimal); big-head MLP; dual heads (don't stack);
hard-example mining / inverse-freq loss / per-row LR / Lookahead / β1 = 0.99 (all within
±0.01 or worse); LUT / tied-LUT unembedder; half-LUT hybrid (vanilla-attn + LUT-FFN); custom
triton wgrad (can't beat torch.compile's fusion ceiling); bf16 LUT weights *in training*
(slower, no tensor-core win — inference-only).

## Gotchas

- **Bit-packing convention.** hybrid_smooth / soft packs bits MSB-first; STE / hard packs
  LSB-first. Naively hard-evaluating a hybrid_smooth checkpoint reads the wrong rows
  (+1.73 bpb). Either bit-reverse the weight tensor on axis 1, or just use `FastMultiHeadLut`
  (which is internally consistent) for deployment-targeted runs.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` roughly halves the reserved footprint of
  soft runs.
- **The fair baseline is UNtied vanilla.** A LUT-LM is structurally forced to be untied;
  comparing against a *tied* vanilla inflates the apparent gap.
