# LUT Transformer Experiments — Summary

**Task:** Byte-level causal language model on fineweb_texts.txt
**Setup:** CONTEXT_SIZE=32, VOCAB_SIZE=257, H100 80GB
**Metric:** Mean cross-entropy loss on 10k held-out test regions (lower is better)

---

## Baseline

| Exp | Description | Val Loss | Steps | Params |
|-----|-------------|----------|-------|--------|
| exp001 | Vanilla transformer (lr=1e-4) | 1.6120 | 100k | 4.87M |
| exp002 | Vanilla transformer (lr=1e-3) | **1.3559** | 100k | 4.87M |
| exp003 | Vanilla + AdamW wd=1e-4 | 1.3545 | 100k | 4.87M |

**Target: 1.3559** (exp002). lr=1e-3 is a major win over lr=1e-4 (+0.256). Weight decay has negligible effect.

---

## Phase 1: LUT Architecture Exploration (exp004–019)

Starting from the notebook architecture, exploring the design space.

### Key findings

**exp004–008: Large-model baseline, LUT unembedder, split LR**

| Exp | Change | Val Loss | Params |
|-----|--------|----------|--------|
| exp004 | LUT transformer, dense unembedder, lr=1e-4 | 1.4628 | 191M |
| exp005 | + LUT unembedder (nalt=3 missing) | 1.6403 | 292M |
| exp006 | + n_alternatives=3 | 1.5031 | 292M |
| exp007 | + sum loss reduction | 1.5111 | 292M |
| exp008 | + split LR (unembedder=1e-3, rest=1e-4) | 1.4869 | 292M |

- LUT unembedder requires n_alternatives≥3 to work (+0.137 vs nalt=1)
- Split LR (unembedder 10× higher) consistently helps
- All ~100k steps, not converged

**exp009–013: Architecture shape**

| Exp | Change | Val Loss | Params |
|-----|--------|----------|--------|
| exp009 | Vanilla-mirror (separate q/k/v/out_proj/ffn), concat PE | 1.4791 | 214M |
| exp010 | Additive PE | 1.4926 | 214M |
| exp011 | No PE | 1.6618 | 214M |
| exp012 | Double FFN (no inner residual) | 1.5064 | 252M |
| exp013 | Double FFN with inner residual | 1.4728 | 252M |

- **Concat PE > additive PE > no PE**: dedicating separate dimensions to position helps LUT routing
- **Double FFN with residual** is the best single improvement: inner residual is essential for gradient flow

**exp014–019: T vs K sweep at d=32**

| Exp | nap (K) | tph (T) | Params | Val Loss |
|-----|---------|---------|--------|----------|
| exp014 | 6 | 16 | 1.45M | 2.0576 |
| exp015 | 6 | 32 | 2.89M | 1.8619 |
| exp016 | 8 | 32 | 11.6M | 1.6754 |
| exp017 | 10 | 32 | 46.2M | 1.6188 |
| exp018 | 8 | 128 | 46.2M | 1.5004 |
| exp019 | 6 | 512 | 46.2M | ~1.51 (stopped) |

**Key result: T beats K at same param budget.** At 46.2M params, K=8/T=128 (1.500) beats K=10/T=32 (1.619) by 0.119. Once pairwise coverage is complete (K≥8 for d=32), adding more tables is more effective than deeper routing.

---

## Phase 2: RankAttention (exp020–047)

Replace SDPA with attention via pairwise rank features: q/k projected to C(d_qk,2) binary comparison features, then SDPA.

| Exp | Change | Val Loss | Params |
|-----|--------|----------|--------|
| exp020 | RankAttention (d=32, K=8, T=128, nalt=3) | 1.5677 | 46.2M |
| exp021 | + d=64, T=64 | 1.5456 | 42.0M |
| exp022 | Smooth LUT, nalt=1, T=64 | 1.5141 | 23.1M |
| exp024 | Smooth LUT + smooth RankAttn (norm) | 1.4371 | 46.2M |
| exp025 | Smooth LUT + hard RankAttn (no norm) | 1.4568 | 46.2M |

exp024 is the best result so far (1.437) — smooth routing + smooth RankAttention with normalized deltas.

**Exploration of smoothing, regularization (exp026–040):** Various noise injections, delta regularization, annealing. exp024 held as best.

**exp040–044: RankAttention d_qk sweep**

| Exp | d_qk | pairs | Val Loss | Params |
|-----|------|-------|----------|--------|
| exp041 | 32 (full d) | 496 | 1.5082 | 46.2M |
| exp042 | 16 (d//2) | 120 | 1.5071 | 46.2M |
| exp044 | 16, 200k steps | 120 | 1.4821 | 46.2M |

d_qk=16 (120 pairs) ≈ full d (496 pairs). Converges better at 200k.

**exp045–046: d sweep**

| Exp | d | Val Loss | Params |
|-----|---|----------|--------|
| exp045 | 64 | 1.5316 | 42.0M |
| exp046 | 16 | 1.5990 | 46.2M |

d=32 remains the sweet spot.

---

## Phase 3: Mini Architecture (exp048–073)

Redesign from scratch: smaller embedding, single FFN, sweep nap/tph. Target: <5M params, 50k steps.

**exp048–057: Single vs double FFN, tph/nap sweep**

| Exp | nap | tph | Params | Val Loss |
|-----|-----|-----|--------|----------|
| exp048 | 8 | 16 | 4.99M | 1.9421 |
| exp049 | 8 | 16 | 5.78M | 1.9412 |
| exp052 | 8 | 32 | 9.98M | 1.7685 |
| exp053 | 6 | 64 | 4.99M | 1.7593 |
| exp054 | 5 | 64 | 2.50M | 1.8294 |
| exp057 | 5 | 128 | 4.99M | 1.7289 |

**exp058: RankProjection ablation**
Replacing all LUTs with pure PairVoting-style projection (no tables): val=1.953 — confirms LUT table structure is essential.

**exp059–063: nap vs tph tradeoff at ~5M params**

| Exp | nap | tph | Val Loss | Params |
|-----|-----|-----|----------|--------|
| exp059 | 4 | 256 | 1.7060 | 4.99M |
| exp060 | 3 | 512 | 1.7058 | 4.99M |
| exp061 | 2 | 1024 | 1.7185 | 4.99M |
| exp062 | 7 | 32 | 1.8271 | 4.99M |
| exp063 | 1 | 2048 | 1.9652 | 1.22M |

**nap=4 and nap=3 essentially tied** — nap=4 is the sweet spot (nap=2 starts degrading, nap=7 much worse).

**exp064–069: Anchor sampling policies**

| Exp | Policy | connected | Val Loss | Params |
|-----|--------|-----------|----------|--------|
| exp059 | random | False | 1.7060 | 4.99M |
| exp064 | random | True | 1.7205 | 4.99M |
| exp065 | disconnected | False | 1.6866 | 4.99M |
| exp066 | full_coverage | False | 1.6839 | 4.99M |
| exp067 | full_coverage | True | 1.6901 | 4.99M |
| exp068 | full_coverage, tph=124 | False | 1.8261 | 2.42M |
| exp069 | full_coverage, tph=512 | False | 1.6228 | 9.98M |

**FULL_COVERAGE wins.** Ensuring all 496 unique pairs are covered across tables gives consistent improvement. tph=512 + full_coverage is the best at 9.8M params.

**exp070–073: Positional encoding variants with FULL_COVERAGE**

| Exp | PE variant | Val Loss | Params |
|-----|-----------|----------|--------|
| exp066 | concat pos (reference) | 1.6839 | 4.99M |
| exp070 | pos from permutation | 1.9411 | 4.99M |
| exp071 | no PE | 1.9286 | 4.99M |
| exp072 | learned permutation | 1.8475 | 5.01M |
| exp073 | sparse (8 dims) | 2.0061 | 2.79M |

Standard concat PE is clearly best. Learned permutation partially helps but doesn't close the gap.

---

## Phase 4: Compact Architecture Refinement (exp074–076)

Switch to embedding_dim=24 (content) + pos_dim=8 (concat), separate pos feed-in.

| Exp | Change | Val Loss | Params |
|-----|--------|----------|--------|
| exp074 | d=32, d_qk=16 (was 8) | 1.6390 | 6.57M |
| exp075 | d=24, pos_dim=8, d_qk=16, d_v=6 | 1.6281 | 5.97M |
| exp076 | exp075 + tph=128 (was 256) | 1.6866 | 4.86M |

**exp075** becomes the new reference: compact, clean separation of content/position, full_coverage sampling.

---

## Phase 5: Alternative Projections (exp077–082)

Testing whether LUT tables can be replaced with more parameter-efficient alternatives.

### FactoredProjection (exp078–079)
Low-rank output: `output = codes @ prototypes`, rank ∈ {1, 4}.

| Exp | rank | Val Loss | Params |
|-----|------|----------|--------|
| exp078 | 1 | ~2.56 (plateau) | 0.73M |
| exp079 | 4 | ~2.56 (plateau) | 2.89M |

Both plateau at ~2.56. **Dead end** — low-rank factoring cannot represent the same function space as full LUT tables.

### SinusoidalProjection (exp080)
Each entry outputs a sinusoidal function: output[j] = Σ A·sin(f·j+φ). Val ~2.59 plateau. Same failure mode.

### PairVoting (exp082)
Each C(d,2) pair has a dedicated output vector; output = (soft rank features) @ vectors. 609k params. Val ~2.56 plateau.

**Conclusion: LUT table structure (full n_outputs × n_entries weight matrix per table) is essential.** All alternatives plateau near ~2.56 regardless of parameterization or rank.

### Sparse tph (exp077)
tph=512 with sparse input (only 12 active dims out of 32): val=1.968 at step 15k then stopped. The sparse routing destroyed information.

---

## Phase 6: Value Dimension & Output Projection (exp083–084)

### exp083: exp075 run to 100k steps
Best val: **1.5985** @ step 86k. Gap to vanilla (1.3559): **0.242**.

### exp084: d_v=16, out_proj tph=512 — **Current SOTA**
Key insight: d_v=6 (d//h) was a bottleneck. Increasing d_v to 16 (=d_qk) forces the out_proj to handle 64-dim input (h*d_v=4×16), requiring tph≥504 for FULL_COVERAGE.

| Component | exp075 | exp084 |
|-----------|--------|--------|
| d_v | 6 | **16** |
| out_proj input | 24 | **64** |
| out_proj tph | 256 | **512** |
| Val loss | 1.6281 | **1.5845** |
| Params | 5.97M | 7.55M |

**+0.044 improvement** — the largest single gain in Phase 5–6. Split LR (inner=1e-4, unembedder=1e-3) is essential.

---

## Phase 7: Attention Variants (exp085–088)

Ablating RankAttention in exp084.

| Exp | Attention | Val Loss | Params | Notes |
|-----|-----------|----------|--------|-------|
| exp084 | RankAttention (hard, t=1.0) | **1.5845** | 7.55M | SOTA |
| exp085 | exp084, uniform lr=1e-3 | ~1.72 (stopped @22k) | 7.55M | Worse than split LR |
| exp086 | L2-normalized SDPA | ~2.36 (stopped @5k) | 7.55M | Much worse |
| exp087 | Mean-centered SDPA | 1.6641 (stopped @27k) | 7.55M | ~0.08 behind exp084 |
| exp088 | Smooth RankAttention | 1.6259 (soft) / **~2.57** (hard) | 7.55M | Cheats on continuous features |

**Key results:**
- **RankAttention is essential**: replacing with normalized or centered SDPA costs ~0.07–0.27 in val loss
- **Split LR matters**: uniform lr=1e-3 for all params hurts; inner LUTs need lr=1e-4
- **Smooth RankAttention collapses**: the model learns to exploit continuous rank values (soft=1.626) but hard evaluation is catastrophic (2.57). The soft/hard gap reaches ~0.94 by end. Cannot be used for inference.

---

## Overall Progression

| Milestone | Exp | Val Loss | Params | Notes |
|-----------|-----|----------|--------|-------|
| Vanilla baseline | exp002 | 1.3559 | 4.87M | Target |
| LUT + RankAttn (smooth) | exp024 | 1.4371 | 46.2M | 30× params |
| LUT + RankAttn (200k) | exp044 | 1.4821 | 46.2M | |
| Mini architecture | exp069 | 1.6228 | 9.98M | FULL_COVERAGE |
| Mini compact | exp075 | 1.6281 | 5.97M | d=24, pos_dim=8 |
| exp075 @ 100k | exp083 | 1.5985 | 5.97M | |
| **d_v=16 redesign** | **exp084** | **1.5845** | **7.55M** | **Current SOTA** |

**Gap to vanilla: 0.229** (exp084 vs exp002)

---

## Key Findings

1. **T > K** (more tables > deeper routing) at same param budget — once pairwise coverage is complete
2. **FULL_COVERAGE sampling** consistently improves over random: ensures all pairs are seen across tables
3. **nap=4 sweet spot**: nap=3 ties, nap=2 degrades, nap=7+ much worse
4. **Split LR essential**: unembedder needs 10× higher LR than inner LUTs
5. **Concat positional encoding**: dedicated position dimensions prevent routing interference
6. **LUT table structure is irreplaceable**: FactoredProjection, SinusoidalProjection, PairVoting all plateau ~2.56 vs LUT's 1.58
7. **d_v bottleneck matters**: increasing d_v from 6→16 gave +0.044, the largest Phase 5–6 gain
8. **RankAttention > SDPA**: rank-based q/k features significantly better than normalized or centered dot-product attention
9. **Smooth mode in RankAttention is harmful**: model overfits to continuous features, hard eval degrades catastrophically
10. **Hard gap to vanilla (0.23)**: unknown how much is fundamental to LUT routing vs architectural limitations not yet explored

---

## Phase 8: Architecture Consolidation (exp089–098)

Moving to d=32 content + pos=16, single FFN, exploring batch/temperature/nap.

### exp089–091: d=32 wider, no FFN ablation

| Exp | Change | Val Loss | Params |
|-----|--------|----------|--------|
| exp089 | d=32, pos=16, mixed tph | 1.5582 | 10.5M |
| exp090 | + n_alternatives=4 | No summary | 10.5M |
| exp091 | – FFN (attn only) | 1.5626 | 9.7M |

FFN adds small but real value (+0.004 over no-FFN at this scale).

### exp092–094: Batch size & temperature sweep (batch=16)

| Exp | Temperature | Val Loss | Notes |
|-----|-------------|----------|-------|
| exp092 | 1.0 | 1.6961 | Batch 16, all else equal |
| exp093 | 0.5 | 1.6853 | **t=0.5 best** |
| exp094 | 0.1 | 1.6964 | Too sharp |

**temperature=0.5 adopted for all future experiments.** Analytically optimal gradient half-width.

### exp095–098: tph=512, nap sweep, longer training

| Exp | nap | Steps | Val Loss | Params |
|-----|-----|-------|----------|--------|
| exp095 | 4 | 50k | 1.6559 | 13.1M |
| exp096 | 3 | 50k | No summary | — |
| exp097 | 6 | 50k | 1.6302 | 52.5M |
| exp098 | 6 | 100k, b=32 | 1.5375 | 52.5M |

nap=6 significantly better than nap=4 at tph=512. Training to 100k helps.

---

## Phase 9: Linear Unembedder & nap Ablation (exp099–102)

### exp099: normalize_weights — stopped early
normalise_weights=True for attention LUTs. ~+0.12 loss vs exp098. Dead end — constrains per-table contribution scale.

### exp100: Linear unembedder — **new reference**
Replace MultiHeadLut unembedder with `nn.Linear(d, vocab_size)`.

| Exp | Unembedder | Val Loss | Params |
|-----|-----------|----------|--------|
| exp097 | MultiHeadLut | 1.6302 | 52.5M |
| **exp100** | **nn.Linear** | **1.6195** | **44.1M** |

Simpler and better — LUT unembedder was overparameterized with unstable outputs (mean=-37, std=27.6).

### nap sweep with linear unembedder

| Exp | nap | Val Loss | Params |
|-----|-----|----------|--------|
| exp101 | 3 | 1.7202 | 5.5M |
| exp102 | 4 | 1.6872 | 11.0M |
| exp100 | 6 | **1.6195** | 44.1M |

nap=6 clear winner. nap=3 viable for tiny models only.

---

## Phase 10: Alternative Projection (exp103)

**RankProjection2**: rank-project input → linear → tanh → inverse rank project (antisymmetric scatter). Replaces all LUTs.

Val=2.1671 at 50k — much worse. Dead end. Removed from codebase.

---

## Phase 11: Residual Stream (exp104–107)

Idea: separate res_stream (dim=128, starts at 0) updated via `up_proj` after each LUT block. Ranking stream is raw LUT chain with no skip connections. Unembedder reads res_stream.

| Exp | up_proj | res_dim | Steps | Val Loss |
|-----|---------|---------|-------|----------|
| exp104 | Linear(32→128), lr=1e-4 | 128 | 50k | 1.6456 |
| exp105 | Linear(32→128), lr=1e-3 | 128 | 50k | 1.6495 |
| exp106 | Linear(32→128), lr=1e-3 | 128 | 100k | 1.5919 |
| exp107 | MLP(32→128→ReLU→64), lr=1e-4 | 64 | 50k | 1.6581 |

All worse than exp100 at 50k. exp106 at 100k beats exp100 (50k) but not competitive vs exp108+. Residual stream alone not the right direction.

---

## Phase 12: FFN + tph Scaling (exp108–112)

### exp108: Add FFN to exp100

Add `Linear(32,128)→ReLU→Linear(128,32)` with residual after attention in each block.

| Exp | Change | Val Loss | Params |
|-----|--------|----------|--------|
| exp100 | Baseline | 1.6195 | 44.1M |
| **exp108** | **+ FFN** | **1.6047** | **44.1M** |

FFN helps (+0.015). New 50k reference.

### exp109: d_qk=32

Val=1.6163, 69.3M params — worse than exp108 despite more params. Larger d_qk is not the right lever.

### tph scaling

| Exp | tph | Batch | Params | Val Loss | Delta vs prev |
|-----|-----|-------|--------|----------|---------------|
| exp108 | 512 | 16 | 44.1M | 1.6047 | — |
| exp110 | 1024 | 16 | 88.1M | 1.5845 | −0.020 |
| exp111 | 2048 | 32 | 176.2M | 1.5021 | −0.082 |
| exp112 | 4096 | 32 | 352.4M | **1.4834** | −0.019 |

Clear scaling law but diminishing returns. 1024→2048 gave −0.082; 2048→4096 gave only −0.019.

**Vanilla baseline (exp003): 1.3545 at 4.87M params.** Gap at exp112: **0.129** with 72× more params.

---

## Overall Progression (ctx=32 era)

| Milestone | Exp | Val Loss | Params | Notes |
|-----------|-----|----------|--------|-------|
| Vanilla baseline | exp002 | 1.3559 | 4.87M | Target (ctx=32) |
| Smooth LUT + RankAttn | exp024 | 1.4371 | 46.2M | |
| d_v=16 redesign | exp084 | 1.5845 | 7.55M | Prior SOTA |
| Linear unembedder | exp100 | 1.6195 | 44.1M | New reference |
| + FFN | exp108 | 1.6047 | 44.1M | |
| tph=4096, b=32 | exp112 | **1.4834** | 352.4M | ctx=32 SOTA |

---

## Phase 13: ctx=128 migration, QK LayerNorm, nap scaling (exp200–246)

Moved to **ctx=128**. New vanilla baseline: 1.3556 @ 4.87M.

**QK LayerNorm (exp218→220)**: LN on Q and K before the dot product is critical for scaling LUT capacity. Without it, tph=256 barely beats tph=128; with it, dramatic improvement. LUT outputs have erratic magnitudes — LN stabilizes the gradient scales.

**nap scaling**: more anchor pairs (larger tables) is the strongest lever. nap=10 (table_dim=1024) dramatically beats nap=5 (32). Sublinear gain: each doubling helps less.

**Best results with residuals** (ctx=128):
| Exp | Config | Params | Val Loss |
|-----|--------|--------|----------|
| **exp233** | Q/K nap=5, V/Out/FFN nap=12, tph=128 | 304M | **1.3562** |
| exp232 | All nap=12, tph=128 | 503M | 1.3633 |
| exp238 | No FFN, OutProj nap=10/tph=256 | 53.5M | 1.3796 |

**Mixed nap finding**: Q/K don't need big tables (nap=5 suffices — attention is a soft routing mechanism). Heavy lifting is in OutProj/FFN (nap=12).

---

## Phase 14: No-residual, ranking-based (exp247–249)

Core idea: residual connections break permutational consistency (`rank(x+y) ≠ f(rank(x), rank(y))`). Remove residuals; add backward ranking-prediction loss to recover gradient flow. See [project_permutational_architecture](project_permutational_architecture.md).

**exp249** architecture: Q/K LUT(nap=5,tph=128) → QK-LN → SDPA → OutProj LUT(nap=5,tph=256) → LN → next layer. No FFN, no residuals. 5.56M params. Reached CE~1.88 at step 6K — first successful pure-ranking 6-layer training.

---

## Phase 15: BitPermutationLUT era (exp299–342)

Architecture family **FullBitPermRankAttn_ctx128**: 6 layers, E=32, 4 heads, d_qk=24, d_v=16, ctx=128. Core primitive is `BitPermutationLUT` (1-bit packed weights, fused CUDA forward/backward, fp8/bf16 latent + STE-gated Adam via `BitPermutationLUTOptimizer`).

**Key design choices**:
- `CANONICAL_FULL_COVERAGE` anchor-sampling policy (default): guarantees coverage of pair pool when slots ≥ C(N,2). Tile-and-repair algorithm.
- `partition_sets`: restricts out_proj input anchors to within-head pairs (the `H·d_v` concat has no cross-head ordering meaning).
- `DominanceCanonicalize` + `SDPA` (replaced earlier `RankAttention`).
- `DominanceToVector` (Borda+LN) after attention and out_proj.

| Exp | Config | Val Loss | Notes |
|-----|--------|----------|-------|
| exp329 | BitPermLUT (in=10, tph=2048, on=10), no partition | **1.379** | CFC default reference |
| exp338 | + partition_sets | 1.401 | Partition alone slightly worse |
| exp332 | Narrower partition config | 1.412 | |
| exp336 | BitPermLUT (in=10, tph=256, on=16) | 1.494 | Small tph hurts |
| exp342 | BitPermLUT (in=4, tph=8192, on=10) | 1.529 | Distill-winner shape, but e2e regresses |

**Distillation framework** (`transformer_exps/distill_exp338/`): extract `(out_proj_input, out_proj_output)` pairs from exp338 per layer, train candidates to match pair-wise sign of Borda-projected E-dim output. Key lessons:
1. Target must be rank of Borda-projected output, not raw 496-dim dominance.
2. `src/spiky/lutorch/determinism.py` added — `set_deterministic(True)` gives bitwise-identical checkpoints.

**Per-layer difficulty** (at 1.3M bit-params): layer 0 = 0.887, layer 5 = 0.986 sign-acc. Layer 0's heavily-local attention (78% within 4 tokens) makes its out_proj input info-dense.

**Layer-0 Pareto** (BitPermLUT): 21M = 0.985; 10.5M = 0.972 (in=7, tph=8192); 2.1M = 0.908.

---

## Phase 16: PermutationalLut (smooth-rational) (exp343–346)

`PermutationalLut` (`src/spiky/lutorch/permutational_lut.py`): fp32 weights, soft-rational forward `0.5·x/(T+|x|)` → scatter_add. Emits E-dim vector directly (no Borda+LN needed). Standard Adam on `nn.Parameter`. `borda_scale_mode='borda'` (default) divides scatter by √(E−1) to match BitPermLUT's post-LN magnitude.

`vote_quant_levels` kwarg adds STE quantization between rational and scatter (tests the "quantized synapse" hypothesis).

| Exp | out_proj | Val Loss | Notes |
|-----|----------|----------|-------|
| exp343 borda | PermLut fp32, in=10, tph=256, on=32 | **1.450** | Smooth-rational closes gap by 30% |
| exp346 | PermLut + vote_quant_levels=16 (4-bit) | 1.470 | K=4 quant costs only 0.02 val-loss |

**Layer-0 distill Pareto (PermLut)**: 21M = 0.9991; **8.4M (in=10, tph=256, on=32) = 0.9956** (best tph-efficient); 5.2M = 0.9917. Much better than BitPermLUT at same bit-count due to fp32 weight expressivity.

---

## Phase 17: MultiBitPermutationLUT (exp347–350+)

New primitive `MultiBitPermutationLUT` (`src/spiky/lutorch/multi_bit_permutation_lut.py`): K-bit packed signed ints (K∈{2,4,8}) in int32 blocks. Semantically matches PermLut+vote_quant_levels=2^K but stored efficiently. Features:
- Rational-pre-quant (T=0.1) in pack kernel.
- **Midrise quantizer** (no level at 0) — fixes "dead zone" bootstrap issue.
- Per-pair bias added in forward to compensate for midrise shift.
- Own optimizer `MultiBitPermutationLUTOptimizer` (bf16 latent + STE gate + multi_bit_pack repack).

Forward perf at same shape as BitPermLUT: within 5% at tph=256–2048; up to 2× faster at tph=8192 (BitPermLUT's warp-coop path not cost-effective there).

| Exp | Config | Val Loss | Notes |
|-----|--------|----------|-------|
| **exp347** | **Hybrid: BitPermLUT q/k/v + MultiBit K=4 out_proj** | **1.432** | **Beats exp346 (same shape PermLut+q16) by 0.04** |
| exp349 | All-MultiBit K=4, qk_tph=192 | ~1.48 | q/k "too heavy" — uniform K=4 hurts |
| exp350 | All-MultiBit K=4, qk_tph=64 | 1.492 | Reduced q/k tph partially recovers but still worse than exp347 |

**Lesson**: q/k don't benefit from K>1 — their output feeds SDPA + DominanceCanonicalize which wants crisp ±-ish dominance. out_proj (wide fan-in, 64→32 across 4 heads) does benefit.

**Matched-shape distill (in=10, tph=256, on=32)**: PermLut fp32 = 0.9956; MultiBit K=4 midrise = 0.9898. Close, with 8× less storage for MultiBit.

---

## Updated Overall Progression (ctx=128 era)

| Milestone | Exp | Val Loss | Storage | Notes |
|-----------|-----|----------|---------|-------|
| Vanilla ctx=128 | — | 1.3556 | 4.87M fp32 params | Target |
| exp233 (ctx=128 LUT SOTA, nap sweep) | exp233 | 1.3562 | 304M fp16 | Matches vanilla with 60× more params |
| No-residual (exp249) | exp249 | ~1.88 @ 6k | 5.56M | Proof-of-concept |
| BitPermLUT + partition | **exp329** | **1.379** | 210M bits ≈ 26MB | BitPermutationLUT baseline |
| Distill-winner shape, e2e | exp342 | 1.529 | 17M bits ≈ 2MB | Shape transfer failed |
| PermLut smooth-rational | exp343 | 1.450 | 50M fp32 = 200MB | |
| PermLut + vote_q16 | exp346 | 1.470 | 50M fp32 = 200MB | |
| **Hybrid MultiBit K=4** | **exp347** | **1.432** | **8.6M bit + 34Mbit ≈ 5.4MB** | **Current best at small tph out_proj** |

**Current frontier**: exp329 is the accuracy target (1.379, all-BitPermLUT). exp347 is the storage-efficient trade-off (1.432 with 5× smaller out_proj).
