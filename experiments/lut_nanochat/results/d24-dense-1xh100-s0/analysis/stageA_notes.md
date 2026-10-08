# d24 dense baseline: Stage A weight-space analysis

**Checkpoint:** `results/d24-dense-1xh100-s0/checkpoints/model_005568.pt` (final step), analysed 2026-10-08.
**Scope:** static weight geometry only. No forward passes and no data.
**Reproduce:** run `analysis/stageA_weight_geometry.py` (runtime **54 s** on the RTX 5090, CUDA), then `analysis/stageA_untrained_rows.py`.
**Outputs:**
- raw numbers: `d24_checkpoint_analysis_stageA/stageA_results.json` and `stageA_untrained_rows.json`;
- plots: in the same folder.

Every number below is *measured* on the stored tensors (cast to fp32/fp64), unless it is explicitly labelled otherwise.

**Environment:** the spiky venv (torch 2.9.1+cu130), *not* the pinned nanochat venv. That venv was never created on this host (`runs/setup_unified_env.sh` was run on Nebius). Pure weight arithmetic does not depend on the pin.

**Metric definitions:**
- **r90 / r99:** the smallest rank holding 90% / 99% of Σσ².
- **PR:** the participation ratio, (Σσ²)²/Σσ⁴.
- **erank:** the entropy effective rank, exp H(σ/Σσ).
- **CKA:** linear CKA on column-centred rows.

## 0. Artifact verification: PASSED

| check | value |
|---|---|
| params (checkpoint) | 1,384,122,122 = the code's own `num_scaling_params()` assertion = the expected value |
| vocab / n_embd / n_layer / heads | 32,768 / 1536 / 24 / 12 (n_kv_head 12) |
| seq len / window pattern | 2048 / SSSL |
| meta step / val_bpb | 5568 / 0.719344 |
| keys vs vendored `GPT` state_dict | 0 missing, 0 unexpected, 0 shape mismatches |

## 1. Token embedding (`wte`) and `lm_head` (untied)

| | wte | lm_head |
|---|---|---|
| row norm, median (p1–p99) | 361.5 (335.4–370.4) | 12.31 (9.31–15.43) |
| row norm, min / max | 30.3 / 397.4 | 1.36 / 18.94 |
| Spearman(row norm, token id) | −0.26 | −0.13 |
| r90 / r99 of 1536 | **1109 / 1483** | **1211 / 1482** |
| PR / erank | 433 / 1409 | 478 / 1482 |
| mean pairwise cosine, raw (4,096 rows) | 0.0215 | 0.0360 |
| mean pairwise cosine, centred | −0.0002 | −0.0002 |

**Row norm vs token id** (see `row_norms.png`):
- `wte` norms are flat, except a dip over the first few hundred ids (bytes and early merges).
- `lm_head` and VE norms *decline slowly* with id. Higher-id BPE tokens, which are later and rarer merges, are ~3–7% smaller.
- So there is no strong rare-token blow-up or collapse.
- **48 token ids sit exactly at init scale in every table.** Their `wte` norm is ≈30, against 0.8·√1536 = 31.4 at init, and their VE norm is ≈1.0, which is the init value. These look like tokens never seen in 5.84 B training tokens (see §2 caveat).

**Spectrum.**
- Both matrices are **high-rank**: 90% of the energy needs ~1100–1200 of 1536 directions.
- There is one dominant direction, s₁/s₂ ≫ 1. That is a mean or "common" component; centring removes it (anisotropy goes from 0.02–0.04 to 0.0002).
- After centring the embeddings are essentially isotropic. There is **no "cone"**.

**wte vs lm_head:**
- Per-token cosine between a token's `wte` row and its `lm_head` row: median **0.023** (p5 −0.063, p95 0.089). That is unrelated per token.
- CKA 0.385; the cosines of the top-256 principal angles average 0.43, none above 0.9.
- So untying bought **genuinely different matrices**: the output space is not a re-scaled input space.

## 2. Value embeddings: 12 × `Embedding(32768, 1536)`, odd layers 1…23 (43.6% of parameters)

**Per table** (all 12 look alike; numbers are ranges across tables):
- Row norm median 64.9 → 73.6, growing with depth. Min ≈0.97 (the init-scale rows); max 114–138.
- r90 **1153–1273**, r99 1468–1505 of 1536.
- PR **1035–1382**, higher in deeper layers.
- Mean pairwise cosine ≈0.000, centred −0.0005.

So each table is **close to full-rank and isotropic**, *more so than `wte`*. Singular values are flat (`spectra.png`): s_min/s_max ≈ 0.1–0.4, against ≈0.08 for `wte`.

**Cross-table redundancy, the decisive question: there is very little.**

| measure | value |
|---|---|
| pairwise CKA, 66 pairs: mean (p5–p95) | **0.062** (0.044–0.092); adjacent VE layers 0.068–0.102 |
| per-token cosine after the best orthogonal (Procrustes) alignment, median over pairs | **0.199** (0.185–0.251) |
| **one shared table + per-layer scalar gate** (best rank-1 over the layer axis) | **9.3%** of variance retained (≈1/12: the 12 tables share almost nothing) |
| **one shared table + per-layer 1536×1536 linear readout** (= rank 1536 of the stacked matrix) | **29.6%** retained |

**Shared rank-r code with per-layer readouts**, i.e. the optimal rank-r of the stacked [32768 × 18432] matrix, uncentred (reconstruction energy):

| r | 32 | 64 | 128 | 256 | 512 | 1024 | 1536 | 3072 |
|---|---|---|---|---|---|---|---|---|
| retained | 1.6% | 2.7% | 4.6% | 7.8% | 13.2% | 22.1% | 29.6% | 47.4% |

- Centred: same to ±0.0003.
- Stacked r90 / r99: 10,753 / 16,367 of 18,432; PR 8,787.
- Plot: `ve_variance_retained_vs_rank.png`.

**Each table alone at rank r** (no sharing), retained:

| r | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|
| retained | 3.7–8.3% | 7.2–13.5% | 13.8–22.2% | 25.8–36.2% | 46.4–57.4% |

Early layers are slightly more compressible than late ones.

**Linear predictability from `wte`** (least-squares `wte → VE_l`, centred), **R² = 0.069–0.095** per layer. The value tables are essentially *not* linear functions of the token embedding.

**Product quantisation**, post hoc k-means on the trained tables (25 iterations; layers 1, 13 and 23), relative Frobenius error:

| groups g × K | 8×256 | 16×256 | 48×256 | 96×256 | 16×1024 | 48×1024 |
|---|---|---|---|---|---|---|
| rel. error | 0.97 | 0.94–0.95 | 0.85–0.86 | **0.74** | 0.92–0.93 | 0.80 |

Even the finest setting (96 sub-codebooks of 16 dims, 256 codes each) leaves **74%** relative error. PQ reconstruction of these tables is poor.

**Robustness check.** Recomputing on the 32,720 trained rows only (dropping the 48 init-scale ids) leaves both the stacked-variance curve and R² unchanged to 3–4 decimals (`stageA_untrained_rows.json`).

**`ve_gate`.**
- 144 parameters = `Linear(12 → n_kv_head=12)`, applied to the *first 12 channels of the residual stream* (`x[..., :12]`).
- It gives one gate per KV head: gate_h = 3·sigmoid(w_h · x[:12]) ∈ (0, 3) (`gpt.py` L81–97). At x = 0 every gate is 1.5.
- Learned weights: per-head row norm grows with depth, from 1.8–2.7 (layer 1) to 2.9–9.6 (layer 23). Mean weight is −0.006 to +0.234.
- So deep layers make the value-embedding mix-in much more input-dependent.
- **How much each table actually contributes cannot be read from weights.** It depends on x, which is Stage B.

**Interpretation, for the Stage 2a plan (not a measurement):**
- The trained value embeddings behave like ~12 independent, near-full-rank, high-entropy lookup tables.
- They are not low-rank, not shared across layers, not predictable from `wte`, and not PQ-friendly.
- *Post-hoc* compression (fit-then-evaluate) will therefore lose most of the table content. Whether that content *matters* to the loss is a separate question, and Stage B and the ablations should answer it.
- The "VE removed entirely" bounding ablation becomes more important, not less. If the tables are high-entropy but the loss barely needs them, a compressed form trained *from scratch* may still work.

## 3. Attention and MLP weights (24 layers; `effective_rank_vs_depth.png`)

**Effective rank, as a fraction of full rank:**
- MLP `c_fc` (6144→1536 rows): r90 **0.74–0.75**, PR 0.73–0.78.
- MLP `c_proj`: r90 0.74–0.75, PR 0.69–0.77; PR rises in the deeper half.
- Attention q/k/v/o: r90 **0.39–0.53**, PR **0.25–0.54**. Attention matrices use roughly half their rank, MLPs three quarters.

**Depth patterns:**
- **q/k effective rank dips sharply at layers 15, 19 and 23** (c_k PR 0.25 / 0.30 / 0.32 against ~0.42 elsewhere). These are full-context layers of the SSSL pattern (L at 3, 7, 11, 15, 19, 23); the dip appears only in the deeper half.
- **`attn.c_proj` rank alternates by parity:** odd layers (which carry a value embedding) are consistently higher, e.g. PR 0.54 vs 0.42.

**MLP unit usage (ReLU²), structural proxies:**
- No hidden unit has outgoing (`c_proj` column) norm below 1% or 5% of the layer median, in any layer. There are 0 exact-zero rows or columns.
- Outgoing norms are tight: median ≈2.60, p5 ≈2.30–2.38.
- Input-row norms: median 3.69–3.76, p5 3.11–3.20.
- **Structurally, no dead units.** Activation-level deadness (units that never fire on data) needs Stage B.

**Learned scalars:**
- `resid_lambdas` (layers 0→23): 0.33, 0.21, 0.53, 0.43, 0.64, 0.56, 0.68, 0.54, 0.77, 0.66, 0.69, 0.54, 0.90, 0.95, 0.95, 0.73, 1.10, 0.86, 0.94, 0.76, 1.04, 0.80, 0.85, 0.57. This is a rising trend from their init of 1.15 → 1.05.
- `x0_lambdas`: 15.73, 9.78, 5.62, 2.30, 3.75, 3.82, −2.32, 1.35, 1.86, 4.60, 0.31, 1.80, 3.64, 4.17, −1.59, −1.78, 5.58, 6.80, −2.99, −1.05, 4.01, 2.29, −1.53, −0.66.
  - These are large compared with their init of 0.20 → 0.05. They are especially strong at layers 0–2, and several are negative.
  - The input embedding is re-injected heavily early on.
- `smear_lambda` 0.206 (init 0); `backout_lambda` 0.179 (init 0.2).

## 4. Overall

**Parameter budget as stored** (tensor bytes sum to 4,227,865,640; the file is 4,227,935,530; container overhead 69,890 B):

| group | params | dtype | bytes |
|---|---|---|---|
| value_embeds (12) | 603,979,776 | bf16 | 1,207,959,552 |
| mlp (24 × 2) | 452,984,832 | fp32 | 1,811,939,328 |
| attn incl. ve_gate | 226,494,144 | fp32 | 905,976,576 |
| wte | 50,331,648 | bf16 | 100,663,296 |
| lm_head | 50,331,648 | fp32 | 201,326,592 |
| scalars / smear | 74 | fp32 | 296 |

**Sanity:**
- No NaN or Inf in any tensor.
- Both zero-initialised projections (`attn.c_proj`, `mlp.c_proj`) moved well off zero (Frobenius 164–239).
- `lm_head` std went 0.001 → 0.317 and `wte` std 0.8 → 9.19.
- Nothing looks untrained apart from the 48 init-scale token rows noted above.

## What could NOT be computed here, and why
- **Actual per-layer value-embedding contribution / gate values:** they depend on activations → Stage B.
- **Activation-level dead MLP units, attention entropy and window usage:** they need forward passes on data → Stage B.
- **PQ on all 12 tables:** measured on 3 (layers 1, 13, 23). They agree within ±0.01, so the other 9 are expected to match (*not measured*).
- **Which tokens the 48 init-scale ids are:** the ids are identified, but they were not decoded to strings. That needs the tokenizer, which is present on this host and identical to the run's; I didn't load it for Stage A.
