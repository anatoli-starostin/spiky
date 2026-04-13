# Transformer Experiments Journal

Dataset: fineweb_texts.txt (byte-level, CONTEXT_SIZE=32, VOCAB_SIZE=257, BOS=256)
Metric: mean cross-entropy loss on 10k held-out test regions
Hardware: NVIDIA H100 80GB

---

## exp001_vanilla_baseline

**Description:** Vanilla causal transformer (SDPA, causal mask). Byte-level LM on fineweb. Establishes the baseline loss target for LUT transformer comparisons.

**Hyperparameters:**
- d_model=256, n_heads=4, num_layers=6, ff_mult=4, dropout=0.0
- optimizer=Adam, lr=1e-4, schedule=warmup_cosine (10% warmup, decay to 10%), n_steps=100k, batch=128

**Parameters:** 4.87M

**Results:**
- best_val_loss: **1.6120** @ step 98k
- final_val_loss: 1.6125
- training_time: 0.19h

**Plots:** [loss.png](exp001_vanilla_baseline/loss.png)

**Interpretation:** Clean convergence. Loss plateaued around 1.61 in the final decay phase with no sign of overfitting. Serves as the reference point for all LUT transformer variants. Superseded by exp002 (better lr).

---

## exp002_vanilla_lr1e3

**Description:** Same as exp001 but lr=0.001. LR sweep to find better baseline.

**Hyperparameters:**
- d_model=256, n_heads=4, num_layers=6, ff_mult=4, dropout=0.0
- optimizer=Adam, lr=1e-3, schedule=warmup_cosine (10% warmup), n_steps=100k, batch=128

**Parameters:** 4.87M

**Results:**
- best_val_loss: **1.3559** @ step 99k
- training_time: 0.19h

**Plots:** [loss.png](exp002_vanilla_lr1e3/loss.png)

**Interpretation:** Large improvement over exp001 (+0.256). Model still improving at step 99k — not fully converged. lr=0.001 is the baseline lr for this series.

---

## exp003_vanilla_wd1e4

**Description:** Same as exp002 but AdamW with weight_decay=1e-4.

**Hyperparameters:**
- d_model=256, n_heads=4, num_layers=6, ff_mult=4, dropout=0.0
- optimizer=AdamW, lr=1e-3, weight_decay=1e-4, schedule=warmup_cosine, n_steps=100k, batch=128

**Parameters:** 4.87M

**Results:**
- best_val_loss: **1.3545** @ step 98k
- training_time: 0.19h

**Plots:** [loss.png](exp003_vanilla_wd1e4/loss.png)

**Interpretation:** Negligible improvement over exp002 (−0.0014). No meaningful overfitting at 100k steps, so weight decay has little effect. Baseline fixed as: Adam, lr=1e-3, no weight decay.

---

## exp004_lut_baseline

**Description:** LUTTransformer matching lutorch_transformer.ipynb architecture exactly. Warmup-cosine lr schedule, 100k steps. lr swept: 1e-3 diverged, 1e-4 stable.

**Hyperparameters:**
- embedding_dim=64, num_layers=6, num_heads=4
- n_anchor_pairs_attn=10, n_anchor_pairs_ffn=12, tables_per_head=96, ffn_tables=96
- smooth_mode=False, n_alternatives=1, calibrate_output=False
- optimizer=Adam, lr=1e-4, schedule=warmup_cosine (10% warmup), n_steps=100k, batch=128

**Parameters:** 191M

**Results:**
- best_val_loss: **1.4628** @ step 98k
- training_time: 0.92h

**Plots:** [loss.png](exp004_lut_baseline/loss.png)

**Interpretation:** LUT model converges to 1.463 vs vanilla baseline 1.356 — 0.107 worse with 39× more parameters. Model not fully converged at 100k steps. The notebook architecture is a starting point; needs tuning to compete with vanilla. Key issues: very high param count (191M) driven by large LUT tables, and slower convergence requiring lower lr than vanilla.

---

## exp005_lut_unembedder

**Description:** Same as exp004 but replaces the dense norm+matmul unembedder with a `MultiHeadLut` (1 head, FFN params: n_anchor_pairs=12, tables=96, n_outputs=vocab_size=257).

**Hyperparameters:** Same as exp004 except unembedder is LUT-based.

**Parameters:** 292M (+101M vs exp004 from the large LUT output projection)

**Bandwidth:**
- virtual: 12.9 MB / dense equivalent: 37,396 MB (ratio 0.000345)

**Results:**
- best_val_loss: **1.6403** @ step 98k
- training_time: 1.05h

**Plots:** [loss.png](exp005_lut_unembedder/loss.png)

**Interpretation:** Significantly worse than exp004 (+0.178 loss). The dense unembedder (norm + matmul against embedding matrix) was doing real work that the LUT unembedder can't match at this scale. With n_anchor_pairs=12, each token reads only 96 out of 4096 table entries — likely too sparse for a direct vocab projection over 257 classes. The dense unembedder in exp004 remains the better choice.

---

## exp006_lut_unembedder_nalt3

**Description:** Same as exp005 but n_alternatives=3 globally. More gradient signal per lookup step.

**Hyperparameters:** Same as exp005 except `n_alternatives=3`.

**Parameters:** 292M (same — n_alternatives doesn't add params, only reads more entries)

**Bandwidth:**
- virtual: 12.9 MB / dense equivalent: 37,396 MB (ratio 0.000345)

**Results:**
- best_val_loss: **1.5031** @ step 99k
- training_time: 1.51h

**Plots:** [loss.png](exp006_lut_unembedder_nalt3/loss.png)

**Interpretation:** Large improvement over exp005 (−0.137). n_alternatives=3 significantly helps the LUT unembedder learn — reading 3 candidate entries per table gives the gradient more signal to update the routing. Still 0.040 behind exp004's dense unembedder, and not yet converged at 100k steps. n_alternatives is a key lever for LUT quality.

---

## exp007_lut_sum_reduction

**Description:** Same as exp006 but sum loss reduction (gradients B*T=4096× larger). Reported loss divided by B*T for comparability.

**Hyperparameters:** Same as exp006 except `loss_reduction=sum`.

**Parameters:** 292M | **Bandwidth:** 12.9 MB virtual

**Results:**
- best_val_loss: **1.5111** @ step 98k
- training_time: 1.51h

**Plots:** [loss.png](exp007_lut_sum_reduction/loss.png)

**Interpretation:** Slightly worse than exp006 (−0.008). Larger gradients from sum reduction did not help. Mean reduction is better.

---

## exp008_lut_unembedder_lr_split

**Description:** Same as exp006 but with split optimizer: unembedder lr=0.001, rest lr=0.0001. Both with warmup-cosine schedule.

**Hyperparameters:** Same as exp006 except unembedder uses 10× higher lr.

**Parameters:** 292M | **Bandwidth:** 12.9 MB virtual

**Results:**
- best_val_loss: **1.4869** @ step 99k
- training_time: 1.52h

**Plots:** [loss.png](exp008_lut_unembedder_lr_split/loss.png)

**Interpretation:** New best for LUT unembedder (−0.016 vs exp006). Higher lr for unembedder helps it learn faster. Still 0.024 behind exp004's dense unembedder, not yet converged. Note: lr=0.01 for unembedder diverged — 0.001 is the sweet spot.

---

## exp009_lut_vanilla_mirror

**Description:** New LUT architecture mirroring vanilla transformer: separate q/k/v/out_proj/ffn/unembedder MultiHeadLuts, SDPA attention, residuals, no norms. Positional encoding concatenated with token embedding. All LUTs: n_anchor_pairs=10, tables_per_head=96, n_alternatives=3. Split lr: unembedder=0.001, rest=0.0001.

**Hyperparameters:**
- embedding_dim=64, num_heads=4, d_head=16, num_layers=6, hidden_dim_ffn=256
- n_anchor_pairs=10, tables_per_head=96, n_alternatives=3, smooth_mode=False
- optimizer=Adam, lr=0.0001 (unembedder: 0.001), schedule=warmup_cosine, n_steps=100k, batch=128

**Parameters:** 214M

**Bandwidth:**
- virtual: 26.8 MB / dense equivalent: 27,393 MB (ratio 0.001)

**Results:**
- best_val_loss: **1.4791** @ step 99k
- training_time: 1.15h

**Plots:** [loss.png](exp009_lut_vanilla_mirror/loss.png)

**Interpretation:** Beats exp008 (−0.008) with a cleaner architecture. SDPA-based attention with LUT Q/K/V projections works comparably to the original LUTAttention approach, with 3× better bandwidth efficiency vs exp004 (26.8 MB vs ~37K MB). Still not converged at 100k steps. Gap vs vanilla (1.356) remains large, but the comparison is about virtual bandwidth budget, not raw parameter count.

---

## exp010_lut_additive_pe

**Description:** Same as exp009 but additive positional encoding: token_emb(d_model) + pos_emb(d_model) instead of concat(token_emb(d/2), pos_emb(d/2)).

**Hyperparameters:** Same as exp009 except PE is additive.

**Parameters:** 214M | **Bandwidth:** 26.8 MB virtual

**Results:**
- best_val_loss: **1.4926** @ step 97k
- training_time: 1.15h

**Plots:** [loss.png](exp010_lut_additive_pe/loss.png)

**Interpretation:** Worse than exp009 (−0.014). Concatenation is better for LUT routing — it reserves dedicated dimensions for token and position signals, making it easier for anchor pairs to specialize. With additive PE both signals compete for the same dimensions, which likely disrupts LUT anchor matching. Concat PE remains the preferred approach.

---

## exp011_lut_no_pe

**Description:** Same as exp009 but no positional encodings. Token embedding uses full d_model. Ablation to quantify PE contribution.

**Hyperparameters:** Same as exp009 except no pos_emb, token_emb dim=d_model.

**Parameters:** 214M | **Bandwidth:** 26.8 MB virtual

**Results:**
- best_val_loss: **1.6618** @ step 98k
- training_time: 1.16h

**Plots:** [loss.png](exp011_lut_no_pe/loss.png)

**Interpretation:** Without PE, loss is 1.662 vs 1.479 with concat PE — a **+0.183 penalty**. Positional information is critical for this task. The model without PE is position-blind and can only use token co-occurrence statistics, not sequence order. Confirms that concat PE in exp009 is essential and well-designed for LUT routing.

---

## exp012_lut_double_ffn

**Description:** Same as exp010 (additive PE) but FFN replaced with two consecutive LUTs (d→d each), no post_processor.

**Hyperparameters:** Same as exp010 except FFN = ffn1(d→d) → ffn2(d→d), outer residual only.

**Parameters:** 252M (+38M vs exp010) | **Bandwidth:** 31.5 MB virtual

**Results:**
- best_val_loss: **1.5064** @ step 97k
- training_time: 1.32h

**Plots:** [loss.png](exp012_lut_double_ffn/loss.png)

**Interpretation:** Worse than single FFN exp010 (−0.014). Two sequential LUTs without an inner residual are harder to optimize — the second LUT gets no direct gradient path to the input. Next: try with residual between ffn1 and ffn2.

---

## exp013_lut_double_ffn_res

**Description:** Same as exp012 but with residual between ffn1 and ffn2: ffn2 receives `x + ffn1_out`, block output is `x + ffn1_out + ffn2_out`.

**Hyperparameters:** Same as exp012 except inner residual added.

**Parameters:** 252M | **Bandwidth:** 31.5 MB virtual

**Results:**
- best_val_loss: **1.4728** @ step 99k
- training_time: 1.32h

**Plots:** [loss.png](exp013_lut_double_ffn_res/loss.png)

**Interpretation:** New LUT best (−0.020 vs exp010, −0.034 vs exp012). Residual between FFN LUTs is essential for optimization — it gives the second LUT a direct gradient path and allows each to specialize incrementally. Still not converged at 100k steps. Gap to vanilla: 0.117.

---

## exp014_lut_small

**Description:** Same as exp013 but with reduced model size: embedding_dim=32, tables_per_head=16, n_anchor_pairs=6. Exploratory ablation to understand how much capacity the LUT model needs.

**Hyperparameters:**
- embedding_dim=32, num_heads=4, num_layers=6
- n_anchor_pairs=6, tables_per_head=16, n_alternatives=3
- smooth_mode=False, additive PE, double FFN with residual
- optimizer=Adam, lr=0.0001 (unembedder: 0.001), schedule=warmup_cosine, n_steps=100k, batch=128

**Parameters:** 1.45M | **Bandwidth:** 2.89 MB virtual / 184.7 MB dense (ratio 0.0156)

**Results:**
- best_val_loss: **2.0576** @ step 98k
- training_time: 0.42h

**Plots:** [loss.png](exp014_lut_small/loss.png)

**Interpretation:** Large regression vs exp013 (+0.585 loss). Reducing embedding_dim from 64→32 and tables_per_head from 96→16 (with nap 10→6) severely limits model capacity. Starting point for a systematic d=32 sweep (exp015–019).

---

## exp015_lut_small_t32

**Description:** Same as exp014 but tables_per_head=32 (was 16). Fixing d=32, nap=6.

**Hyperparameters:** Same as exp014 except tables_per_head=32.

**Parameters:** 2.89M | **Bandwidth:** 1.0 MB virtual

**Results:**
- best_val_loss: **1.8619** @ step 90k
- training_time: 0.41h

**Plots:** [loss.png](exp015_lut_small_t32/loss.png)

**Interpretation:** Doubling tables (16→32) gives −0.196. Model starts converging earlier (best @ 90k vs 98k). T is a strong lever even at K=6.

---

## exp016_lut_small_nap8

**Description:** Same as exp015 but n_anchor_pairs=8 (was 6). d=32, tables_per_head=32.

**Hyperparameters:** Same as exp015 except n_anchor_pairs=8.

**Parameters:** 11.6M | **Bandwidth:** 1.0 MB virtual

**Results:**
- best_val_loss: **1.6754** @ step 98k
- training_time: 0.40h

**Plots:** [loss.png](exp016_lut_small_nap8/loss.png)

**Interpretation:** K=6→8 gives −0.187. Theoretically motivated: with d=32 and T=32 tables, K=6 leaves ~5 input dimension pairs with systematic blind spots (never jointly observed); K=8 reduces this to near-zero. Large gain confirms the coverage bottleneck.

---

## exp017_lut_small_nap10

**Description:** Same as exp016 but n_anchor_pairs=10 (was 8). d=32, tables_per_head=32.

**Hyperparameters:** Same as exp016 except n_anchor_pairs=10.

**Parameters:** 46.2M | **Bandwidth:** 1.0 MB virtual

**Results:**
- best_val_loss: **1.6188** @ step 95k
- training_time: 0.43h

**Plots:** [loss.png](exp017_lut_small_nap10/loss.png)

**Interpretation:** K=8→10 gives −0.057 — diminishing returns vs K=6→8. Coverage was already nearly complete at K=8; extra table entries are less useful than more tables.

---

## exp018_lut_k8_t128

**Description:** Same as exp016 but tables_per_head=128 (was 32). d=32, nap=8.

**Hyperparameters:** Same as exp016 except tables_per_head=128.

**Parameters:** 46.2M | **Bandwidth:** 23.1 MB virtual

**Results:**
- best_val_loss: **1.5004** @ step 99k
- training_time: 0.90h

**Plots:** [loss.png](exp018_lut_k8_t128/loss.png)

**Interpretation:** At the same 46.2M param budget, K=8/T=128 (1.500) beats K=10/T=32 (1.619) by 0.119 — **T beats K**. More tables provide better additive decomposition coverage; finer routing per table has diminishing returns once pairwise coverage is complete. Still not converged at 99k. Approaches exp013 (1.473, d=64) despite d=32.

---

## exp019_lut_k6_t512

**Description:** Same as exp018 but n_anchor_pairs=6 (was 8), tables_per_head=512 (was 128). d=32. Tests whether pushing T further at smaller K continues to help.

**Hyperparameters:** Same as exp018 except n_anchor_pairs=6, tables_per_head=512.

**Parameters:** 46.2M | **Bandwidth:** 92.3 MB virtual

**Results:**
- val_loss @ step 64k: ~1.552 (run killed — too slow, projected ~1.50–1.52 final)
- training_time: N/A (stopped early)

**Interpretation:** Killed at step 64k. Trajectory suggested ~1.50–1.52 final — similar to exp018 but at 4× the inference bandwidth. K=6 routing is too coarse even with T=512: the coverage blind spots per table can't be fully compensated by more tables when K<8 leaves individual routing decisions too noisy. exp018 (K=8/T=128) remains the better operating point for d=32.

---

## exp020_rank_attention

**Description:** Same as exp018 but replaces SDPA with RankAttention (pairwise rank features for q/k).

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5677** @ 100k

---

## exp021_rank_attn_d64

**Description:** Same as exp020 but embedding_dim=64, tables_per_head=64. RankAttention with d_head=16 (120 pairs).

**Parameters:** 42.0M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5456** @ 100k

---

## exp022_smooth_nalt1

**Description:** d=32, K=8, T=64, n_alternatives=1, smooth_mode=True. Tests smooth routing with minimal alternatives.

**Parameters:** 23.1M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5141** @ 100k

---

## exp023_smooth_nalt1_t128

**Description:** Same as exp022 but tables_per_head=128. d=32, K=8, n_alternatives=1, smooth_mode=True.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4371** @ 100k

---

## exp024_smooth_rank_norm

**Description:** Same as exp023 but with RankAttention (normalize_deltas=True) replacing SDPA.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4371** @ 100k

---

## exp025_smooth_rank_nonorm

**Description:** Same as exp023 but with RankAttention (normalize_deltas=False) replacing SDPA.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4568** @ 100k

---

## exp026_smooth_delta_reg

**Description:** Same as exp023 but with delta regularizer (delta_reg_weight=0.01) to force large anchor deltas.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4368** @ 100k

---

## exp027_smooth_output_scale_noise

**Description:** Same as exp023 but with output_scale_noise=0.5 (U[0.5, 1.5] per batch*head during training).

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4674** @ 100k

---

## exp028_smooth_osn_nalt3

**Description:** Same as exp027 but with n_alternatives=3.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4590** @ 100k

---

## exp029_smooth_input_scale_noise

**Description:** Same as exp023 but with input_scale_noise=0.5 (U[0.5, 1.5] applied to input per batch element).

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4439** @ 100k

---

## exp030_rank_input_noise

**Description:** RankAttention + input_scale_noise=0.5 on both LUTs and RankAttention, nalt=1, smooth_mode=True.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4614** @ 100k

---

## exp031_rank_nalt3_lut_noise

**Description:** RankAttention (no input noise) + LUT input_scale_noise=0.5, nalt=3, smooth_mode=True.

**Parameters:** 46.2M | **Steps:** 100k

**Results:** No summary (likely stopped early)

---

## exp032_smooth_anneal50k

**Description:** Same as exp023 but anneals smooth_mode=False at step 50000.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4956** @ 100k

---

## exp033_uncertainty_bias001

**Description:** Same as exp023 but uncertainty_bias=0.01. Smooth mode throughout.

**Results:** No summary

---

## exp034_uncertainty_bias01

**Description:** Same as exp023 but uncertainty_bias=0.1. Smooth mode throughout.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4723** @ 100k

---

## exp035_ub01_isn05

**Description:** Same as exp034 but input_scale_noise=0.5.

**Results:** No summary

---

## exp036_ub01_nalt3

**Description:** Same as exp034 but n_alternatives=3.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.4975** @ 100k

---

## exp037_rank_single_ffn

**Description:** RankAttention (temperature=0.1), smooth_mode=False, n_alternatives=1, single FFN.

**Results:** No summary

---

## exp038_rank_single_ffn_t1

**Description:** RankAttention (temperature=1.0), smooth_mode=False, n_alternatives=1, single FFN.

**Parameters:** 39.9M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5594** @ 100k

---

## exp039_rank_double_ffn_t1

**Description:** RankAttention (temperature=1.0), smooth_mode=False, n_alternatives=1, double FFN.

**Results:** No summary

---

## exp040_rank_smooth_lut_hard

**Description:** RankAttention (temperature=1.0), smooth_mode=False, n_alternatives=1, RankAttn smooth t=0.1 + LUT non-smooth double FFN.

**Results:** No summary

---

## exp041_rank_dqk_full

**Description:** RankAttention d_qk=embedding_dim=32 (496 pairs), d_v=d_head=8, smooth RankAttn t=0.1, LUT non-smooth, double FFN.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5082** @ 100k

---

## exp042_rank_dqk_half_hard

**Description:** RankAttention d_qk=d//2=16 (120 pairs), d_v=d_head=8, hard RankAttn t=1.0, LUT non-smooth, double FFN.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5071** @ 100k

---

## exp043_rank_dqk_half_hard_nalt3

**Description:** Same as exp042 but n_alternatives=3.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5359** @ 100k

---

## exp044_rank_dqk_half_hard_200k

**Description:** Same as exp042 but 200k steps.

**Parameters:** 46.2M | **Steps:** 200k

**Results:**
- best_val_loss: **1.4821** @ 200k

---

## exp045_rank_d64

**Description:** embedding_dim=64, RankAttention d_qk=d_head=16 (120 pairs), d_v=16, hard t=1.0, LUT non-smooth, tph=64, double FFN, 100k steps.

**Parameters:** 42.0M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5316** @ 100k

---

## exp046_rank_d16

**Description:** embedding_dim=16, n_heads=4, RankAttention d_qk=d_v=16 (120 pairs), hard t=1.0, LUT non-smooth. q/k/v tph=64, out/ffn tph=256, unembedder tph=128. ~46M params.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5990** @ 100k

---

## exp047_rank_d16_8l

**Description:** Same as exp046 but 8 layers.

**Parameters:** 46.2M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5887** @ 100k

---

## exp048_mini

**Description:** Minimalistic: embedding_dim=32, n_heads=4, d_qk=d_v=8, single FFN, nap=8, tph=16, hard RankAttn t=1.0, 50k steps.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.9421** @ 50k

---

## exp049_mini_double_ffn

**Description:** Minimalistic + double FFN. embedding_dim=32, n_heads=4, nap=8, tph=16, 50k steps.

**Parameters:** 5.8M | **Steps:** 50k

**Results:**
- best_val_loss: **1.9412** @ 50k

---

## exp050_mini_recursive_ffn

**Description:** Minimalistic + RecursiveMultiHeadLut (n_iters=2) for out_proj and ffn. Same params as exp048.

**Results:** No summary

---

## exp051_mini_recursive_no_sharing

**Description:** Same as exp050 but always separate refine_lut — no weight reuse. All LUTs get independent refinement weights.

**Parameters:** 8.9M | **Steps:** 50k

**Results:**
- best_val_loss: **1.8813** @ 50k

---

## exp052_mini_tph32

**Description:** Same as exp048 but tph=32.

**Parameters:** 10.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.7685** @ 50k

---

## exp053_mini_tph64_nap6

**Description:** tph=64, nap=6. Same net table entries as exp048.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.7593** @ 50k

---

## exp054_mini_tph64_nap5

**Description:** Super tiny: tph=64, nap=5. ~2.5M params.

**Parameters:** 2.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.8294** @ 50k

---

## exp055_mini_tph64_nap5_posemb

**Description:** Same as exp054 but pos embeddings initialized uniform(-0.1, 0.1).

**Results:** No summary

---

## exp056_mini_tph32_nap6_posemb

**Description:** Same as exp055 but tph=32, nap=6.

**Results:** No summary

---

## exp057_mini_tph128_nap5

**Description:** tph=128, nap=5. Same table entries as exp048.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.7289** @ 50k

---

## exp058_rank_projection

**Description:** RankProjection instead of LUTs for all projections. d=32, h=4, d_head=8, M=496 pairs, smooth_mode=False.

**Parameters:** 0.6M | **Steps:** 50k

**Results:**
- best_val_loss: **1.9526** @ 50k

---

## exp059_mini_tph256_nap4

**Description:** tph=256, nap=4. Same budget as exp048.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.7060** @ 50k

---

## exp060_mini_tph512_nap3

**Description:** tph=512, nap=3. Same budget as exp048.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.7058** @ 50k

---

## exp061_mini_tph1024_nap2

**Description:** tph=1024, nap=2. Same budget as exp048.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.7185** @ 50k

---

## exp062_mini_tph32_nap7

**Description:** tph=32, nap=7. Same budget as exp048.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.8271** @ 50k

---

## exp063_mini_tph2048_nap1

**Description:** tph=496, nap=1, FULL_COVERAGE: one table per unique pair.

**Parameters:** 1.2M | **Steps:** 50k

**Results:**
- best_val_loss: **1.9652** @ 50k

---

## exp064_mini_tph256_nap4_connected

**Description:** Same as exp059 but connected_anchors_mode=True.

**Results:** No summary

---

## exp065_mini_tph256_nap4_disconnected

**Description:** Same as exp059 but DISCONNECTED anchor sampling.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6866** @ 50k

---

## exp066_mini_tph256_nap4_fullcoverage

**Description:** Same as exp059 but FULL_COVERAGE anchor sampling.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6839** @ 50k

---

## exp067_mini_tph256_nap4_disconnected_fullcoverage

**Description:** Same as exp059 but DISCONNECTED_FULL_COVERAGE anchor sampling.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6901** @ 50k

---

## exp068_mini_tph124_nap4_fullcoverage

**Description:** FULL_COVERAGE, nap=4, tph=124 — minimum tables for complete pair coverage.

**Parameters:** 2.4M | **Steps:** 50k

**Results:**
- best_val_loss: **1.8261** @ 50k

---

## exp069_mini_tph512_nap4_fullcoverage

**Description:** FULL_COVERAGE, nap=4, tph=512 — each pair appears ~4x on average.

**Parameters:** 9.8M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6228** @ 50k

---

## exp070_mini_tph256_nap4_fc_posperm

**Description:** Same as exp066 but with PositionalPermutation applied to q and k before attention.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.9411** @ 50k

---

## exp071_mini_tph256_nap4_fc_nopos

**Description:** Same as exp066 but with NO positional encoding.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.9286** @ 50k

---

## exp072_mini_tph256_nap4_fc_learnedperm

**Description:** Same as exp070 but with LearnedSoftPermutations instead of fixed random permutations.

**Parameters:** 5.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.8475** @ 50k

---

## exp073_mini_tph256_nap4_fc_sparse8

**Description:** Fork of exp066 with sparse_output_dim=8 applied to out_proj, ffn, and unembedder.

**Parameters:** 2.8M | **Steps:** 50k

**Results:**
- best_val_loss: **2.0061** @ 50k

---

## exp074_mini_tph256_nap4_fc_dqk16

**Description:** Fork of exp066 with d_qk=16 instead of d_head=8.

**Parameters:** 6.6M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6390** @ 50k

---

## exp075_mini_embed24_pos8_dqk16

**Description:** embedding_dim=24, positional_dim=8 (separate, concatenated inside layers), d_qk=16, d_v=6. tph=256, nap=4, FULL_COVERAGE.

**Parameters:** 6.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6281** @ 50k

---

## exp076_mini_embed24_pos8_dqk16_tph128

**Description:** Fork of exp075 with mixed tph: q/k/v tph=256, out_proj/ffn/unembedder tph=128.

**Parameters:** 3.4M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6866** @ 50k

---

## exp077_mini_embed24_pos8_dqk16_tph512_sparse12

**Description:** Fork of exp075 with tph=512, nap=4, sparse_output_dim=12.

**Results:** No summary

---

## exp078_mini_fp_rank1

**Description:** Fork of exp075 with FactoredProjection rank=1 everywhere.

**Results:** No summary

---

## exp079_mini_fp_rank4

**Description:** Fork of exp075 with FactoredProjection rank=4 everywhere.

**Results:** No summary

---

## exp080_mini_sinusoidal

**Description:** Fork of exp075 with SinusoidalProjection everywhere.

**Results:** No summary

---

## exp081_mini_double_ffn32

**Description:** Fork of exp075 with 2-layer FFN: ffn1(24→32) + ffn2(32→24).

**Results:** No summary

---

## exp082_mini_pairvoting

**Description:** Fork of exp075 replacing all MultiHeadLut with PairVoting (learnable output vector per input pair).

**Results:** No summary

---

## exp083_exp075_100k

**Description:** exp075 run for 100k steps.

**Parameters:** 6.0M | **Steps:** 100k

**Results:**
- best_val_loss: **1.6281** @ 100k

**Note:** Same result as 50k — likely a summary.json copy issue.

---

## exp084_dv16_outproj512

**Description:** Fork of exp075. d_v increased from 6 to 16. out_proj uses tph=512.

**Parameters:** 7.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.5845** @ 50k

---

## exp085_dv16_outproj512_lr1e3

**Description:** Fork of exp084 with lr=0.001 for all params.

**Results:** No summary

---

## exp086_dv16_normalized_attn

**Description:** Fork of exp084. Replaces RankAttention with standard SDPA, Q and K L2-normalized.

**Parameters:** 7.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.5845** @ 50k

---

## exp087_dv16_centered_attn

**Description:** Fork of exp086. Q and K mean-centered before standard SDPA.

**Parameters:** 7.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.5845** @ 50k

---

## exp088_dv16_rank_smooth

**Description:** Fork of exp084. RankAttention with smooth_mode=True.

**Parameters:** 7.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6259** @ 50k

---

## exp089_d32_pos16_wider

**Description:** embedding_dim=32, positional_dim=16. q/k/v tph=384, out_proj tph=512, ffn/unembedder tph=256. d_qk=16, d_v=16, nap=4, FULL_COVERAGE.

**Parameters:** 10.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.5582** @ 50k

---

## exp090_d32_pos16_nalt4

**Description:** Fork of exp089. n_alternatives=4.

**Results:** No summary

---

## exp091_d32_pos16_no_ffn

**Description:** Fork of exp089. FFN removed — attention-only transformer.

**Parameters:** 9.7M | **Steps:** 50k

**Results:**
- best_val_loss: **1.5626** @ 50k

---

## exp092_batch16

**Description:** Fork of exp089. batch_size=16.

**Parameters:** 10.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6961** @ 50k

---

## exp093_batch16_t05

**Description:** Fork of exp092. RankAttention temperature=0.5.

**Parameters:** 10.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6853** @ 50k

---

## exp094_batch16_t01

**Description:** Fork of exp093. RankAttention temperature=0.1.

**Parameters:** 10.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6964** @ 50k

---

## exp095_no_ffn_tph512

**Description:** Fork of exp091. batch_size=16, tph=512 everywhere, temperature=0.5.

**Parameters:** 13.1M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6559** @ 50k

---

## exp096_no_ffn_tph512_nap3

**Description:** Fork of exp095. nap=3, temperature=0.5. No FFN, tph=512, batch=16.

**Results:** No summary

---

## exp097_no_ffn_tph512_nap6

**Description:** Fork of exp095. nap=6, temperature=0.5. No FFN, tph=512, batch=16.

**Parameters:** 52.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6302** @ 50k

---

## exp098_no_ffn_tph512_nap6_b32_100k

**Description:** Fork of exp097. batch_size=32, 100k steps. Saves checkpoint.pt at end.

**Parameters:** 52.5M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5375** @ 100k

---

## exp099_nap6_normalize_weights

**Description:** Fork of exp098. normalise_weights=True for attention LUTs. Stopped early — much worse (~+0.12 loss).

**Results:** Stopped early, no summary

---

## exp100_nap6_linear_unemb

**Description:** Fork of exp097. Unembedder replaced with nn.Linear(d, vocab_size). nap=6, tph=512, temperature=0.5, no FFN, batch=16, 50k steps.

**Parameters:** 44.1M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6195** @ 50k

---

## exp101_nap3_linear_unemb

**Description:** Fork of exp100. nap=3. Linear unembedder.

**Parameters:** 5.5M | **Steps:** 50k

**Results:**
- best_val_loss: **1.7202** @ 50k

---

## exp102_nap4_linear_unemb

**Description:** Fork of exp100. nap=4. Linear unembedder.

**Parameters:** 11.0M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6872** @ 50k

---

## exp103_rank_proj2

**Description:** LUTs replaced by RankProjection2 (rank-project → linear → inverse rank project). smooth_mode=False. Linear unembedder. batch=16, 50k steps.

**Parameters:** 47.0M | **Steps:** 50k

**Results:**
- best_val_loss: **2.1671** @ 50k

**Interpretation:** Much worse than LUTs — dead end. RankProjection2 removed from codebase.

---

## exp104_residual_stream

**Description:** Fork of exp100. Separate residual stream (res_dim=128) updated via up_proj(Linear 32→128) after each LUT block. Ranking stream is raw LUT chain with no skip connections. res_stream starts at zero.

**Parameters:** 44.1M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6456** @ 50k

---

## exp105_residual_stream_lr

**Description:** Fork of exp104. up_proj layers use lr=0.001 (same as unembedder, was 0.0001).

**Parameters:** 44.1M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6495** @ 50k

---

## exp106_residual_stream_100k

**Description:** Fork of exp105. 100k steps.

**Parameters:** 44.1M | **Steps:** 100k

**Results:**
- best_val_loss: **1.5919** @ 100k

---

## exp107_residual_mlp

**Description:** Fork of exp104. up_proj replaced by 2-layer MLP (32→128→ReLU→64) per layer. res_dim=64.

**Parameters:** 44.1M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6581** @ 50k

---

## exp108_ffn

**Description:** Fork of exp100. FFN added after attention: Linear(32,128)→ReLU→Linear(128,32) with residual. nap=6, tph=512, temperature=0.5, batch=16, 50k steps.

**Parameters:** 44.1M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6047** @ 50k

**Interpretation:** FFN helps significantly — new best at 50k over exp100 baseline.

---

## exp109_dqk32

**Description:** Fork of exp108. d_qk=32 (was 16). d_v=16 unchanged.

**Parameters:** 69.3M | **Steps:** 50k

**Results:**
- best_val_loss: **1.6163** @ 50k

**Interpretation:** Larger d_qk hurts despite more params. exp108 remains better.

---

## exp110_tph1024

**Description:** Fork of exp108. tables_per_head=1024 (was 512). FFN unchanged.

**Parameters:** 88.1M | **Steps:** 50k

**Results:**
- best_val_loss: **1.5845** @ 50k

---

## exp111_tph2048

**Description:** Fork of exp110. tables_per_head=2048, batch_size=32.

**Parameters:** 176.2M | **Steps:** 50k

**Results:**
- best_val_loss: **1.5021** @ 50k

---

## exp112_tph4096

**Description:** Fork of exp111. tables_per_head=4096, batch_size=32.

**Parameters:** 352.4M | **Steps:** 50k

**Results:**
- best_val_loss: **1.4834** @ 50k

**Interpretation:** Diminishing returns — 2×params gave only −0.019 vs −0.082 from tph=1024→2048. Vanilla baseline (exp003) is 1.3545 at 4.87M params.

---

## Batch 32 — Fused CUDA Kernels & Architecture Exploration (exp165–172)

### exp165_fused_kernel

**Description:** Same as exp164 but with fused CUDA kernel for LUTAttentionV2. 1H, tph=512, nap=4, 12 layers.

**Parameters:** 3.19M | **Steps:** 50k | **BS:** 32

**Results:**
- best_val_loss: **1.5317** @ 48k

---

### exp166_tph512_layers4

**Description:** tph=512, nap=4, 4 layers.

**Parameters:** 1.09M | **Steps:** 50k | **BS:** 32

**Results:**
- best_val_loss: **1.5730** @ 49k

---

### exp167_tph256_nap5

**Description:** tph=256, nap=5, 6 layers.

**Parameters:** 1.62M | **Steps:** 50k | **BS:** 32

**Results:**
- best_val_loss: **1.5492** @ 49k

---

### exp168_tph256_nap5_bs128

**Description:** Same as exp167 but batch_size=128.

**Parameters:** 1.62M | **Steps:** 50k | **BS:** 128

**Results:**
- best_val_loss: **1.5048**

---

### exp169_meta_attention

**Description:** Chained layers without residuals, meta-attention over layer outputs with learned layer positional embedding.

**Parameters:** 1.88M | **Steps:** 50k | **BS:** 32

**Results:**
- best_val_loss: **1.5894**

**Interpretation:** Meta-attention over layer outputs didn't improve over standard residual connections.

---

### exp170_4head_outproj

**Description:** 4-headed LUT attention + 1-headed out_proj LUT with per-head LayerNorm + post-outproj LayerNorm, residual connection.

**Parameters:** 3.19M | **Steps:** 50k | **BS:** 32

**Results:**
- best_val_loss: **1.5504** @ 48k

---

### exp171_e64_tph128

**Description:** 4-headed attn + out_proj, embedding_dim=64, tph=256, nap=4, 100K steps.

**Parameters:** 3.25M | **Steps:** 100k | **BS:** 32

**Results:**
- best_val_loss: **1.5266**

---

### exp172_excl_sets

**Description:** 4H attn(tph=128,nap=4) + out_proj(tph=512,nap=4), exclusion_sets on attention (no within-group pairs), quadratic self-excitement.

**Parameters:** 2.01M | **Steps:** 50k | **BS:** 32

**Results:**
- best_val_loss: **1.5742**

**Interpretation:** Exclusion sets didn't help. Quadratic self-excitement didn't improve over linear.

---

## Batch 33 — LUTAttentionV3 Softmax Architecture (exp173–186)

V3 architecture: LUT-based score generation → softmax → LUT values → matmul → LUT out_proj.
Classic transformer attention pattern but with all projections replaced by LUTs.

### exp173_v3_softmax

**Description:** V3 softmax, score(nap=6,tph=256), v(nap=4,tph=256), outproj(nap=5,tph=768), d_v=16, H=4, linear unembedder.

**Parameters:** 3.26M | **Steps:** 50k | **BS:** 32

**Results:**
- best_val_loss: **1.5491**

---

### exp177_v3_nap6_tph128_bs128

**Description:** All nap=6, attn tph=128, v tph=128, outproj tph=512, bs128.

**Parameters:** 9.65M | **Steps:** 50k | **BS:** 128

**Results:**
- best_val_loss: **1.4455**

**Interpretation:** Best result at time — nap=6 everywhere with large outproj.

---

### exp181_v3_v256_op768nap5

**Description:** V3 softmax, v(nap4,tph256), outproj(nap5,tph768), bs128.

**Parameters:** 6.51M | **Steps:** 50k | **BS:** 128

**Results:**
- best_val_loss: **1.4584**

---

### exp182_v3_sdiff_op1536

**Description:** V3 softmax with SignedDiff pairs processor (cat([x_i-x_j, rpe])), op_tph=1536.

**Parameters:** 5.72M | **Steps:** 50k | **BS:** 128

**Results:**
- best_val_loss: **1.4657**

---

### exp183_v3_sdiff_op1536_100k

**Description:** Same as exp182, 100K steps.

**Parameters:** 5.72M | **Steps:** 100k | **BS:** 128

**Results:**
- best_val_loss: **1.4478**

---

### exp184_v3_v256_op768nap5_100k

**Description:** Same as exp181, 100K steps. Concat pairs processor (default).

**Parameters:** 6.51M | **Steps:** 100k | **BS:** 128

**Results:**
- best_val_loss: **1.4411**

**Interpretation:** Best V3 result. Concat processor slightly better than SignedDiff at matched steps. Gap to vanilla baseline (1.3559) is ~0.085.

---

### exp185_v3_high_nap

**Description:** High nap everywhere: score(nap8,tph128), v(nap6,tph128), op(nap6,tph256), dropout=0.3, constant lr, bs64.

**Parameters:** 7.10M | **Steps:** 50k | **BS:** 64

**Results:**
- best_val_loss: **1.5740**

**Interpretation:** Constant lr + dropout hurt. But bottleneck sweep showed nap helps more than tph at 8K steps.

---

### exp186_v3_high_nap_100k

**Description:** Same high nap config, warmup+cosine, no dropout, bs128, 100K.

**Parameters:** 7.10M | **Steps:** 100k | **BS:** 128

**Results:**
- best_val_loss: **1.4576**

---

## Batch 34 — Sweeps & Analysis

### Parameter Sweeps

**sweep_v3** (5K steps, bs32): Found out_proj is the most important LUT. More budget → better loss. 8 heads surprisingly strong.

**sweep_v3_bottleneck** (8K steps, bs32): Scaling one component at a time from exp184 base.
Key finding: **nap > tph** — increasing entries per table consistently beats more tables at same params.
Value LUT is the biggest bottleneck. Score attention barely matters beyond tph=128.

### Pairs Processor Sweep

Tested 5 pair aggregation methods for attention scores:
1. **SignedDiff** (x_i-x_j+rpe): Best overall (1.7288 on op1536)
2. AbsDiff (|x_i-x_j|+rpe): Good with large capacity (1.7219 on op2048)
3. Concat (default): Solid baseline (1.7423)
4. LinearComb (learned): Similar to concat
5. Sum (x_i+x_j+rpe): Worst — too lossy

**Conclusion:** SignedDiff wins at 5K sweeps but concat is slightly better at longer training (exp184 vs exp183 at matched steps).

### tph Formula

Derived rule of thumb: `tph ≈ input_dim × (input_dim - 1) / nap` for ~2× pair coverage. Works well for nap=4 but nap itself matters more than tph for expressiveness.

---

## Batch 35 — LayerNorm & Optimizer Analysis (exp187–192)

### LayerNorm Ablation (25K steps, bs32)

| Exp | Config | Val Loss |
|-----|--------|----------|
| exp187 | No attn_norm, post-outproj norm only | **1.5480** |
| exp188 | Both norms (attn_norm + post-outproj) | 1.5566 |
| exp189 | No norms at all | 1.5767 |

**Key findings from analysis:**
- Without attn_norm: value grad/weight ratio 21× healthier, gradient imbalance reduced from 64× to 3×
- Without any norms: 28.6% dead weights, activation collapse (output std grows 1→13)
- Post-outproj norm alone is the sweet spot: stable residual stream + healthy gradients

### Optimizer Comparison

**Optimisation sweep** (10K steps from exp184 checkpoint, lr=0.0001):

| Config | Improvement |
|--------|-------------|
| SGD+momentum nalt1 | -0.0021 (best) |
| SGD nalt1 | -0.0011 |
| Adam nalt1 | -0.0011 |
| n_alternatives=3 | consistently worse |

**From-scratch comparison (25K, bs32):**
- exp191 (no attn_norm + SGD_mom): **1.5455** (best)
- exp187 (no attn_norm + Adam): 1.5480
- exp190 (both norms + SGD_mom): 1.5603

### exp192_adam_cosine_restarts

**Description:** Adam with CosineAnnealingWarmRestarts (T0=10K, Tmult=2), bs64, no attn_norm, 100K.

**Parameters:** 6.5M | **Steps:** 100k | **BS:** 64

**Results:**
- best_val_loss: **1.4768**

**Interpretation:** Cosine restarts didn't beat standard cosine decay (exp184: 1.4411).

### Vanilla Baseline Comparison

**exp002_retrain:** Vanilla transformer (d_model=256, 4 heads, 6 layers, FFN 4×), 100K steps.
- **best_val_loss: 1.3556** at 4.87M params
- Attention entropy much lower (0.06–0.49) vs LUT (0.68–1.17) — vanilla achieves sharper attention patterns

**Analysis of exp184 (best LUT model):**
- Value LUT gradients 25× smaller than score gradients (raw mean) — but after fixing the ratio metric (log scale, excluding near-zero weights), all components have balanced gradients around 10⁻⁴
- 0.3% dead weights — minimal
- The gap to vanilla (0.085) appears to be a capacity/expressiveness issue rather than optimization

---

## Batch 36 — HyperLUT (exp193–195)

HyperLUT: replaces MultiHeadLut with pairwise comparisons → MLP (Linear → GELU → Linear).
Key advantage: MLP sees all M comparisons at once (vs LUT where each table sees only nap pairs independently). Avoids 2^nap exponential table size.

### exp193_hyper_lut

**Description:** HyperLUT, hidden_dim=64 uniform, sigmoid soft function, temp=0.1.

**Parameters:** 2.34M | **Steps:** 25k | **BS:** 32

**Results:**
- best_val_loss: **1.6574**

---

### exp194_hyper_lut_wide_op

**Description:** HyperLUT, per-component hidden: score=32, value=64, outproj=256, rational soft function, temp=0.1.

**Parameters:** 4.32M | **Steps:** 25k | **BS:** 32

**Results:**
- best_val_loss: **1.6111**

---

### exp195_hyper_lut_big

**Description:** HyperLUT: score(1024 pairs, hid=32), value(all 496 pairs, hid=64), outproj(all 2016 pairs, hid=128), rational, 50K steps, bs64.

**Parameters:** 3.15M | **Steps:** 50k | **BS:** 64

**Results:** Running.

---
