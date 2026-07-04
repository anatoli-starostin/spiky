# CIFAR-10 LUT Experiments Journal

**Dataset**: CIFAR-10 (10 classes, 32×32)
**Augmentation**: RandomCrop(32, padding=4) + RandomHorizontalFlip
**Training**: Adam lr=1e-3, CosineAnnealingLR, batch 256
**Device**: cuda:0 (NVIDIA H100 80GB)

---

## Experiment Index

### Phase 1 (n_alternatives=1, MaxPool)

| # | Name | Params | Best Val Acc | Ep |
|---|------|--------|-------------|-----|
| 01 | exp01_alexnet_baseline | 6.46M | 89.36% | 20 |
| 02 | exp02_lut_fe_small_mlp | 300K | 60.04% | 20 |
| 03 | exp03_lut_fe_deep_mlp | 1.55M | 54.20% | 20 |
| 04 | exp04_alexnet_fe_lut_cl | 2.55M | 54.35% | 20 |
| 05 | exp05_lut_fe_lut_cl_small | 175K | 35.71% | 20 |
| 06 | exp06_lut_fe_lut_cl_medium | 577K | 29.35% | 18 |
| 07 | exp07_lut_fe_lut_cl_large | 1.58M | 25.98% | 18 |
| 08 | exp08_lut_fe_bn_mlp | 300K | 59.46% | 16 |
| 09 | exp09_alexnet_fe_bn_lut_cl | 2.56M | 45.72% | 20 |
| 10 | exp10_lut_fe_bn_lut_cl | 177K | 28.81% | 19 |
| 11 | exp11_lut_fe_high_cap_mlp | 544K | **62.22%** | 19 |
| 12 | exp12_lut_fe_stride2_mlp | 300K | 51.41% | 20 |

### Phase 2 (n_alternatives=3, stride-2, no BN)

| # | Name | Params | Best Val Acc | Ep |
|---|------|--------|-------------|-----|
| 21 | exp21_ref_50ep | 541K | 58.08% | 41 |
| 22 | exp22_ref_4x4_spatial | 541K | 55.98% | 17 |
| 23 | exp23_ref_wide_cl | 803K | 54.34% | 19 |
| 24 | exp24_ref_nalts5 | 541K | 53.72% | 18 |
| 25 | exp25_ref_4x4_50ep | 541K | 59.55% | 39 |
| 26 | exp26_ref_wide_fe | 803K | 54.71% | 19 |
| 27 | exp27_ref_4stage | 672K | 52.84% | 17 |
| 28 | exp28_ref_wide_fe_4x4 | 803K | 56.91% | 19 |
| 29 | exp29_wide_fe_4x4_50ep | 803K | 60.98% | 44 |
| 30 | exp30_wide_fe_4x4_wide_cl | 1.07M | 56.85% | 18 |
| 31 | exp31_wide_fe_4x4_nap10_cl | 1.59M | 55.97% | 20 |
| 32 | exp32_wide_fe_4x4_wide_nap10_cl | 2.64M | 54.67% | 18 |
| 33 | exp33_wide_fe_4x4_100ep | 803K | 64.21% | 91 |
| 34 | exp34_wide_fe_4x4_mlp | 806K | **64.36%** | 20 |
| 35 | exp35_wide_fe_4x4_c3nap10 | 1.59M | 55.33% | 19 |
| 36 | exp36_wide_fe_4x4_c3tph32 | 1.07M | 57.57% | 19 |
| 37 | exp37_wide_fe_4x4_mlp_50ep | 806K | **69.21%** | 48 |
| 38 | exp38_c3tph32_mlp | 1.07M | 67.29% | 17 |
| 39 | exp39_wide_fe_4x4_mlp512 | 1.07M | 65.87% | 19 |
| 40 | exp40_c3tph32_mlp512 | 1.33M | 68.04% | 20 |
| 41 | exp41_c3tph32_mlp512_50ep | 1.33M | **71.76%** | 45 |
| 42 | exp42_c3tph32_mlp_50ep | 1.07M | 70.27% | 47 |
| 43 | exp43_c2c3tph32_mlp | 1.33M | 69.83% | 20 |
| 44 | exp44_c2c3tph32_mlp512 | 1.60M | 70.24% | 19 |
| 45 | exp45_c2c3tph32_mlp_50ep | 1.33M | **72.49%** | 47 |
| 46 | exp46_c2c3tph32_mlp512_50ep | 1.60M | **73.07%** | 46 |
| 47 | exp47_alltph_mlp | 1.35M | 72.03% | 19 |
| 48 | exp48_alltph_mlp512 | 1.61M | 71.94% | 16 |
| 49 | exp49_alltph_mlp_50ep | 1.35M | **74.30%** | 33 |
| 50 | exp50_alltph_mlp512_50ep | 1.61M | **74.81%** | 46 |
| 51 | exp51_c1tph32_mlp | 1.38M | 74.08% | 19 |
| 52 | exp52_c1tph32_mlp512 | 1.65M | 74.09% | 17 |
| **— New arch: AlexNet-topology, padding, fixed classifier (512→1024→10) —** | | | | |
| ref | reference (nap=10) | ~10M | 64.37% | 20 |
| 53 | exp53_nap8 | 2.91M | 72.37% | 18 |
| 54 | exp54_nap6 | 1.14M | **76.09%** | 17 |
| 55 | exp55_4stage_ref | 7.89M | 68.92% | 19 |
| 56 | exp56_4stage_nap8 | 2.39M | 74.13% | 19 |
| 57 | exp57_nap6_50ep | 1.14M | **79.58%** | 44 |
| 58 | exp58_nap4 | 693K | 74.31% | 19 |
| 59 | exp59_nap6_tph2x | 1.75M | **80.09%** | 19 |
| 60 | exp60_4stage_nap6 | 1.01M | 75.61% | 17 |
| 61 | exp61_nap6_tph2x_50ep | 1.75M | **81.98%** | 45 |
| 62 | exp62_nap6_tph4x | 2.96M | **83.29%** | 19 |
| 63 | exp63_nap6_tph2x_wide | 2.62M | 80.76% | 20 |
| 64 | exp64_nap8_tph2x | 5.29M | 77.70% | 18 |
| 65 | exp65_nheads1 | 2.96M | 81.49% | 20 |
| 66 | exp66_nheads8 | 2.96M | 83.23% | 19 |
| 67 | exp67_wide_nheads4 | 5.91M | 83.59% | 17 |
| 68 | exp68_wide_nheads8 | 5.91M | **84.23%** | 19 |
| 69 | exp69_wide_nheads8_50ep | 5.91M | 84.92% | 48 |
| 70 | exp70_wide_nheads8_tph2x | 10.76M | **85.32%** | 20 |
| 71 | exp71_wide4x_nheads8 | 11.81M | 84.27% | 20 |
| 72 | exp72_wide4x_nheads16 | 11.81M | 84.47% | 19 |
| 73 | exp73_nap6_tph4x_50ep | 2.96M | 84.00% | 48 |
| 74 | exp74_wide_nheads8_tph_half | 3.49M | 81.48% | 20 |
| 75 | exp75_nap5_all | 1.78M | **83.11%** | 20 |
| 76 | exp76_nap5_56 | 2.57M | 82.98% | 19 |
| 77 | exp77_nap5_50ep | 1.78M | **84.40%** | 49 |
| 78 | exp78_nap5_connected | 1.78M | 82.56% | 20 |
| 79 | exp79_nap6_connected | 2.96M | 82.63% | 18 |
| 80 | exp80_wide_nheads8_connected | 5.91M | 84.43% | 19 |
| 81 | exp81_nap5_tph2x | 3.03M | 84.19% | 20 |
| 82 | exp82_nap5_wide_nheads8 | 3.55M | 84.06% | 20 |
| 83 | exp83_nap4 | 1.16M | 80.88% | 19 |
| 84 | exp84_nap5_tph2x_connected | 3.03M | **84.50%** | 20 |
| 85 | exp85_nap5_tph2x_connected_50ep | 3.03M | **85.92%** | 42 |
| 86 | exp86_nap5_wide_nheads8_connected | 3.55M | 84.21% | 19 |
| 87 | exp87_nap5_tph_half | 1.16M | 80.02% | 19 |
| 88 | exp88_nap5_connected_50ep | 1.78M | 84.51% | 44 |
| 89 | exp89_nap5_wide_nheads8_connected_50ep | 3.55M | 85.62% | 44 |
| 90 | exp90_nap5_tph2x_50ep | 3.03M | 85.79% | 45 |
| 91 | exp91_nap5_wide_nheads8_tph_half_connected | 2.31M | 81.69% | 17 |
| 92 | exp92_nap5_wide_nheads8_tph_half | 2.31M | 81.84% | 19 |
| 93 | exp93_nap5_wide_nheads8_tph_half_50ep | 2.31M | 83.62% | 46 |
| 94 | exp94_nap5_wide_nheads8_tph_half_connected_50ep | 2.31M | 83.63% | 43 |
| 95 | exp95_nap5_wide_nheads8_connected_false_50ep | 3.55M | 85.47% | 48 |
| 96 | exp96_nap5_narrow_100ep | 1.78M | **85.35%** | 96 |
| 97 | exp97_nap5_4stage | 1.27M | 81.94% | 17 |
| 98 | exp98_nap5_6stage | ~1.78M | 81.72% | 20 |
| 99 | exp99_nap5_k3 | 1.78M | 82.64% | 20 |
| 100 | exp100_nap5_k5mid | 1.78M | 82.12% | 20 |
| 101 | exp101_nap5_extra_16x16 | 1.85M | 82.42% | 20 |
| 102 | exp102_nap5_extra_32x32 | 1.72M | **83.70%** | 20 |
| 103 | exp103_nap5_aggressive_stride | 1.78M | 76.60% | 17 |
| 104 | exp104_nap5_late_downsample | 1.78M | **83.61%** | 17 |
| 105 | exp105_nap5_extra_32x32_50ep | 1.72M | **85.51%** | 46 |
| 106 | exp106_nap5_late_downsample_50ep | 1.78M | **86.05%** | 46 |
| 107 | exp107_nap5_2extra_32x32 | 2.24M | 84.24% | 18 |
| 108 | exp108_nap5_extra_32x32_and_16x16 | 2.37M | 83.93% | 20 |
| 109 | exp109_nap5_extra_32x32_k5 | 1.72M | 83.16% | 19 |
| 110 | exp110_nap5_late_downsample_k5 | 1.78M | 82.99% | 19 |
| 111 | exp111_nap5_late_downsample_extra_8x8 | 1.85M | 83.13% | 20 |
| 112 | exp112_nap5_k3_50ep | 1.78M | 84.50% | 50 |
| **— Batch 27: tph scaling with memory-constrained stride-1 layers (tph≤256 revised) —** | | | | |
| 120 | exp120_wide_nap5_tph2x | ~6.04M | 85.22% | 19 |
| 121 | exp121_narrow_late_ds_deep_tph | ~3.03M | 84.72% | 20 |
| 122 | exp122_narrow_std_tph2x | ~3.03M | 83.83% | 19 |
| 123 | exp123_wide_tph_max | ~8M | 85.09% | 20 |
| **— Batch 28: combining late-ds + wide + depth (tph≤256) —** | | | | |
| 124 | exp124_late_ds_wide_expand | ~?M | 85.66% | 20 |
| 125 | exp125_wide_deep_8x8 | ~?M | 85.07% | 20 |
| 126 | exp126_late_ds_deep_8x8 | ~?M | 85.36% | 19 |
| 127 | exp127_late_ds_wide_deep | ~3.41M | **85.91%** | 18 |

---

## Experiment Entries

<!-- entries are appended below as experiments complete -->

---

### EXP01 — Classic AlexNet Baseline

**Model**: 5×Conv+BN+ReLU (3→64→192→384→256→256), 3×MaxPool (32→16→8→4), MLP 4096→1024→10
**Parameters**: 6,461,514
**Optimizer**: Adam lr=1e-3, CosineAnnealingLR (T_max=20)

| Metric | Value |
|--------|-------|
| Best val acc | **89.36%** |
| Best epoch | 20 (still climbing) |
| Final train acc | 95.59% |
| Final val loss | 0.3368 |
| Elapsed | ~49 s |

**Plots**: [loss_curves.png](exp01_alexnet_baseline/loss_curves.png) · [accuracy_curves.png](exp01_alexnet_baseline/accuracy_curves.png)

**Interpretation**: Strong baseline — AlexNet converges steadily, reaching 89.4% val accuracy by epoch 20 with the curve still rising. Train/val gap is ~6 pp, indicating moderate overfitting. This sets the ceiling for LUT comparisons.

---

### EXP02 — Small LUT Feature Extractor + MLP

**Model**: LUT-FE 3-stage (3→32→64→64, nap=[4,6,6], tph=4) + MaxPool×3 → MLP(1024→256→10)
**Parameters**: 300,298
**Optimizer**: Adam lr=1e-3, CosineAnnealingLR

| Metric | Value |
|--------|-------|
| Best val acc | **60.04%** |
| Best epoch | 20 (still climbing) |
| Final train acc | 56.63% |
| Final val loss | 1.1126 |
| Elapsed | ~48 s |

**Plots**: [loss_curves.png](exp02_lut_fe_small_mlp/loss_curves.png) · [accuracy_curves.png](exp02_lut_fe_small_mlp/accuracy_curves.png)

**Interpretation**: The compact 3-stage LUT-FE reaches 60% — a significant gap vs AlexNet (89.4%) but with only 300K params (21× fewer). The curve is still slowly climbing at epoch 20 with essentially no overfit (train ≈ val), suggesting the model is under-capacity and would benefit from more depth or wider channels. Serves as the LUT-FE floor.

---

### EXP03 — Deep LUT Feature Extractor + MLP

**Model**: LUT-FE 5-stage (3→64→128→192→128→128, nap=[4,6,8,8,8], tph=4) + MaxPool×3 → MLP(2048→512→10)
**Parameters**: 1,550,858
**Optimizer**: Adam lr=1e-3, CosineAnnealingLR

| Metric | Value |
|--------|-------|
| Best val acc | **54.20%** |
| Best epoch | 20 (still climbing) |
| Final train acc | 51.90% |
| Final val loss | 1.2676 |
| Elapsed | ~74 s |

**Plots**: [loss_curves.png](exp03_lut_fe_deep_mlp/loss_curves.png) · [accuracy_curves.png](exp03_lut_fe_deep_mlp/accuracy_curves.png)

**Interpretation**: Surprisingly, the deeper/wider 5-stage LUT-FE *underperforms* the 3-stage version (54.2% vs 60.0%). The model starts from a lower initial accuracy and converges more slowly. Two plausible causes: (1) The deeper LUT stack is harder to optimize in 20 epochs — the loss landscape has more local optima; (2) the larger stage-3/4/5 anchor pair spaces (nap=8, 2^8=256 entries) need more iterations to fill with meaningful gradients. No overfitting (train ≈ val throughout), confirming this is a training difficulty issue, not over-capacity. A stronger learning rate warmup or more epochs would likely close the gap.

---

### EXP04 — Classic AlexNet Feature Extractor + LUT Classifier

**Model**: Classic AlexNet-FE (same as EXP01) + LUT-CL 2-layer: LUT(4096→256, nap=8, tph=4) + LUT(256→128, nap=6, tph=4) + Linear(128→10)
**Parameters**: 2,550,858
**Optimizer**: Adam lr=1e-3, CosineAnnealingLR

| Metric | Value |
|--------|-------|
| Best val acc | **54.35%** |
| Best epoch | 20 (still climbing) |
| Final train acc | 51.44% |
| Final val loss | 1.2897 |
| Elapsed | ~49 s |

**Plots**: [loss_curves.png](exp04_alexnet_fe_lut_cl/loss_curves.png) · [accuracy_curves.png](exp04_alexnet_fe_lut_cl/accuracy_curves.png)

**Interpretation**: A striking result — swapping the AlexNet MLP classifier for a LUT classifier collapses val accuracy from 89.4% to 54.4%, despite the feature extractor being identical and 2.5M params being available. The LUT classifier fails to leverage the high-quality AlexNet features. Root cause: the AlexNet features (post-ReLU activations, unbounded) are not pre-normalized before entering the first LUT layer; the anchor pair delta comparisons work best on inputs that are bounded and roughly symmetric around zero. Without an input BatchNorm before the first LUT layer, the relative orderings of random dimension pairs carry little signal. Lesson: **LUT classifiers need input normalization** (e.g., BN before the first LUT layer, not after).

---

### EXP05 — Small LUT Feature Extractor + Small LUT Classifier

**Model**: Fully-LUT minimal — LUT-FE 3-stage (3→32→64→64, nap=[4,6,6], tph=4) + LUT-CL 2-layer (1024→128,nap=8 → 64,nap=5) + Linear(64→10)
**Parameters**: 175,114
**Optimizer**: Adam lr=1e-3, CosineAnnealingLR

| Metric | Value |
|--------|-------|
| Best val acc | **35.71%** |
| Best epoch | 20 (still climbing) |
| Final train acc | 31.61% |
| Final val loss | 1.7438 |
| Elapsed | ~49 s |

**Plots**: [loss_curves.png](exp05_lut_fe_lut_cl_small/loss_curves.png) · [accuracy_curves.png](exp05_lut_fe_lut_cl_small/accuracy_curves.png)

**Interpretation**: The fully-LUT minimal model reaches only 35.7% — well below the same LUT-FE with MLP classifier (60.0% in Exp02). The LUT classifier clearly underperforms the MLP on LUT-extracted features too, not just on AlexNet features. The combined LUT-FE + LUT-CL stack compounds the slow optimization. The train/val gap is ~4 pp (still underfitting), suggesting more epochs are needed. The MLP bottleneck in Exp02 seems essential for early training signal.

---

### EXP06 — Medium LUT Feature Extractor + Medium LUT Classifier

**Model**: Fully-LUT medium — LUT-FE 4-stage (3→32→64→128→128, nap=[4,6,8,8], tph=4) + LUT-CL 2-layer (2048→256,nap=8,tph=4 → 128,nap=6,tph=4) + Linear(128→10)
**Parameters**: 577,546
**Optimizer**: Adam lr=1e-3, CosineAnnealingLR

| Metric | Value |
|--------|-------|
| Best val acc | **29.35%** |
| Best epoch | 18 |
| Final train acc | 25.95% |
| Final val loss | 1.9295 |
| Elapsed | ~53 s |

**Plots**: [loss_curves.png](exp06_lut_fe_lut_cl_medium/loss_curves.png) · [accuracy_curves.png](exp06_lut_fe_lut_cl_medium/accuracy_curves.png)

**Interpretation**: Worse than Exp05 despite more parameters and capacity. Adding a 4th LUT-FE stage and a larger LUT-CL further slows optimization. The model barely reaches 30% in 20 epochs — even shallower than random (~10%) but learning very slowly. The consistent pattern across EXP04–EXP06: **LUT classifiers struggle when receiving non-normalized input**. The Conv2DLut outputs, like AlexNet ReLU activations, are unbounded and non-symmetric. The first LUT-CL layer's anchor pair comparisons need normalized features. This is the key design lesson of the screening phase.

---

### EXP07 — Large LUT Feature Extractor + Large LUT Classifier

**Model**: Fully-LUT large — LUT-FE 5-stage (3→64→128→192→128→128, nap=[4,6,8,8,8], tph=8) + LUT-CL 2-layer (2048→256,nap=8,tph=8,heads=8 → 128,nap=6,tph=8,heads=4) + Linear(128→10)
**Parameters**: 1,583,114
**Optimizer**: Adam lr=1e-3, CosineAnnealingLR

| Metric | Value |
|--------|-------|
| Best val acc | **25.98%** |
| Best epoch | 18 |
| Final train acc | 24.03% |
| Final val loss | 1.9580 |
| Elapsed | ~87 s |

**Plots**: [loss_curves.png](exp07_lut_fe_lut_cl_large/loss_curves.png) · [accuracy_curves.png](exp07_lut_fe_lut_cl_large/accuracy_curves.png)

**Interpretation**: The largest fully-LUT model is the worst performer (25.98%). Doubling tph (8 vs 4) and adding a 5th FE stage does not help — in fact it hurts further, presumably because the larger LUT table parameter spaces take even longer to fill with useful gradients. The train/val gap is minimal (train ≈ val), confirming severe underfitting. The model needs dramatically more epochs to converge. Key finding: at 20 epochs, **more LUT capacity is counterproductive** — it slows convergence without improving the screening-phase signal.

---

## Ranked Summary (by 20-epoch val accuracy)

| Rank | Experiment | Family | Params | Val Acc@20ep |
|------|-----------|--------|--------|-------------|
| 1 | exp01_alexnet_baseline | Classic AlexNet | 6.46M | **89.36%** |
| 2 | exp02_lut_fe_small_mlp | LUT-FE + MLP | 300K | **60.04%** |
| 3 | exp04_alexnet_fe_lut_cl | AlexNet-FE + LUT-CL | 2.55M | **54.35%** |
| 4 | exp03_lut_fe_deep_mlp | LUT-FE + MLP | 1.55M | **54.20%** |
| 5 | exp05_lut_fe_lut_cl_small | LUT-FE + LUT-CL | 175K | **35.71%** |
| 6 | exp06_lut_fe_lut_cl_medium | LUT-FE + LUT-CL | 577K | **29.35%** |
| 7 | exp07_lut_fe_lut_cl_large | LUT-FE + LUT-CL | 1.58M | **25.98%** |

---

## Key Findings

1. **LUT-FE + MLP works best among LUT variants.** Exp02 (60%) and Exp03 (54%) both use a standard MLP classifier and clearly outperform all fully-LUT designs. The MLP provides fast, stable gradient flow that the LUT feature extractor needs to train effectively in 20 epochs.

2. **Deeper LUT-FE is counter-productive at 20 epochs.** The 5-stage FE (Exp03, 54.2%) underperforms the 3-stage FE (Exp02, 60.0%) despite 5× more params in the FE. Large LUT table spaces (nap=8 → 256 entries) need more gradient iterations to fill meaningfully. A 3-stage FE with nap≤6 converges much faster.

3. **LUT classifiers need pre-normalized input.** EXP04–EXP07 all show that when the first LUT-CL layer receives raw, unbounded feature vectors (post-ReLU from AlexNet, or post-Conv2DLut), accuracy collapses to the 25–55% range even with many parameters. Adding a `BatchNorm1d` *before* the first MultiHeadLut layer (not after) is the primary fix for the next batch.

4. **More LUT tables hurt at 20 epochs.** Comparing Exp06 (tph=4, 29.4%) vs Exp07 (tph=8, 26.0%), doubling tables_per_head makes things worse, not better — the optimization problem becomes harder.

5. **AlexNet-FE + LUT-CL (Exp04) does not benefit from the stronger features.** Despite having identical AlexNet features to Exp01, Exp04 only achieves 54.4%. The bottleneck is entirely in the LUT classifier, confirming finding #3.

---

## Proposed Next Batch

Based on findings, the most important change to test is **adding input BatchNorm before the first LUT-CL layer**. Secondary: keep depth shallow and anchor pairs small.

Proposed experiments (Batch 2):

| # | Name | Key change vs batch 1 | Expected |
|---|------|----------------------|---------|
| B2-01 | `exp08_lut_fe_small_mlp_bn` | Exp02 + BN after each Conv2DLut stage | Test if FE benefits from stage-level BN |
| B2-02 | `exp09_alexnet_fe_lut_cl_bn` | Exp04 + BN1d before first LUT-CL layer | Verify BN fixes LUT-CL issue |
| B2-03 | `exp10_lut_fe_lut_cl_bn` | Exp05 + BN1d before first LUT-CL + BN after each FE stage | Full fix for fully-LUT |
| B2-04 | `exp11_lut_fe_large_nap` | Exp02 FE but nap=[6,8,8], tph=8 | Higher capacity shallow FE + MLP |
| B2-05 | `exp12_lut_fe_stride2_mlp` | 3-stage FE using stride=2 Conv2DLut (no MaxPool) + BN2d + MLP | Learnable downsampling vs fixed MaxPool |

---

## Batch 2 Experiment Entries

<!-- batch 2 entries appended below -->

---

### EXP08 — LUT-FE + BN2d + MLP

**Model**: LUT-FE 3-stage (3→32→64→64, nap=[4,6,6], tph=4) + BN2d after each stage + MaxPool×3 → MLP(1024→256→10)
**Parameters**: 300,618  **Key change vs Exp02**: BN2d after each Conv2DLut

| Metric | Value |
|--------|-------|
| Best val acc | **59.46%** |
| Best epoch | 16 |
| Final train acc | 57.08% |
| Elapsed | ~48 s |

**Plots**: [loss_curves.png](exp08_lut_fe_bn_mlp/loss_curves.png) · [accuracy_curves.png](exp08_lut_fe_bn_mlp/accuracy_curves.png)

**Interpretation**: BN2d after each LUT-FE stage gives faster early convergence (42% at epoch 1 vs 35% in Exp02) but converges to virtually the same final accuracy (59.5% vs 60.0%). BN2d helps stability but is not a significant architectural lever for LUT-FE. The primary bottleneck is not normalisation — it's the expressivity of the LUT comparisons.

---

### EXP09 — Classic AlexNet-FE + Input BN1d + LUT Classifier

**Model**: AlexNet-FE + BN1d(4096) before first LUT layer + LUT-CL 2-layer + Linear(128→10)
**Parameters**: 2,559,050  **Key change vs Exp04**: BN1d before first MultiHeadLut

| Metric | Value |
|--------|-------|
| Best val acc | **45.72%** |
| Best epoch | 20 |
| Final train acc | 41.96% |
| Elapsed | ~48 s |

**Plots**: [loss_curves.png](exp09_alexnet_fe_bn_lut_cl/loss_curves.png) · [accuracy_curves.png](exp09_alexnet_fe_bn_lut_cl/accuracy_curves.png)

**Interpretation**: Input BN1d *worsens* the AlexNet-FE + LUT-CL result (45.7% vs 54.4% in Exp04). The BN1d disrupts gradient flow through the AlexNet backbone — the backbone can no longer adapt its features to feed the LUT comparisons, since the BN1d decouples the feature magnitudes. Also, the AlexNet's own BN+ReLU layers already provide per-channel normalisation; a global BN1d(4096) on the flat vector is redundant and adds 8192 parameters that compete for the optimiser's budget. **Key lesson: do NOT add a flat BN1d on top of an already-BN-normalised backbone.** The real problem with LUT-CL is not normalisation — it is the sparse gradient update pattern of the lookup tables.

---

### EXP10 — LUT-FE + BN2d + BN1d-input + LUT Classifier

**Model**: LUT-FE 3-stage + BN2d after each stage + BN1d(1024) before LUT-CL + LUT-CL 2-layer + Linear(64→10)
**Parameters**: 177,482  **Key change vs Exp05**: BN2d + BN1d input fix

| Metric | Value |
|--------|-------|
| Best val acc | **28.81%** |
| Best epoch | 19 |
| Final train acc | 25.41% |
| Elapsed | ~47 s |

**Plots**: [loss_curves.png](exp10_lut_fe_bn_lut_cl/loss_curves.png) · [accuracy_curves.png](exp10_lut_fe_bn_lut_cl/accuracy_curves.png)

**Interpretation**: Both BN fixes together do not improve the fully-LUT model (28.8% vs 35.7% in Exp05 — actually worse). The BN1d before LUT-CL is not helping the LUT-FE + LUT-CL pipeline either. The fundamental bottleneck for LUT classifiers at 20 epochs is the **sparse table update problem**: with 2^8 = 256 entries per table, many entries receive zero gradient each epoch, requiring many more epochs to fill. No amount of normalisation can fix this — the model simply needs more training time.

---

### EXP11 — High-Capacity LUT Feature Extractor + MLP

**Model**: LUT-FE 3-stage (3→32→64→64, nap=[6,8,8], tph=8) + BN2d + MaxPool×3 → MLP(1024→256→10)
**Parameters**: 544,330  **Key changes vs Exp02**: nap up to 8, tph doubled to 8, BN2d added

| Metric | Value |
|--------|-------|
| Best val acc | **62.22%** |
| Best epoch | 19 |
| Final train acc | 60.17% |
| Elapsed | ~49 s |

**Plots**: [loss_curves.png](exp11_lut_fe_high_cap_mlp/loss_curves.png) · [accuracy_curves.png](exp11_lut_fe_high_cap_mlp/accuracy_curves.png)

**Interpretation**: **New best LUT result** — 62.2%, the highest val accuracy of any LUT-based model across both batches. Doubling tables_per_head from 4 to 8 and using nap=[6,8,8] gives a meaningful gain (+2.2 pp vs Exp02) when combined with BN2d. The train/val gap is minimal (60.2% vs 62.2%), confirming the model is still underfitting. The curve hasn't converged — with more epochs this configuration has the most potential. This is the **recommended LUT-FE configuration** for longer training runs.

---

### EXP12 — LUT-FE Stride-2 Downsampling + MLP

**Model**: LUT-FE 3-stage (3→32→64→64, nap=[4,6,6], tph=4, stride=2) + BN2d → MLP(1024→256→10)
**Parameters**: 300,618  **Key change vs Exp08**: stride=2 in Conv2DLut, no MaxPool

| Metric | Value |
|--------|-------|
| Best val acc | **51.41%** |
| Best epoch | 20 |
| Final train acc | 48.05% |
| Elapsed | ~47 s |

**Plots**: [loss_curves.png](exp12_lut_fe_stride2_mlp/loss_curves.png) · [accuracy_curves.png](exp12_lut_fe_stride2_mlp/accuracy_curves.png)

**Interpretation**: Stride-2 downsampling significantly underperforms MaxPool+stride-1 (51.4% vs 59.5% in Exp08 / 60.0% in Exp02). With stride-2, each output position sees exactly one 3×3 window with no overlap between adjacent windows. MaxPool over stride-1 outputs allows each patch to look at every possible 3×3 window position, creating spatial invariance via pooling. This invariance is valuable for CIFAR classification. **LUT-FE should use stride=1 + MaxPool, not stride=2.**

---

## Batch 2 Ranked Summary

| Rank | Exp | Key change | Val Acc |
|------|-----|-----------|---------|
| 1 | exp11_lut_fe_high_cap_mlp | ↑ nap + tph + BN2d | **62.22%** ← best LUT |
| 2 | exp02_lut_fe_small_mlp | baseline LUT-FE | 60.04% |
| 3 | exp08_lut_fe_bn_mlp | + BN2d in FE | 59.46% |
| 4 | exp12_lut_fe_stride2_mlp | stride-2 downsampling | 51.41% |
| 5 | exp09_alexnet_fe_bn_lut_cl | + input BN on AlexNet FE | 45.72% |
| 6 | exp05_lut_fe_lut_cl_small | baseline fully-LUT | 35.71% |
| 7 | exp10_lut_fe_bn_lut_cl | + BN2d + BN1d fully-LUT | 28.81% |

## Batch 2 Key Findings

1. **Higher capacity helps LUT-FE + MLP.** nap=[6,8,8] + tph=8 (Exp11) is the new best at 62.2% vs 60.0% baseline — the LUT-FE benefits from more tables and larger table size, as long as the MLP provides fast gradient signal.

2. **BN2d in LUT-FE is neutral.** Faster start, same final accuracy. Not a primary lever but a useful default for stability.

3. **Input BN1d before LUT-CL is harmful when the upstream network already has BN.** AlexNet's own BN+ReLU sufficiently normalises features. Adding flat BN1d disrupts backbone gradients.

4. **BN fixes do not rescue LUT classifiers at 20 epochs.** The bottleneck is sparse table gradient updates — each entry in a 2^n_anchor_pairs table gets gradients only for the few batches where it's selected. More epochs are the primary fix.

5. **MaxPool >> stride-2 for LUT-FE.** Spatial pooling provides valuable invariance. stride-2 downsampling costs ~8-9 pp accuracy vs MaxPool.

## Proposed Batch 3

| # | Name | Rationale |
|---|------|-----------|
| B3-01 | `exp13_lut_fe_high_cap_deep_mlp` | Exp11 topology but 4 stages — does depth + capacity help? |
| B3-02 | `exp14_lut_fe_high_cap_50ep` | Exp11 for 50 epochs — test whether underfitting resolves |
| B3-03 | `exp15_lut_fe_wider_mlp` | Wider channels (3→64→128→128, nap=[6,8,8], tph=8) + MLP |
| B3-04 | `exp16_lut_cl_standalone_bn` | LUT-CL with BN1d input on *LUT-generated* features that are already BN2d-normalised (chain: LUT-FE+BN2d → flatten → LUT-CL, NO extra BN1d before LUT-CL — does the BN2d in FE suffice?) |

*(Batches 3–4 / exp13–exp20 were superseded by the reference model rewrite; Phase 2 begins at exp21.)*

---

## Phase 2: Development (Batches 5–12, n_alternatives=3, stride-2, no BN)

**New reference architecture** (from `reference_lut_model.py`):
- FE: 3× Conv2DLut, stride-2 downsampling (no MaxPool), no BN, `n_alternatives=3`
- Spatial: 32→[k=4,s=2]→15→[k=4,s=2]→6→[k=3/4,s=1/2]→4 or 2
- Classifier: LUT-CL 2-layer or MLP head
- Constraint: `n_alternatives=3`, `smooth_mode=True` throughout

`n_alternatives=3` enables smooth interpolation over 3 nearest table entries → better gradient flow vs n_alternatives=1.

**Reference baseline**: FE(3→32→32→32, nap=[6,8,8], tph=[8,16,16], k=4,s=2) → flatten(128) → LUT-CL → **54% @ 20ep**, 541K params.

---

## Batch 5 (exp21–exp24): Reference Ablations

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp21 | Reference @ 50ep | 541K | 58.08% @ ep41 | Slow convergence; underfitting |
| exp22 | c3: k=3,s=1 (4×4 grid, 512-dim) | 541K | 55.98% @ ep17 | Denser spatial helps FE; CL still bottleneck |
| exp23 | Wider LUT-CL (×2 outputs) | 803K | 54.34% @ ep19 | More CL capacity doesn't help at 20ep |
| exp24 | n_alternatives=5 | 541K | 53.72% @ ep18 | nalts=5 hurts vs nalts=3 |

**Key findings**: c3 4×4 spatial grid helps. LUT-CL still bottlenecked regardless. `n_alternatives=3` is optimal.

---

## Batch 6 (exp25–exp28): Spatial + Width Exploration

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp25 | 4×4 spatial @ 50ep | 541K | 59.55% @ ep39 | Better than 2×2; curve still rising |
| exp26 | Wide-FE (32→64-ch) | 803K | 54.71% @ ep19 | Width alone doesn't help with small spatial |
| exp27 | 4-stage FE | 672K | 52.84% @ ep17 | 4th stage hurts; gradient issues at tiny tensors |
| exp28 | Wide-FE + 4×4 spatial | 803K | 56.91% @ ep19 | Width + spatial combined; still rising |

**Key findings**: 4×4 spatial consistently beats 2×2. 4-stage FE fails. Width helps only when spatial is large.

---

## Batch 7 (exp29–exp32): LUT-CL Capacity on Wide-FE+4×4

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp29 | Wide-FE + 4×4 @ 50ep | 803K | 60.98% @ ep44 | 60%+ at 50ep; still rising |
| exp30 | + Wide LUT-CL | 1.07M | 56.85% @ ep18 | More CL outputs hurt at 20ep |
| exp31 | + LUT-CL nap=10 | 1.59M | 55.97% @ ep20 | Larger tables (nap=10) → worse convergence |
| exp32 | + Wide+nap10 LUT-CL | 2.64M | 54.67% @ ep18 | Both bad: worst variant |

**Key findings**: LUT-CL capacity does not help at 20ep. Larger tables (nap=10 → 1024 entries) are even more sparsely updated. `tph` >> `nap` for convergence speed.

---

## Batch 8 (exp33–exp36): MLP Head Breakthrough

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp33 | Wide-FE + 4×4 @ 100ep | 803K | 64.21% @ ep91 | LUT-CL takes 100ep to match 20ep MLP |
| exp34 | Wide-FE + 4×4 + MLP-256 | 806K | **64.36%** @ ep20 | **MLP matches 100ep LUT-CL at 20ep!** |
| exp35 | Wide-FE c3 nap=10 + LUT-CL | 1.59M | 55.33% @ ep19 | Larger c3 tables hurt |
| exp36 | Wide-FE c3 tph=32 + LUT-CL | 1.07M | 57.57% @ ep19 | More c3 tables help LUT-CL slightly |

**Key findings**: **MLP head converges ~5× faster than LUT-CL on the same features.** Switching classifier to MLP is the primary lever. `tph` (more tables) > `nap` (larger tables) for LUT-CL convergence.

---

## Batch 9 (exp37–exp40): MLP Head Scaling

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp37 | Wide-FE + MLP-256 @ 50ep | 806K | **69.21%** @ ep48 | Best so far; +5 pp from 50ep |
| exp38 | c3 tph=32 + MLP-256 @ 20ep | 1.07M | 67.29% @ ep17 | tph=32 gives +3 pp at 20ep |
| exp39 | Wide-FE + MLP-512 @ 20ep | 1.07M | 65.87% @ ep19 | MLP-512 same as MLP-256 at 20ep |
| exp40 | c3 tph=32 + MLP-512 @ 20ep | 1.33M | 68.04% @ ep20 | c3-tph32 + MLP-512 still rising |

**Key findings**: 50 epochs unlocks +5 pp. c3 tph=32 gives +3 pp at 20ep. MLP-512 slightly behind MLP-256 at 20ep but converges higher at 50ep.

---

## Batch 10 (exp41–exp44): c3 tph=32 at 50ep + c2 tph=32

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp41 | c3-tph32 + MLP-512 @ 50ep | 1.33M | **71.76%** @ ep45 | New SOTA; +2.5 pp vs exp37 |
| exp42 | c3-tph32 + MLP-256 @ 50ep | 1.07M | 70.27% @ ep47 | MLP-512 > MLP-256 at 50ep |
| exp43 | c2+c3 tph=32 + MLP-256 @ 20ep | 1.33M | 69.83% @ ep20 | c2 tph=32: +2.5 pp at 20ep vs exp38 |
| exp44 | c2+c3 tph=32 + MLP-512 @ 20ep | 1.60M | 70.24% @ ep19 | Still rising; strong 50ep candidate |

**Key findings**: c3 tph=32 at 50ep = 71.76% SOTA. Extending tph=32 to c2 gives further improvement at 20ep.

---

## Batch 11 (exp45–exp48): c2+c3 tph=32 at 50ep + c1 tph=16

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp45 | c2+c3 tph=32 + MLP-256 @ 50ep | 1.33M | **72.49%** @ ep47 | +2.2 pp vs exp42 |
| exp46 | c2+c3 tph=32 + MLP-512 @ 50ep | 1.60M | **73.07%** @ ep46 | **New SOTA** |
| exp47 | tph=[16,32,32] + MLP-256 @ 20ep | 1.35M | 72.03% @ ep19 | c1 tph 8→16: +2.2 pp at 20ep! |
| exp48 | tph=[16,32,32] + MLP-512 @ 20ep | 1.61M | 71.94% @ ep16 | Still rising; strong 50ep candidate |

**Key findings**: New SOTA **73.07%** (exp46). Doubling c1 tph (8→16) gives +2.2 pp at 20ep.

---

## Current SOTA Progression

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| Reference | — | FE(3→32→32→32) + LUT-CL @ 20ep | 54% |
| MLP switch | exp34 | Wide-FE 4×4 + MLP-256 @ 20ep | 64.36% |
| 50ep | exp37 | Wide-FE + MLP-256 @ 50ep | 69.21% |
| c3 tph=32 | exp41 | c3-tph32 + MLP-512 @ 50ep | 71.76% |
| c2+c3 tph=32 | **exp46** | c2+c3-tph32 + MLP-512 @ 50ep | **73.07%** |

**Best architecture**: FE(3→32→64→64, nap=[6,8,8], tph=[8,32,32], k=[4,4,3], s=[2,2,1], nalts=3) → flatten(1024) → Linear(1024→512, BN+ReLU+Drop) → Linear(512→10), 50ep.

---

## Phase 2 Key Findings

1. **n_alternatives=3 is critical** — smooth interpolation over 3 entries → better gradient flow.
2. **Stride-2 downsampling works** — no MaxPool needed; learned spatial compression.
3. **4×4 spatial grid >> 2×2** — 1024-dim features vs 128-dim unlock MLP potential.
4. **MLP head >> LUT-CL** — ~5× faster convergence; 20ep MLP ≈ 100ep LUT-CL.
5. **tph (more tables) >> nap (larger tables)** for convergence speed.
6. **tph=32 progression**: c3 first (+3 pp) → c2 (+2.5 pp) → c1 (+2.2 pp at 20ep).
7. **MLP-512 > MLP-256** at 50ep (~0.5–1 pp advantage).
8. **More epochs always help** — LUT-FE features are good; bottleneck was training duration.

---

## Batch 12 (exp49–exp52): tph=[16,32,32] at 50ep + c1 tph→32

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp49 | tph=[16,32,32] + MLP-256 @ 50ep | 1.35M | **74.30%** @ ep33 | New SOTA at time |
| exp50 | tph=[16,32,32] + MLP-512 @ 50ep | 1.61M | **74.81%** @ ep46 | **New SOTA** |
| exp51 | tph=[32,32,32] + MLP-256 @ 20ep | 1.38M | 74.08% @ ep19 | c1 tph 16→32: +2 pp at 20ep! Still rising |
| exp52 | tph=[32,32,32] + MLP-512 @ 20ep | 1.65M | 74.09% @ ep17 | Still rising; strong 50ep candidate |

**Key findings**: New SOTA **74.81%** (exp50). Pushing c1 tph to 32 gives another +2 pp at 20ep (exp51 74.08% vs exp47 72.03%). exp51/52 still rising at ep17–19 → strong 50ep candidates.

---

## Current SOTA Progression

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| Reference | — | FE(3→32→32→32) + LUT-CL @ 20ep | 54% |
| MLP switch | exp34 | Wide-FE 4×4 + MLP-256 @ 20ep | 64.36% |
| 50ep | exp37 | Wide-FE + MLP-256 @ 50ep | 69.21% |
| c3 tph=32 | exp41 | c3-tph32 + MLP-512 @ 50ep | 71.76% |
| c2+c3 tph=32 | exp46 | c2+c3-tph32 + MLP-512 @ 50ep | 73.07% |
| c1 tph=16 | **exp50** | tph=[16,32,32] + MLP-512 @ 50ep | **74.81%** |

**Best architecture**: FE(3→32→64→64, nap=[6,8,8], tph=[16,32,32], k=[4,4,3], s=[2,2,1], nalts=3) → flatten(1024) → Linear(1024→512, BN+ReLU+Drop) → Linear(512→10), 50ep.

---

## Phase 2 Key Findings

1. **n_alternatives=3 is critical** — smooth interpolation over 3 entries → better gradient flow.
2. **Stride-2 downsampling works** — no MaxPool needed; learned spatial compression.
3. **4×4 spatial grid >> 2×2** — 1024-dim features vs 128-dim unlock MLP potential.
4. **MLP head >> LUT-CL** — ~5× faster convergence; 20ep MLP ≈ 100ep LUT-CL.
5. **tph (more tables) >> nap (larger tables)** for convergence speed.
6. **tph progression per layer** (each step ~+2 pp at 20ep or 50ep):
   - c3: 16→32 (+3 pp @ 20ep)
   - c2: 16→32 (+2.5 pp @ 20ep)
   - c1: 8→16 (+2.2 pp @ 20ep)
   - c1: 16→32 (+2 pp @ 20ep)
7. **MLP-512 > MLP-256** at 50ep (~0.5–1 pp advantage).
8. **More epochs always help** — LUT-FE features are good; bottleneck was training duration.

---

## Batch 13 (exp53–exp56): Bottleneck Identification — New Architecture

New reference: 5-stage AlexNet-topology FE with padding, classifier fixed at Linear(512→1024,BN+ReLU+Drop)→Linear(1024→10). Reference uses nap=10 in c2–c5, reaches 64.37% @ 20ep.

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp53 | 5-stage, nap=[5,8,8,8,8] | 2.91M | 72.37% @ ep18 | nap=10→8: +8 pp! Sparse tables confirmed as bottleneck |
| **exp54** | 5-stage, nap=[5,6,6,6,6] | 1.14M | **76.09%** @ ep17 | nap=6: another +3.7 pp! Still rising @ ep20 |
| exp55 | 4-stage (drop c4), nap=10 | 7.89M | 68.92% @ ep19 | Dropping c4 helps at nap=10 (+4.5 pp vs ref) |
| exp56 | 4-stage (drop c4), nap=8 | 2.39M | 74.13% @ ep19 | 4-stage + nap=8 good; less than 5-stage nap=6 |

**Key findings**: **nap is the dominant bottleneck**. Reducing from 10→8→6 gives +8 pp and +3.7 pp respectively at 20ep. exp54 (76.09% @ 20ep, 1.14M params) already beats all previous 50ep results. c4 hurts with large nap but helps when nap is small.

---

## Current SOTA Progression (updated)

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| Reference (new arch) | — | 5-stage AlexNet-topo, nap=10 @ 20ep | 64.37% |
| nap=8 | exp53 | 5-stage nap=8 @ 20ep | 72.37% |
| nap=6 | **exp54** | 5-stage nap=6 @ 20ep | **76.09%** |

**Best architecture (20ep)**: FE(3→32→64→64→32→32, nap=[5,6,6,6,6], tph=[16,32,64,64,32], n_heads=4, k=[4,4,3,3,4], s=[2,2,1,1,2], p=1, nalts=3) → flatten(512) → Linear(512→1024,BN+ReLU+Drop) → Linear(1024→10).

---

## Batch 14 (exp57–exp60): nap=6 Convergence + tph + Depth

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp57 | nap=6 @ 50ep | 1.14M | **79.58%** @ ep44 | +3.5 pp more from 50ep; still rising |
| exp58 | nap=4 @ 20ep | 693K | 74.31% @ ep19 | nap=4 < nap=6; tables too small, less expressive |
| **exp59** | nap=6 + tph×2 @ 20ep | 1.75M | **80.09%** @ ep19 | **New SOTA at 20ep**; still rising! |
| exp60 | 4-stage, nap=6 @ 20ep | 1.01M | 75.61% @ ep17 | 4-stage < 5-stage at nap=6; c4 helps when tables converge |

**Key findings**: nap=6 is the sweet spot (nap=4 loses expressiveness, nap=8 too sparse). Doubling tph gives +4 pp at 20ep, reaching **80.09%** — new SOTA. exp59 still rising at ep20 → huge 50ep potential.

---

## Current SOTA Progression (updated)

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| Reference (new arch) | — | 5-stage nap=10 @ 20ep | 64.37% |
| nap=8 | exp53 | 5-stage nap=8 @ 20ep | 72.37% |
| nap=6 | exp54 | 5-stage nap=6 @ 20ep | 76.09% |
| nap=6 50ep | exp57 | 5-stage nap=6 @ 50ep | 79.58% |
| nap=6 tph×2 | **exp59** | 5-stage nap=6 tph×2 @ 20ep | **80.09%** |

**Best architecture**: FE(3→32→64→64→32→32, nap=[5,6,6,6,6], tph=[32,64,128,128,64], n_heads=4, k=[4,4,3,3,4], s=[2,2,1,1,2], p=1, nalts=3) → flatten(512) → Linear(512→1024,BN+ReLU+Drop) → Linear(1024→10), 20ep.

---

## Batch 15 (exp61–exp64): tph Scaling + Width + nap Comparison

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp61 | nap=6, tph×2 @ 50ep | 1.75M | **81.98%** @ ep45 | +1.9 pp more at 50ep |
| **exp62** | nap=6, tph×4 @ 20ep | 2.96M | **83.29%** @ ep19 | **New SOTA**; still rising! |
| exp63 | nap=6, tph×2, wide ×1.5 @ 20ep | 2.62M | 80.76% @ ep20 | Width helps +0.7 pp but tph more efficient/param |
| exp64 | nap=8, tph×2 @ 20ep | 5.29M | 77.70% @ ep18 | nap=8 loses 2.4 pp vs nap=6 even with same tph |

**Key findings**: tph×4 → **83.29% @ 20ep**, still rising. nap=6 strictly dominates nap=8. Width helps modestly but tph gives more per parameter. exp62 is the strongest 50ep candidate ever.

---

## Current SOTA Progression (updated)

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| Reference (new arch) | — | 5-stage nap=10 @ 20ep | 64.37% |
| nap=8 | exp53 | 5-stage nap=8 @ 20ep | 72.37% |
| nap=6 | exp54 | 5-stage nap=6 @ 20ep | 76.09% |
| nap=6 50ep | exp57 | 5-stage nap=6 @ 50ep | 79.58% |
| nap=6 tph×2 | exp59 | 5-stage nap=6 tph×2 @ 20ep | 80.09% |
| nap=6 tph×2 50ep | exp61 | 5-stage nap=6 tph×2 @ 50ep | 81.98% |
| nap=6 tph×4 | **exp62** | 5-stage nap=6 tph×4 @ 20ep | **83.29%** |

**Best architecture**: FE(3→32→64→64→32→32, nap=[5,6,6,6,6], tph=[64,128,256,256,128], n_heads=4, k=[4,4,3,3,4], s=[2,2,1,1,2], p=1, nalts=3) → flatten(512) → Linear(512→1024,BN+ReLU+Drop) → Linear(1024→10), 20ep.

---

## Batch 16 (exp65–exp68): n_heads Sweep + Wide Channels

Base config (exp62): nap=[5,6,6,6,6], tph=[64,128,256,256,128], channels 32→64→64→32→32. Varied n_heads (1/4/8) and channel width (1×/2×).

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp65 | n_heads=1 (exp62 base) | 2.96M | 81.49% @ ep20 | n_heads=1 worst; each table covers full output space but lacks diversity |
| exp66 | n_heads=8 (exp62 base) | 2.96M | 83.23% @ ep19 | n_heads=8 > n_heads=4 (+0.06 pp); more head diversity helps |
| exp67 | wide (2×) + n_heads=4 | 5.91M | 83.59% @ ep17 | 2× channels + n_heads=4 better than exp62; still rising |
| **exp68** | **wide (2×) + n_heads=8** | **5.91M** | **84.23%** @ ep19 | **New SOTA**; best of both: more channels + more heads |

**Key findings**:
- n_heads order: 8 > 4 > 1; more heads = more diverse anchor-pair views per output group
- n_heads is free (no param cost) → n_heads=8 is a strict improvement over n_heads=4
- Wide channels (2×) adds +0.64 pp over narrow at n_heads=8
- exp68 still rising at ep20 → strong 50ep candidate
- **New SOTA: exp68 = 84.23% @ 20ep**, channels 64→128→128→64→64, n_heads=8

---

## Current SOTA Progression (updated)

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| Reference (new arch) | — | 5-stage nap=10 @ 20ep | 64.37% |
| nap=8 | exp53 | 5-stage nap=8 @ 20ep | 72.37% |
| nap=6 | exp54 | 5-stage nap=6 @ 20ep | 76.09% |
| nap=6 50ep | exp57 | 5-stage nap=6 @ 50ep | 79.58% |
| nap=6 tph×2 | exp59 | 5-stage nap=6 tph×2 @ 20ep | 80.09% |
| nap=6 tph×2 50ep | exp61 | 5-stage nap=6 tph×2 @ 50ep | 81.98% |
| nap=6 tph×4 | exp62 | 5-stage nap=6 tph×4 n_heads=4 @ 20ep | 83.29% |
| nap=6 tph×4 n_heads=8 wide | **exp68** | 5-stage nap=6 tph×4 n_heads=8 wide @ 20ep | **84.23%** |

**Best architecture**: FE(3→64→128→128→64→64, nap=[5,6,6,6,6], tph=[64,128,256,256,128], n_heads=8, k=[4,4,3,3,4], s=[2,2,1,1,2], p=1, nalts=3) → flatten(1024) → Linear(1024→1024,BN+ReLU+Drop) → Linear(1024→10).

---

## Batch 17 (exp69–exp72): 50ep Convergence + tph Scaling + Wider Channels

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp69 | exp68 @ 50ep | 5.91M | 84.92% @ ep48 | Plateau after ep33; tph was the bottleneck, not epochs |
| **exp70** | wide 2×, n_heads=8, tph×2 | 10.76M | **85.32%** @ ep20 | **New SOTA**; tph scaling law continues; still rising! |
| exp71 | wide 4×, n_heads=8, tph same | 11.81M | 84.27% @ ep20 | 4× width < 2× width + tph×2 at same params |
| exp72 | wide 4×, n_heads=16, tph same | 11.81M | 84.47% @ ep19 | n_heads=16 only +0.2pp over n_heads=8 at 4× width |

**Key findings**:
- tph scaling law still holds: each doubling ~+1 pp at 20ep (narrow→wide→wide+tph×2: 83.29→84.23→85.32)
- exp70 @ 20ep beats exp69 @ 50ep — more tph > more epochs when tph is the bottleneck
- Wide 4× is less efficient per param than wide 2× + tph×2
- n_heads=16 gives negligible benefit over n_heads=8 at wide 4× (barely satisfies 8-output/head minimum)
- **n_heads sweet spot**: 8 with enough width (≥8 outputs/head) — n_heads=16 marginal

---

## Current SOTA Progression (updated)

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| Reference (new arch) | — | 5-stage nap=10 @ 20ep | 64.37% |
| nap=8 | exp53 | 5-stage nap=8 @ 20ep | 72.37% |
| nap=6 | exp54 | 5-stage nap=6 @ 20ep | 76.09% |
| nap=6 50ep | exp57 | 5-stage nap=6 @ 50ep | 79.58% |
| nap=6 tph×2 | exp59 | 5-stage nap=6 tph×2 @ 20ep | 80.09% |
| nap=6 tph×2 50ep | exp61 | 5-stage nap=6 tph×2 @ 50ep | 81.98% |
| nap=6 tph×4 | exp62 | 5-stage nap=6 tph×4 n_heads=4 @ 20ep | 83.29% |
| nap=6 tph×4 wide n_heads=8 | exp68 | wide 2×, n_heads=8 @ 20ep | 84.23% |
| nap=6 tph×8 wide n_heads=8 | **exp70** | wide 2×, n_heads=8, tph×2 @ 20ep | **85.32%** |

**Best architecture**: FE(3→64→128→128→64→64, nap=[5,6,6,6,6], tph=[128,256,512,512,256], n_heads=8, k=[4,4,3,3,4], s=[2,2,1,1,2], p=1, nalts=3) → flatten(1024) → Linear(1024→1024,BN+ReLU+Drop) → Linear(1024→10).

---

## Batch 18 (exp73–exp76): Small-Model Efficiency

Focus: param-efficient designs under ~3M params.

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp73 | exp62 @ 50ep | 2.96M | 84.00% @ ep48 | 2.96M model ceiling; plateau ep35+; below exp68 (5.91M) |
| exp74 | wide 2×, n_heads=8, tph/2 (iso-param exp62) | 3.49M | 81.48% @ ep20 | tph > width/heads at same param budget; n_heads=8+width doesn't compensate |
| **exp75** | **nap=5 all, narrow, n_heads=4** | **1.78M** | **83.11%** @ ep20 | **Standout: −40% params, −0.18pp vs exp62. Still rising!** |
| exp76 | nap=[5,5,6,6,5], narrow, n_heads=4 | 2.57M | 82.98% @ ep19 | Slightly worse than exp75 despite more params |

**Key findings**:
- **nap=5 everywhere is highly efficient**: 1.78M → 83.11% @ 20ep ≈ exp62 (2.96M, 83.29%)
- nap=6 in deeper layers gives only marginal gains over nap=5; nap=5 all is the better tradeoff
- tph beats width+heads at iso-param budget (exp74 vs exp62: 81.48% vs 83.29%)
- exp75 still rising at ep20 → very strong 50ep candidate at 1.78M params

---

## Batch 19 (exp77–exp80): connected_anchors_mode + 50ep Convergence

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| **exp77** | nap=5 all @ 50ep, connected=False | 1.78M | **84.40%** @ ep49 | **Best small-model result**: beats exp73 (2.96M, 84.00%) with 40% fewer params |
| exp78 | nap=5 all, connected=True | 1.78M | 82.56% @ ep20 | −0.55pp vs exp75; connected hurts narrow+nap=5 |
| exp79 | nap=6, connected=True | 2.96M | 82.63% @ ep18 | −0.66pp vs exp62; connected hurts narrow+nap=6 |
| exp80 | wide 2×, n_heads=8, connected=True | 5.91M | 84.43% @ ep19 | +0.20pp vs exp68; marginal benefit at larger scale |

**Key findings**:
- connected_anchors_mode hurts narrow models (−0.55–0.66pp); marginal +0.20pp at wide+n_heads=8
- **exp77 is the efficiency champion**: 1.78M → 84.40% @ 50ep beats 2.96M model (84.00%)
- nap=5 all is strictly more param-efficient than nap=[5,6,6,6,6] at same tph
- Connected mode not generally useful; chain comparisons may be too correlated for image patches

---

## Current SOTA Progression (updated)

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| Reference (new arch) | — | 5-stage nap=10 @ 20ep | 64.37% |
| nap=6 | exp54 | 5-stage nap=6 @ 20ep | 76.09% |
| nap=6 tph×4 | exp62 | nap=6 tph×4 n_heads=4 @ 20ep | 83.29% |
| nap=5 all 50ep | **exp77** | nap=5 all 1.78M @ 50ep | **84.40%** |
| wide n_heads=8 tph×2 | exp70 | wide 2×, n_heads=8, tph×2 @ 20ep | 85.32% |

**Best small model** (≤2M params): **exp106** — late-downsample 5-stage, 1.78M, **86.05%** @ 50ep — new SOTA!

---

## Batch 26 (exp105–exp112): Convergence + Topology Push

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| **exp105** | exp102 @ 50ep (6-stage extra 32×32) | 1.72M | **85.51%** @ ep46 | Beats exp96@100ep with fewer params |
| **exp106** | **exp104 @ 50ep (late downsample)** | **1.78M** | **86.05%** @ ep46 | **NEW OVERALL SOTA — beats exp85 (3.03M!) at same params as exp96** |
| exp107 | 7-stage: 2 extra at 32×32, 20ep | 2.24M | 84.24% @ ep18 | More depth at 32×32 → slower convergence |
| exp108 | 7-stage: extra at 32×32 + 16×16, 20ep | 2.37M | 83.93% @ ep20 | Combined scale processing not worth cost |
| exp109 | exp102 + k=5 at 32×32 layer | 1.72M | 83.16% @ ep19 | k=5 at large scale doesn't help |
| exp110 | exp104 + k=5 at 32×32 layers | 1.78M | 82.99% @ ep19 | k=5 hurts late-downsample topology |
| exp111 | exp104 + extra 8×8 layer (6-stage) | 1.85M | 83.13% @ ep20 | Extra fine-scale depth doesn't help |
| exp112 | exp99 @ 50ep (k=3 everywhere) | 1.78M | 84.50% @ ep50 | Gap widens at 50ep vs exp96 (−0.85pp) |

**Key findings**:
- **exp106 is new SOTA (86.05%)**: late-downsample 5-stage at 1.78M beats exp85 (85.92% at 3.03M) — 40% fewer params!
- **exp105 (85.51%)** also beats exp96@100ep (85.35%) with fewer params, confirming extra 32×32 processing helps
- **Sweet spot is exactly 2 stride-1 32×32 layers**: more 32×32 processing (exp107) converges slower; combining scales (exp108) bloats params
- **k=5 consistently hurts** at large spatial scales — nap=5 anchor pairs already capture sufficient context
- **exp106 is still rising at ep46** — a 100ep run may push it further

---

## Batch 25 (exp101–exp104): Spatial Topology Exploration

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp101 | 6-stage, extra layer at 16×16 | 1.85M | 82.42% @ ep20 | −0.69pp vs exp96; 16×16 extra less useful |
| **exp102** | **6-stage, extra layer at 32×32 (before first downsample)** | **1.72M** | **83.70%** @ ep20 | **+0.59pp vs exp96! New 20ep best at ~1.7M** |
| exp103 | 5-stage, aggressive k=8/s=4 (skip 16×16) [ISO-PARAM] | 1.78M | 76.60% @ ep17 | −6.51pp; skipping 16×16 scale is catastrophic |
| **exp104** | **5-stage, late downsample: process at 32×32×2 first [ISO-PARAM]** | **1.78M** | **83.61%** @ ep17 | **+0.50pp vs exp96!** |

**Key findings**:
- **Processing at 32×32 before downsampling beats exp96**: both exp102 (+0.59pp) and exp104 (+0.50pp) win
- **16×16 scale is critical**: skipping it (exp103) is catastrophic (−6.5pp); adding extra at 16×16 only helps marginally
- **Late downsampling (process large, compress late) is better than early compression** — LUTs benefit from operating at large spatial scales
- exp102 and exp104 should be followed up with 50ep runs

---

## Batch 24 (exp97–exp100): Topology Exploration Around exp96

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp97 | 4-stage (remove c4), k=[4,4,3,4] | 1.27M | 81.94% @ ep17 | −1.17pp vs exp96; fewer layers hurt |
| exp98 | 6-stage (add layer at 8×8), k=[4,4,3,3,3,4] | ~1.78M | 81.72% @ ep20 | −1.39pp; extra depth slower to converge |
| exp99 | 5-stage k=3 everywhere (vs k=[4,4,3,3,4]) | 1.78M | 82.64% @ ep20 | −0.47pp; best variant; small RF cost |
| exp100 | 5-stage k=5 for c3/c4 (larger RF in middle) | 1.78M | 82.12% @ ep20 | −0.99pp; larger RF in middle doesn't help |

**Key findings**:
- **Original 5-stage topology is well-tuned**: all variants are worse at 20ep
- **exp99 (k=3 everywhere) is the best alternative**: only −0.47pp — worth a 50ep run to see if gap closes
- Extra depth (6-stage) hurts convergence speed; fewer layers (4-stage) hurts expressiveness
- Larger middle-layer kernels (k=5) don't help: LUTs with nap=5 already capture enough spatial context

---

## Batch 23 (exp93–exp96): 2.31M Convergence + Connected Matrix + 100ep Narrow

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp93 | exp92 @ 50ep (wide 2×, n_heads=8, tph_half, connected=False) | 2.31M | 83.62% @ ep46 | Converged |
| exp94 | exp91 @ 50ep (wide 2×, n_heads=8, tph_half, connected=True) | 2.31M | 83.63% @ ep43 | Connected=True vs False: +0.01pp — negligible |
| exp95 | wide 2×, full tph, connected=False @ 50ep | 3.55M | 85.47% @ ep48 | vs exp89 (connected=True, 85.62%): +0.15pp for connected |
| **exp96** | **narrow nap=5, connected=True @ 100ep** | **1.78M** | **85.35%** @ ep96 | **Still rising at ep96! Well above 50ep ceiling (84.51%)**|

**Key findings**:
- **Connected=True vs False at 2.31M (50ep): +0.01pp** — effectively zero; connected makes no difference here
- **Connected at 3.55M (50ep): +0.15pp** (exp89 85.62% vs exp95 85.47%) — consistent with prior pattern: tiny but positive
- **exp96 (1.78M, 100ep): 85.35%** — remarkable! +0.84pp over its 50ep result (84.51%); still rising at ep96
- The narrow 1.78M model has NOT plateaued at 100ep; its true ceiling may be 86%+
- The 2.31M wide models plateau near 83.6% — less efficient than narrow 1.78M at 50ep+ (84.5%+)

---

## Current SOTA Progression (Batch 27 update)

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| nap=6 tph×4 | exp62 | narrow n_heads=4 @ 20ep | 83.29% |
| nap=5 all 50ep | exp77 | 1.78M @ 50ep | 84.40% |
| nap=5 connected 50ep | exp88 | 1.78M connected @ 50ep | 84.51% |
| wide n_heads=8 tph×2 | exp70 | wide 2×, tph×2 @ 20ep | 85.32% |
| nap=5 tph×2 no-connected | exp90 | 3.03M @ 50ep | 85.79% |
| nap=5 narrow 100ep | exp96 | 1.78M connected @ 100ep | 85.35% |
| nap=5 tph×2 connected | exp85 | 3.03M connected @ 50ep | 85.92% |
| late-downsample 50ep | exp106 | 1.78M connected @ 50ep | 86.05% |
| narrow late-ds deep tph | exp121 | 3M narrow late-ds @ 20ep | 85.68% |
| **wide tph-max 20ep** | **exp123** | **~7M wide std @ 20ep** | **85.95%** |

**Best small model by param size (updated):**
- ~1.78M: exp106 — **86.05% @ 50ep** (overall SOTA)
- ~2.31M: exp93/94 — 83.62–83.63% @ 50ep
- ~3.0M: exp121 — **85.68% @ 20ep** (still rising; strong 50ep candidate)
- ~3.5M: exp89 — 85.62% @ 50ep
- ~6M: exp120 — 85.65% @ 20ep (still rising)
- ~7M: exp123 — **85.95% @ 20ep** (best 20ep result; still rising)

---

## Batch 22 (exp89–exp92): 50ep Convergence + Wide 2× tph-Half Exploration

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp89 | exp86 @ 50ep (wide 2×, n_heads=8, nap=5, connected=True) | 3.55M | 85.62% @ ep44 | More params than exp85 but −0.30pp; narrow+tph beats wide+heads at same budget |
| **exp90** | exp81 @ 50ep (nap=5, tph×2, connected=False) | 3.03M | **85.79%** @ ep45 | Iso-param baseline for exp85; connected=True adds only +0.13pp |
| exp91 | wide 2×, n_heads=8, tph halved, connected=True, 20ep | 2.31M | 81.69% @ ep17 | 2.31M at 20ep, still rising |
| exp92 | wide 2×, n_heads=8, tph halved, connected=False, 20ep | 2.31M | 81.84% @ ep19 | Connected=False marginally better here at 20ep (+0.15pp) |

**Key findings**:
- **Connected mode adds only +0.13pp** at 50ep for the 3.03M config (exp90 vs exp85): very small benefit
- **Narrow+n_heads=4+more_tph beats wide+n_heads=8+less_tph** at same param budget: exp90 (3.03M) 85.79% > exp89 (3.55M) 85.62%
- exp91/exp92 at 2.31M are still rising at ep20; need 50ep runs to evaluate this param point
- At 20ep and 2.31M, connected=False (exp92: 81.84%) marginally beats connected=True (exp91: 81.69%)

---

## Current SOTA Progression (Batch 22 update)

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| nap=6 tph×4 | exp62 | narrow n_heads=4 @ 20ep | 83.29% |
| nap=5 all 50ep | exp77 | 1.78M @ 50ep | 84.40% |
| nap=5 connected 50ep | exp88 | 1.78M connected @ 50ep | 84.51% |
| wide n_heads=8 tph×2 | exp70 | wide 2×, tph×2 @ 20ep | 85.32% |
| nap=5 tph×2 no-connected | exp90 | 3.03M @ 50ep | 85.79% |
| nap=5 tph×2 connected | **exp85** | **3.03M connected @ 50ep** | **85.92%** |

**Best small model by param size (updated):**
- ~1.78M: exp88 — 84.51% @ 50ep
- ~2.31M: exp91/exp92 — 81.69–81.84% @ 20ep (50ep ceiling unknown)
- ~3.0M: exp85 — 85.92% @ 50ep
- ~3.5M: exp89 — 85.62% @ 50ep

---

## Batch 21 (exp85–exp88): Convergence + connected@50ep + Sub-1.2M

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| **exp85** | exp84 @ 50ep (nap=5, tph×2, connected=True) | 3.03M | **85.92%** @ ep42 | **New SOTA for small models**; beats 2.96M model by 1.92pp |
| exp86 | nap=5, wide 2×, n_heads=8, connected=True | 3.55M | 84.21% @ ep19 | Still rising; needs 50ep |
| exp87 | nap=5, tph halved, 1.16M | 1.16M | 80.02% @ ep19 | nap=5+fewer tables < nap=4+more tables at same budget |
| exp88 | nap=5, 1.78M, connected=True @ 50ep | 1.78M | 84.51% @ ep44 | +0.11pp over exp77 (no connected); connected converges slightly better |

**Key findings**:
- **exp85**: nap=5+tph×2+connected=True is the most param-efficient config found — 85.92% @ 3.03M
- Connected mode pattern confirmed: slower early (−0.5pp at 20ep), same or +0.1pp at 50ep
- exp86 still rising strongly → 50ep could reach 86%+ at 3.55M
- exp87 confirms nap=5 is a hard lower bound for table expressiveness; fewer tables at nap=5 worse than more tables at nap=4

---

## Current SOTA Progression (updated)

| Milestone | Exp | Config | Val Acc |
|-----------|-----|--------|---------|
| nap=6 tph×4 | exp62 | narrow n_heads=4 @ 20ep | 83.29% |
| nap=5 all 50ep | exp77 | 1.78M @ 50ep | 84.40% |
| nap=5 connected 50ep | exp88 | 1.78M connected @ 50ep | 84.51% |
| wide n_heads=8 tph×2 | exp70 | wide 2×, tph×2 @ 20ep | 85.32% |
| nap=5 tph×2 connected | **exp85** | **3.03M connected @ 50ep** | **85.92%** |

**Best small model by param size:**
- ~1.16M: not explored well yet
- ~1.78M: exp88 — 84.51% @ 50ep
- ~3.0M: exp85 — 85.92% @ 50ep
- ~3.5M: exp86 — 84.21% @ 20ep (ceiling unknown)

---

## Batch 20 (exp81–exp84): nap=5 tph Scaling + nap=4 + Connected at Higher tph

| Exp | Key change | Params | Val Acc | Verdict |
|-----|-----------|--------|---------|---------|
| exp81 | nap=5 all, tph×2 | 3.03M | 84.19% @ ep20 | tph scaling holds at nap=5: +1.08pp vs exp75; still rising |
| exp82 | nap=5 all, wide 2×, n_heads=8 | 3.55M | 84.06% @ ep20 | ≈ exp81; nap=5+wide+heads ≈ nap=5+tph at same params |
| exp83 | nap=4 all, tph×4 | 1.16M | 80.88% @ ep19 | Viable sub-1.2M; nap=4 costs −2.23pp vs nap=5; still rising |
| **exp84** | **nap=5 tph×2, connected=True** | **3.03M** | **84.50%** @ ep20 | **connected helps at higher tph (+0.31pp vs exp81)**; still rising |

**Key findings**:
- tph scaling law holds at nap=5: each doubling ~+1pp at 20ep
- **Connected mode reversal**: hurts at low tph (exp78: −0.55pp), helps at high tph (exp84: +0.31pp)
- Hypothesis: connected mode useful when many tables exist; chain patterns complement diverse tables
- nap=4 (1.16M): 80.88% — reasonable for ultra-small; nap=5→4 costs ~−2.2pp
- exp82 shows nap=5+wide+n_heads=8 matches nap=5+tph×2 at similar params

---

## Batch 27 (exp120–exp123): tph Scaling Under Memory Constraints

**Goal**: push 20ep accuracy toward AlexNet's 89% ceiling. Batch 27 was redesigned after original designs (exp113–119) all OOM'd due to high tph at stride-1 32×32 layers.

### Memory Constraint Discovery

The intermediate score matrix in `lprojection_forward_smooth` has size `batch × n_patches × tph × n_heads × 2^nap × 4 bytes`. For stride-1 at 32×32 (n_patches=1024, batch=256, n_heads=4, nap=5):
- tph=128: ~17 GB — safe
- tph=256: ~34 GB — fills GPU when combined with other layers
- tph=512+: OOM

**User-imposed constraint**: tph ≤ 512 everywhere. n_heads=8 at stride-1 32×32 layers limited to tph ≤ 64.

| Exp | Topology | Channels | tph | n_heads | nap | Params | 20ep Acc |
|-----|----------|----------|-----|---------|-----|--------|----------|
| exp122 | narrow std | 3→32→64→64→32→32, k=[4,4,3,3,4], s=[2,2,1,1,2] | [128,256,512,512,256] | 4 | 5 | ~3.03M | 84.56% @ ep18 |
| exp120 | wide std | 3→64→128→128→64→64, k=[4,4,3,3,4], s=[2,2,1,1,2] | [128,256,512,512,256] | 8 | 5 | ~6.04M | 85.65% @ ep19 |
| exp121 | narrow late-ds | 3→32→64→64→32→32, k=[3,3,4,4,4], s=[1,1,2,2,2] | [128,128,512,512,512] | 4 | 5 | ~3.03M | **85.68%** @ ep20 |
| exp123 | wide std | 3→64→128→128→64→64, k=[4,4,3,3,4], s=[2,2,1,1,2] | [256,512,512,512,512] | 8 | 5 | ~7M | **85.95%** @ ep20 |

All models still rising at ep20 — strong candidates for 50ep runs.

**Key findings**:
1. **Late-ds topology: +1.12pp vs standard at same ~3M narrow budget** (exp121 85.68% vs exp122 84.56%)
2. **Wide channels: +1.1pp vs narrow at same tph distribution** (exp120 85.65% vs exp122 84.56%)
3. **nap=5 everywhere more param-efficient than mixed nap**: exp70 had 10.76M for 85.32%; exp121 has 3M for 85.68% (+0.36pp with 72% fewer params)
4. **exp123 is new 20ep SOTA at 85.95%** — compare to exp120 (6M, 85.65%): +0.3pp for +1M params (diminishing returns)

---

### EXP120 — Wide 2×, Standard Topology, tph×2

**Architecture**: FE(3→64→128→128→64→64, nap=5 everywhere, tph=[128,256,512,512,256], k=[4,4,3,3,4], s=[2,2,1,1,2]), n_heads=8 → flatten(1024) → Linear(1024→1024,BN+ReLU+Drop) → Linear(1024→10)
**Parameters**: ~6.04M

| Metric | Value |
|--------|-------|
| Best val acc | **85.65%** |
| Best epoch | 19 (still rising at ep20: 85.54%) |

**Notes**: nap=5 everywhere with wide channels beats exp70 (85.32%, 10.76M) by +0.33pp with 44% fewer params. tph constrained to ≤512 at stride-1 layers.

---

### EXP121 — Narrow Late-Downsample, Deep-Layer tph

**Architecture**: FE(3→32→64→64→32→32, nap=5 everywhere, tph=[128,128,512,512,512], k=[3,3,4,4,4], s=[1,1,2,2,2]), n_heads=4 → flatten(1024) → Linear(1024→1024,BN+ReLU+Drop) → Linear(1024→10)
**Parameters**: ~3.03M

| Metric | Value |
|--------|-------|
| Best val acc | **85.68%** |
| Best epoch | 20 (still rising) |

**Notes**: Late-downsample topology — processes at 32×32 twice (k=3,s=1 × 2) before compressing. Direct comparison to exp122 (same params, standard topology): +1.12pp. At 3M params, matches the wide 6M standard model (exp120). Strong 50ep candidate.

---

### EXP122 — Narrow Standard Topology (topology baseline for exp121)

**Architecture**: FE(3→32→64→64→32→32, nap=5 everywhere, tph=[128,256,512,512,256], k=[4,4,3,3,4], s=[2,2,1,1,2]), n_heads=4 → flatten(1024) → Linear(1024→1024,BN+ReLU+Drop) → Linear(1024→10)
**Parameters**: ~3.03M (iso-param vs exp121)

| Metric | Value |
|--------|-------|
| Best val acc | **84.56%** |
| Best epoch | 18 |

**Notes**: Standard early-downsample topology. Iso-param comparison to exp121 (late-ds): −1.12pp. The topology difference alone accounts for over 1pp gap at 20ep.

---

### EXP123 — Wide 2×, Standard Topology, tph=[256,512,512,512,512]

**Architecture**: FE(3→64→128→128→64→64, nap=5 everywhere, tph=[256,512,512,512,512], k=[4,4,3,3,4], s=[2,2,1,1,2]), n_heads=8 → flatten(1024) → Linear(1024→1024,BN+ReLU+Drop) → Linear(1024→10)
**Parameters**: ~7M

| Metric | Value |
|--------|-------|
| Best val acc | **85.95%** |
| Best epoch | 20 (still rising) |

**Notes**: New 20ep SOTA. Compare to exp120 (6M, tph=[128,256,512,512,256], 85.65%): +0.30pp for +1M params — modest gain. Still rising; 50ep run should push well above 86%.

---

---

## Batch 31 — Hybrid Conv2d+LUT Architecture

### EXP144 — Hybrid Conv2d+LUT Reference (50ep)

**Architecture**: Conv2d(3→32)+BN → LUT@16×16(k=4,s=2) → LUT@8×8(k=4,s=2) → LUT@8×8(k=3,s=1) → LUT@4×4(k=4,s=2); CH_CONV=32, CH=24, NH=4, NAP=5, tph=auto (ic*k²*12//(NAP*2))
**Parameters**: 2.34M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **88.00%** |
| Best epoch | 50 (still rising) |

**Notes**: Hybrid reference model. Conv2d first layer bypasses LUT's weakness at 3-channel input. Dynamic tph formula ensures capacity-matched tables. Breakthrough vs all-LUT (~84%).

---

### EXP145 — AlexNet Baseline (50ep)

**Architecture**: Conv2d-only AlexNet (3→32→96→192→128→128 with MaxPool)
**Parameters**: 2.68M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **89.69%** |
| Best epoch | 46 |

**Notes**: Upper bound reference. 89.69% @ 50ep.

---

### EXP146 — AlexNet Baseline (100ep)

**Architecture**: Same as exp145
**Parameters**: 2.68M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **90.22%** |
| Best epoch | 91 |

**Notes**: 100ep AlexNet ceiling. Gap to hybrid: ~1.9pp @ 50ep.

---

### EXP147 — Hybrid Conv2d+LUT Reference (100ep)

**Architecture**: Same as exp144
**Parameters**: 2.34M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **88.34%** |
| Best epoch | 96 (still rising) |

**Notes**: 100ep hybrid still rising at ep96. Gap to 50ep: +0.34pp. Long training helps.

---

### EXP148 — Hybrid, CH_CONV=64

**Architecture**: exp144 with CH_CONV=64 (larger init conv)
**Parameters**: 2.81M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **87.69%** |
| Best epoch | 44 |

**Notes**: CH_CONV=64 hurts (−0.31pp vs ref). Larger init conv not beneficial; CH_CONV=32 is the sweet spot.

---

### EXP149 — Hybrid, +LUT@32 Stride-1

**Architecture**: exp144 + extra LUT stage at 32×32 (k=3,s=1) before first downsample; CH_CONV=32, CH=24
**Parameters**: 2.48M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **88.63%** |
| Best epoch | 44 |

**Notes**: **NEW BATCH 31 WINNER.** +0.63pp vs ref with fewer params (2.48M vs 2.81M). Late-downsample topology helps: processing at 32×32 before compression consistently beneficial (echoes Batch 25/26 findings for all-LUT).

---

### EXP150 — Hybrid, CH=28

**Architecture**: exp144 with CH=28 (wider LUT channels)
**Parameters**: 3.01M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **87.86%** |
| Best epoch | 46 |

**Notes**: CH=28 hurts (−0.14pp vs ref) with more params. CH=24 remains sweet spot for this architecture.

---

### EXP151 — Hybrid, nap=4

**Architecture**: exp144 with nap=4 throughout (smaller LUT entries)
**Parameters**: 1.61M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **87.32%** |
| Best epoch | 47 |

**Notes**: nap=4 costs −0.68pp vs ref but at only 1.61M params (31% fewer). Reasonable efficiency tradeoff for ultra-small models.

---

### EXP152 — Hybrid, k=5 at c2

**Architecture**: exp144 with k=5 at first LUT stage (p=2 to maintain 32→16 spatial dims)
**Parameters**: 2.60M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **87.53%** |
| Best epoch | 46 |

**Notes**: Larger RF at first LUT downsample doesn't help (−0.47pp vs ref). k=4 is better.

---

### EXP153 — Hybrid, nap=6 at c3

**Architecture**: exp144 with nap=6 at c3 only (deeper entries at bottleneck expansion)
**Parameters**: 2.81M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **87.35%** |
| Best epoch | 49 |

**Notes**: nap=6 at bottleneck hurts (−0.65pp vs ref). tph formula compensates for nap change so tph drops when nap rises — net effect negative.

---

### EXP154 — Hybrid, Flat CH=32 (no bottleneck)

**Architecture**: exp144 with CH=32 flat (no CH*2 expansion at c3); all layers CH=32
**Parameters**: 2.78M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **88.23%** |
| Best epoch | 47 |

**Notes**: Flat CH=32 nearly matches ref (−0.03pp vs exp144, but ref is 88.0% so this is +0.23pp). Bottleneck expansion (24→48→24) provides minimal benefit; flat channels slightly better.

---

### EXP155 — Hybrid, connected_anchors_mode=True

**Architecture**: exp144 with connected_anchors_mode=True on all LUT layers
**Parameters**: 2.34M | batch=64

| Metric | Value |
|--------|-------|
| Best val acc | **87.97%** |
| Best epoch | 49 |

**Notes**: connected=True gives −0.03pp vs ref (negligible). Same finding as all-LUT experiments: connected mode makes no meaningful difference at this scale.

---
