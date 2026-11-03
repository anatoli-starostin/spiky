# d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000

**Question.** Does the whiten (Muon-analogue) table optimizer help when *everything else* is Muon too? Earlier `../d24-lutffn-confn1-1xh100-s0-whiten500/` (tables on whiten, projections on AdamW) showed no gain over AdamW in 500 steps. Hypothesis (Anatoly): the idea may only work if applied everywhere - Muon on the attention/block matrices (already the case in all LUT runs), Muon on the ProjectionMHL projections, plus whitening on the LUT tables.

## Setup
Identical to `../d24-lutffn-confn1-1xh100-s0-whiten500/` (v2 recipe, `--no-lr-rescale`, `NUM_ITERATIONS=5568` schedule, `--max-steps=1001`, seed 42, batch 2^20, fp8) plus **`--lut-ffn-proj-fp8-muon`**: ProjectionMHL compress/decompress GEMMs in fp8 (fp32 masters), projection *weights* on Muon at matrix-lr (0.02, no LR rescale); biases and confidence scalars stay AdamW. Tables: `--lut-table-optimizer=whiten` (same untuned RMS 0.3 / beta 0.9 / eps 1e-3). Launcher: `nanochat/runs/lut_ffn_d24_whiten_projmuon1000_1xh100.sh`; wandb group `nanochat_lut_ffn`, tag `whiten-projmuon1000`.

**Checkpoints** at steps 250, 500, 750 and 1000 are kept on the worker (`SAVE_EVERY=250`, `KEEP_CKPTS=4`, not committed); continue to 5568 with the same launcher and `TRAIN_EXTRA="--max-steps=5569"`.

## Confound and planned control (B)
`--lut-ffn-proj-fp8-muon` also changes the projections' precision (fp8 vs fp32 island) and the optimizer, so a gain here cannot be attributed to the whitened tables alone. Control B (same flag, AdamW tables) is to be run afterwards (a separate experiment) if this run looks interesting.

## Baselines
`../d24-lutffn-confn1-1xh100-s0-full-v2/` (AdamW, first 1000 steps from its wandb history; repro500 only covers 500), `../d24-lutffn-confn1-1xh100-s0-v2repro500/` (AdamW, to 500), `../d24-lutffn-confn1-1xh100-s0-whiten500/` (tables only), reference v2. Noise caveat as in the repro500 README (single seed; AdamW repro vs v2 differ up to 0.12 loss at step ~200).

## Acceptance criteria (to be judged in Slack by Anatoly)
train/loss windows, val/bpb at 250/500/750/1000, CORE at 500 and 1000, s/step, memory, stability vs the baselines above. Result below.

## Result (run finished: 1001 steps, exit 0, no restarts/resume needed)
wandb: `d24-lutffn-confn1-1xh100-s0-whiten-projmuon1000-d07ba1ff`. Compare files: `compare_projmuon1000_vs_adamw_whiten500.{csv,png}` (per-step + 10-step smoothed losses; v2 ref and projmuon cover 0-1000, repro500 / whiten500 only 0-500).

### train/loss, mean over step windows
| window | AdamW v2 ref | AdamW repro500 | whiten500 | **whiten+projMuon** | vs ref | vs repro | vs whiten500 |
|---|---|---|---|---|---|---|---|
| 1-10 | 9.838 | 9.842 | 9.891 | 9.892 | +0.054 | +0.050 | +0.001 |
| 91-100 | 4.635 | 4.667 | 4.622 | 4.531 | -0.104 | -0.136 | -0.092 |
| 241-250 | 3.633 | 3.510 | 3.504 | 3.465 | -0.167 | -0.044 | -0.039 |
| 491-500 | 3.188 | 3.132 | 3.129 | 3.119 | -0.068 | -0.013 | -0.010 |
| 741-750 | 3.011 | - | - | 3.023 | +0.013 | - | - |
| 991-1000 | 2.961 | - | - | 2.953 | -0.008 | - | - |

### val/bpb
| step | ref | repro500 | whiten500 | whiten+projMuon |
|---|---|---|---|---|
| 0 | 3.1808 | 3.1714 | 3.1714 | 3.1714 |
| 250 | 1.0814 | 1.0560 | 1.0538 | **1.0431** |
| 500 | 0.9662 | 0.9603 | 0.9588 | **0.9549** |
| 750 | 0.9237 | - | - | **0.9158** |
| 1000 | 0.9004 | - | - | **0.8942** |

### CORE (core_metric)
| step | ref | repro500 | whiten500 | whiten+projMuon |
|---|---|---|---|---|
| 500 | 0.0988 | 0.0973 | 0.0919 | 0.0977 |
| 1000 | 0.1489 | - | - | 0.1396 |

### Speed / memory / stability
| | ref | repro500 | whiten500 | whiten+projMuon |
|---|---|---|---|---|
| s/step (mean, steps 11+) | 14.54 | 14.21 | 14.49 | 14.24 |
| tok/s (mean) | 72.1k | 73.8k | 72.4k | 73.7k |
| peak memory (allocated / reserved, MiB) | 64377 / 65458 | n/a | 63225 / 65460 | **57820 / 60044** |
| train time to step 500 / 1000 (s) | - / 14394 | 6964 / - | 7100 / - | ~7000 / 14093 |

* Stability: no NaN, no spikes (>4 sd of per-step scatter after step 30) in the projmuon run; max rise of the 10-step smoothed loss 0.007. The v2 ref has a jump at step ~197 (3.91 -> 4.09, smoothed rise 0.039) that is absent in repro500, whiten500 and this run. Per-step scatter std: ref 0.065, repro 0.090, whiten500 0.089, projmuon 0.066.
* First divergence: the two whiten runs share seed/data and differ only by the projection flag; their 10-step smoothed losses differ by >0.01 from step 13 (>0.05 from step 19). Against the AdamW runs they differ from the very first steps (the whiten table init/step already differs: window 1-10 is 9.89 vs 9.84), so "first divergence step" vs AdamW is not informative (step ~10).
* Projmuon is below whiten500 in 100% of smoothed steps 10-500, below repro500 in 98%, below ref in 94%.

## Verdict (honest)
Muon-everywhere (whiten tables + fp8 Muon projections) is **modestly better than every baseline in the first 1000 steps, but the margin is small and shrinking**: val/bpb -0.0053 / -0.0079 / -0.0062 vs the AdamW ref at 500 / 750 / 1000 (-0.0054 vs repro500 and -0.0039 vs whiten500 at 500); the loss gap vs repro500 is -0.044 at ~250 and only -0.013 at ~500, and vs the ref the window means cross zero around step 750 (+0.013) and are -0.008 at 1000. The larger gaps against the v2 ref in 91-250 are mostly the ref's early spike at ~200 (repro500 is itself 0.12 lower there), not an effect of Muon. The clearest gain is early (steps ~50-300: loss -0.09..-0.14 at 91-100 and val/bpb -0.013 at 250 vs repro500), which is consistent with faster initial progress but not evidence of a better asymptote.
CORE is flat to worse: 0.0977 at 500 (equal to AdamW within noise, above whiten500) but **0.1396 at 1000 vs 0.1489** for the ref (-0.009), despite lower bpb; CORE at this scale is noisy and not monotone with bpb, so it does not support a gain.
Efficiency: speed equal to the repro (14.24 vs 14.21 s/step; ~2% faster than ref/whiten500 runs) and peak memory ~5.5-6.5 GiB lower (57.8 vs 63-64 GiB), plausibly from fp8 projections + no AdamW state for the projection matrices.
Caveats: single seed (s0); the ref's early spike inflates the ref comparison for steps ~200-300 (use repro500 for steps <=500); **the fp8-projection + Muon change is confounded with the whitened tables** (whiten500 had fp32-island AdamW projections), so the ~0.004 bpb gain vs whiten500 at 500 cannot be attributed to Muon vs fp8; control B (AdamW tables + `--lut-ffn-proj-fp8-muon`) is still needed to separate them. Differences of ~0.005 bpb are of the order of seed-to-seed noise (repro vs ref differ by 0.006 bpb at 500 for unrelated reasons). Conclusion: no evidence of a decisive win; at best a small, possibly transient gain with lower memory.

## Checkpoints (worker only, not committed)
`/home/lab-ds-h100-anatoly/work/nanochat_base/base_checkpoints/d24_lutffn_confn1_1xh100_whiten_projmuon1000/` : `model_`, `optim_*_rank0`, `meta_` for steps 000250, 000500, 000750, 001000 (all verified present); also visible via `checkpoints/` in this results dir (git-ignored).
