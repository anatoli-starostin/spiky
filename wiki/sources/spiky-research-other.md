# Sources checklist: `research/lut_distillation`, `exp/lif-mhl-experiments`, `research/lut-trm-sudoku`, `research/recursive-reasoning-models`

Generated 2026-11-04 by a script from `anatoli-starostin/spiky` (read-only). Refs: `research/lut_distillation` @ `5ea5f4b5`, `exp/lif-mhl-experiments` @ `ccd0616c`, `research/lut-trm-sudoku` @ `2175d690`, `research/recursive-reasoning-models` @ `7d156bfe`.

Line format: `- [ ] [path](url) — what it is; goes to: <Primitives|Optimisation|Images|LM small|LM NanoChat|Spikes|undecided|left out>`. Nothing is ticked. Folder links use /tree/, files /blob/. A grouped line (`path (+N more)`) stands for N files of one module group, listed in brackets. Family lines (`prefix*`) stand for many sibling experiment folders (count and 3 examples given). Branch-only paths are linked on their branch, not on main. `MODIFIED vs main` = file exists on main and the branch changes it. 'goes to' is a guess from names and the ideas map, to be confirmed.

Items in this file: **28**.

---

## A. research/lut_distillation (no README on branch; diagnostics + one distillation run)

- [ ] [experiments/lut_distillation/diag_eval_protocol/](https://github.com/anatoli-starostin/spiky/tree/research/lut_distillation/experiments/lut_distillation/diag_eval_protocol) — diagnostics of eval protocol: val difficulty, overlap, head packing (14 files); goes to: LM small
- [ ] [experiments/lut_distillation/exp_block3_h4c48_nap7_tph128/](https://github.com/anatoli-starostin/spiky/tree/research/lut_distillation/experiments/lut_distillation/exp_block3_h4c48_nap7_tph128) — block-3 FFN distillation run (H4, nap7, tph128) (5 files); goes to: LM small
- [ ] [experiments/lut_distillation/diag_eval_protocol/eval_windows.py (+6 more)](https://github.com/anatoli-starostin/spiky/blob/research/lut_distillation/experiments/lut_distillation/diag_eval_protocol/eval_windows.py) — scripts: eval windows, head anatomy/packing, protocol matrix, val holdout [7 files: eval_windows.py, protocol_matrix.py, val_overlap_scored.py]; goes to: LM small

## B. exp/lif-mhl-experiments (LIF-detector multi-head LUTs, walker2d distillation)

- [x] [experiments/lif_detectors_mhl/README.md](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/README.md) — LIFDetectorsMHL - LIF-detector front-end for HyperplaneMultiHeadLUT; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/RESULTS_product_bucket.md](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/RESULTS_product_bucket.md) — ProductBucketLIFMHL - mixed-radix product-of-detectors sweep; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/mhl_detectors_math.tex](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/mhl_detectors_math.tex) — Multi-Head LUT Detectors:\\ Anchor-Pairs, Hyperplane, and LIF; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/mhl_detectors_math.pdf](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/mhl_detectors_math.pdf) — mhl detectors math; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/competitive_bucket_lif_mhl.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/competitive_bucket_lif_mhl.py) — CompetitiveBucketLIFMHL - a lateral-inhibition (winner-take-all) population variant of the bucket LIF MHL; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/distill_walker2d.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/distill_walker2d.py) — Repro: distill the real trained Walker2d int4 LUT actor into LIFDetectorsMHL (output-action; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/distill_walker2d_bucket.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/distill_walker2d_bucket.py) — Distill the real Walker2d int4 LUT actor into BucketLIFDetectorsMHL (M=16 time buckets); goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/distill_walker2d_competitive.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/distill_walker2d_competitive.py) — Distill the frozen int4 Walker2d LUT actor into CompetitiveBucketLIFMHL (post-hoc column WTA); goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/distill_walker2d_product.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/distill_walker2d_product.py) — Distill the frozen int4 Walker2d LUT actor into ProductBucketLIFMHL (mixed-radix product of; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/distill_walker2d_rollout.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/distill_walker2d_rollout.py) — Distill the frozen int4 Walker2d LUT actor into RolloutLIFGroupsMHL (genuine time-stepped LIF; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/distill_walker2d_temporal.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/distill_walker2d_temporal.py) — Distill the frozen int4 Walker2d LUT actor into TemporalInhibitionBucketLIFMHL (causal WTA scan); goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/distill_walker2d_ttfs.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/distill_walker2d_ttfs.py) — Distill the real Walker2d int4 LUT actor into PureLIFDetectorsMHL (TTFS variant). Same; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/lif_detectors_mhl.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/lif_detectors_mhl.py) — LIFDetectorsMHL - a LIF-detector drop-in for the HyperplaneMultiHeadLUT index front-end; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/pure_lif_detectors_mhl.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/pure_lif_detectors_mhl.py) — PureLIFDetectorsMHL - a pure time-to-first-spike (TTFS) LIF-detector front-end for the multi-head LUT; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/rollout_lif_groups_mhl.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/rollout_lif_groups_mhl.py) — RolloutLIFGroupsMHL - a genuine 2-layer time-stepped spiking net (no bucket/table addressing); goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/temporal_inhibition_bucket_lif_mhl.py](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/temporal_inhibition_bucket_lif_mhl.py) — TemporalInhibitionBucketLIFMHL - causal (recurrent) lateral-inhibition bucket-LIF population; goes to: Spikes
- [ ] [experiments/lif_detectors_mhl/test_competitive_bucket_lif_mhl.py (+4 more)](https://github.com/anatoli-starostin/spiky/blob/exp/lif-mhl-experiments/experiments/lif_detectors_mhl/test_competitive_bucket_lif_mhl.py) — tests for LIF detector MHL variants [5 files: test_competitive_bucket_lif_mhl.py, test_pure_lif_detectors_mhl.py, test_temporal_inhibition_bucket_lif_mhl.py]; goes to: Spikes

## C. research/lut-trm-sudoku (Tiny Recursive Model on sudoku, baselines for LUT variants)

- [x] [doc/research/hrm.md](https://github.com/anatoli-starostin/spiky/blob/research/lut-trm-sudoku/doc/research/hrm.md) — Hierarchical Reasoning Model (HRM) - notes, and how it might combine with; goes to: LM small
- [x] [doc/research/trm.md](https://github.com/anatoli-starostin/spiky/blob/research/lut-trm-sudoku/doc/research/trm.md) — Tiny Recursive Models (TRM) - literature review and LUT integration ideas; goes to: LM small
- [ ] [doc/research/figures/trm_architecture.tex](https://github.com/anatoli-starostin/spiky/blob/research/lut-trm-sudoku/doc/research/figures/trm_architecture.tex) — trm architecture; goes to: LM small
- [ ] [doc/research/figures/trm_architecture.pdf](https://github.com/anatoli-starostin/spiky/blob/research/lut-trm-sudoku/doc/research/figures/trm_architecture.pdf) — trm architecture; goes to: LM small
- [ ] [experiments/lut_trm_sudoku/exp_a01_stock_trm_sudoku_baseline/](https://github.com/anatoli-starostin/spiky/tree/research/lut-trm-sudoku/experiments/lut_trm_sudoku/exp_a01_stock_trm_sudoku_baseline) — stock TRM sudoku baseline (setup, launch, config) (5 files); goes to: LM small
- [ ] [experiments/lut_trm_sudoku/exp_a02_fast_baseline/](https://github.com/anatoli-starostin/spiky/tree/research/lut-trm-sudoku/experiments/lut_trm_sudoku/exp_a02_fast_baseline) — fast TRM sudoku baseline with subsample test (6 files); goes to: LM small

## D. research/recursive-reasoning-models (reading notes only)

- [x] [doc/research/hrm.md](https://github.com/anatoli-starostin/spiky/blob/research/recursive-reasoning-models/doc/research/hrm.md) — notes on Hierarchical Reasoning Model (HRM); goes to: LM small
- [x] [doc/research/trm.md](https://github.com/anatoli-starostin/spiky/blob/research/recursive-reasoning-models/doc/research/trm.md) — notes on Tiny Recursive Model (TRM); goes to: LM small
