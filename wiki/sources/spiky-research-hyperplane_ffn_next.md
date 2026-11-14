# Sources checklist: `research/hyperplane_ffn_next` (experiments/hyperplane_ffn, ~340 folders)

Generated 2026-11-04 by a script from `anatoli-starostin/spiky` (read-only). Refs: `research/hyperplane_ffn_next` @ `0c35e3be`.

Line format: `- [ ] [path](url) — what it is; goes to: <Primitives|Optimisation|Images|LM small|LM NanoChat|Spikes|undecided|left out>`. Nothing is ticked. Folder links use /tree/, files /blob/. A grouped line (`path (+N more)`) stands for N files of one module group, listed in brackets. Family lines (`prefix*`) stand for many sibling experiment folders (count and 3 examples given). Branch-only paths are linked on their branch, not on main. `MODIFIED vs main` = file exists on main and the branch changes it. 'goes to' is a guess from names and the ideas map, to be confirmed.

Items in this file: **37**.

---

## 1. Top-level summaries and notes (experiments/hyperplane_ffn)

- [ ] [experiments/hyperplane_ffn/BACKLOG.md](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/experiments/hyperplane_ffn/BACKLOG.md) — Ideas captured but not built or queued. Nothing here is running. Each; goes to: LM small
- [ ] [experiments/hyperplane_ffn/FFN_GRID_SUMMARY.md](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/experiments/hyperplane_ffn/FFN_GRID_SUMMARY.md) — FFN-LUT paper grid - consolidated results (untied-vanilla zero-line); goes to: LM small
- [ ] [experiments/hyperplane_ffn/FFN_SWEEP_SUMMARY.md](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/experiments/hyperplane_ffn/FFN_SWEEP_SUMMARY.md) — FFN-LUT cost/quality sweep - results (task f551912a); goes to: LM small
- [ ] [experiments/hyperplane_ffn/INNER48_SWEEP_SUMMARY.md](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/experiments/hyperplane_ffn/INNER48_SWEEP_SUMMARY.md) — inner=48 v_lut sweep - results (task ae659700); goes to: LM small
- [ ] [experiments/hyperplane_ffn/SPARSITY_SWEEP_SUMMARY.md](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/experiments/hyperplane_ffn/SPARSITY_SWEEP_SUMMARY.md) — Sparsity sweep - where routing sparsity stops being free, and how best; goes to: LM small
- [ ] [experiments/hyperplane_ffn/SWEEP_ffn_lut.md](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/experiments/hyperplane_ffn/SWEEP_ffn_lut.md) — FFN-slot LUT sweep (exp036, exp042, exp043-exp069); goes to: LM small
- [x] [experiments/hyperplane_ffn/TERNARY_16K_SWEEP.md](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/experiments/hyperplane_ffn/TERNARY_16K_SWEEP.md) — Ternary FFN - 16K star / one-axis-at-a-time sweep (for the paper); goes to: LM small
- [ ] [experiments/hyperplane_ffn/VSWEEP_SUMMARY.md](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/experiments/hyperplane_ffn/VSWEEP_SUMMARY.md) — v_lut overnight sweep - results & conclusions; goes to: LM small
- [ ] [experiments/hyperplane_ffn/FFN_GRID_plots.png (+6 more)](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/experiments/hyperplane_ffn/FFN_GRID_plots.png) — summary plots: sweeps, frontier, top-k vs penalty, finale vs dense [7 files: FFN_GRID_plots.png, finale_vs_dense.png, vsweep_summary.png]; goes to: LM small
- [ ] [experiments/hyperplane_ffn/analysis/](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/experiments/hyperplane_ffn/analysis) — 144k three-way val-bpb comparison plot script and figures (3 files); goes to: LM small
- [ ] [experiments/hyperplane_ffn/benchmark/](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/experiments/hyperplane_ffn/benchmark) — FFN-slot inference benchmark harness and kernels (27 files); goes to: Primitives
- [ ] [experiments/hyperplane_ffn/mdn/](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/experiments/hyperplane_ffn/mdn) — MDN Gaussian unembedder prototype: backbone, unigram, sweeps, journal (13 files); goes to: LM small
- [ ] [experiments/hyperplane_ffn/probes/](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/experiments/hyperplane_ffn/probes) — probe: CompressionMHL autoencoder (1 files); goes to: LM small
- [ ] [experiments/hyperplane_ffn/tools/](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/experiments/hyperplane_ffn/tools) — FLOPs/bandwidth profiler and FFN grid plotting (2 files); goes to: LM small

## 2. Experiment folder families

- [ ] [experiments/hyperplane_ffn/exp0NN_*](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/experiments/hyperplane_ffn) — numbered hyperplane/CompressionMHL exps 010-107 (single-stream, tied, head sweeps) [97 folders, 694 files; e.g. exp010_hyperplane_full_model_16k, exp058_compressionmhl_B3_3h_64-64_nap6_g0_tied_4k, exp107_singlestream_manyheads_qktph32]; goes to: LM small
- [ ] [experiments/hyperplane_ffn/exp_n_NNNN_*](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/experiments/hyperplane_ffn) — n-series FFN-replacement runs (grid, long 48k/144k, ablations) [157 folders, 889 files; e.g. exp_n_0001_D5champ_H12_d32_16k, exp_n_0080_H8_d48_tph64_batched_bs1p5x_16k, exp_n_0159_long144k_from_0128]; goes to: LM small
- [ ] [experiments/hyperplane_ffn/exp_g_NNNN_*](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/experiments/hyperplane_ffn) — g-series runs (light MHL, ternary, hybrid, hyperplane-pure) [50 folders, 347 files; e.g. exp_g_0008_H16_d48_tph64_16k, exp_g_0030_exp_g_0029_ternary_4k, exp_g_0055_ternary_t32_inputonly_headdecomp_nap7_tph64]; goes to: LM small

## 3. doc/lutorch/ffn_replacement (paper drafts, ternary anchoring)

- [ ] [doc/lutorch/ffn_replacement/bench_combined.pdf](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/bench_combined.pdf) — figure: combined FFN-slot benchmark; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/ffn_replacement.pdf](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/ffn_replacement.pdf) — compiled draft of FFN-to-LUT paper; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/ffn_replacement.tex](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/ffn_replacement.tex) — draft of FFN-to-LUT paper (earlier version of main's paper); goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/fig16k.pdf](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/fig16k.pdf) — figure: 16k-step val bpb sweep; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/long_runs_fig.pdf](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/long_runs_fig.pdf) — figure: 48k long runs; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/sparsity_fig.pdf](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/sparsity_fig.pdf) — figure: routing sparsity sweep; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/superlong_fig.pdf](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/superlong_fig.pdf) — figure: 144k super-long runs; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/ternary_anchoring.pdf](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/ternary_anchoring.pdf) — compiled ternary anchoring note; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/ternary_anchoring.tex](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/ternary_anchoring.tex) — note: ternary anchoring for hyperplane LUT routing; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/ternary_sweep_fig.pdf](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/ternary_sweep_fig.pdf) — figure: ternary 16k sweep; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/make_bench_figs.py (+4 more)](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/make_bench_figs.py) — figure-making scripts for paper drafts [5 files: make_bench_figs.py, make_fig16k.py, make_superlong_fig.py]; goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/bench_data/](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/bench_data) — bench data (2 files); goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/long_run_data/](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/long_run_data) — long run data (4 files); goes to: LM small
- [ ] [doc/lutorch/ffn_replacement/superlong_data/](https://github.com/anatoli-starostin/spiky/tree/research/hyperplane_ffn_next/doc/lutorch/ffn_replacement/superlong_data) — superlong data (2 files); goes to: LM small

## 4. library changes (src/spiky/lutorch)

- [ ] [src/spiky/lutorch/compression_mhl.py](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/src/spiky/lutorch/compression_mhl.py) — MODIFIED vs main: CompressionMHL: compress, LUT, decompress bottleneck; goes to: Primitives
- [ ] [src/spiky/lutorch/fast_multi_head_lut.py](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/src/spiky/lutorch/fast_multi_head_lut.py) — MODIFIED vs main: FastMultiHeadLut - multi-head LUT primitive for LUTGPT; goes to: Primitives
- [ ] [src/spiky/lutorch/hyperplane_code_lut.py](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/src/spiky/lutorch/hyperplane_code_lut.py) — HyperplaneCodeLUT - a fully-soft, code-scoring hyperplane unembedder; goes to: Primitives
- [x] [src/spiky/lutorch/ternary_hyperplane_multi_head_lut.py](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/src/spiky/lutorch/ternary_hyperplane_multi_head_lut.py) — ternary-weight hyperplane multi-head LUT; goes to: Primitives
- [ ] [src/spiky/lutorch/tests/test_compression_mhl.py](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/src/spiky/lutorch/tests/test_compression_mhl.py) — MODIFIED vs main: Tests for CompressionMultiHeadLUT (CompressionMHL) - the compress / FastMHL; goes to: Primitives
- [ ] [src/spiky/lutorch/tests/test_hyperplane_code_lut.py](https://github.com/anatoli-starostin/spiky/blob/research/hyperplane_ffn_next/src/spiky/lutorch/tests/test_hyperplane_code_lut.py) — CPU tests for HyperplaneCodeLUT (the code-scoring hyperplane unembedder); goes to: Primitives
