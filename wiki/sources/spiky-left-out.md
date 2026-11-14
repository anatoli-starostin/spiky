# Sources checklist: reviewed-but-left-out research branches (branch line + main docs only)

Generated 2026-11-04 by a script from `anatoli-starostin/spiky` (read-only). Refs: `research/neurodarwinism` @ `5ebb7b1a`, `research/lut-to-spiking` @ `15235ef5`, `research/mdn_gaussian_unembedder` @ `d6a6ec49`, `research/lookupffn` @ `93de410d`, `research/lac_cell_stationary` @ `672344d6`, `research/drawer-memory` @ `bfaa2b63`.

Line format: `- [ ] [path](url) — what it is; goes to: <Primitives|Optimisation|Images|LM small|LM NanoChat|Spikes|undecided|left out>`. Nothing is ticked. Folder links use /tree/, files /blob/. A grouped line (`path (+N more)`) stands for N files of one module group, listed in brackets. Family lines (`prefix*`) stand for many sibling experiment folders (count and 3 examples given). Branch-only paths are linked on their branch, not on main. `MODIFIED vs main` = file exists on main and the branch changes it. 'goes to' is a guess from names and the ideas map, to be confirmed.

Items in this file: **22**.

---
- [ ] [research/neurodarwinism (branch)](https://github.com/anatoli-starostin/spiky/tree/research/neurodarwinism) — evolving spiking networks (neurodarwinism) incl. exp001-012, left out; goes to: left out
- [ ] [experiments/evolution/ideas.md](https://github.com/anatoli-starostin/spiky/blob/research/neurodarwinism/experiments/evolution/ideas.md) — Evolution ideas: Hypothesis: evolution may PRESERVE COMPOSITIONALITY; goes to: left out
- [ ] [experiments/evolution/spnet_growth_notes.md](https://github.com/anatoli-starostin/spiky/blob/research/neurodarwinism/experiments/evolution/spnet_growth_notes.md) — Population-in-one-SPNet - notes for the Mode-2 harness; goes to: left out
- [ ] [experiments/neurodarwinism/README.md](https://github.com/anatoli-starostin/spiky/blob/research/neurodarwinism/experiments/neurodarwinism/README.md) — neurodarwinism: Research chapter: can a spiking (LIF) network be evolved - not; goes to: left out
- [ ] [research/lut-to-spiking (branch)](https://github.com/anatoli-starostin/spiky/tree/research/lut-to-spiking) — LUT to spiking-network conversion and walker2d RSNN distill, left out; goes to: left out
- [ ] [doc/research/lut_to_spiking.md](https://github.com/anatoli-starostin/spiky/blob/research/lut-to-spiking/doc/research/lut_to_spiking.md) — Initializing spiking networks from trained LUT tables - construction, cost, and a; goes to: left out
- [ ] [experiments/lut_to_spiking/README.md](https://github.com/anatoli-starostin/spiky/blob/research/lut-to-spiking/experiments/lut_to_spiking/README.md) — branch `research/lut-to-spiking` · implementer: gpustar (RTX 5090); goes to: left out
- [ ] [experiments/walker2d_lut_distill/README.md](https://github.com/anatoli-starostin/spiky/blob/research/lut-to-spiking/experiments/walker2d_lut_distill/README.md) — Walker2d int4-LUT → recurrent E/I spiking net (surrogate-gradient distillation); goes to: left out
- [ ] [research/mdn_gaussian_unembedder (branch)](https://github.com/anatoli-starostin/spiky/tree/research/mdn_gaussian_unembedder) — MDN Gaussian unembedder (shares hyperplane_ffn exps), left out; goes to: left out
- [ ] [experiments/hyperplane_ffn/SWEEP_ffn_lut.md](https://github.com/anatoli-starostin/spiky/blob/research/mdn_gaussian_unembedder/experiments/hyperplane_ffn/SWEEP_ffn_lut.md) — FFN-slot LUT sweep (exp036, exp042, exp043-exp069); goes to: left out
- [x] [research/lookupffn (branch)](https://github.com/anatoli-starostin/spiky/tree/research/lookupffn) — LookupFFN module prototype, left out; goes to: left out
- [x] [research/lookupffn/README.md](https://github.com/anatoli-starostin/spiky/blob/research/lookupffn/research/lookupffn/README.md) — A clean, heavily-commented, pure-PyTorch reference implementation of the LookupFFN; goes to: left out
- [ ] [research/lac_cell_stationary (branch)](https://github.com/anatoli-starostin/spiky/tree/research/lac_cell_stationary) — LAC cell-stationary kernel research (also pcalm_lut commits), left out; goes to: left out
- [ ] [research/lac_cell_stationary/ONEHOT_GEMM.md](https://github.com/anatoli-starostin/spiky/blob/research/lac_cell_stationary/research/lac_cell_stationary/ONEHOT_GEMM.md) — One-hot GEMM at the study shape - does the old H100 "14-22×; goes to: left out
- [ ] [research/lac_cell_stationary/README.md](https://github.com/anatoli-starostin/spiky/blob/research/lac_cell_stationary/research/lac_cell_stationary/README.md) — LUT gather dataflows on an RTX 5090 - five arms against the; goes to: left out
- [ ] [research/pcalm_lut/DERIVATION.md](https://github.com/anatoli-starostin/spiky/blob/research/lac_cell_stationary/research/pcalm_lut/DERIVATION.md) — PC (A) and PC-ALM (B) over a stack of paired forward/backward LightMHL; goes to: left out
- [ ] [research/pcalm_lut/DERIVATION_v2.md](https://github.com/anatoli-starostin/spiky/blob/research/lac_cell_stationary/research/pcalm_lut/DERIVATION_v2.md) — PC (A, symmetric) and PC-ALM (B, constraint) over paired forward/backward LightMHL; goes to: left out
- [ ] [research/pcalm_lut/VAE_METRICS.md](https://github.com/anatoli-starostin/spiky/blob/research/lac_cell_stationary/research/pcalm_lut/VAE_METRICS.md) — What the VAE literature reports on Fashion-MNIST, and whether our MSE can; goes to: left out
- [ ] [research/drawer-memory (branch)](https://github.com/anatoli-starostin/spiky/tree/research/drawer-memory) — drawer memory (out of scope), left out; goes to: left out
- [ ] [experiments/drawer_memory/runs/PROGRESS.md (+18 more)](https://github.com/anatoli-starostin/spiky/blob/research/drawer-memory/experiments/drawer_memory/runs/PROGRESS.md) — progress logs of drawer-memory runs [19 files: PROGRESS*.md]; goes to: left out
- [ ] [experiments/drawer_memory/README.md](https://github.com/anatoli-starostin/spiky/blob/research/drawer-memory/experiments/drawer_memory/README.md) — drawer_memory: (anchor-pair sign bits → table cells) - as a replacement for; goes to: left out
- [ ] [experiments/drawer_memory/report/drawer_memory_report.pdf](https://github.com/anatoli-starostin/spiky/blob/research/drawer-memory/experiments/drawer_memory/report/drawer_memory_report.pdf) — drawer memory report.pdf; goes to: left out
