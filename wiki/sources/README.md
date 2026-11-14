# Detailed sources checklists (item level)

Generated 2026-11-04 by a script (not included in the repository). Main @ `3ac72cca`.

| file | covers | items |
|---|---|---|
| [spiky-main.md](spiky-main.md) | spiky main (`src/spiky/{lut_fused,lutorch,lutorch_ex}`, doc, workbooks, examples, experiments, claude) | 215 |
| [spiky-research-lut_nanochat.md](spiky-research-lut_nanochat.md) | `research/lut_nanochat`: what it adds over main (nanochat integration line) | 69 |
| [spiky-research-fused-ablations.md](spiky-research-fused-ablations.md) | `research/lutorch_ex_fused_ablations` and `research/dominance_stack_ablation` | 26 |
| [spiky-research-ffn_replacement_fix.md](spiky-research-ffn_replacement_fix.md) | `research/ffn_replacement_fix` (fixed eval, runs_corrected, lut_ablation, distill, old lutorch line) | 116 |
| [spiky-research-hyperplane_ffn_next.md](spiky-research-hyperplane_ffn_next.md) | `research/hyperplane_ffn_next` (experiments/hyperplane_ffn, ~340 folders) | 37 |
| [spiky-research-other.md](spiky-research-other.md) | `research/lut_distillation`, `exp/lif-mhl-experiments`, `research/lut-trm-sudoku`, `research/recursive-reasoning-models` | 28 |
| [spiky-research-walker2d.md](spiky-research-walker2d.md) | `research/walker2d-lut`, `live/walker2d-viz`, `sync/walker2d-viz-from-live` | 25 |
| [spiky-left-out.md](spiky-left-out.md) | reviewed-but-left-out research branches (branch line + main docs only) | 22 |
| [calibrate-branch.md](calibrate-branch.md) | `feature/lutorch-calibrate-output-normalize-weights` (memory bank, notes, papers, experiments, lutorch modules, docs; duplicates marked) | 471 |
| **total** | | **1009** |

Paths verified against their branch trees at generation: all present.

How to use: a box is ticked in the pull request that lands that item's content in the wiki (the pull request is named after the line). 'goes to' is a name-based guess, to be confirmed per item; `dup of` marks copies of a canonical file. The generating scripts are not part of the repository; edit the lists by hand.

## Items by 'goes to' guess (spiky main + research files only)

| section | items |
|---|---|
| LM small | 203 |
| Primitives | 151 |
| Spikes | 62 |
| left out | 52 |
| LM NanoChat | 38 |
| Optimisation | 23 |
| undecided | 7 |
| Images | 2 |

## Skipped on purpose

- Build artifacts, binaries, checkpoints, logs, images, `.npz`/`.pt`/`.csv`/`.json` data files: not listed (folders that contain them are listed as folders; plots/figures only as one group line where useful).
- Paired `workbooks/*.py` twins of the notebooks; `.png` diagrams under `doc/` (not listed individually).
- Vendored: `experiments/lut_nanochat/nanochat/` = karpathy/nanochat @ 92d63d4 (one line; only the files tagged `[lut_nanochat]` and LUT-specific scripts are listed).
- Not in scope for this part: `src/spiky/spnet`, `src/spiky/util`, `native/spiky/{spnet,misc,synapse_growth,connections_manager,torch_utils}`, `native/common` (libraries outside the three named modules), `landing/` internals (one line each), `.github`.
- On main, `experiments/ffn_replacement/runs/` (23 folders) is listed one line per folder (<= 25 siblings); everything inside run folders (configs, metrics, logs) is covered by the folder line.
- Branch diffs are taken against main (three-dot, minus blobs identical to current main or to a named sibling branch): `research/dominance_stack_ablation` content shared with `research/lutorch_ex_fused_ablations` is listed once; `research/walker2d-lut` excludes files identical to `research/neurodarwinism` / `research/lut-to-spiking`; `research/mdn_gaussian_unembedder` excludes `research/hyperplane_ffn_next` content; `research/lac_cell_stationary` excludes `research/ffn_replacement_fix` content.
- `live/walker2d-viz`: the deletions of `landing/walker2d-viz/server` actor files relative to main are not listed. `sync/walker2d-viz-from-live` is already merged into main (no own files).
- Docs inside experiment run folders (README/RESULT of each run, exp_c* READMEs on walker2d, exp0NN READMEs on hyperplane) are covered by the folder or family line, except analysis/summary docs with their own checkbox.
- Left-out branches: PROGRESS logs of `research/drawer-memory` grouped into one line; `.txt` artifacts, run result folders and per-experiment READMEs deeper than 3 path levels omitted.
- Secrets: no file contents are copied; descriptions are first headings or docstring lines only. 

## Calibrate branch (471 items)

Groups: (a) memory banks 96, (b) notes 126, (c) papers 16, (d) experiments 127 (numbered folders grouped into families), (e) lutorch and other src 50, (f) docs, notebooks, scripts, skills/commands/hooks 57. 115 lines are marked `dup of` (the export holds each doc up to three times). Skipped: data files (json, csv, logs, images), caches, licence and ignore files; per-experiment `train.py` copies are covered by the family lines.
