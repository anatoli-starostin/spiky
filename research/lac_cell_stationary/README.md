# Cell-stationary LUT inference kernels on an RTX 5090

Does a **cell-stationary** LUT kernel — one where a thread owns table cells, holds their
values in registers for the life of the kernel, and lets the whole batch stream past —
beat the **output-stationary gather** the repo already uses?

Short answer: **no, by two orders of magnitude, and the reason is not the one the design
predicts.** The design's premise is that the table read is the expensive part, so making
the tables stationary and paying extra arithmetic is a good trade. On this GPU the table
read is nearly free, because the 64 MiB table set fits inside the 96 MB L2; what the
cell-stationary form adds — a compare per (thread, token) over a grid of
`n_t x R x n/K` threads, and a cross-thread reduction to put the per-table partials back
together — is what costs. The hardware argument for the systolic LAC is not refuted by
this; what is refuted is that a GPU can stand in for it.

Everything here is inference only, top-1 only, int8 tables, fp32 accumulation. Nothing was
retrained and nothing left the machine.

## Contents

| file | what it is |
|---|---|
| `lac_kernels.cu` | all five kernels: the `gather` baseline and `cs_v0`/`cs_v1`/`cs_v2`/`cs_v3`, plus the two ablation kernels used to attribute runtime |
| `lac.py` | build (JIT, `~/.cache`), python wrappers, the torch reference, burn-in + CUDA-event timing |
| `probe_model.py` | step 0: print the real LUT shape in the checkpoint instead of assuming it |
| `extract_real.py` | one real forward; saves the layer's real top-1 indices, real margin coefficients and int8-quantised tables |
| `index_stats.py` | the index statistics: per-table utilisation, cross-table hot-row alignment, empirical intra-thread dedup vs the birthday bound, distinct rows per token tile |
| `smoke.py`, `smoke2.py` | correctness gate for every compiled configuration |
| `diag_attrib.py` | runtime attribution by ablation (scan / reduction / flush) |
| `bench.py` | the sweep and the per-B ladder for both shapes and both coefficient modes |
| `bench_l2.py` | the L2 amortisation measurement, done without ncu |
| `ptxas_table.py` | registers / spills / SMEM per compiled kernel, from the build log |
| `plot_bench.py` | the figures |

## Reproducing

```sh
cd research/lac_cell_stationary
python probe_model.py        # what shape is the real layer
python extract_real.py       # -> artifacts/real_layer.pt
python index_stats.py        # -> artifacts/index_stats.json
python smoke.py && python smoke2.py     # correctness, all configurations
python diag_attrib.py        # runtime attribution
python bench.py              # -> artifacts/bench.json
python bench_l2.py           # -> artifacts/l2.json
python plot_bench.py         # -> figs/
```

<!-- RESULTS -->
