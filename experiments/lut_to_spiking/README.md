# lut_to_spiking — building spiking circuits from trained LUT tables

*Tracking issue: [#74](https://github.com/anatoli-starostin/spiky/issues/74) ·
branch `research/lut-to-spiking` · implementer: gpustar (RTX 5090).
Methodology: [`claude/experiment-methodology.md`](../../claude/experiment-methodology.md).*

**The findings live in [`doc/research/lut_to_spiking.md`](../../doc/research/lut_to_spiking.md).**
This README is the map of the code.

## Status

Roadmap step 1 of #74 (**"can we build a population of spiking neurons that acts
according to one table?"**) is **done, and validated on a real trained model with real
data: 100.00% exact**. Steps 2–3 (a whole MHL, a whole transformer) are **not** started —
the open blocker is the `sum` over `tables_per_head`, which a latency code cannot express.

## Layout note

Unlike `hyperplane_ffn/` and `lut_trm_sudoku/`, this chapter is **flat rather than one
folder per experiment**. It is not a series of training runs — it is one instrumented
probe series (`t01` … `t13`) over a shared harness, and the scripts import each other
(`snn_harness`, `table_io`, `paths`, `exp025_model`). Splitting them into per-experiment
folders would break those imports for no gain. Each script is self-contained, prints its
own numbers, and can be run on its own.

## Shared modules

| file | what it is |
|---|---|
| `paths.py` | path resolution; `EXP025_CKPT` / `NANOCHAT_ROOT` / `EXP011_CKPT` are env-overridable |
| `snn_harness.py` | hand-authored `ChunkOfConnections`, one `SynapseMeta` per `(weight, delay)`, latency injection via `sparse_input`, first-spike readout |
| `table_io.py` | loads a real trained LUT table and gives it latency semantics |
| `exp025_model.py` | reconstructs the exp025 model from its checkpoint (acceptance test: val bpb 1.2409 vs recorded 1.2408) |

## The probe series

**Part 1 — the primitive** (no checkpoint needed)

| script | question | answer |
|---|---|---|
| `t01_calibrate.py` | does `out = in + delay` hold exactly? | yes, `t_in + delay + 1`, all delays |
| `t02_physics.py` | threshold-k, integration window, inhibition | window is 1–3 ticks; inhibition 1 tick wide; over-inhibition *fires* the neuron |
| `t03_window_latch.py` | can a latch hold a veto? | yes but ragged and costly |
| `t04_if_neuron.py` | **can the engine be made leak-free?** | **yes** — `cf_2=cf_1=cf_0=a=b=d=0`; comparator = 1 neuron / 2 synapses / ≤1 spike |

**Part 2 — the cost of one table** (needs the exp011 checkpoint)

| script | question | answer |
|---|---|---|
| `t05_families.py` | which min-plus families can approximate a table? | none well |
| `t06_costfidelity.py` | cost/fidelity frontier + **random-table control** | trained ≈ random ⇒ incompressible |
| `t07_spnet_table.py` | build both constructions in SPNet | exact: 100%; fitted: 59% |
| `t08_headlevel.py` | does per-table error survive the 256-table sum? | no — cheap circuit ≈ ignoring the input |
| `t09_figure.py` | → `lut2spiking_costfidelity.png` | |

**Part 3 — a real table on real data** (needs the exp025 checkpoint + a nanochat checkout)

| script | question | answer |
|---|---|---|
| `t10_realdata.py` | acceptance test + capture real inputs at layer 3 `out_proj` | val bpb 1.2409; `X = [8192, 384]` |
| `t11_real_table_spiking.py` | **does the spiking circuit reproduce the real table on real inputs?** | **100.00%** (order-coded input) |
| `t12_bpb_resolution.py` | what does coarse timing cost the whole model? | 64 ticks (6 bit) → **+0.3 mb** |
| `t13_figure_real.py` | → `real_table_spiking.png` | |

## Not in git

Checkpoints and the 113 MB activation capture `real_capture_layer3.pt` are excluded by
`.gitignore` (`experiments/**/*.pt`). The capture regenerates from `t10_realdata.py` in
about two minutes. Result JSONs (`t06/t07/t11_results.json`) and both figures **are**
committed.
