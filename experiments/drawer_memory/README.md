# drawer_memory

**Drawer memory** — a hash-addressed associative memory written at inference, using spiky's LUT addressing
(anchor-pair sign bits → table cells) — as a replacement for softmax attention. Proof of concept on Apple M1/M4
laptops (CPU/MPS), small models (2–4 layers, ≤6M parameters).

- **Report:** [`report/drawer_memory_report.html`](report/drawer_memory_report.html) (PDF alongside) — abstract;
  1 data and tasks; 2 model and training (what is trained, how bits per byte are measured); 3 key findings;
  4 every batch of experiments with learning curves and tables. Built from the run logs by `build_report.py`.
- **Plain-language explainer:** [`report/explainer.html`](report/explainer.html) (drawers = cells, cabinets = tables).

## Mechanism (one paragraph)

Per head, a shared projection gives each token a 32-dim address vector; each of `tph` tables compares `nap` fixed
random coordinate pairs, and the sign bits select one of `2**nap` cells. Every token adds its value vector (and a
count) into its cells; every token reads its own cells and divides the summed values by the summed counts. This is
causal normalised linear attention whose feature map is a one-hot per table; training uses spiky's hard forward with a
soft full-cell surrogate gradient. Per token, a fixed number of rows is touched instead of the full state.
Variants: neighbour cell, confidence score, soft-to-hard address annealing, bounded soft read (top-4 or flip-2 cells per
table), decay / EMA writes, local-window hybrids, normalised + gated read, delta writes (`lut_delta`, with DeltaNet
`linear_delta` as its dense baseline), large tables with a coarse-to-fine bit curriculum — see the report.

## Files

| file | role |
|---|---|
| `models.py` | mixers: softmax (+RoPE), linear (elu+1), drawer memory (`LutMemMixer`) and variants, local window, hybrids |
| `train.py` | synthetic associative recall (MQAR, optionally with gaps or key updates); one run per call |
| `train_lm.py` | character/byte-level language modelling on Tiny Shakespeare or enwik8 |
| `run_grid.py` | named grids of runs (`pilot` … `e8confirm`), run sequentially with a thermal cool-down |
| `diag.py` | address diagnostics on saved checkpoints (store/read agreement, collisions, entropy, bit statistics) |
| `bench_decode.py`, `bench_cost.py` | measured per-token decode cost (recurrent form, CPU) |
| `cap_sim.py` | capacity at equal cost per token, without training (random keys, noisy queries) |
| `prof_delta.py` | training-time profile of the delta mixers |
| `thermal.py` | CPU temperature logging via `macmon` (Apple silicon) |
| `build_report.py`, `report_notes.json` | builds the HTML report: data from `runs/`, interpretation from the notes |
| `runs/<run>/` | `config.json`, `metrics.csv`, `summary.json` (+ `diag.csv`, `sample.txt`) per run; checkpoints are not in git |

## Reproduce

```bash
pip install torch numpy                      # PyTorch 2.14.1 was used
# Tiny Shakespeare -> data/tinyshakespeare.txt (see spiky.util.tiny_shakespeare.TINYSHAKESPEARE_URL)
# enwik8 -> data/enwik8/enwik8 (unzip http://mattmahoney.net/dc/enwik8.zip)
python train.py --mixer lut_hard_tied --n_pairs 32 --feat 256 --nap 4 --seed 0
python train_lm.py --mixer lut_hard_tied --nap 6 --bitflip 2 --corpus enwik8 --steps 1500 --name demo
python run_grid.py night                     # any grid name defined in run_grid.py
python bench_cost.py && python build_report.py
```

`thermal.py` needs `macmon` (`brew install macmon`); on other machines remove the `Thermal` calls or stub them.
