# lutorch_ex report

A self-contained report on the `lutorch_ex` library: (1) tables addressed by an LSH, (2) the ProjectionLUT
wrapper, (3) cartridges — one contract, three generations — with their math, usage, quantisation and the
deployment path, (4) shared machinery, (5) the 48k seven-configuration sweep, (6) CPU and GPU benchmarks,
(7) open items, plus a one-page reference appendix. **Draft v2 (written from scratch 2026-10-05); iterated
with Anatoli.** The previous draft is kept as `sources/draft_v1.tex` for reference.

## Build

```
make          # build lutorch_ex_report.pdf (latexmk -pdf, multi-pass)
make clean    # remove aux + PDF
```

The compiled **`lutorch_ex_report.pdf` is committed alongside the source** and must be rebuilt and
re-committed on every `.tex` change, so the PDF always tracks the `.tex`.

## Generated inputs

- `fig_curves.pdf` ← `make_fig_curves.py`: the seven `val_bpb` learning curves, read from
  `experiments/lutorch_ex/*/metrics.csv` (both the eval-only and the full-history CSV layouts).
- `bench_cpu.json`, `bench_cuda.json` ← `bench_report.py`: the benchmark table of §6, measured with the
  library's own `spiky.lutorch_ex.bench` harness (`--device cpu|cuda`). Re-run on the target hardware;
  the JSON records device name, torch version, batch sizes, warmup and repeats.

Both scripts run with the repo venv: `../../../.venv/bin/python <script>`.

## Sources

Everything the report draws on is vendored under `sources/` (see `sources/README.md`): the Spiking
Manifesto, the LUT-mechanisms note, the ablation table, the quantisation note. Library code:
`src/spiky/lutorch_ex/`.
