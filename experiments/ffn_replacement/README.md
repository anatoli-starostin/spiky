# ffn_replacement

Everything supporting the paper **"Replacing Transformer Feed-Forward Layers with Lookup
Tables"** (Anatoly Starostin, DeepSpike Inc.) — and nothing else.

The paper replaces each transformer block's dense 4× MLP feed-forward layer with a
multi-table lookup layer: the input is reprojected into a small subspace, routed through
per-head tables by anchor-pair sign comparisons, and the selected rows are gathered,
summed and restored to full width by a learned output decompression.

Scope note: the ternary / quantized-hyperplane line of work is a **separate paper**
(`ternary_anchoring`) and is deliberately **not** here — not its experiments, not its
figures, not its kernel path. Nothing under this directory references it.

## Layout

| path | what |
|---|---|
| `paper/` | The paper: `ffn_replacement.tex`, the built `ffn_replacement.pdf`, its five figures, and the three figure generators. |
| `runs/` | The 23 experiment directories the paper cites. |
| `benchmark/` | The FFN-slot inference harness and CUDA kernels behind §8 (Table 6, Figure 5). |
| `benchmark/paper_timings/` | RTX 5090 timings: `results.json` + the phase-breakdown figure. |
| `benchmark/paper_timings_h100/` | The H100 counterpart. |
| `tools/` | `measure_flops_bandwidth.py` (the FLOPs/vBW columns) and a legacy grid plot. |
| `FFN_GRID_SUMMARY.md` | Consolidated 16k grid table (read the caveats at its top). |

## The 23 runs

Each directory holds `config.json`, `metrics.csv`, `summary.json`, `train.py` and
`loss.png`; most also carry `flops_bandwidth.txt`, and six carry their own summary `.md`.

**Baselines** — `exp_n_0135_untied_vanilla_baseline_16k` (the 1.20144 zero-line, §3) ·
`exp_n_0151_long48k_untied_vanilla` · `exp_n_0157_long144k_vanilla`

**§4 straightforward LUT** — `exp_n_0136_fastmhl_raw_H4_nap8_tph128`
(note: built from its `raw_nap`/`raw_tph`/`raw_n_heads` keys — R=512, k=8 — *not* the
inert `lut_*` keys also present in its config)

**§5 output compression** — `exp_n_0138_outcompress_only_H4_nap8_tph128`

**§6 input-reprojection sweep (Table 4)** — `exp_n_0084_untied_nheads4` ·
`exp_n_0118_ffnsw_S2a_nap9_FULL16k` · `exp_n_0119_ffnsw_S2a_nap9_tph128_16k` ·
`exp_n_0121_ffnsw_nap8_tph128_16k` (the anchor) · `exp_n_0126_grid_H4d48_nap7_tph64` ·
`exp_n_0127_grid_H4d48_nap7_tph128` · `exp_n_0128_grid_H4d48_nap8_tph64` ·
`exp_n_0129_grid_H4d48_nap8_tph256` · `exp_n_0130_grid_H4d48_nap10_tph64` ·
`exp_n_0131_grid_H2d96_nap8_tph128` · `exp_n_0132_grid_H8d24_nap8_tph128` ·
`exp_n_0133_grid_H4d48_nap10_tph128` · `exp_n_0137_grid_H1d192_nap8_tph128` ·
`exp_n_0153_H2d96_nap7_tph64`

**§7 longer horizons** — 48k: `exp_n_0152_long48k_tiny_H4d48_nap7_tph64` ·
`exp_n_0155_long48k_from_0127` · `exp_n_0156_long48k_from_0128` — 144k:
`exp_n_0158_long144k_from_0127`

## Checkpoints are NOT included

`.gitignore` excludes `experiments/**/*.pt`. Trained checkpoints exist only for
`exp_n_0126`, `0127` and `0128` (≈520 MiB total) and live on the machine that trained
them. Consequence: `run_bench.py --load-checkpoint` and the `paper_timings` scripts
cannot be re-run from a fresh clone without retraining those three. The measured numbers
are preserved in `benchmark/paper_timings*/results.json`, so the paper itself is
reproducible from what is committed; only re-measurement is not.

## Rebuilding the paper

```bash
cd paper && latexmk -pdf ffn_replacement.tex
```

### Which script builds which figure

All five figures regenerate from committed data.

| fig | file | generator | reads |
|---|---|---|---|
| 1 | `fig16k.pdf` | `paper/make_fig16k.py` | nothing — the 17 (params, bpb) points are inlined in the script |
| 2 | `FFN_GRID_plots.png` | `paper/make_ffn_grid.py` | nothing — the grid values are inlined in the script |
| 3 | `long_runs_fig.pdf` | `paper/make_long_runs_fig.py` | `../runs/exp_n_015{1,2,5,6}*/metrics.csv` |
| 4 | `superlong_fig.pdf` | `paper/make_superlong_fig.py` | `../runs/exp_n_015{7,8}*/metrics.csv` |
| 5 | `bench_combined.pdf` | `paper/make_bench_figs.py` | `../benchmark/paper_timings{,_h100}/results.json` |

```bash
cd paper
python make_fig16k.py          # -> fig16k.pdf
python make_ffn_grid.py        # -> FFN_GRID_plots.png
python make_long_runs_fig.py   # -> long_runs_fig.pdf
python make_superlong_fig.py   # -> superlong_fig.pdf
python make_bench_figs.py      # -> bench_combined.pdf
```

**No GPU and no checkpoints are needed for any of these** — Figure 5 redraws the
committed timing JSON rather than re-measuring. To actually re-measure, see
"Re-running the benchmarks" below, which *does* need the checkpoints.

Two caveats worth knowing before trusting a regenerated figure:

- **Figures 1 and 2 have no data inputs.** Their values are hardcoded in the scripts,
  so they redraw the same numbers regardless of what `runs/*/summary.json` says. If a
  run is ever re-scored, neither will follow — edit the tables by hand. The two also
  differ by convention: `make_fig16k.py` uses the paper's Table 4 `best_val_bpb`, while
  `make_ffn_grid.py` uses `final_val_bpb` (matching `FFN_GRID_SUMMARY.md`). Figures 3,
  4 and 5 *are* driven by committed run data and do follow it.
- **Re-rendering is not byte-reproducible across machines.** The figures redraw with
  identical content but not identical bytes — matplotlib/freetype version differences
  shift the layout by a pixel or two. What is committed here are the original bytes the
  published PDF was built from; regenerate only when you mean to replace them.

Self-contained: no `\input`, no bibliography file, five stock packages, and all five
figures sit beside the `.tex`. Verified to build to 13 pages with every reference
resolved.

> Keep the paper's `FFN_GRID_plots.png` in `paper/`. A **different** 3-panel figure of
> the same name exists elsewhere in the research history; if both are ever reachable on
> one graphics path, LaTeX picks by name and silently embeds the wrong one. For the same
> reason `tools/plot_ffn_grid.py` writes `ffn_grid_overview_3panel.png`, not that name.

## Re-running the benchmarks

```bash
cd benchmark
python run_bench.py --exp exp_n_0126_grid_H4d48_nap7_tph64 --gather-impl cuda-fused
python paper_timings/phase_split.py --load-checkpoint   # writes results.json
python paper_timings/make_figure.py                     # renders the figure from it
```

Paths default to this folder (`runs/` for experiments); `--root` overrides. Read
`benchmark/README.md` first — the warm-up discipline there is not optional, and ignoring
it has flipped the sign of a headline result.
