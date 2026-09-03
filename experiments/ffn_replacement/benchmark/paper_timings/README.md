# Paper timings — FFN-slot inference numbers and the phase-breakdown figure

Scripts that regenerate the RTX 5090 inference numbers and the figure used in the paper.
They run on top of the benchmark harness one directory up (`../model.py`,
`../gather_fused.py`, `../bench.py`), so run them from the `benchmark/` directory.

```bash
cd experiments/ffn_replacement/benchmark

# citable totals + ratios, and the torch.compile comparison
python paper_timings/interleaved.py --load-checkpoint

# per-phase splits; writes results.json
python paper_timings/phase_split.py --load-checkpoint

# figure, rendered FROM results.json (not from hardcoded numbers)
python paper_timings/make_figure.py
```

Both timing scripts take `--root`, `--exps`, `--baseline`, `--batch`, `--seq`, `--iters`
and default to the three CompressionMHL grid cells against the untied vanilla baseline.
`--load-checkpoint` is required for anything you intend to quote: it loads the trained
weights. `interleaved.py --no-compile` skips the `torch.compile` variants.

## Citable conditions

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 (sm_120) |
| torch / CUDA | 2.9.1+cu130 / CUDA 13.3 |
| workload | **batch 48 × seq 512 = 24,576 tokens** per slot call |
| dtype | **bf16** — hybrid-v2 storage: dense weights *stored* bf16 (never autocast), LUT tables bf16, LUT input kept fp32 so the native routing stays alive. Vanilla baseline bf16 throughout. |
| gather path | `--gather-impl cuda-fused` (fused routing+gather, bf16 table), routing regime auto-dispatched v1/v1/v2 |
| timing | `torch.no_grad()`, 30 iterations per sample, median of 11 interleaved rounds (`interleaved.py`) or 7 repetitions (`phase_split.py`) |
| clock | measured warmed: SM ~630 MHz idle → **2880–3040 MHz** under load (max 3210) |
| checkpoints | trained (`--load-checkpoint`); val bpb 1.20694 / 1.19471 / 1.20228 |

## Measured (this repo, the run committed in `results.json`)

Slot totals, interleaved:

| variant | median ms | vs vanilla |
|---|---|---|
| vanilla dense (eager) | 0.3636 | 1.00× |
| `exp_n_0126` fused | **0.1824** | **0.50×** |
| `exp_n_0127` fused | **0.2749** | **0.76×** |
| `exp_n_0128` fused | **0.1899** | **0.52×** |
| vanilla (compiled) | 0.3309 | 0.91× |
| `exp_n_0127` (compiled) | 0.2778 | 0.76× |

Phase split (fused path); phases 2 and 3 are **one kernel** and are reported together:

| model | total | compress 384→192 | routing+gather | decompress 192→384 | other |
|---|---|---|---|---|---|
| `exp_n_0126` | 0.1798 | 0.0344 (19.1%) | **0.0845 (47.0%)** | 0.0505 (28.1%) | 0.0105 |
| `exp_n_0127` | 0.2708 | 0.0342 (12.6%) | **0.1727 (63.8%)** | 0.0507 (18.7%) | 0.0132 |
| `exp_n_0128` | 0.1875 | 0.0342 (18.2%) | **0.0916 (48.8%)** | 0.0504 (26.9%) | 0.0114 |
| vanilla | 0.3429 | 384→1536+GELU 0.2086 (60.8%) | — | 1536→384 0.1298 (37.8%) | 0.0045 |

`phase_split.py` also prints the **separable** phase-2 vs phase-3 breakdown by re-running
on `cuda-bf16` (native routing + a separate gather), since the fused kernel cannot split
them. Fusion saves 0.038 / 0.177 / 0.107 ms over running them separately.

## Read this before quoting a number

1. **`torch.compile` speeds up the BASELINE, not the LUT models.** Vanilla gets ~1.10×;
   `exp_n_0127` gets nothing (it graph-breaks 4× at the pybind11 custom op, so there is
   nothing for inductor to fuse). Against a *compiled* vanilla the LUT ratios become
   **0.55× / 0.83× / 0.57×** instead of 0.50 / 0.76 / 0.52. Both are honest; say which
   baseline you used. `interleaved.py` prints both.
2. **A cold first measurement inflates by ~10%.** The per-call 60-iteration burn-in is
   *not* sufficient alone: measured cold-first the vanilla slot reads 0.378 ms and warmed
   0.344. Whichever model is timed first otherwise eats the ramp. `phase_split.py` does a
   300-call global warm-up first and prints the SM clock before and after it.
3. **~4–6% drift over a long session.** Re-measuring the identical vanilla slot at the end
   of a sequential run comes back that much slower. So **absolute ms taken minutes apart
   are not comparable** — that is precisely why `interleaved.py` exists and why its
   ratios, not `phase_split.py`'s absolute ms, are the citable ones. `phase_split.py`
   prints its own drift figure so you can see how much its numbers moved.
4. **`[min–max]` spreads include occasional high outliers** from scheduling; medians are
   quoted for that reason. Treat a ratio as a good central estimate, not a tight bound,
   unless the harness reports the intervals as `disjoint`.
5. **"other"** in the phase split is reshape/dtype glue — the difference between the phase
   sum and the independently measured total. It is printed rather than folded into a
   phase so the accounting stays honest.

## Files

| file | what it is |
|---|---|
| `interleaved.py` | Slot totals in alternating rounds + the `torch.compile` comparison. **The citable numbers.** |
| `phase_split.py` | Per-phase ms/% using real captured intermediates; global warm-up and drift check; writes `results.json`. |
| `make_figure.py` | Renders `ffn_phase_split.png` from `results.json`. |
| `results.json` | The measured run tabulated above. |
| `ffn_phase_split.png` | The figure. |
