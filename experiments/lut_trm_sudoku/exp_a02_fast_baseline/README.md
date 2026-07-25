# exp_a02 — fast-iteration TRM Sudoku baseline (Phase A variant of #73)

*Umbrella #72 · issue #73 · branch `research/lut-trm-sudoku` · implementer: gpustar (RTX 5090).*
Methodology: [`claude/experiment-methodology.md`](../../../claude/experiment-methodology.md).

## Purpose / hypothesis

The full stock baseline ([exp_a01](../exp_a01_stock_trm_sudoku_baseline/)) is ~5 h — too slow to
iterate on LUT swaps. This is the **standard fast A/B baseline** for Phase B: a config that still
exercises the recursion + deep supervision (so a LUT-in-the-recursion swap is genuinely tested) but
finishes in **~25–40 min**. Lower accuracy is fine — the point is a stable, repeatable signal.
Expected **val_exact ~40–60%** (vs the paper's 87.4% at full training).

**Phase-B success criterion, restated for this baseline:** a LUT halting-head (or other) swap that
**matches this fast baseline's val_exact at equal-or-better it/s**.

## What's cheaper vs stock (exp_a01), and why

| knob | stock (exp_a01) | fast (exp_a02) | effect |
|---|---|---|---|
| recursion | H_cycles=3, L_cycles=6 (21 evals/step) | **H_cycles=2, L_cycles=3 (8 evals/step)** | ~2.5× faster/step; less activation memory |
| batch | 512 (32GB-forced) | **768** | cheaper recursion frees the memory → back to stock batch |
| epochs | 50000 | **10000** | time-box |
| eval set | full 422,786 test | **5000-puzzle subset** (`subsample_test.py`) | eval ~7 batches (secs) not ~826 (mins) |
| optimizer | — | AdamATan2 drop-in (as exp_a01) | Blackwell sm_120 |

Recursion still runs (H=2 improvement passes × L=3 latent steps + deep supervision, EMA on), so the
LUT-in-recursion behaviour we want to test is preserved — just shallower.

## Files

- `subsample_test.py` — builds `data/sudoku-extreme-1k-aug-1000-testsub5k/` (train symlinked to the
  full augmented train set; test = first 5000 puzzles). Full dataset left intact.
- `config.json` — machine-readable config + all deviations.
- `adam_atan2.py` — the pure-PyTorch optimizer drop-in (copy of exp_a01's; also already installed in
  the TRM venv).
- `setup.sh` — one-time prep (just runs `subsample_test.py`; deps + full dataset already exist from
  exp_a01, so **no network needed**).
- `launch.sh` — the fast run (detached, persistent tmux, unbuffered log).

## Status

Committed before launch (agree→commit→go). The run is *in-cage GPU work — frictionless, no new
network* (deps + dataset already present). Results (`metrics.csv`/`summary.json`, never checkpoints)
committed after completion.
