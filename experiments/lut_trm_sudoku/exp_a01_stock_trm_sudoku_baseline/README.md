# exp_a01 — stock TRM Sudoku-Extreme baseline (Phase A of #73)

*Umbrella #72 · experiment issue #73 · branch `research/lut-trm-sudoku` · implementer: gpustar (RTX 5090).*
Methodology: [`claude/experiment-methodology.md`](../../../claude/experiment-methodology.md) — one idea →
one issue → one branch; each experiment its own folder; **agree → commit → go** (code committed
*before* the run, results after); commit+push directly to the research branch (no PR); checkpoints
never in git.

## Hypothesis / goal

Reproduce the **stock TRM** Sudoku-Extreme baseline as-is, to establish a trustworthy anchor on our
hardware before any LUT change (Phase B). Target: **~87.4% exact accuracy (±2%)** — the paper's
attention-free MLP-Mixer TRM number (Table 4 of arXiv:2510.04871). Record **accuracy + throughput**
(steps/s, wall-clock, param count) so Phase B (LUT halting head) has an equal-hardware comparison.

## What is being run (the stock reference, unmodified)

- **Reference:** Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks*,
  arXiv:2510.04871 · code `SamsungSAILMontreal/TinyRecursiveModels` (MIT, archived read-only
  2026-04-01). Background: [`doc/research/trm.md`](../../../doc/research/trm.md).
- **Variant:** **TRM-MLP** (`arch.mlp_t=True`, `arch.pos_encodings=none`) — the attention-free
  MLP-Mixer variant, which is the *best* on Sudoku (87.4% vs 74.7% for the attention variant) and,
  crucially, **needs no flash-attn** (see hardware note).
- **Recursion:** `L_layers=2`, `H_cycles=3` (= T=3), `L_cycles=6` (= n=6), deep supervision
  N_sup≤16, **EMA on** — exactly the paper's Sudoku setting.
- **Optim:** `lr=1e-4`, `puzzle_emb_lr=1e-4`, `weight_decay=1.0`, `puzzle_emb_weight_decay=1.0`,
  optimizer `adam-atan2`. `epochs=50000`, `eval_interval=5000`.
- **Data:** Sudoku-Extreme, **1000 train puzzles + 1000 augmentations**, ~423K test.

See `config.json` for the machine-readable copy.

## Hardware / reproducibility note (Blackwell sm_120)

The reference README installs **torch nightly cu126**, which predates solid Blackwell (sm_120)
support. We **deliberately keep the box's existing `torch 2.9.1+cu130`** (verified: CUDA available,
device `RTX 5090`, capability `(12, 0)` = sm_120) rather than downgrade to cu126 — cu130 supports
Blackwell. This is the *only* intentional deviation from the reference recipe; everything else is
stock. `adam-atan2` builds a small CUDA extension and must compile for sm_120 (expected fine under
cu130; flagged as the one build risk). `flash-attn` is **not** installed — unnecessary for the MLP
path (it would be the real Blackwell headache, deferred to if/when the attention variant is needed
for Maze/ARC).

## How to run

`setup.sh` and `launch.sh` in this folder are the exact, reviewable recipe (committed *before* the
run per agree→commit→go). `setup.sh` contains the **network steps that require Anatoli's approval**
(they cross the sandbox: a `git clone`, dependency installs, and the dataset build — the cage has no
network). Once those are approved/run, `launch.sh` starts training detached in a persistent tmux
session with its own log, and copies `metrics.csv` / `summary.json` back into this folder on
completion (checkpoints stay out of git).

## Status

**SETUP COMMITTED — execution is BLOCKED on network approvals** (clone the reference repo, install
the handful of missing pure-Python deps + `adam-atan2`, build the Sudoku-Extreme dataset). These are
batched in `setup.sh`; see the Slack report on #73. The training run itself is in-cage GPU work
(frictionless) once the code + data are in place.
