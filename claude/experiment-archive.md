# Where the historical experiment data lives

*How to find and verify the full experimental record behind [thesis.md](thesis.md) and
[experiment-journey.md](experiment-journey.md).*

## The archive branch

The complete pre-2026-07 experimental record lives on **one non-obvious branch** of this
repo:

```
feature/lutorch-calibrate-output-normalize-weights
```

The name looks like a stray feature branch, but it is actually **the archive** — the sole
surviving copy of the experimental record from the earlier (Nebius-era) work. It is kept
**read-only as history**; do not add new work to it.

## What's on it

- **~1,366 experiment directories** under `nanochat_exps/`, `transformer_exps/`,
  `transformer_paper_exps/`, and `lut_lm/` — each with `config.json`, `metrics.csv`,
  `summary.json` (and plots). These are the source of truth for any experiment number or bpb
  figure quoted in the thesis / journey notes.
- The experiment **journals** — `.../experiments.md`, `nanochat_exps/EXPERIMENTS.md`,
  `research_report_draft.md` — plus the top-level `eval_*` / `bench_*` / `diag_*` / `probe_*`
  scripts.
- `doc/`, `docs/`, and the code as it stood at that point in history.

## The one exception: no checkpoints

**Model checkpoints are NOT in git** (no `.pt` / `.pth` / `.safetensors` / `.ckpt` tracked —
too large). The original checkpoints are effectively gone; only configs, metrics, and
summaries survive. **To reproduce a result, re-run from its `config.json`** — not from a
saved checkpoint.

## How to look things up

You do not need to check the whole branch out. From inside a clone of this repo:

```bash
# make sure you have the branch
git fetch origin feature/lutorch-calibrate-output-normalize-weights

# find an experiment by number/name
git ls-tree -r --name-only \
  origin/feature/lutorch-calibrate-output-normalize-weights | grep <exp>

# read a single file without checking out
git show \
  'origin/feature/lutorch-calibrate-output-normalize-weights:<path>'

# or, if you need the whole tree at once, use a worktree:
git worktree add ../spiky-archive \
  origin/feature/lutorch-calibrate-output-normalize-weights
```

Note: `main` does **not** contain this data — the archive is only on that branch. Verify any
specific experiment id or figure here before quoting it as established fact.

## Going forward

This branch is the **old** archive. New research does not append to it; going forward each
idea is developed on its own branch and only decisive results are curated into `main`. The
exact multi-machine workflow is being defined separately and will be documented alongside
these notes.
