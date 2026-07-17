# hyperplane_ffn

First idea of the structured research chapter (see `claude/experiment-methodology.md`).

- **Tracking issue:** anatoli-starostin/spiky#61 — the idea's origin and status log.
- **Branch:** `research/hyperplane_ffn` (this branch).
- **Baseline anchor:** `exp001_untied_vanilla_baseline` — untied-vanilla MinimalGPT,
  val_bpb ≈ **1.2014** on the local ClimbMix subset. Every experiment here compares against it.

Each `exp*/` folder is one run (`config.json`, `metrics.csv`, `summary.json`, plots);
checkpoints are gitignored. Progress notes go on issue #61.
