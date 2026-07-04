---
name: New folder per experiment
description: When forking an existing experiment, always create a new dedicated folder unless the user explicitly says to overwrite — never reuse the source experiment's directory.
type: feedback
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
When the user asks to "fork", "modify", or "try X starting from expN", create a brand-new experiment folder (e.g. `nanochat_exps/expM_<descriptor>/` or `transformer_exps/expM_<descriptor>/`) with its own `train.py`, `config.json`, `metrics.csv`, `summary.json`, etc. Never overwrite or reuse the source experiment's directory.

**Why:** Each experiment's outputs (metrics, plots, summary, checkpoint) must remain immutable so prior runs stay reproducible and comparable. Overwriting them destroys the journal. The user stated this preference explicitly on 2026-04-30 after I asked whether to fork exp061 in-place or create exp062.

**How to apply:** Default to creating a new folder for any experiment fork. The only exception is when the user explicitly says "edit in place", "overwrite", "redo expN" or otherwise unambiguously approves reusing an existing folder. When in doubt, ask — don't overwrite.
