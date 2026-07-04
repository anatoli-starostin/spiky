---
name: Always show complete experiment description before launching
description: Before launching any transformer experiment, present the full config/description and wait for user approval instead of launching immediately.
type: feedback
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
Before launching any transformer experiment, present the full proposed description (config, architectural changes vs baseline, expected param count) and wait for explicit user confirmation before running the training.

**Why:** Autonomous launches burned time on configs that missed the user's intent — e.g., the user meant PermutationalLut for Q/K/V/out_proj, but the launched fork reused MultiHeadLut. Two hours of training wasted.

**How to apply:** For any new `transformer_exps/expN*` fork, show: (1) what it forks from, (2) exact diffs in config.json, (3) module choices (PermLut vs MultiHeadLut), (4) expected param count and training duration. Then stop and wait.
