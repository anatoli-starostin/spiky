---
name: Monitor messages must report step / val / delta
description: When monitoring a training run via the Monitor tool, each event reply should include step, validation metric, and delta-from-reference — not just "Continuing."
type: feedback
originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---
When monitoring a long-running training job (e.g. nanochat_exps runs), each monitor-event reply must include:

1. The current **step** (or otherwise the meaningful x-axis position)
2. The current **validation metric** (e.g. val_bpb)
3. The **delta vs the reference comparator** (e.g. exp265 at the same step, or other relevant baseline)

Do not respond with just "Continuing.", "No response needed", "running to completion", or any other empty-acknowledgement text — EVERY eval event, including routine mid-run ones, must carry step + val + delta. The user explicitly called out "No response needed — running to completion" as wrong (2026-05-21). No exceptions for "boring" evals.

**Why:** the user follows the run from Telegram and the monitor message is the only signal they see. Saying just "Continuing" is uninformative; they need the numbers to judge whether to interrupt or let it run.

**How to apply:**
- When a `[VAL] step X: bpb=...` event arrives, format as: `step X: bpb=Y (Δ +/-Z vs <reference> at same step)`.
- Pull the reference value (one-time, cache mentally for the session) at the start of monitoring.
- If a reference at the same step isn't available, state that and report bpb + trend (vs previous eval).
- This applies to every monitor message during a training run, not just the first or last.
