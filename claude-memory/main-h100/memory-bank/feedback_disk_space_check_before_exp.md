---
name: disk-space-check-before-exp
description: "Run `df -h /` before launching any experiment; warn the user if root filesystem is getting full."
metadata: 
  node_type: memory
  type: feedback
  originSessionId: fa43f8ba-4262-4d5d-ab44-b1dfbc584286
---

Before launching any experiment (training run, sweep, distillation, etc.), run `df -h /` and check free space on the root filesystem. If free space is low (rough thresholds: <30 GB free, or >85% used), surface a warning to the user *before* starting the run, since experiments write checkpoints + logs that can be multiple GB each.

**Why:** This box has only one writable disk (~242 GB root, ~133 GB free as of 2026-05-13). `/mnt/cloud-metadata` looks like 1 TB free but is a read-only metadata mount — no usable bulk storage. User asked for this check on 2026-05-13 after discovering ~/.cache (35 G) and other caches were eating root.

**How to apply:** Fold the `df -h /` check into the pre-launch flow alongside [[show-exp-description-before-launch]] — present the experiment config *and* current disk usage, then wait for approval. If free space is low, also suggest deletable caches (e.g. `~/.cache/{uv,pip,huggingface}` are regeneratable; `~/spiky_old` may be stale).

See also: [[launching-experiments]], [[show-exp-description-before-launch]], [[new-folder-per-experiment]].
