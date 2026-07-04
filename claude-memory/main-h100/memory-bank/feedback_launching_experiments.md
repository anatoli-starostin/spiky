---
name: How to launch transformer experiments
description: Correct way to launch long-running training experiments in background
type: feedback
---

Always launch experiments like this:

```bash
.venv/bin/python -u transformer_exps/<exp_dir>/train.py > transformer_exps/<exp_dir>/stdout.log 2>&1 &
echo "PID: $!"
```

Then monitor with `tail -f`:
```bash
tail -f transformer_exps/<exp_dir>/stdout.log
```

**Why:** Use `dangerouslyDisableSandbox: true` on all commands. Use `python -u` (unbuffered) so output appears immediately. Use shell `&` (not `run_in_background: true`) — the background task runner kills processes unexpectedly. Redirect to file with `>` (not `tee`).

**How to apply:** Any time a new experiment needs to be launched.
