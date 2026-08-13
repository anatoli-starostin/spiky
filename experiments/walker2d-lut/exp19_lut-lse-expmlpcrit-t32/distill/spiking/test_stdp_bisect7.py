"""Seventh: does do_sort_by_target_id=True fix the crash?

The chunk validator flags exactly ONE violation in every config, passing ones included:
"Target neuron IDs not sorted within groups" -- ChunkOfConnections spec rule 9. That is a
direct consequence of _grow_explicit's do_sort_by_target_id default of False, which
es_harness and steady_state.py both inherit. If the native forward build relies on rule 9,
flipping this one argument is a caller-side fix with no engine change at all.
"""
import subprocess
import sys

HERE = __file__.replace("test_stdp_bisect7.py", "test_stdp.py")

CONFIGS = [(2, True), (2, False), (4, False), (20, False)]   # (n_meta, sources span)

for n_meta, span in CONFIGS:
    row = []
    for sort_targets in (False, True):
        cmd = [sys.executable, HERE, "--case", "B", "--n-meta", str(n_meta),
               "--max-syn", "640"]
        if not span:
            cmd.append("--one-meta-per-source")
        if sort_targets:
            cmd.append("--sort-targets")
        r = subprocess.run(cmd, capture_output=True, text=True)
        row.append("PASS " if "PASS" in (r.stdout + r.stderr) else "CRASH")
    lbl = "sources SPAN metas " if span else "one meta per source"
    print(f"  metas {n_meta:2d}  {lbl}  sort=False -> {row[0]}   sort=True -> {row[1]}",
          flush=True)
