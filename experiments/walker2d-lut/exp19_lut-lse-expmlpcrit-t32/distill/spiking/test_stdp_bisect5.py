"""Fifth bisection: is the trigger ONE SOURCE SPANNING TWO METAS, not the meta count?

The capacity estimator writes ONE slot per source:
    capacity_estimations[source_neuron_id - forward_shift] = capacity;
If a source ever yields two ROOT blocks (one per meta) instead of one chained list, the
second write OVERWRITES the first instead of adding, the net under-allocates, and
create_forward_groups runs off the end -> illegal address at connections_manager.cu:126.

Test: hold the META COUNT high but give every source synapses in exactly ONE meta, so no
source can ever span metas. If that PASSES while the interleaved version crashes, the
trigger is spanning, not counting -- and es_harness's 20 metas working is no longer a
contradiction.
"""
import subprocess
import sys

HERE = __file__.replace("test_stdp_bisect5.py", "test_stdp.py")

for n_meta in (2, 4, 20):
    for span in (True, False):
        cmd = [sys.executable, HERE, "--case", "B", "--n-meta", str(n_meta),
               "--max-syn", "640"]
        if not span:
            cmd.append("--one-meta-per-source")
        r = subprocess.run(cmd, capture_output=True, text=True)
        txt = r.stdout + r.stderr
        v = "PASS " if "PASS" in txt else "CRASH"
        lbl = "sources SPAN metas" if span else "one meta per source"
        print(f"  metas {n_meta:2d}  {lbl:20s} -> {v}", flush=True)
