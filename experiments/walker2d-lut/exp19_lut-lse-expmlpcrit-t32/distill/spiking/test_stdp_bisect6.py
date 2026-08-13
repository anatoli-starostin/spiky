"""Sixth: with spanning eliminated, is the meta-count result STABLE and monotonic?

bisect5 gave 1 PASS, 2 CRASH, 4 PASS, 20 PASS at identical topology (check_span shows 0
spanning sources and the same 80 sublists in every one of those runs). Non-monotonic is
either a real narrow bug or run-to-run flakiness -- repeat each config to tell them apart.
"""
import subprocess
import sys

HERE = __file__.replace("test_stdp_bisect6.py", "test_stdp.py")
REPEATS = 3

for n_meta in (1, 2, 3, 4, 5, 8, 20):
    out = []
    for _ in range(REPEATS):
        r = subprocess.run([sys.executable, HERE, "--case", "B", "--n-meta", str(n_meta),
                            "--max-syn", "640", "--one-meta-per-source"],
                           capture_output=True, text=True)
        out.append("PASS" if "PASS" in (r.stdout + r.stderr) else "CRASH")
    print(f"  metas {n_meta:2d} (no spanning) -> {'  '.join(out)}", flush=True)
