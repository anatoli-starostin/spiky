"""Fourth bisection: GROUP SIZE.

es_harness runs explicit wiring over 20 delay metas without trouble, and the one setup
knob it differs on is synapse_group_size = 2 (this test used 8). The original steady_state
stage-two crash also reported a DIFFERENT cuda error at group size 2 than at 8, which is
the tell that group size is a live variable and not a bystander.
"""
import subprocess
import sys

HERE = __file__.replace("test_stdp_bisect4.py", "test_stdp.py")

for gs in (1, 2, 4, 8, 16):
    for n_meta in (1, 2, 20):
        r = subprocess.run([sys.executable, HERE, "--case", "B", "--gs", str(gs),
                            "--n-meta", str(n_meta), "--max-syn", "640"],
                           capture_output=True, text=True)
        txt = r.stdout + r.stderr
        if "PASS" in txt:
            v = "PASS "
        else:
            e = [ln.strip() for ln in txt.splitlines() if "Error" in ln]
            v = "CRASH " + (e[-1].split("error")[-1].strip()[:60] if e else "")
        print(f"  group_size {gs:2d}  metas {n_meta:2d} -> {v}", flush=True)
