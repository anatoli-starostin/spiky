"""Regenerate the CPU-safe analytics (1-3) for every experiment that has data.

Post-hoc file reading only -- no network is built or run, so this is safe to invoke at any
time, including while a GPU experiment is training. (4) teacher_student.py is NOT run here
because it needs GPU rollouts; it is queued separately behind the training runs.

    python run_all.py
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
env = dict(os.environ, PYTHONPATH=HERE + ":" + os.path.dirname(HERE),
           MPLCONFIGDIR=os.environ.get("MPLCONFIGDIR", "/tmp/mpl"))

for script in ("traj_diversity.py", "delay_utilization.py", "weight_structure.py"):
    print(f"\n{'=' * 70}\n== {script}\n{'=' * 70}")
    r = subprocess.run([sys.executable, "-u", os.path.join(HERE, script)],
                       env=env, capture_output=True, text=True)
    # `"Matplotlib" in l is False` was a chained comparison -- Python reads it as
    # ("Matplotlib" in l) and (l is False), which is never true, so this filter silently
    # swallowed EVERY line of output and run_all.py printed nothing but its own banners.
    print("\n".join(l for l in (r.stdout + r.stderr).splitlines()
                    if "matplotlib" not in l.lower()))
    if r.returncode != 0:
        print(f"!! {script} exited {r.returncode}")
print(f"\nfigures + summaries in {os.path.join(HERE, 'out')}")
