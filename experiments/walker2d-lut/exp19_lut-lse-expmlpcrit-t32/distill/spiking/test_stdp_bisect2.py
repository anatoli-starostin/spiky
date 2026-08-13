"""Second bisection: which factor of CASE B actually trips create_forward_groups?

CASE B = explicit wiring + 20 plastic metas. It changes THREE things at once versus the
passing CASE A: the meta COUNT, the fact that those metas are PLASTIC, and the DELAY
DIVERSITY (metas 1..20 carry delays 1..20). Vary one at a time.
"""
import subprocess
import sys

HERE = __file__.replace("test_stdp_bisect2.py", "test_stdp.py")

VARIANTS = [
    ("20 plastic, delays 1..20  (= CASE B)",      ["--n-meta", "20"]),
    ("20 FROZEN  (lr=0), delays 1..20",           ["--n-meta", "20", "--lr", "0"]),
    ("20 plastic, ALL delay 1",                   ["--n-meta", "20", "--same-delay"]),
    ("20 FROZEN,  ALL delay 1",                   ["--n-meta", "20", "--lr", "0",
                                                   "--same-delay"]),
    ("2  plastic, delays 1..2",                   ["--n-meta", "2"]),
    ("4  plastic, delays 1..4",                   ["--n-meta", "4"]),
    ("8  plastic, delays 1..8",                   ["--n-meta", "8"]),
    ("1  plastic, delay 1        (= CASE A)",     ["--n-meta", "1"]),
]

for desc, extra in VARIANTS:
    r = subprocess.run([sys.executable, HERE, "--case", "B"] + extra,
                       capture_output=True, text=True)
    txt = r.stdout + r.stderr
    if "PASS" in txt:
        verdict = "PASS"
    else:
        err = [ln.strip() for ln in txt.splitlines()
               if "Error" in ln and "cuda" in ln.lower()]
        verdict = "CRASH  " + (err[-1][:100] if err else f"rc={r.returncode}")
    print(f"  {desc:42s} -> {verdict}", flush=True)
