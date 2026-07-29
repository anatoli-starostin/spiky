"""exp_c16 — evaluate the two seed-0 replicates and separate run noise from seed effect (#75).

The comparison that matters is a single subtraction: |A - B| at a FIXED seed is pure
run-to-run nondeterminism, and everything exp_c12..exp_c15 has called "seed-sd" is really
that plus the seed. If |A - B| is small, the seed numbers mean what they say. If it is
comparable to the seed spreads, they do not.

Three reference points are quoted: exp_c15's own seed-0 run (same config, but trained
CONCURRENTLY with two others, so it also probes whether contention matters), exp_c15's
3-seed spread, and the unexplained exp_c11-vs-exp_c14 same-seed gap of 951.
"""
import json, os, re, subprocess, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
REPS = ("a", "b")
C15_SEED0 = 4152.7          # same config+seed, but trained 3-concurrent
C15_SEED_SD = 178.0         # exp_c15's across-seed sd, the thing under test
C11_C14_GAP = 951.6         # 5146.9 - 4195.3, the unexplained same-seed discrepancy
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
COV = re.compile(r"row-cov\s+([\d.]+)%", re.M)


def run_eval(actor, label):
    t0 = time.time()
    print(f"  [{time.strftime('%H:%M:%S', time.gmtime())}] evaluating {label}", flush=True)
    p = subprocess.Popen(
        [PY, "-u", os.path.join(C09, "eval_cpu.py"), actor,
         "--episodes", "100", "--forward-mode", "hard", "--progress", label],
        cwd=C09, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("    ["):
            print(line, flush=True)
        elif "CPU-reference" in line:
            print(f"    -> {line.strip()}", flush=True)
    err = p.stderr.read()
    if p.wait() != 0:
        print(f"    FAILED: {err[-300:]}", flush=True)
        return False
    print(f"    done in {time.time() - t0:.0f}s", flush=True)
    return True


def main():
    got = []
    for r in REPS:
        actor = f"lut_sac_c16_rep_{r}_actor.npz"
        log = os.path.join(HERE, f"cell_rep_{r}.log")
        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        d = DONE.search(txt)
        if not d:
            print(f"  (replicate {r}: still training)", flush=True)
            continue
        ev = os.path.join(C09, actor.replace("_actor.npz", "_cpueval.json"))
        if not os.path.exists(ev):
            if not run_eval(actor, f"replicate-{r}"):
                continue
        e = json.load(open(ev))
        covs = COV.findall(txt)
        got.append(dict(rep=r, mean=e["cpu_reference_mean"], std=e["cpu_reference_std"],
                        mjx=float(d.group(1)),
                        cov=float(covs[-1]) if covs else float("nan")))

    if len(got) < 2:
        print(f"\nonly {len(got)}/2 replicates ready — not concluding yet")
        return

    a, b = got[0], got[1]
    delta = abs(a["mean"] - b["mean"])
    print(f"\n=== THE REPLICATE TEST: same config, same seed 0, two sequential runs ===")
    for g in got:
        print(f"  replicate {g['rep']}:  CPU-ref {g['mean']:7.1f} +/- {g['std']:6.1f} "
              f"(episodes)   best MJX {g['mjx']:7.1f}   cov {g['cov']:.1f}%")
    print(f"\n  |A - B| = {delta:.1f}   <- PURE RUN-TO-RUN NONDETERMINISM (seed is fixed)")

    print(f"\n=== what that does to the seed numbers ===")
    print(f"  exp_c15 across-seed sd (torch init)          {C15_SEED_SD:8.1f}")
    print(f"  this replicate gap at a FIXED seed           {delta:8.1f}")
    frac = delta / C15_SEED_SD if C15_SEED_SD else float("inf")
    print(f"  ratio                                        {frac:8.2f}x")
    if frac < 0.5:
        verdict = ("run noise is SMALL relative to the seed spread — the seed numbers "
                   "measure real seed effects")
    elif frac < 1.5:
        verdict = ("run noise is COMPARABLE to the seed spread — 'seed-sd' has been "
                   "measuring mostly nondeterminism, not the seed")
    else:
        verdict = ("run noise EXCEEDS the seed spread — the seed comparisons are "
                   "dominated by nondeterminism and cannot support their conclusions")
    print(f"  VERDICT: {verdict}")

    print(f"\n=== other reference points ===")
    d15 = abs(a["mean"] - C15_SEED0), abs(b["mean"] - C15_SEED0)
    print(f"  exp_c15 seed 0 (same config+seed, 3-CONCURRENT): {C15_SEED0:.1f}")
    print(f"    |A - c15s0| = {d15[0]:.1f}   |B - c15s0| = {d15[1]:.1f}")
    print(f"    (these two runs had the GPU to themselves; if the concurrent one is a")
    print(f"     clear outlier, contention matters as well as nondeterminism)")
    print(f"  unexplained exp_c11 vs exp_c14 same-seed gap:  {C11_C14_GAP:.1f}")
    print(f"    -> this replicate gap is {delta / C11_C14_GAP:.2f}x that, so ordinary "
          f"run noise {'DOES' if delta > 0.5 * C11_C14_GAP else 'does NOT'} account for it")

    json.dump(dict(replicates=got, delta=float(delta), c15_seed0=C15_SEED0,
                   c15_seed_sd=C15_SEED_SD, ratio_to_seed_sd=float(frac),
                   c11_c14_gap=C11_C14_GAP, verdict=verdict),
              open(os.path.join(HERE, "replicate_results.json"), "w"), indent=1)
    print("\nwrote replicate_results.json")


if __name__ == "__main__":
    main()
