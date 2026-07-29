"""exp_c17 — did deterministic GPU ops fix it? (#75)

Two tests, in order of sensitivity:

  1. CHECKPOINT IDENTITY. Compare the two trained checkpoints tensor by tensor. This is the
     real test: two returns can coincide by luck, 28,032 trained weights cannot. A single
     differing float after 10,000 iterations means the run is still nondeterministic.
  2. CPU-reference return, for continuity with exp_c16's |A - B| = 999.1.

A subtlety worth stating: bit-identical checkpoints prove determinism was achieved, but
NON-identical checkpoints do not by themselves prove the flag did nothing — they could
diverge less. So the report includes the magnitude of the divergence, not just a boolean.
"""
import json, os, re, subprocess, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
REPS = ("a", "b")
C16_DELTA = 999.1          # the nondeterministic replicate gap this is trying to kill
C16_RUN_SD = 662.6
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)


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
    paths, got = {}, []
    for r in REPS:
        actor = f"lut_sac_c17_det_{r}_actor.npz"
        log = os.path.join(HERE, f"cell_det_{r}.log")
        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        d = DONE.search(txt)
        if not d:
            print(f"  (deterministic run {r}: still training)", flush=True)
            continue
        paths[r] = os.path.join(C09, actor)
        ev = os.path.join(C09, actor.replace("_actor.npz", "_cpueval.json"))
        if not os.path.exists(ev):
            if not run_eval(actor, f"det-{r}"):
                continue
        got.append(dict(rep=r, mean=json.load(open(ev))["cpu_reference_mean"],
                        std=json.load(open(ev))["cpu_reference_std"],
                        mjx=float(d.group(1))))

    if len(paths) < 2:
        print(f"\nonly {len(paths)}/2 runs ready — not concluding")
        return

    # ---- 1. checkpoint identity, the sensitive test -------------------------
    za, zb = np.load(paths["a"]), np.load(paths["b"])
    print("\n=== CHECKPOINT COMPARISON (the real test) ===")
    print(f"{'tensor':<14}{'elements':>10}{'max|Δ|':>13}{'mean|Δ|':>13}"
          f"{'differing':>12}")
    tensors, all_same, worst = {}, True, 0.0
    for k in ("w", "b", "weights", "log_T_soft", "log_T_sel"):
        a_, b_ = np.asarray(za[k], np.float64), np.asarray(zb[k], np.float64)
        dif = np.abs(a_ - b_)
        n_dif = int((a_ != b_).sum())
        same = n_dif == 0
        all_same &= same
        worst = max(worst, float(dif.max()))
        tensors[k] = dict(elements=int(a_.size), max_abs=float(dif.max()),
                          mean_abs=float(dif.mean()), n_differing=n_dif,
                          identical=same)
        print(f"{k:<14}{a_.size:>10,}{dif.max():>13.3e}{dif.mean():>13.3e}"
              f"{n_dif:>9,}{'' if same else ' *'}")
    total_el = sum(t["elements"] for t in tensors.values())
    total_dif = sum(t["n_differing"] for t in tensors.values())
    print(f"{'TOTAL':<14}{total_el:>10,}{worst:>13.3e}{'':>13}{total_dif:>9,}")

    # ---- 2. return, for continuity with exp_c16 ----------------------------
    print("\n=== CPU-reference return ===")
    for g in got:
        print(f"  deterministic {g['rep']}:  {g['mean']:7.1f} +/- {g['std']:6.1f}   "
              f"best MJX {g['mjx']:7.1f}")
    delta = abs(got[0]["mean"] - got[1]["mean"]) if len(got) == 2 else float("nan")
    print(f"\n  |A - B| = {delta:.1f}   (exp_c16 without the flag: {C16_DELTA:.1f})")

    # ---- verdict ------------------------------------------------------------
    print("\n=== VERDICT ===")
    if all_same:
        v = ("CONFIRMED AND FIXED: the two checkpoints are BIT-FOR-BIT IDENTICAL across "
             f"all {total_el:,} parameters. Deterministic GPU ops remove the "
             "nondeterminism entirely, so the atomics-based scatter path was the source. "
             "Every A/B in exp_c12..exp_c16 can now be redone on a footing where a fixed "
             "seed reproduces.")
    elif delta == delta and delta < 0.1 * C16_DELTA:
        v = (f"MOSTLY FIXED: checkpoints still differ ({total_dif:,} of {total_el:,} "
             f"elements, max|Δ| {worst:.3e}) but the return gap collapsed from "
             f"{C16_DELTA:.0f} to {delta:.1f}. Something residual remains -- likely a "
             "second nondeterministic op the flag does not cover.")
    else:
        v = (f"NOT FIXED: {total_dif:,} of {total_el:,} parameters still differ "
             f"(max|Δ| {worst:.3e}) and |A - B| = {delta:.1f} against exp_c16's "
             f"{C16_DELTA:.0f}. Deterministic ops did NOT remove it, so the scatter-add "
             "atomics were not the (only) source. Next suspects, in order: nondeterministic "
             "reductions inside the MJX rollout; XLA autotuning picking different kernels "
             "per process; and any host-side ordering in the replay buffer.")
    print(v)

    json.dump(dict(tensors=tensors, all_identical=bool(all_same),
                   total_elements=total_el, total_differing=total_dif,
                   worst_abs=worst, returns=got, delta=float(delta),
                   c16_delta=C16_DELTA, c16_run_sd=C16_RUN_SD, verdict=v),
              open(os.path.join(HERE, "determinism_results.json"), "w"), indent=1)
    print("\nwrote determinism_results.json")


if __name__ == "__main__":
    main()
