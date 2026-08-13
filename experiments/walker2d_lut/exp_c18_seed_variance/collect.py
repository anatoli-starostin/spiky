"""exp_c18 — evaluate the 6 deterministic seeds and report the seed spread (#75). MJX venv.

Every policy is evaluated with the deterministic 100-episode CPU reference IN HARD MODE,
which is the mode it was trained in. That is not a detail: an earlier round of this work
evaluated hard-trained policies through the soft path and produced numbers that meant
nothing. The eval mode is pinned here rather than passed in, so it cannot drift.

The reported spread is a SEED spread and nothing else. Under exp_c17's determinism flags
each seed names exactly one run, so re-running seed 3 reproduces this number bit for bit;
what varies across the six rows is the seed alone.
"""
import json, os, re, subprocess, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
SEEDS = (0, 1, 2, 3, 4, 5)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)

# For context in the writeup, not for arithmetic:
C16_RUN_SD = 662.6      # same-seed run-to-run sd BEFORE determinism (exp_c16)
C15_MEAN, C15_SD = 4318.5, 178.0   # the 3-seed pre-determinism measurement this replaces


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
        print(f"    FAILED: {err[-400:]}", flush=True)
        return False
    print(f"    done in {time.time() - t0:.0f}s", flush=True)
    return True


def main():
    got, pending = [], []
    for s in SEEDS:
        actor = f"lut_sac_c18_seed{s}_actor.npz"
        log = os.path.join(HERE, f"cell_seed{s}.log")
        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        d = DONE.search(txt)
        if not d:
            pending.append(s)
            print(f"  (seed {s}: still training)", flush=True)
            continue
        ev = os.path.join(C09, actor.replace("_actor.npz", "_cpueval.json"))
        if not os.path.exists(ev) and not run_eval(actor, f"seed{s}"):
            pending.append(s)
            continue
        j = json.load(open(ev))
        # row coverage as logged during training, for the dead-row question
        cov = re.findall(r"row-cov\s+([\d.]+)%", txt)
        got.append(dict(seed=s, mean=j["cpu_reference_mean"], std=j["cpu_reference_std"],
                        mjx_best=float(d.group(1)),
                        train_row_cov=float(cov[-1]) if cov else None))

    if pending:
        print(f"\n{len(got)}/{len(SEEDS)} seeds ready — not concluding "
              f"(waiting on {pending})")
        return

    means = np.array([g["mean"] for g in got], np.float64)
    mu, sd = float(means.mean()), float(means.std(ddof=1))
    lo, hi = float(means.min()), float(means.max())
    arg_lo = got[int(means.argmin())]["seed"]
    arg_hi = got[int(means.argmax())]["seed"]
    # mean within-run episode sd: a different quantity from the seed sd, and confusing the
    # two is the error this whole chapter has been correcting for.
    ep_sd = float(np.mean([g["std"] for g in got]))

    print("\n=== exp_c18 — hyperplane x hard, anchor_pairs init, 6 deterministic seeds ===")
    print(f"{'seed':>6}{'CPU-ref 100ep':>16}{'ep-sd':>9}{'best MJX (20ep)':>18}"
          f"{'train row-cov':>15}")
    for g in sorted(got, key=lambda g: g["seed"]):
        cov = f"{g['train_row_cov']:.1f}%" if g["train_row_cov"] is not None else "n/a"
        print(f"{g['seed']:>6}{g['mean']:>16.1f}{g['std']:>9.1f}"
              f"{g['mjx_best']:>18.1f}{cov:>15}")
    print(f"\n  6-seed mean +/- sd : {mu:.1f} +/- {sd:.1f}")
    print(f"  min-max range      : {lo:.1f} (seed {arg_lo}) .. {hi:.1f} (seed {arg_hi})"
          f"   spread {hi - lo:.1f}")
    print(f"  mean within-run ep-sd: {ep_sd:.1f}   (100-episode spread of ONE policy — "
          f"not the seed spread)")

    # ---- what the number means ------------------------------------------------
    print("\n=== reading ===")
    notes = []
    notes.append(
        f"Each row is now reproducible: determinism was verified in exp_c17 "
        f"(bit-identical checkpoints), so re-running any seed returns its number exactly. "
        f"The {sd:.0f} sd is therefore seed sensitivity, not run noise.")
    if sd > 300:
        notes.append(
            f"The spread is LARGE: sd {sd:.0f} on a mean of {mu:.0f} is "
            f"{100 * sd / mu:.0f}% of the score, and the min-max gap is {hi - lo:.0f}. "
            f"Any A/B on this config that moves the mean by less than ~{2 * sd / np.sqrt(6):.0f} "
            f"at 6 seeds is not measurable. See diag_seeds.py for the why.")
    else:
        notes.append(
            f"The spread is MODEST: sd {sd:.0f} ({100 * sd / mu:.0f}% of the mean). "
            f"At 6 seeds the standard error is {sd / np.sqrt(6):.0f}, so differences of "
            f"~{2 * sd / np.sqrt(6):.0f} and up are resolvable.")
    notes.append(
        f"exp_c15 measured this same config at 3 seeds WITHOUT determinism and got "
        f"{C15_MEAN:.1f} +/- {C15_SD:.1f}. That sd was contaminated by run noise "
        f"(exp_c16: {C16_RUN_SD:.0f} at a fixed seed) and should be read as superseded, "
        f"not as agreeing or disagreeing with this one.")
    for n in notes:
        print(f"  - {n}")

    json.dump(dict(seeds=got, mean=mu, sd=sd, min=lo, max=hi, spread=hi - lo,
                   argmin_seed=arg_lo, argmax_seed=arg_hi, mean_episode_sd=ep_sd,
                   n_seeds=len(got), notes=notes,
                   c15_mean=C15_MEAN, c15_sd=C15_SD, c16_run_sd=C16_RUN_SD),
              open(os.path.join(HERE, "seed_variance_results.json"), "w"), indent=1)
    print("\nwrote seed_variance_results.json")


if __name__ == "__main__":
    main()
