"""exp_c19 — the MLP control's 6-seed spread, and the LUT comparison (#75). MJX venv.

The number this exists to produce is NOT the MLP's score. It is the RATIO of the two seed
sds. If the MLP's spread is comparable to the LUT's, the LUT spread is SAC-on-Walker2d and
says nothing about the representation; if the MLP's is much smaller, the spread is the
LUT's own. Both outcomes are useful, so the verdict below states which one was observed
rather than looking for one of them.

One statistical caution is applied rather than mentioned: comparing two sds from 6 samples
each is weak. The F-test's 5% critical value at (5, 5) degrees of freedom is 5.05, so a
variance ratio has to exceed ~5x (i.e. an sd ratio of ~2.25x) before it is distinguishable
from chance. That threshold is applied to the verdict, not left to the reader.
"""
import json, os, re, subprocess, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C18 = os.path.join(HERE, "..", "exp_c18_seed_variance")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
SEEDS = (0, 1, 2, 3, 4, 5)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
F_CRIT_5_5 = 5.05          # F(0.95; 5, 5): the variance ratio needed at n=6 per group


def run_eval(actor, label):
    t0 = time.time()
    print(f"  [{time.strftime('%H:%M:%S', time.gmtime())}] evaluating {label}", flush=True)
    p = subprocess.Popen(
        [PY, "-u", os.path.join(HERE, "eval_cpu_mlp.py"), actor,
         "--episodes", "100", "--progress", label],
        cwd=HERE, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"),
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
        actor = f"mlp_sac_c19_seed{s}_actor.npz"
        log = os.path.join(HERE, f"cell_seed{s}.log")
        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        d = DONE.search(txt)
        if not d:
            pending.append(s)
            print(f"  (MLP seed {s}: still training)", flush=True)
            continue
        ev = os.path.join(HERE, actor.replace("_actor.npz", "_cpueval.json"))
        if not os.path.exists(ev) and not run_eval(actor, f"mlp-s{s}"):
            pending.append(s)
            continue
        j = json.load(open(ev))
        got.append(dict(seed=s, mean=j["cpu_reference_mean"], std=j["cpu_reference_std"],
                        params=j["params"], mjx_best=float(d.group(1))))

    if pending:
        print(f"\n{len(got)}/{len(SEEDS)} MLP seeds ready — not concluding "
              f"(waiting on {pending})")
        return

    means = np.array([g["mean"] for g in got], np.float64)
    mu, sd = float(means.mean()), float(means.std(ddof=1))
    lo, hi = float(means.min()), float(means.max())

    print("\n=== exp_c19 — MLP-actor SAC control, 6 deterministic seeds ===")
    print(f"{'seed':>6}{'CPU-ref 100ep':>16}{'ep-sd':>9}{'best MJX (20ep)':>18}")
    for g in sorted(got, key=lambda g: g["seed"]):
        print(f"{g['seed']:>6}{g['mean']:>16.1f}{g['std']:>9.1f}{g['mjx_best']:>18.1f}")
    print(f"\n  6-seed mean +/- sd : {mu:.1f} +/- {sd:.1f}")
    print(f"  min-max range      : {lo:.1f} .. {hi:.1f}   spread {hi - lo:.1f}")
    print(f"  actor params       : {got[0]['params']:,} (LUT: 28,032)")

    # ---- the comparison this run exists for --------------------------------
    lut = None
    lut_path = os.path.join(C18, "seed_variance_results.json")
    if os.path.exists(lut_path):
        lut = json.load(open(lut_path))

    verdict = None
    if lut:
        l_mu, l_sd = lut["mean"], lut["sd"]
        ratio = (l_sd / sd) if sd > 0 else float("inf")
        var_ratio = ratio ** 2
        print("\n=== LUT vs MLP, seed sensitivity ===")
        print(f"{'actor':<22}{'mean':>10}{'seed sd':>10}{'range':>22}{'cv':>8}")
        l_range = f"{lut['min']:.0f}-{lut['max']:.0f}"
        print(f"{'LUT (hyperplane/hard)':<22}{l_mu:>10.1f}{l_sd:>10.1f}"
              f"{l_range:>22}{100 * l_sd / l_mu:>7.1f}%")
        print(f"{'MLP (2x256)':<22}{mu:>10.1f}{sd:>10.1f}"
              f"{f'{lo:.0f}-{hi:.0f}':>22}{100 * sd / mu:>7.1f}%")
        print(f"\n  sd ratio LUT/MLP = {ratio:.2f}x   (variance ratio {var_ratio:.2f}x; "
              f"F(0.95;5,5) = {F_CRIT_5_5:.2f})")
        if var_ratio > F_CRIT_5_5:
            verdict = (f"THE LUT IS GENUINELY MORE SEED-SENSITIVE. Variance ratio "
                       f"{var_ratio:.1f}x clears the F(0.95;5,5) threshold of "
                       f"{F_CRIT_5_5:.2f}, so the LUT's spread is not simply "
                       f"SAC-on-Walker2d variance -- the representation contributes.")
        elif var_ratio < 1 / F_CRIT_5_5:
            verdict = (f"THE MLP IS THE MORE SEED-SENSITIVE OF THE TWO (variance ratio "
                       f"{var_ratio:.2f}x). Whatever the LUT's spread is, it is not "
                       f"worse than the standard baseline's.")
        else:
            verdict = (f"NOT DISTINGUISHABLE at 6 seeds each. The variance ratio "
                       f"{var_ratio:.2f}x sits inside [{1/F_CRIT_5_5:.2f}, "
                       f"{F_CRIT_5_5:.2f}], the band an F-test cannot separate from "
                       f"chance at this sample size. The honest reading is that the "
                       f"LUT's spread is CONSISTENT WITH ordinary SAC-on-Walker2d seed "
                       f"variance; separating them would need ~15-20 seeds per arm.")
        print(f"\n  VERDICT: {verdict}")
        print(f"\n  (Mean scores are NOT the comparison here and the MLP has "
              f"{got[0]['params'] / 28032:.1f}x the actor parameters; this run was "
              f"commissioned to measure SPREAD, not to rank the two actors.)")
    else:
        print("\n(exp_c18 results not found — MLP spread reported alone)")

    json.dump(dict(seeds=got, mean=mu, sd=sd, min=lo, max=hi, spread=hi - lo,
                   actor_params=got[0]["params"], n_seeds=len(got),
                   lut_mean=(lut["mean"] if lut else None),
                   lut_sd=(lut["sd"] if lut else None),
                   sd_ratio_lut_over_mlp=((lut["sd"] / sd) if lut and sd else None),
                   f_crit_5_5=F_CRIT_5_5, verdict=verdict),
              open(os.path.join(HERE, "mlp_control_results.json"), "w"), indent=1)
    print("\nwrote mlp_control_results.json")


if __name__ == "__main__":
    main()
