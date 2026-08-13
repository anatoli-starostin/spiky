"""exp_c14 — evaluate the 3 hyperplane seeds and compare with anchors (#75).

The point of this experiment is a single number: the hyperplane arm's seed-sd. exp_c13
measured the anchors arm at 131 -> 474 (median) -> 1850 and attributed that spread to the
frozen draw BEING the architecture. Hyperplanes learn their boundaries, so if that
attribution is right their seed-sd should be much smaller. If it is not, the explanation
is wrong and the spread is coming from LUT-SAC itself.
"""
import json, os, re, subprocess, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
C13 = os.path.join(HERE, "..", "exp_c13_anchors_multiseed", "multiseed_results.json")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
SEEDS = (0, 1, 2)
OLD_SINGLE_SEED = 5146.9        # the original one-seed reference this replaces
COV = re.compile(r"row-cov\s+([\d.]+)%", re.M)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)


def run_eval(actor, label, i, n):
    t0 = time.time()
    print(f"  [{time.strftime('%H:%M:%S', time.gmtime())}] {i}/{n} evaluating {label}",
          flush=True)
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
    got, t_eval = [], time.time()
    for i, s in enumerate(SEEDS, 1):
        actor = f"lut_sac_c14_hyperplane_hard_s{s}_actor.npz"
        log = os.path.join(HERE, f"cell_hyperplane_hard_s{s}.log")
        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        if not DONE.search(txt):
            print(f"  (skip seed {s}: still running)", flush=True)
            continue
        ev = os.path.join(C09, actor.replace("_actor.npz", "_cpueval.json"))
        if not os.path.exists(ev):
            if not run_eval(actor, f"hyperplane/s{s}", i, len(SEEDS)):
                continue
        e = json.load(open(ev))
        covs = COV.findall(txt)
        got.append(dict(seed=s, mean=e["cpu_reference_mean"], std=e["cpu_reference_std"],
                        cov=float(covs[-1]) if covs else float("nan")))
    eval_s = time.time() - t_eval
    if not got:
        print("nothing to aggregate")
        return

    m = np.array([g["mean"] for g in got])
    hyp_mean = float(m.mean())
    hyp_sd = float(m.std(ddof=1)) if len(m) > 1 else float("nan")
    hyp_ep = float(np.mean([g["std"] for g in got]))

    print(f"\n=== hyperplane x hard nap6/tph32, {len(got)} seeds ===")
    for g in got:
        print(f"  seed {g['seed']}: {g['mean']:7.1f} +/- {g['std']:6.1f} (episodes)"
              f"   cov {g['cov']:.1f}%")
    print(f"  3-seed mean {hyp_mean:.1f} | seed-sd {hyp_sd:.1f} | "
          f"mean ep-sd {hyp_ep:.1f} | range {m.min():.0f}-{m.max():.0f}")
    print(f"  (the single-seed number this replaces: {OLD_SINGLE_SEED})")

    rows = json.load(open(C13))
    a_sds = np.array([r["seed_std"] for r in rows])
    best = max(rows, key=lambda r: r["seed_mean"])
    print(f"\n=== seed sensitivity: learned boundaries vs a frozen draw ===")
    print(f"  hyperplane seed-sd            {hyp_sd:8.1f}")
    print(f"  anchors seed-sd  min/med/max  {a_sds.min():8.1f} / "
          f"{np.median(a_sds):.1f} / {a_sds.max():.1f}  (9 configs)")
    n_above = int((a_sds > hyp_sd).sum())
    print(f"  {n_above}/9 anchors configs are MORE seed-sensitive than hyperplane")
    ratio = np.median(a_sds) / hyp_sd if hyp_sd else float("inf")
    print(f"  median anchors seed-sd is {ratio:.1f}x the hyperplane's")
    print("  VERDICT: " + ("learned hyperplanes ARE less seed-sensitive — consistent "
                           "with boundaries being trained rather than drawn"
                           if hyp_sd < np.median(a_sds) else
                           "hyperplane is NOT less seed-sensitive — the frozen-draw "
                           "explanation for the anchors spread does not hold"))

    # Headline, both sides now multi-seed. Separation in units of the combined spread:
    # sqrt(sd_a^2 + sd_h^2) is the sd of the DIFFERENCE of two independent means.
    gap = hyp_mean - best["seed_mean"]
    comb = float(np.hypot(best["seed_std"], hyp_sd))
    print(f"\n=== headline on equal footing (both 3-seed) ===")
    print(f"  hyperplane x hard nap6/tph32   {hyp_mean:7.1f} +/- {hyp_sd:.1f}  "
          f"(28,034 params, 3,648 active)")
    print(f"  best anchors nap{best['nap']}/tph{best['tph']}      "
          f"{best['seed_mean']:7.1f} +/- {best['seed_std']:.1f}  "
          f"({best['total']:,} params, {best['act12']:,} active)")
    print(f"  gap {gap:.1f} = {gap / comb:.2f} combined sd "
          f"(combined sd {comb:.1f}); anchors reaches "
          f"{100 * best['seed_mean'] / hyp_mean:.0f}% of the hyperplane mean")

    json.dump(dict(seeds=got, mean=hyp_mean, seed_std=hyp_sd, ep_std=hyp_ep,
                   old_single_seed=OLD_SINGLE_SEED, eval_seconds=eval_s,
                   anchors_best=dict(nap=best["nap"], tph=best["tph"],
                                     mean=best["seed_mean"], seed_std=best["seed_std"]),
                   gap=gap, combined_sd=comb),
              open(os.path.join(HERE, "hyperplane_multiseed.json"), "w"), indent=1)
    print(f"\n{len(got)} evals in {eval_s:.0f}s -> hyperplane_multiseed.json")


if __name__ == "__main__":
    main()
