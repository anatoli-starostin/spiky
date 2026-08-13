"""exp_c20 — did seed 4's routing carry the win? (#75). MJX venv.

Reads both arms and reports the comparison that actually answers the question:

    arm A  seed 4's frozen routing  + 3 fresh table/critic/exploration seeds
    arm B  seed 5's frozen routing  + the SAME 3 fresh seeds

A vs B isolates the routing, because everything else about the two arms is identical --
including the freezing itself, whose cost would otherwise be attributed to the routing.
A vs seed 4's own 5286.6 is reported too, but as a secondary number: that comparison also
changes the table content and removes the joint optimisation, so it confounds three things
at once.

Verdict logic is stated in advance, in the code, so it is not chosen after seeing the data:
  * A ~= 5300 and clearly above B  -> the routing carries the win
  * A ~= B, both well below 5287   -> the win is in the joint solution, not the routing
  * A > B but both depressed       -> the routing carries PART of it and freezing costs the
                                     rest; the gap A - B is the routing's transferable share
"""
import json, os, re, subprocess, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
C18 = os.path.join(HERE, "..", "exp_c18_seed_variance")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
SEEDS = (100, 101, 102)
ARMS = (("from4", 4, "seed 4's routing (the winner)"),
        ("from5", 5, "seed 5's routing (pack control)"))
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
SEED4 = 5286.6
PACK_MEAN, PACK_SD = 4112.3, 159.2      # exp_c18's five non-outlier seeds


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
    got, pending = {a: [] for a, _, _ in ARMS}, []
    for arm, src, _lab in ARMS:
        for s in SEEDS:
            actor = f"lut_sac_c20_{arm}_s{s}_actor.npz"
            log = os.path.join(HERE, f"cell_{arm}_s{s}.log")
            txt = open(log, errors="replace").read() if os.path.exists(log) else ""
            d = DONE.search(txt)
            if not d:
                pending.append(f"{arm}/s{s}")
                continue
            ev = os.path.join(C09, actor.replace("_actor.npz", "_cpueval.json"))
            if not os.path.exists(ev) and not run_eval(actor, f"{arm}-s{s}"):
                pending.append(f"{arm}/s{s}")
                continue
            j = json.load(open(ev))
            got[arm].append(dict(seed=s, src=src, mean=j["cpu_reference_mean"],
                                 std=j["cpu_reference_std"], mjx_best=float(d.group(1))))

    if pending:
        n = sum(len(v) for v in got.values())
        print(f"\n{n}/{len(SEEDS)*len(ARMS)} runs ready — not concluding "
              f"(waiting on {pending})")
        return

    print("\n=== exp_c20 — frozen transplanted routing, table content relearned ===")
    summ = {}
    for arm, src, lab in ARMS:
        v = np.array([g["mean"] for g in got[arm]], np.float64)
        summ[arm] = dict(mean=float(v.mean()), sd=float(v.std(ddof=1)),
                         min=float(v.min()), max=float(v.max()), src=src, label=lab)
        print(f"\n  {lab}   (routing frozen from exp_c18 seed {src})")
        print(f"{'fresh seed':>12}{'CPU-ref 100ep':>16}{'ep-sd':>9}{'best MJX':>11}")
        for g in sorted(got[arm], key=lambda g: g["seed"]):
            print(f"{g['seed']:>12}{g['mean']:>16.1f}{g['std']:>9.1f}"
                  f"{g['mjx_best']:>11.1f}")
        print(f"{'mean +/- sd':>12}{v.mean():>16.1f} +/- {v.std(ddof=1):.1f}"
              f"   range {v.min():.1f}-{v.max():.1f}")

    a_, b_ = summ["from4"], summ["from5"]
    gap = a_["mean"] - b_["mean"]
    # pooled sd of two 3-sample groups, and the 95% CI half-width for their difference
    sp = np.sqrt((a_["sd"] ** 2 + b_["sd"] ** 2) / 2)
    se = sp * np.sqrt(2 / len(SEEDS))
    ci = 2.776 * se          # t(0.975, df=4)
    print("\n=== THE COMPARISON THAT ISOLATES THE ROUTING ===")
    print(f"  arm A  seed 4's routing : {a_['mean']:7.1f} +/- {a_['sd']:.1f}")
    print(f"  arm B  seed 5's routing : {b_['mean']:7.1f} +/- {b_['sd']:.1f}")
    print(f"  A - B = {gap:+.1f}   95% CI [{gap-ci:+.1f}, {gap+ci:+.1f}]  "
          f"(t, df=4, pooled sd {sp:.1f})")
    print("\n  reference points, NOT the primary comparison:")
    print(f"    seed 4 trained jointly, unfrozen : {SEED4:.1f}")
    print(f"    exp_c18 pack (5 non-outliers)    : {PACK_MEAN:.1f} +/- {PACK_SD:.1f}")
    print(f"    freezing penalty on the pack routing: "
          f"{b_['mean'] - PACK_MEAN:+.1f} (arm B vs the pack it came from)")

    print("\n=== VERDICT ===")
    routing_matters = (gap - ci) > 0
    recovers = a_["mean"] > SEED4 - 2 * max(a_["sd"], 1.0)
    if routing_matters and recovers:
        v = (f"THE ROUTING CARRIES THE WIN. Arm A reaches {a_['mean']:.0f}, statistically "
             f"indistinguishable from seed 4's own {SEED4:.0f}, and beats the pack-routing "
             f"control by {gap:+.0f} with the CI excluding zero. Seed 4's learned "
             f"addressing is transferable and its table content is interchangeable -- the "
             f"win is in WHERE the policy routes, not in the particular table it filled in.")
    elif routing_matters:
        v = (f"THE ROUTING CARRIES PART OF IT. Arm A beats the control by {gap:+.0f} "
             f"(CI excludes zero), so seed 4's addressing is genuinely better, but at "
             f"{a_['mean']:.0f} it does not recover the {SEED4:.0f} it had when trained "
             f"jointly. The shortfall of {SEED4 - a_['mean']:.0f} is the part that needed "
             f"the addressing and the table to co-adapt.")
    elif abs(gap) <= ci:
        # Deliberately NOT phrased as "the routing does not transfer". A non-significant
        # difference of means at n=3 per arm is not evidence of absence, and this outcome
        # is bimodal (exp_c18: five seeds at ~4112, one at 5287) -- a t-test on a bimodal
        # variable spends its power estimating a mean no run sits near. See basin.py,
        # which asks the binary question this data is actually shaped for.
        v = (f"THIS TEST CANNOT TELL. Arm A {a_['mean']:.0f} +/- {a_['sd']:.0f} and arm B "
             f"{b_['mean']:.0f} +/- {b_['sd']:.0f} give A - B = {gap:+.0f} with CI "
             f"[{gap-ci:+.0f}, {gap+ci:+.0f}], which contains zero and is far too wide to "
             f"exclude anything -- it is consistent both with no effect and with the full "
             f"{SEED4 - PACK_MEAN:.0f} being transferred. Do NOT read this as 'the routing "
             f"does not transfer'. The outcome is bimodal, so the informative question is "
             f"whether each run reached the fast-gait basin, not where the means fell: "
             f"see basin.py.")
    else:
        v = (f"UNEXPECTED: the PACK routing beat seed 4's by {-gap:.0f} (CI excludes "
             f"zero). Frozen routing quality does not track the joint result at all, "
             f"which would itself be worth understanding before drawing any conclusion "
             f"about addressing.")
    print(v)
    if b_["mean"] < PACK_MEAN - 200:
        print(f"\n  NOTE: arm B came in {PACK_MEAN - b_['mean']:.0f} below the pack it was "
              f"taken from, so freezing the addressing costs real return on its own. That "
              f"is exactly why arm B exists; read A against B, not against 5286.6.")

    json.dump(dict(arms={k: dict(summ[k], runs=got[k]) for k in summ},
                   gap=float(gap), ci_half_width=float(ci), pooled_sd=float(sp),
                   seed4_joint=SEED4, pack_mean=PACK_MEAN, pack_sd=PACK_SD,
                   verdict=v, seeds=list(SEEDS)),
              open(os.path.join(HERE, "transplant_results.json"), "w"), indent=1)
    print("\nwrote transplant_results.json")


if __name__ == "__main__":
    main()
