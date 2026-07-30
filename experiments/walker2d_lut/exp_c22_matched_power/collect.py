"""exp_c22 — LUT vs param-matched MLP at n=12 (#75). MJX venv.

EVERY THRESHOLD BELOW IS FIXED BEFORE THE DATA EXISTS. That is the point of writing them
into the file rather than deciding them afterwards, and it is why the verdict can and will
come back "tie" if that is what the numbers say.

  performance : Welch's two-sample t-test on the 12 CPU-reference scores per arm (Welch,
                not Student, because exp_c19 gave every reason to expect unequal variances
                and Welch costs nothing when they happen to be equal). Significant at
                p < 0.05, two-sided. Effect size reported as Hedges' g (Cohen's d with the
                small-sample correction), since d is biased upward at n=12.
  reliability : variance ratio var(MLP)/var(LUT) against F(0.95; 11, 11) = 2.8179, two-sided
                at the 5% level, so the ratio must exceed 2.8179 or fall below 1/2.8179.
  stability   : mean retention = final score / that run's own best. Reported two ways --
                the cross-evaluator form used in exp_c19 (CPU-reference final over best MJX
                proxy) for continuity, and a within-evaluator form (final MJX over best MJX,
                both from the training history) which is the cleaner measure because it
                does not mix two different evaluators in one ratio.

The bottom line is then read off a fixed 2x2: better on performance, on reliability, on
both, or neither.
"""
import json, os, re, subprocess, sys, time

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C09 = os.path.join(D, "exp_c09_lut_sac")
C18 = os.path.join(D, "exp_c18_seed_variance")
C19 = os.path.join(D, "exp_c19_mlp_sac_control")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")

SEEDS = tuple(range(12))
NEW_LUT = (6, 7, 8, 9, 10, 11)      # 0-5 are reused from exp_c18
HIDDEN = 153
LUT_PARAMS, MLP_PARAMS = 28032, 28164
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)

ALPHA = 0.05
F_CRIT = 2.8179                      # F(0.95; 11, 11)


def run_eval(script, cwd, actor, label, extra=()):
    t0 = time.time()
    print(f"  [{time.strftime('%H:%M:%S', time.gmtime())}] evaluating {label}", flush=True)
    p = subprocess.Popen(
        [PY, "-u", script, actor, "--episodes", "100", "--progress", label, *extra],
        cwd=cwd, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    for line in p.stdout:
        line = line.rstrip()
        if "CPU-reference" in line:
            print(f"    -> {line.strip()}", flush=True)
    err = p.stderr.read()
    if p.wait() != 0:
        print(f"    FAILED: {err[-400:]}", flush=True)
        return None
    print(f"    done in {time.time() - t0:.0f}s", flush=True)
    return True


def collect_arm(name):
    """Returns (rows, pending). Each row: seed, score, ep_sd, best_mjx, final_mjx."""
    rows, pending = [], []
    for s in SEEDS:
        if name == "lut":
            if s in NEW_LUT:
                tag, log = f"lut_sac_c22_lut_s{s}", os.path.join(HERE, f"cell_lut_s{s}.log")
            else:
                tag, log = (f"lut_sac_c18_seed{s}",
                            os.path.join(C18, f"cell_seed{s}.log"))
            script, cwd = os.path.join(C09, "eval_cpu.py"), C09
            extra, home = ("--forward-mode", "hard"), C09
        else:
            tag = f"mlp_sac_c22_mlp{HIDDEN}_s{s}"
            log = os.path.join(HERE, f"cell_mlp_s{s}.log")
            script, cwd = os.path.join(C19, "eval_cpu_mlp.py"), C19
            extra, home = (), C19

        txt = open(log, errors="replace").read() if os.path.exists(log) else ""
        d = DONE.search(txt)
        if not d:
            pending.append(f"{name}/s{s}")
            continue
        ev = os.path.join(home, f"{tag}_cpueval.json")
        if not os.path.exists(ev) and not run_eval(script, cwd, f"{tag}_actor.npz",
                                                   f"{name}-s{s}", extra):
            pending.append(f"{name}/s{s}")
            continue
        j = json.load(open(ev))
        hist = json.load(open(os.path.join(home, f"{tag}.json")))["history"]
        mjx = [h["mjx_return"] for h in hist]
        rows.append(dict(seed=s, score=j["cpu_reference_mean"], ep_sd=j["cpu_reference_std"],
                         params=j["params"], best_mjx=float(d.group(1)),
                         final_mjx=float(mjx[-1]), max_mjx=float(max(mjx))))
    return rows, pending


def describe(rows):
    v = np.array([r["score"] for r in rows], np.float64)
    return dict(n=len(v), mean=float(v.mean()), sd=float(v.std(ddof=1)),
                min=float(v.min()), max=float(v.max()),
                cv=float(v.std(ddof=1) / v.mean()),
                retention_cross=float(np.mean([r["score"] / r["best_mjx"] for r in rows])),
                retention_within=float(np.mean([r["final_mjx"] / r["max_mjx"]
                                                for r in rows])),
                values=v.tolist())


def main():
    lut, p1 = collect_arm("lut")
    mlp, p2 = collect_arm("mlp")
    if p1 or p2:
        print(f"\nLUT {len(lut)}/12, MLP {len(mlp)}/12 ready — not concluding "
              f"(waiting on {(p1 + p2)[:8]}{' …' if len(p1+p2) > 8 else ''})")
        return

    L, M = describe(lut), describe(mlp)
    print(f"\n=== exp_c22 — LUT vs param-matched MLP, n=12 each ===")
    print(f"  LUT actor {LUT_PARAMS:,} params | MLP 2x{HIDDEN} = {MLP_PARAMS:,} params "
          f"(+{100*(MLP_PARAMS-LUT_PARAMS)/LUT_PARAMS:.2f}%)")
    print(f"\n{'seed':>5}{'LUT':>10}{'MLP':>10}")
    for a_, b_ in zip(sorted(lut, key=lambda r: r["seed"]),
                      sorted(mlp, key=lambda r: r["seed"])):
        print(f"{a_['seed']:>5}{a_['score']:>10.1f}{b_['score']:>10.1f}")
    print(f"\n{'arm':<10}{'n':>4}{'mean':>10}{'sd':>9}{'min':>9}{'max':>9}{'cv':>8}"
          f"{'ret(x-eval)':>13}{'ret(within)':>13}")
    for nm, S in (("LUT", L), ("MLP", M)):
        print(f"{nm:<10}{S['n']:>4}{S['mean']:>10.1f}{S['sd']:>9.1f}{S['min']:>9.1f}"
              f"{S['max']:>9.1f}{S['cv']:>8.3f}{S['retention_cross']:>13.3f}"
              f"{S['retention_within']:>13.3f}")

    a = np.array(L["values"]); b = np.array(M["values"])
    # ---- performance: Welch ------------------------------------------------
    t, p = stats.ttest_ind(a, b, equal_var=False)
    df = ((a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b)) ** 2
          / ((a.var(ddof=1)/len(a))**2/(len(a)-1) + (b.var(ddof=1)/len(b))**2/(len(b)-1)))
    sp = np.sqrt(((len(a)-1)*a.var(ddof=1) + (len(b)-1)*b.var(ddof=1)) / (len(a)+len(b)-2))
    d_cohen = (a.mean() - b.mean()) / sp
    J = 1 - 3 / (4 * (len(a) + len(b)) - 9)          # Hedges' small-sample correction
    g = d_cohen * J
    perf_win = p < ALPHA
    print(f"\n=== PERFORMANCE (pre-registered: Welch two-sided, p < {ALPHA}) ===")
    print(f"  mean difference LUT - MLP = {a.mean() - b.mean():+.1f}")
    print(f"  Welch t = {t:+.3f}, df = {df:.1f}, p = {p:.4g}   -> "
          f"{'SIGNIFICANT' if perf_win else 'not significant'}")
    print(f"  Hedges' g = {g:+.3f} (Cohen's d {d_cohen:+.3f}, corrected)")

    # ---- reliability: F ----------------------------------------------------
    vr = b.var(ddof=1) / a.var(ddof=1)
    rel_win = vr > F_CRIT
    rel_lose = vr < 1 / F_CRIT
    print(f"\n=== RELIABILITY (pre-registered: var(MLP)/var(LUT) vs F(0.95;11,11) = "
          f"{F_CRIT}) ===")
    print(f"  sd  LUT {a.std(ddof=1):.1f}  vs  MLP {b.std(ddof=1):.1f}")
    print(f"  variance ratio MLP/LUT = {vr:.3f}   thresholds [{1/F_CRIT:.3f}, {F_CRIT:.3f}]"
          f"   -> " + ("LUT more reliable" if rel_win else
                       "MLP more reliable" if rel_lose else "indistinguishable"))

    # ---- bottom line, from the fixed 2x2 -----------------------------------
    better_perf = perf_win and a.mean() > b.mean()
    worse_perf = perf_win and a.mean() < b.mean()
    if better_perf and rel_win:
        bl = ("THE LUT WINS ON BOTH. At matched parameters and n=12 it scores higher "
              "(p < 0.05) and is significantly less seed-sensitive. The exp_c19 result was "
              "not a capacity artefact.")
    elif better_perf:
        bl = ("THE LUT WINS ON PERFORMANCE ONLY. Higher mean at p < 0.05, but the variance "
              f"ratio {vr:.2f} does not clear {F_CRIT}, so the reliability edge seen at "
              "n=6 is not confirmed at matched parameters.")
    elif rel_win:
        bl = (f"THE LUT WINS ON RELIABILITY ONLY. Variance ratio {vr:.2f} clears "
              f"{F_CRIT}, but the means are statistically indistinguishable "
              f"(p = {p:.3g}). The LUT is the more DEPENDABLE actor at equal parameters, "
              "not the stronger one.")
    elif worse_perf:
        bl = (f"THE MLP WINS ON PERFORMANCE. At matched parameters it scores "
              f"{b.mean()-a.mean():+.0f} higher (p = {p:.3g}). exp_c19's favourable "
              "reading of the LUT was carried by the MLP's excess capacity, not helped "
              "by it.")
    else:
        bl = (f"IT IS A TIE. Neither the mean difference (p = {p:.3g}) nor the variance "
              f"ratio ({vr:.2f}, thresholds [{1/F_CRIT:.2f}, {F_CRIT:.2f}]) clears its "
              "pre-registered threshold at n=12. At matched parameters the two actors are "
              "not distinguishable on this task.")
    print(f"\n=== BOTTOM LINE ===\n  {bl}")

    print(f"\n  For reference, exp_c19's UNMATCHED comparison (MLP 2x256, 73,484 params, "
          f"n=6): LUT 4308.0 +/- 500.1 vs MLP 3450.5 +/- 1506.5, variance ratio 9.1x.")

    json.dump(dict(lut=dict(L, runs=lut), mlp=dict(M, runs=mlp),
                   lut_params=LUT_PARAMS, mlp_params=MLP_PARAMS, hidden=HIDDEN,
                   welch_t=float(t), welch_df=float(df), p_value=float(p),
                   hedges_g=float(g), cohens_d=float(d_cohen),
                   variance_ratio_mlp_over_lut=float(vr), f_crit=F_CRIT, alpha=ALPHA,
                   performance_significant=bool(perf_win),
                   reliability_lut_better=bool(rel_win), bottom_line=bl),
              open(os.path.join(HERE, "matched_power_results.json"), "w"), indent=1)
    print("\nwrote matched_power_results.json")


if __name__ == "__main__":
    main()
