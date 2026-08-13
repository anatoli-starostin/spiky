"""exp_c21 — what did doubling the budget buy on seed 4? (#75). MJX venv.

Three things, in order:

  1. IDENTITY CHECK at the 10k mark. Under determinism, a 20k run's state at iteration
     10,000 must be bit-identical to the 10k run's final state -- `iters` only bounds the
     loop and nothing downstream of it touches the RNG chain. Asserting that (rather than
     assuming it) does two jobs: it proves the two runs are the same trajectory, so the
     10k->20k delta is a genuine within-run gain and not a comparison across different
     runs; and it re-verifies exp_c17's determinism fix on a fresh 20k run.
  2. THE SCORES. Deterministic 100-episode CPU-reference eval in hard mode at 20k, against
     seed 4's 10k number of 5286.6. The 10k checkpoint is evaluated too -- and if the
     identity check passed, that eval must return 5286.6, which is a third check.
  3. HAS IT CONVERGED BY 20k? The same early-vs-late movement measure exp_c18 used, now
     with a late window of 18,000-20,000, plus the 8,000-10,000 window so the question
     "was it still moving at 10k?" and "is it still moving at 20k?" are answered on one
     trajectory rather than across runs.
"""
import json, os, re, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
C09 = os.path.join(D, "exp_c09_lut_sac")
C03 = os.path.join(D, "exp_c03_distillation")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
TAG = "lut_sac_c21_seed4_20k"
REF_10K = 5286.6           # exp_c18 seed 4, 10,000 iters
PACK = "4112.3 +/- 159.2"  # exp_c18's five non-outlier seeds
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
N_OBS, OBS_SEED = 20000, 0


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
        return None
    print(f"    done in {time.time() - t0:.0f}s", flush=True)
    return json.load(open(os.path.join(
        C09, actor.replace("_actor.npz", "_cpueval.json"))))


def ev_for(actor, label):
    p = os.path.join(C09, actor.replace("_actor.npz", "_cpueval.json"))
    return json.load(open(p)) if os.path.exists(p) else run_eval(actor, label)


def load_obs():
    st = json.load(open(os.path.join(C03, "dataset_stats.json")))
    om = np.asarray(st["obs_mean"], np.float64)
    osd = np.asarray(st["obs_std"], np.float64)
    o = np.load(os.path.join(C03, "obs.npy"), mmap_mode="r")
    idx = np.sort(np.random.default_rng(OBS_SEED).choice(len(o), N_OBS, replace=False))
    return (np.asarray(o[idx], np.float64) - om) / (osd + 1e-6)


def bits(x, w, b):
    return (np.einsum("bd,tnd->btn", x, w) + b[None]) > 0


def angle(w0, w1):
    a_, b_ = w0.reshape(-1, w0.shape[-1]), w1.reshape(-1, w1.shape[-1])
    cos = ((a_ * b_).sum(-1)
           / (np.linalg.norm(a_, axis=-1) * np.linalg.norm(b_, axis=-1) + 1e-12))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    log = os.path.join(HERE, "cell_seed4_20k.log")
    txt = open(log, errors="replace").read() if os.path.exists(log) else ""
    if not DONE.search(txt):
        print("  (20k run still training — not concluding)")
        return
    best20 = float(DONE.search(txt).group(1))

    # ---- 1. identity at the 10k mark ---------------------------------------
    print("=== 1. IS THE 20k RUN THE SAME TRAJECTORY AS THE 10k RUN? ===")
    p10 = os.path.join(C09, f"{TAG}_at10000_actor.npz")
    p18 = os.path.join(C09, "lut_sac_c18_seed4_actor.npz")
    identical = None
    if os.path.exists(p10) and os.path.exists(p18):
        za, zb = np.load(p10), np.load(p18)
        n_dif = tot = 0
        worst = 0.0
        for k in ("w", "b", "weights", "log_T_soft", "log_T_sel"):
            a_, b_ = np.asarray(za[k], np.float64), np.asarray(zb[k], np.float64)
            n_dif += int((a_ != b_).sum()); tot += a_.size
            worst = max(worst, float(np.abs(a_ - b_).max()))
        identical = n_dif == 0
        print(f"  20k run @ iter 10,000  vs  exp_c18 seed 4 final: "
              f"{n_dif:,} of {tot:,} elements differ, max|Δ| {worst:.3e}")
        print("  -> BIT-IDENTICAL. The 10k->20k gain below is a within-run gain on one "
              "trajectory, and exp_c17's determinism fix is re-confirmed on a fresh run."
              if identical else
              "  -> NOT identical. Something outside `iters` differs between the two runs; "
              "treat the 10k->20k comparison as across runs, not within one.")
    else:
        print("  (missing checkpoint — cannot check)")

    # ---- 2. the scores ------------------------------------------------------
    print("\n=== 2. SCORES (deterministic 100-episode CPU reference, hard mode) ===")
    e20 = ev_for(f"{TAG}_actor.npz", "20k")
    e10 = ev_for(f"{TAG}_at10000_actor.npz", "10k") if os.path.exists(p10) else None
    if e20 is None:
        print("  eval failed — stopping")
        return
    s20 = e20["cpu_reference_mean"]
    s10 = e10["cpu_reference_mean"] if e10 else None
    print(f"  seed 4 @ 10,000 iters (exp_c18)   : {REF_10K:7.1f}")
    if s10 is not None:
        print(f"  seed 4 @ 10,000 iters (this run)  : {s10:7.1f} +/- "
              f"{e10['cpu_reference_std']:.1f}"
              + ("   (matches, as it must)" if abs(s10 - REF_10K) < 0.5
                 else f"   MISMATCH vs {REF_10K:.1f}"))
    print(f"  seed 4 @ 20,000 iters             : {s20:7.1f} +/- "
          f"{e20['cpu_reference_std']:.1f}   best MJX {best20:.1f}")
    gain = s20 - REF_10K
    print(f"\n  10k -> 20k gain: {gain:+.1f}  ({100 * gain / REF_10K:+.1f}%)")
    print(f"  for scale: the exp_c18 pack sat at {PACK}, and seed 4's edge over it "
          f"was +1174")

    # ---- 3. still moving at 20k? -------------------------------------------
    print("\n=== 3. HAS THE ADDRESSING CONVERGED BY 20k? ===")
    sn = os.path.join(C09, f"{TAG}_snaps.npz")
    mv = None
    if os.path.exists(sn):
        x = load_obs()
        zs = np.load(sn)
        its = np.asarray(zs["iters"])
        W, B = np.asarray(zs["w"], np.float64), np.asarray(zs["b"], np.float64)
        seg = []
        prev = bits(x, W[0], B[0])
        for i in range(1, len(its)):
            d = int(its[i] - its[i - 1])
            cur = bits(x, W[i], B[i])
            seg.append(dict(it=int(its[i]),
                            rot=float(angle(W[i - 1], W[i]).mean() * 500 / d),
                            flip=float((prev != cur).mean() * 500 / d)))
            prev = cur

        def win(lo, hi):
            g = [s for s in seg if lo < s["it"] <= hi]
            return (float(np.mean([s["rot"] for s in g])),
                    float(np.mean([s["flip"] for s in g]))) if g else (np.nan, np.nan)

        wins = [("early   500-2,500", win(500, 2500)),
                ("mid   8,000-10,000", win(8000, 10000)),
                ("late 18,000-20,000", win(18000, 20000))]
        print(f"{'window':<22}{'rotation/500':>14}{'bit-flip/500':>15}")
        for lab, (r, f) in wins:
            print(f"{lab:<22}{r:>13.2f}°{100*f:>14.2f}%")
        e_rot, e_flip = wins[0][1]
        m_rot, m_flip = wins[1][1]
        l_rot, l_flip = wins[2][1]
        mv = dict(early_rot=e_rot, early_flip=e_flip, mid_rot=m_rot, mid_flip=m_flip,
                  late_rot=l_rot, late_flip=l_flip,
                  late_over_early=float(l_flip / e_flip) if e_flip else None,
                  late_over_mid=float(l_flip / m_flip) if m_flip else None,
                  segments=seg)
        print(f"\n  late/early = {l_flip/e_flip:.2f}   late/mid(10k) = {l_flip/m_flip:.2f}")
        if l_flip > 0.6 * m_flip:
            verd = (f"STILL MOVING. The final 2,000 iterations rewrite {100*l_flip:.2f}% "
                    f"of address bits per 500, only {l_flip/m_flip:.2f}x the rate at the "
                    f"10k mark. Doubling the budget did not reach a resting point either; "
                    f"the addressing is not converging so much as decaying slowly.")
        elif l_flip > 0.15 * e_flip:
            verd = (f"SLOWING BUT NOT SETTLED. Late churn is {l_flip/m_flip:.2f}x the 10k "
                    f"rate and {l_flip/e_flip:.2f}x the early rate -- markedly quieter, "
                    f"still non-zero.")
        else:
            verd = (f"CONVERGED. Late churn is {l_flip/e_flip:.2f}x the early rate; the "
                    f"addressing has settled well inside 20k, so 20k is a resting point "
                    f"where 10k was not.")
        print(f"  {verd}")
    else:
        verd = None
        print("  (no snapshots)")

    # ---- reading ------------------------------------------------------------
    print("\n=== READING ===")
    if gain > 300:
        rd = (f"MORE BUDGET PAYS. {gain:+.0f} on top of an already-outlying seed. The "
              f"10k horizon was leaving real return on the table for this run.")
    elif gain > 100:
        rd = (f"MODEST GAIN of {gain:+.0f}. Doubling the budget helps, but far less than "
              f"the +1174 that separated seed 4 from the pack in the first place -- the "
              f"basin matters more than the budget.")
    elif gain > -100:
        rd = (f"FLAT: {gain:+.0f} for twice the compute. Seed 4 had essentially converged "
              f"in RETURN by 10k even though its addressing had not converged in "
              f"GEOMETRY. Those are different questions and this run separates them: the "
              f"addressing kept moving without buying anything.")
    else:
        rd = (f"IT GOT WORSE: {gain:+.0f}. The extra budget degraded a good policy, which "
              f"points at late-training instability rather than under-training -- the "
              f"same failure mode exp_c19 measured in the MLP arm.")
    print(f"  {rd}")

    json.dump(dict(score_20k=s20, score_10k_thisrun=s10, score_10k_ref=REF_10K,
                   gain=float(gain), best_mjx_20k=best20,
                   checkpoint_identical_at_10k=identical, movement=mv,
                   convergence_verdict=verd, reading=rd),
              open(os.path.join(HERE, "long_run_results.json"), "w"), indent=1)
    print("\nwrote long_run_results.json")


if __name__ == "__main__":
    main()
