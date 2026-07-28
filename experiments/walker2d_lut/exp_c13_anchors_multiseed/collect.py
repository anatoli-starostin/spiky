"""exp_c13 — evaluate all 27 runs and aggregate ACROSS SEEDS (#75).

Every finished run gets the deterministic 100-episode CPU-reference eval in hard mode
(skipping any already evaluated), then configs are reported as mean +/- std ACROSS THE
THREE SEEDS. That std is the number that retires exp_c12's one-seed caveat: exp_c12
quoted the within-run episode spread, which says how consistent one trained policy is,
NOT how reproducible the training is. They are different quantities and the second is
usually much larger.

Both are printed. `ep-sd` is the mean over seeds of each run's episode-level std --
policy consistency. `seed-sd` is the std of the three seeds' means -- training
reproducibility.

Target: hyperplane x hard = 5146.9 +/- 28.2 at 28,034 total params (ONE seed, old
sampler -- so the comparison below is still one-sided until that arm is reseeded too).
"""
import json, os, re, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
OBS, OUT, ACT_OUT = 17, 12, 6
TARGET = 5146.9
SEEDS = (0, 1, 2)
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
COV = re.compile(r"row-cov\s+([\d.]+)%", re.M)


def params(nap, tph, heads=1):
    T, K = heads * tph, 2 ** nap
    table = T * K * OUT
    addr = T * nap * OBS + T * nap          # frozen for anchors, but still stored
    return table, addr, table + addr + 2    # +2 learnable temperatures


def active(nap, tph, addressing, out_per_cell=OUT, heads=1):
    """Values TOUCHED per inference step in hard mode -- one row per table fires.

    reads are independent of nap (more rows is memory, not work); addressing is
    nap*tph*2 element reads for anchors vs nap*tph*17 MACs for a dense hyperplane.
    out_per_cell=6 gives the DEPLOYED cost (the 6 log-sigmas are training-only).
    """
    T = heads * tph
    reads = T * out_per_cell
    addr = T * nap * (2 if addressing == "anchors" else OBS)
    return reads, addr, reads + addr


def run_eval(actor, label, done_n, todo_n):
    """Evaluate one run, streaming the child's live progress lines through."""
    t0 = time.time()
    print(f"  [{time.strftime('%H:%M:%S', time.gmtime())}] run {done_n + 1}/{todo_n}"
          f"  evaluating {label} ...", flush=True)
    p = subprocess.Popen(
        [PY, "-u", os.path.join(C09, "eval_cpu.py"), actor,
         "--episodes", "100", "--forward-mode", "hard", "--progress", label],
        cwd=C09, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("    ["):
            print(line, flush=True)
        elif "CPU-reference" in line:
            print(f"    -> {line.strip()}", flush=True)
        tail.append(line)
    err = p.stderr.read()
    if p.wait() != 0:
        print(f"    FAILED ({label}): {err[-300:] or chr(10).join(tail[-5:])}", flush=True)
        return False
    print(f"    done {label} in {time.time() - t0:.0f}s", flush=True)
    return True


def main():
    runs, pending = {}, []
    for f in sorted(os.listdir(HERE)):
        m = re.match(r"cell_nap(\d+)_tph(\d+)_s(\d+)\.log$", f)
        if not m:
            continue
        nap, tph, seed = (int(m.group(i)) for i in (1, 2, 3))
        txt = open(os.path.join(HERE, f), errors="replace").read()
        if not DONE.search(txt):
            print(f"  (skip nap{nap}/tph{tph}/s{seed}: still running)", flush=True)
            continue
        actor = f"lut_sac_c13_nap{nap}_tph{tph}_s{seed}_actor.npz"
        runs[(nap, tph, seed)] = (actor, txt)
        if not os.path.exists(os.path.join(
                C09, actor.replace("_actor.npz", "_cpueval.json"))):
            pending.append((nap, tph, seed))

    if pending:
        print(f"{len(pending)} run(s) to evaluate", flush=True)
    n_done = 0
    for (nap, tph, seed), (actor, _) in sorted(runs.items()):
        if (nap, tph, seed) in pending:
            if run_eval(actor, f"nap{nap}/tph{tph}/s{seed}", n_done, len(pending)):
                n_done += 1

    # ---- aggregate across seeds -------------------------------------------
    rows = []
    for nap in (6, 7, 8):
        for tph in (32, 64, 128):
            got = []
            for seed in SEEDS:
                key = (nap, tph, seed)
                if key not in runs:
                    continue
                actor, txt = runs[key]
                ev = os.path.join(C09, actor.replace("_actor.npz", "_cpueval.json"))
                if not os.path.exists(ev):
                    continue
                e = json.load(open(ev))
                covs = COV.findall(txt)
                got.append(dict(seed=seed, mean=e["cpu_reference_mean"],
                                std=e["cpu_reference_std"],
                                cov=float(covs[-1]) if covs else float("nan")))
            if not got:
                continue
            means = np.array([g["mean"] for g in got])
            t, a_, tot = params(nap, tph)
            r12, ad, at12 = active(nap, tph, "anchors")
            r6, _, at6 = active(nap, tph, "anchors", ACT_OUT)
            rows.append(dict(
                nap=nap, tph=tph, table=t, addr=a_, total=tot,
                rd12=r12, rd6=r6, act_addr=ad, act12=at12, act6=at6,
                n_seeds=len(got), seed_mean=float(means.mean()),
                # ddof=1: these three seeds are a SAMPLE of the seed distribution,
                # not the population. With n=3 the difference is not cosmetic.
                seed_std=float(means.std(ddof=1)) if len(means) > 1 else float("nan"),
                seed_min=float(means.min()), seed_max=float(means.max()),
                ep_std=float(np.mean([g["std"] for g in got])),
                cov=float(np.mean([g["cov"] for g in got])),
                per_seed={g["seed"]: g["mean"] for g in got}))

    rows.sort(key=lambda r: -r["seed_mean"])
    hr, ha, ht = active(6, 32, "hyperplane")
    _, _, htot = params(6, 32)

    print("\n=== anchors x hard, lutorch BALANCED sampler, 3 seeds — "
          "sorted by across-seed mean ===")
    print("  seed-sd = std of the 3 seeds' means (training reproducibility).")
    print("  ep-sd   = mean within-run episode std (single-policy consistency).")
    print("  These are DIFFERENT quantities; exp_c12 only ever quoted the second.")
    print(f"{'nap':>4}{'tph':>5}{'rows':>6}{'total par':>11}{'rd12':>7}{'rd6':>6}"
          f"{'addr':>7}{'act12':>7}{'act6':>7}{'spars':>8}"
          f"{'CPU-ref (3 seeds)':>22}{'ep-sd':>8}{'range':>15}{'cov':>7}  vs tgt")
    for r in rows:
        sp = 100.0 * r["act12"] / r["total"]
        n = "" if r["n_seeds"] == 3 else f" (n={r['n_seeds']})"
        print(f"{r['nap']:>4}{r['tph']:>5}{2 ** r['nap']:>6}{r['total']:>11,}"
              f"{r['rd12']:>7,}{r['rd6']:>6,}{r['act_addr']:>7,}"
              f"{r['act12']:>7,}{r['act6']:>7,}{sp:>7.2f}%"
              f"{r['seed_mean']:>14.1f} ± {r['seed_std']:<5.1f}"
              f"{r['ep_std']:>8.1f}"
              f"{r['seed_min']:>8.0f}-{r['seed_max']:<6.0f}"
              f"{r['cov']:>6.1f}%{100.0 * r['seed_mean'] / TARGET:>7.0f}%{n}")

    print(f"\nREFERENCE  hyperplane x hard nap6/tph32 (ONE seed, OLD sampler): "
          f"{htot:,} total | active12 {hr:,} reads + {ha:,} MAC = {ht:,} "
          f"({100.0 * ht / htot:.2f}%) | CPU-ref {TARGET}")

    if rows:
        best = rows[0]
        verb = "REACHED" if best["seed_mean"] >= TARGET else "NOT REACHED"
        print(f"\n{verb}: best nap{best['nap']}/tph{best['tph']} = "
              f"{best['seed_mean']:.1f} ± {best['seed_std']:.1f} across seeds "
              f"({100 * best['seed_mean'] / TARGET:.0f}% of target) | "
              f"{best['act12']:,} active12 ({best['act12'] / ht:.2f}x), "
              f"{best['act6']:,} active6")
        # Is the seed spread bigger than the gaps the ranking rests on?
        if len(rows) > 1:
            gaps = [rows[i]["seed_mean"] - rows[i + 1]["seed_mean"]
                    for i in range(len(rows) - 1)]
            sds = [r["seed_std"] for r in rows if r["n_seeds"] > 1]
            if gaps and sds:
                print(f"\nRANKING RELIABILITY: median adjacent gap "
                      f"{np.median(gaps):.0f}, median seed-sd {np.median(sds):.0f} "
                      f"-> {'gaps are INSIDE the noise' if np.median(gaps) < np.median(sds) else 'gaps EXCEED the noise'}")

    for axis, fixed, vals in (("nap", ("tph", 32), (6, 7, 8)),
                              ("tph", ("nap", 6), (32, 64, 128))):
        sel = [r for r in rows if r[fixed[0]] == fixed[1]]
        if len(sel) > 1:
            sel.sort(key=lambda r: r[axis])
            note = ("active reads constant, memory grows" if axis == "nap"
                    else "the axis that DOES cost active reads")
            print(f"\n{axis} axis at FIXED {fixed[0]}={fixed[1]} ({note}):")
            for r in sel:
                per = "  ".join(f"s{s}:{v:.0f}" for s, v in sorted(r["per_seed"].items()))
                print(f"  {axis}{r[axis]:<4}: {r['seed_mean']:7.1f} ± "
                      f"{r['seed_std']:<6.1f}  act12 {r['act12']:>6,}   [{per}]")

    json.dump(rows, open(os.path.join(HERE, "multiseed_results.json"), "w"), indent=1)
    print("\nwrote multiseed_results.json")


if __name__ == "__main__":
    main()
