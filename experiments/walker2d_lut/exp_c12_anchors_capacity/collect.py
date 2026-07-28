"""exp_c12 — evaluate every finished sweep cell and print the capacity table (#75).

Runs the deterministic 100-episode CPU-reference eval in HARD mode for each completed
cell (skipping any already evaluated), then prints the table sorted by CPU-ref score.

Baseline nap6/tph32 = 4302.4 +/- 49.9 is reused from exp_c11 rather than rerun.
Target: hyperplane x hard = 5146.9 +/- 28.2 at 28,032 params.
"""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
OBS, OUT = 17, 12
TARGET = 5146.9
BASE = dict(nap=6, tph=32, mean=4302.4, std=49.9, best_mjx=4290.8, cov=99.7,
            source="exp_c11 (not rerun)")
DONE = re.compile(r"^done: best MJX ([-\d.]+)", re.M)
ITER = re.compile(r"row-cov\s+([\d.]+)%", re.M)


def params(nap, tph, heads=1):
    T, K = heads * tph, 2 ** nap
    table = T * K * OUT
    addr = T * nap * OBS + T * nap          # frozen for anchors, but still stored
    return table, addr, table + addr + 2    # +2 learnable temperatures


def active(nap, tph, addressing, heads=1):
    """ACTIVE work per inference step in HARD mode — the metric that matters for
    sparse / in-memory-compute hardware, where only touched weights cost anything.

    table reads : exactly ONE row per table fires, so heads*tph*OUT values are read.
      **Independent of nap.** Raising nap multiplies stored rows (memory) without
      touching a single extra value at inference; only tph scales active reads.

    addressing  : anchors    -> nap*tph*heads*2 element reads (each comparator looks
                               at exactly two input coordinates: x[a] vs x[b]);
                  hyperplane -> nap*tph*heads*OBS multiply-accumulates (a dense
                               affine over the whole observation per address bit).
    """
    T = heads * tph
    reads = T * OUT
    addr = T * nap * (2 if addressing == "anchors" else OBS)
    return reads, addr, reads + addr


def main():
    rows = []
    t, a_, tot = params(BASE["nap"], BASE["tph"])
    ar, aa, at = active(BASE["nap"], BASE["tph"], "anchors")
    rows.append(dict(BASE, table=t, addr=a_, total=tot,
                     act_reads=ar, act_addr=aa, act_total=at))

    for f in sorted(os.listdir(HERE)):
        m = re.match(r"cell_nap(\d+)_tph(\d+)\.log$", f)
        if not m:
            continue
        nap, tph = int(m.group(1)), int(m.group(2))
        txt = open(os.path.join(HERE, f), errors="replace").read()
        d = DONE.search(txt)
        if not d:
            print(f"  (skip nap{nap}/tph{tph}: still running)", flush=True)
            continue
        covs = ITER.findall(txt)
        actor = f"lut_sac_c12_nap{nap}_tph{tph}_actor.npz"
        ev = os.path.join(C09, actor.replace("_actor.npz", "_cpueval.json"))
        if not os.path.exists(ev):
            print(f"  evaluating nap{nap}/tph{tph} ...", flush=True)
            r = subprocess.run(
                [PY, "-u", os.path.join(C09, "eval_cpu.py"), actor,
                 "--episodes", "100", "--forward-mode", "hard"],
                cwd=C09, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"),
                capture_output=True, text=True)
            for line in r.stdout.splitlines():
                if "CPU-reference" in line:
                    print("   ", line.strip(), flush=True)
            if r.returncode != 0:
                print(f"    FAILED: {r.stderr[-300:]}", flush=True)
                continue
        e = json.load(open(ev))
        t, a_, tot = params(nap, tph)
        ar, aa, at = active(nap, tph, "anchors")
        rows.append(dict(nap=nap, tph=tph, table=t, addr=a_, total=tot,
                         act_reads=ar, act_addr=aa, act_total=at,
                         mean=e["cpu_reference_mean"], std=e["cpu_reference_std"],
                         best_mjx=float(d.group(1)),
                         cov=float(covs[-1]) if covs else float("nan"),
                         source="exp_c12"))

    rows.sort(key=lambda r: -r["mean"])
    hr, ha, ht = active(6, 32, "hyperplane")
    _, _, htot = params(6, 32)

    print("\n=== anchors x hard capacity sweep — sorted by CPU-reference ===")
    print("  ACTIVE = values touched per inference step in hard mode. Raising nap adds")
    print("  stored rows (memory) but touches NOT ONE extra value; only tph scales it.")
    print(f"{'nap':>4}{'tph':>5}{'rows':>6}{'total par':>11}{'act.rd':>8}"
          f"{'act.addr':>9}{'act.tot':>9}{'sparsity':>10}"
          f"{'CPU-ref (100 ep)':>21}{'cov':>7}  vs tgt")
    for r in rows:
        vs = 100.0 * r["mean"] / TARGET
        sp = 100.0 * r["act_total"] / r["total"]
        star = "  <- baseline" if str(r["source"]).startswith("exp_c11") else ""
        print(f"{r['nap']:>4}{r['tph']:>5}{2 ** r['nap']:>6}{r['total']:>11,}"
              f"{r['act_reads']:>8,}{r['act_addr']:>9,}{r['act_total']:>9,}"
              f"{sp:>9.2f}%{r['mean']:>13.1f} ± {r['std']:<5.1f}"
              f"{r['cov']:>6.1f}%{vs:>7.0f}%{star}")

    print(f"\nREFERENCE  hyperplane x hard nap6/tph32: {htot:,} total params | "
          f"active {hr:,} reads + {ha:,} MAC = {ht:,} ({100.0 * ht / htot:.2f}% "
          f"sparsity) | CPU-ref {TARGET}")
    print("  hyperplane addressing is DENSE: nap*tph*17 multiply-accumulates.")
    print("  anchors addressing is 2 element reads per bit — 8.5x less addressing work.")

    best = rows[0]
    ba = best["act_total"]
    if best["mean"] >= TARGET:
        print(f"\nREACHED: nap{best['nap']}/tph{best['tph']} -> {best['mean']:.1f} "
              f"at {best['total']:,} total params ({best['total'] / htot:.1f}x) "
              f"but only {ba:,} ACTIVE ({ba / ht:.2f}x the hyperplane active cost)")
    else:
        print(f"\nNOT REACHED: best nap{best['nap']}/tph{best['tph']} = "
              f"{best['mean']:.1f} ({100 * best['mean'] / TARGET:.0f}% of target) | "
              f"{best['total']:,} total ({best['total'] / htot:.1f}x), {ba:,} active "
              f"({ba / ht:.2f}x)")

    # Separate the two axes: does nap alone (free at inference) close the gap?
    at_tph32 = [r for r in rows if r["tph"] == 32]
    if len(at_tph32) > 1:
        at_tph32.sort(key=lambda r: r["nap"])
        print("\nnap axis at FIXED tph=32 (active cost constant, memory grows):")
        for r in at_tph32:
            print(f"  nap{r['nap']}: {r['mean']:7.1f}   total {r['total']:>8,}   "
                  f"active {r['act_total']:>6,} (unchanged by nap)")
    json.dump(rows, open(os.path.join(HERE, "capacity_sweep_results.json"), "w"),
              indent=1)
    print("wrote capacity_sweep_results.json")


if __name__ == "__main__":
    main()
