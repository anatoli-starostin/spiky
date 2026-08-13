"""exp_c12 — evaluate every finished sweep cell and print the capacity table (#75).

Runs the deterministic 100-episode CPU-reference eval in HARD mode for each completed
cell (skipping any already evaluated), then prints the table sorted by CPU-ref score.

Baseline nap6/tph32 = 4302.4 +/- 49.9 is reused from exp_c11 rather than rerun.
Target: hyperplane x hard = 5146.9 +/- 28.2 at 28,032 params.
"""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
C09 = os.path.join(HERE, "..", "exp_c09_lut_sac")
PY = os.path.expanduser("~/projects/walker2d_mjx/.venv/bin/python")
OBS, OUT = 17, 12
ACT_OUT = 6          # the 6 action means; the 6 log-sigmas are training-only
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


def deployable(nap, tph, addressing, heads=1):
    """Same accounting, but counting only what DETERMINISTIC INFERENCE touches.

    Each cell stores 12 values: 6 action means and 6 log-sigmas. The sigmas exist
    only for the SAC entropy term during training -- eval_cpu.py takes
    tanh(y[:, :6]) and never reads them. A deployed policy therefore stores and
    reads 6 values per cell, halving BOTH the table memory and the active reads.
    Addressing is unchanged: the same rows must still be selected.
    """
    T = heads * tph
    reads = T * ACT_OUT
    addr = T * nap * (2 if addressing == "anchors" else OBS)
    return reads, addr, reads + addr


def run_eval(actor, label, done_n, todo_n):
    """Evaluate one cell, streaming the child's live progress lines through.

    The 100 episodes are stepped in LOCKSTEP inside a single process, so there is
    no per-episode completion to count down: the child reports step/1000, how many
    walkers have fallen, and the running mean. `done_n/todo_n` is the overall
    cell counter across the sweep.
    """
    t0 = time.time()
    print(f"  [{time.strftime('%H:%M:%S', time.gmtime())}] cell {done_n + 1}/{todo_n}"
          f"  evaluating {label} (100-ep deterministic CPU reference) ...", flush=True)
    p = subprocess.Popen(
        [PY, "-u", os.path.join(C09, "eval_cpu.py"), actor,
         "--episodes", "100", "--forward-mode", "hard", "--progress", label],
        cwd=C09, env=dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    tail = []
    for line in p.stdout:
        line = line.rstrip()
        if line.startswith("    ["):                 # live progress from the child
            print(line, flush=True)
        elif "CPU-reference" in line:
            print(f"    -> {line.strip()}", flush=True)
        tail.append(line)
    err = p.stderr.read()
    if p.wait() != 0:
        print(f"    FAILED ({label}): {err[-300:] or chr(10).join(tail[-5:])}",
              flush=True)
        return False
    print(f"    done {label} in {time.time() - t0:.0f}s "
          f"({done_n + 1}/{todo_n} cells evaluated)", flush=True)
    return True


def main():
    rows = []
    t, a_, tot = params(BASE["nap"], BASE["tph"])
    ar, aa, at = active(BASE["nap"], BASE["tph"], "anchors")
    dr, da, dt_ = deployable(BASE["nap"], BASE["tph"], "anchors")
    rows.append(dict(BASE, table=t, addr=a_, total=tot,
                     act_reads=ar, act_addr=aa, act_total=at,
                     dep_reads=dr, dep_addr=da, dep_total=dt_))

    # First pass: which finished cells still need evaluating, so the live readout
    # can quote a real "cell i/N" instead of counting up to an unknown total.
    pending = []
    for f in sorted(os.listdir(HERE)):
        m = re.match(r"cell_nap(\d+)_tph(\d+)\.log$", f)
        if not m:
            continue
        nap, tph = int(m.group(1)), int(m.group(2))
        actor = f"lut_sac_c12_nap{nap}_tph{tph}_actor.npz"
        if not os.path.exists(os.path.join(
                C09, actor.replace("_actor.npz", "_cpueval.json"))):
            pending.append((nap, tph))
    if pending:
        print(f"{len(pending)} cell(s) to evaluate: "
              + ", ".join(f"nap{n}/tph{t_}" for n, t_ in pending), flush=True)
    n_done = 0

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
            if not run_eval(actor, f"nap{nap}/tph{tph}", n_done, len(pending)):
                continue
            n_done += 1
        e = json.load(open(ev))
        t, a_, tot = params(nap, tph)
        ar, aa, at = active(nap, tph, "anchors")
        dr, da, dt_ = deployable(nap, tph, "anchors")
        rows.append(dict(nap=nap, tph=tph, table=t, addr=a_, total=tot,
                         act_reads=ar, act_addr=aa, act_total=at,
                         dep_reads=dr, dep_addr=da, dep_total=dt_,
                         mean=e["cpu_reference_mean"], std=e["cpu_reference_std"],
                         best_mjx=float(d.group(1)),
                         cov=float(covs[-1]) if covs else float("nan"),
                         source="exp_c12"))

    rows.sort(key=lambda r: -r["mean"])
    hr, ha, ht = active(6, 32, "hyperplane")
    hdr, hda, hdt = deployable(6, 32, "hyperplane")
    _, _, htot = params(6, 32)

    print("\n=== anchors x hard capacity sweep — sorted by CPU-reference ===")
    print("  ACTIVE = values touched per inference step in hard mode. Raising nap adds")
    print("  stored rows (memory) but touches NOT ONE extra value; only tph scales it.")
    print("  12/cell = as trained (6 action means + 6 log-sigmas). 6/cell = as")
    print("  DEPLOYED: deterministic inference reads tanh(mean) and never touches a")
    print("  sigma, so half the table is dead weight at inference. Addressing is the")
    print("  same either way — the same row still has to be selected.")
    print(f"{'nap':>4}{'tph':>5}{'rows':>6}{'total par':>11}"
          f"{'rd12':>7}{'rd6':>6}{'addr':>7}{'act12':>7}{'act6':>7}"
          f"{'spars12':>9}{'spars6':>8}"
          f"{'CPU-ref (100 ep)':>21}{'cov':>7}  vs tgt")
    for r in rows:
        vs = 100.0 * r["mean"] / TARGET
        sp = 100.0 * r["act_total"] / r["total"]
        # deployed total params drops with the table: 6/cell instead of 12/cell
        dep_tot = r["table"] // 2 + r["addr"] + 2
        sp6 = 100.0 * r["dep_total"] / dep_tot
        star = "  <- baseline" if str(r["source"]).startswith("exp_c11") else ""
        print(f"{r['nap']:>4}{r['tph']:>5}{2 ** r['nap']:>6}{r['total']:>11,}"
              f"{r['act_reads']:>7,}{r['dep_reads']:>6,}{r['act_addr']:>7,}"
              f"{r['act_total']:>7,}{r['dep_total']:>7,}"
              f"{sp:>8.2f}%{sp6:>7.2f}%{r['mean']:>13.1f} ± {r['std']:<5.1f}"
              f"{r['cov']:>6.1f}%{vs:>7.0f}%{star}")

    print(f"\nREFERENCE  hyperplane x hard nap6/tph32: {htot:,} total params | "
          f"active12 {hr:,} reads + {ha:,} MAC = {ht:,} ({100.0 * ht / htot:.2f}% "
          f"sparsity) | active6 {hdr:,} + {hda:,} = {hdt:,} | CPU-ref {TARGET}")
    print("  hyperplane addressing is DENSE: nap*tph*17 multiply-accumulates. It is")
    print("  89% of its active cost at 12/cell and 94% at 6/cell — dropping the sigmas")
    print("  barely helps a dense addresser, because the table was never its bottleneck.")
    print("  anchors addressing is 2 element reads per bit — 8.5x less addressing work.")

    best = rows[0]
    ba, bd = best["act_total"], best["dep_total"]
    if best["mean"] >= TARGET:
        print(f"\nREACHED: nap{best['nap']}/tph{best['tph']} -> {best['mean']:.1f} "
              f"at {best['total']:,} total params ({best['total'] / htot:.1f}x) "
              f"but only {ba:,} ACTIVE ({ba / ht:.2f}x the hyperplane active cost); "
              f"deployed 6/cell {bd:,} ({bd / hdt:.2f}x)")
    else:
        print(f"\nNOT REACHED: best nap{best['nap']}/tph{best['tph']} = "
              f"{best['mean']:.1f} ({100 * best['mean'] / TARGET:.0f}% of target) | "
              f"{best['total']:,} total ({best['total'] / htot:.1f}x), {ba:,} active12 "
              f"({ba / ht:.2f}x), {bd:,} active6 ({bd / hdt:.2f}x)")

    # Separate the two axes: does nap alone (free at inference) close the gap?
    at_tph32 = [r for r in rows if r["tph"] == 32]
    if len(at_tph32) > 1:
        at_tph32.sort(key=lambda r: r["nap"])
        print("\nnap axis at FIXED tph=32 (active reads constant, memory grows):")
        for r in at_tph32:
            print(f"  nap{r['nap']}: {r['mean']:7.1f}   total {r['total']:>8,}   "
                  f"act12 {r['act_total']:>6,}  act6 {r['dep_total']:>6,}"
                  f"   (reads unchanged by nap; only addressing ticks up)")

    # tph axis at fixed nap: the axis that actually costs active reads.
    at_nap6 = sorted((r for r in rows if r["nap"] == 6), key=lambda r: r["tph"])
    if len(at_nap6) > 1:
        print("\ntph axis at FIXED nap=6 (the axis that DOES cost active reads):")
        for r in at_nap6:
            print(f"  tph{r['tph']:<4}: {r['mean']:7.1f}   total {r['total']:>8,}   "
                  f"act12 {r['act_total']:>6,}  act6 {r['dep_total']:>6,}")
    json.dump(rows, open(os.path.join(HERE, "capacity_sweep_results.json"), "w"),
              indent=1)
    print("wrote capacity_sweep_results.json")


if __name__ == "__main__":
    main()
