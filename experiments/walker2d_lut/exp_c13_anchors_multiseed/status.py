"""exp_c13 — compact chat-ready snapshot of the 27-run sweep (#75). Read-only."""
import os, re, time

HERE = os.path.dirname(os.path.abspath(__file__))
TOTAL_RUNS, ITERS = 27, 10000
DONE = re.compile(r"^done: best MJX ([-\d.]+) over ([\d,]+) env-steps in ([\d.]+) min", re.M)
PROG = re.compile(r"^\[\s*(\d+)/\s*(\d+)\]\s+steps\s+([\d,]+)\s+\|\s+MJX ret\s+([-\d.]+)"
                  r".*?\|\s+best\s+([-\d.]+)\s+\|\s+([\d.]+)m", re.M)
LAUNCH = re.compile(r"^=== launch (\d+)/(\d+)", re.M)


def bar(frac, width=22):
    n = int(round(width * max(0.0, min(1.0, frac))))
    return "#" * n + "." * (width - n)


def main():
    done, running = [], []
    for f in sorted(os.listdir(HERE)):
        m = re.match(r"cell_nap(\d+)_tph(\d+)_s(\d+)\.log$", f)
        if not m:
            continue
        nap, tph, seed = (int(m.group(i)) for i in (1, 2, 3))
        txt = open(os.path.join(HERE, f), errors="replace").read()
        d = DONE.search(txt)
        if d:
            done.append((nap, tph, seed, float(d.group(1)), float(d.group(3))))
            continue
        p = PROG.findall(txt)
        it, best, mins = (int(p[-1][0]), float(p[-1][4]), float(p[-1][5])) if p else (0, 0.0, 0.0)
        running.append((nap, tph, seed, it, best, mins))

    log = os.path.join(HERE, "run_sweep.log")
    launched = 0
    if os.path.exists(log):
        ls = LAUNCH.findall(open(log, errors="replace").read())
        launched = int(ls[-1][0]) if ls else 0
    queued = TOTAL_RUNS - launched

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"exp_c13 anchors x hard capacity sweep — balanced sampler, 3 seeds")
    print(f"{ts}   [{bar(len(done) / TOTAL_RUNS)}]  {len(done)}/{TOTAL_RUNS} done"
          f"  {len(running)} running  {queued} queued")
    if running:
        print()
        for nap, tph, seed, it, best, mins in sorted(running):
            print(f"  nap{nap}/tph{tph:<3} s{seed}  [{bar(it / ITERS, 16)}] "
                  f"{it:>5,}/{ITERS:,}  best MJX {best:>7.1f}  {mins:>5.1f}m")
    if done:
        # eta from measured wall-clock, not a guess: waves left x mean cell time
        mean_min = sum(d[4] for d in done) / len(done)
        waves_left = (TOTAL_RUNS - len(done) + 2) // 3
        print(f"\n  mean cell {mean_min:.0f} min | ~{waves_left} wave(s) left "
              f"-> ~{waves_left * mean_min / 60:.1f} h")


if __name__ == "__main__":
    main()
