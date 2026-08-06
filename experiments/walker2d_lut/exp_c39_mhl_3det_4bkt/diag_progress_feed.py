"""exp_c39 diagnosis — keep `diag_status.json` moving while the transplant runs.

The phase bar renders `diag_status.json`, but the phases are written by hand between steps,
so a phase containing a 33-minute GPU job sits at one number for 33 minutes and reads as
wedged. It is the same reporting flaw as the warmup-inflated ETA on the sweep bars: a
progress number that cannot move during the longest thing it covers is not a progress
number.

This feeds the transplant's real iteration counts into the status file, so the bar advances
continuously across the phase instead of jumping at its end.

Usage:
  python diag_progress_feed.py [--lo 55] [--hi 85] [--interval 45]
"""
import argparse
import json
import os
import re
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STATUS = os.path.join(HERE, "diag_status.json")
ITER = re.compile(r"\[\s*(\d+)/(\d+)\]", re.M)
CPU = re.compile(r"CPU-reference \d+-ep deterministic:\s+([-\d.]+)")
CELLS = {"D": ("winner FRONT-END + loser table", 0, 2),
         "E": ("loser front-end + winner TABLE", 0, 2),
         "F": ("winner FRONT-END + loser table", 1, 2)}


def scan():
    prog, notes, done = [], [], 0
    for name, (desc, seed, aseed) in sorted(CELLS.items()):
        log = os.path.join(HERE, f"sp_{name}.log")
        if not os.path.exists(log):
            prog.append(0.0)
            continue
        txt = open(log, errors="replace").read()
        cpu = CPU.findall(txt)
        if cpu:
            prog.append(1.0)
            done += 1
            notes.append(f"{name} ({desc}): CPU-ref {float(cpu[-1]):.0f}")
            continue
        ms = ITER.findall(txt)
        if ms:
            it, tot = int(ms[-1][0]), int(ms[-1][1])
            prog.append(it / max(1, tot))
            notes.append(f"{name} ({desc}): {it:,}/{tot:,}")
        else:
            prog.append(0.0)
            notes.append(f"{name} ({desc}): starting")
    return (sum(prog) / max(1, len(prog))), notes, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=55.0)
    ap.add_argument("--hi", type=float, default=85.0)
    ap.add_argument("--interval", type=int, default=45)
    a = ap.parse_args()
    while True:
        frac, notes, done = scan()
        st = dict(phase="final", pct=a.lo + (a.hi - a.lo) * frac,
                  notes=["Transplant test running — the decisive init-vs-RL-stream "
                         "experiment"] + notes, done=False)
        json.dump(st, open(STATUS, "w"), indent=1)
        if done == len(CELLS) or os.path.exists(os.path.join(HERE, "SPLIT_DONE")):
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
