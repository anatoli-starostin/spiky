"""Rescue attempt-2's learning curve from its logs into a committable metrics.csv.

attempt2_plain_lse_logspace_init/ holds ONLY .log files: the run was stopped early, so
ppo.py never wrote its ppo_s*.json, and spiky's global .gitignore excludes `*.log`. Without
this the arm would vanish from the repository — and it is the decisive control in the exp17
story (a log-space init that reproduced exp10's starting statistics exactly and STILL
plateaued, which is what proved the fault was the readout and not the initialisation).

Parses the per-10-update log rows into the same metrics.csv schema the rest of the chapter
uses, so the curve survives in git.

Usage:  python rescue_attempt2_metrics.py
"""
import csv
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = (0, 1, 2)
ROW = re.compile(
    r"\[upd\s+([\d,]+)/([\d,]+)\]\s+ep_ret\s+(-?[\d.]+)\s+\(max\s+(-?[\d.]+),\s+"
    r"len\s+(-?[\d.]+)\)\s+\|\s+([\d,]+)\s+env-steps/s\s+\|\s+lr\s+([\d.e+-]+)\s+\|\s+"
    r"kl\s+([\d.]+)")


def main():
    for folder in ("attempt1_additive_init", "attempt2_plain_lse_logspace_init"):
        d = os.path.join(HERE, folder)
        if not os.path.isdir(d):
            continue
        rows = []
        for s in SEEDS:
            f = os.path.join(d, f"ppo_s{s}.log")
            if not os.path.exists(f):
                continue
            for line in open(f, errors="replace"):
                m = ROW.search(line)
                if m:
                    upd = int(m.group(1).replace(",", ""))
                    rows.append([s, upd, upd * 8192 * 32, float(m.group(3)),
                                 float(m.group(4)), float(m.group(5)),
                                 float(m.group(7)), float(m.group(8))])
        if not rows:
            print(f"{folder}: no log rows found — skipped")
            continue
        out = os.path.join(d, "metrics.csv")
        with open(out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["seed", "update", "env_steps", "ep_ret_mean", "ep_ret_max",
                        "ep_len_mean", "lr", "kl"])
            w.writerows(rows)
        last = {s: max((r for r in rows if r[0] == s), key=lambda r: r[1]) for s in SEEDS
                if any(r[0] == s for r in rows)}
        print(f"{folder}: wrote {out} ({len(rows)} rows)")
        for s, r in last.items():
            print(f"   seed {s}: last update {r[1]}, ep_ret {r[3]:.1f}")


if __name__ == "__main__":
    main()
