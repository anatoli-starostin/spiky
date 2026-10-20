"""Decode cost per token vs memory size, for the accuracy-vs-cost curve (CPU, batch 64, context 1024).
Writes runs/bench_cost/results.csv. Same step implementations as bench_decode.py."""
import csv
import os

import torch

from bench_decode import Drawers, DrawersFlip, DrawersTopK, Linear, bench

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    out = os.path.join(HERE, "runs", "bench_cost")
    os.makedirs(out, exist_ok=True)
    B, reps, rows = 64, 200, []
    cfgs = [(f"linear F={f}", f, lambda f=f: Linear(B, f)) for f in (32, 64, 128, 256)]
    cfgs += [(f"drawers {t}x{b} (F={t * 2 ** b})", t * 2 ** b, lambda t=t, b=b: Drawers(B, t, b))
             for t, b in ((4, 4), (8, 4), (16, 4), (16, 2), (64, 2), (4, 6))]
    cfgs += [(f"drawers {t}x{b} top-4 soft (F={t * 2 ** b})", t * 2 ** b, lambda t=t, b=b: DrawersTopK(B, t, b))
             for t, b in ((16, 4), (4, 6))]
    cfgs += [(f"drawers {t}x{b} flip-2 soft (F={t * 2 ** b})", t * 2 ** b, lambda t=t, b=b: DrawersFlip(B, t, b))
             for t, b in ((4, 6), (16, 4))]
    for name, F, make in cfgs:
        sec, nbytes = bench(make, B, reps)
        rows.append(dict(mixer=name, memory_F=F, us_per_token=round(sec * 1e6 / B, 2),
                         state_kb_per_seq=round(nbytes / B / 1024, 1)))
        print(rows[-1], flush=True)
    with open(os.path.join(out, "results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
