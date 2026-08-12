"""exp012: is the fast episode path BIT-IDENTICAL to the shared harness.run_episode?

The gate that has to pass before the fast path is allowed anywhere near a run. Same net, same
batch, same seed, both paths; then exact equality on

  * the first-spike offsets   (integers -- exact equality, max abs diff must be 0)
  * the silence flags         (offset == readout_window)
  * the per-member MSE        (float -- EXACT equality, not allclose)

Also checks the assumption the sparse-id cache rests on: that the engine hands back the same
neuron ids every time a net of the same shape is built. If that were false the cache would
serve a stale id tensor and every episode after the first would be silently wrong.

    python tiny_equiv_check.py [--pool 512] [--batch 256] [--reps 3]
"""
import argparse
import json
import time

import numpy as np
import torch

import tiny_snn as T
from data import load, sample_batch
from harness import LatencyEncoder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=int, default=512)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--w-max", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    pool = [T.random_genome(rng, a.w_max) for _ in range(a.pool)]

    # ---- the cache's premise: same shape in, same ids out, every build
    H1 = T.build(pool, device=a.device, seed=1)
    H2 = T.build(pool, device=a.device, seed=1)
    ids_stable = all(np.array_equal(x, y) for x, y in zip(H1["ids"], H2["ids"]))
    print(f"neuron ids stable across rebuilds: {ids_stable}")

    worst = dict(offsets=0, silence=0, mse=0.0)
    rows = []
    for rep in range(a.reps):
        Xb, Yb, _ = sample_batch(Xp, Yp, a.batch, a.seed, rep)
        H = T.build(pool, device=a.device, seed=1)

        s_old = T.score(H, Xb, Yb, enc, fast=False)
        s_new = T.score(H, Xb, Yb, enc, fast=True)

        d_off = int(np.max(np.abs(s_old["first"] - s_new["first"])))
        sil_o = s_old["first"] >= T.READOUT_WINDOW
        sil_n = s_new["first"] >= T.READOUT_WINDOW
        d_sil = int((sil_o != sil_n).sum())
        d_mse = float(np.max(np.abs(s_old["mse"] - s_new["mse"])))
        exact = bool(np.array_equal(s_old["mse"], s_new["mse"]))
        rows.append(dict(rep=rep, max_abs_diff_offsets=d_off, silence_flag_mismatches=d_sil,
                         max_abs_diff_mse=d_mse, mse_bitwise_equal=exact,
                         n_offsets_compared=int(s_old["first"].size)))
        worst["offsets"] = max(worst["offsets"], d_off)
        worst["silence"] = max(worst["silence"], d_sil)
        worst["mse"] = max(worst["mse"], d_mse)
        print(f"  rep {rep}: offsets maxdiff {d_off}  silence mismatches {d_sil}  "
              f"mse maxdiff {d_mse:.3e}  bitwise-equal {exact}  "
              f"({s_old['first'].size:,} offsets compared)")

    # ---- timing, both paths, on the same net
    def timeit(fast, n=3):
        H = T.build(pool, device=a.device, seed=1)
        T.score(H, Xb, Yb, enc, fast=fast)                     # warm
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            Hn = T.build(pool, device=a.device, seed=1)
            T.score(Hn, Xb, Yb, enc, fast=fast)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / n

    t_old = timeit(False)
    t_new = timeit(True)

    ok = bool(worst["offsets"] == 0 and worst["silence"] == 0 and worst["mse"] == 0.0
              and all(r["mse_bitwise_equal"] for r in rows) and ids_stable)
    out = dict(config=vars(a), ids_stable=ids_stable, reps=rows, worst=worst,
               round_s_old=t_old, round_s_new=t_new, speedup=t_old / t_new, ok=ok)
    print(f"\nfull round (build + score):  old {t_old * 1000:7.1f} ms   "
          f"new {t_new * 1000:7.1f} ms   speedup {t_old / t_new:.2f}x")
    print(f"\nEQUIVALENT={ok}   worst offsets {worst['offsets']}, "
          f"silence {worst['silence']}, mse {worst['mse']:.3e}")
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(out), f, indent=1)
        print(f"wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
