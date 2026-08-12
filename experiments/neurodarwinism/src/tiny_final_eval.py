"""exp012: score the run's PROPERLY SELECTED model on held-out data.

WHY THIS EXISTS. tiny_evolve tracks `best` = the genome with the lowest score on the round's
TRAINING BATCH, and records that genome's held-out number alongside. That is honest (held-out
never enters selection) but it is not a running minimum: a genome that wins on the batch can
score worse on held-out than an earlier winner, so `best["heldout_mse"]` can go UP, and taking
the minimum of it over the run would be selecting on held-out through the back door.

The number to report is the held-out score of the model selection actually ends on: the pool
member with the best EWMA fitness at the final round, chosen on training batches alone. That
is what this reads out of the checkpoint.

    python tiny_final_eval.py --ckpt <dir>/ck_s0.npz --seed 0 --out <dir>/final_leader.json
"""
import argparse
import json
import os

import numpy as np

import tiny_snn as T
from data import load
from harness import LatencyEncoder
from tiny_evolve import load_ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    pool, ewma, age, rnd, hist, best, _ = load_ckpt(a.ckpt)
    _, _, Xp, Yp, Xv, Yv = load(a.batch, seed=a.seed)
    T.fit_target_stats(Yp)
    enc = LatencyEncoder(Xp)
    tgt = T.target_offsets(Yv)

    finite = np.where(np.isfinite(ewma))[0]
    lead = int(finite[np.argmin(ewma[finite])])
    H = T.build([pool[lead]], device=a.device)
    s = T.score(H, Xv, Yv, enc)
    first = s["first"][:, 0, :]
    r, ceil = T.affine_ceiling_and_r(first, tgt)

    per = []
    for d in range(first.shape[1]):
        p, t = first[:, d], tgt[:, d]
        rr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else 0.0
        per.append(dict(dim=d, r=rr, mse=float(((p - t) ** 2).mean()),
                        bias2=float((p.mean() - t.mean()) ** 2),
                        scale_err=float((p.std() - t.std()) ** 2)))
    b2 = float(np.mean([x["bias2"] for x in per]))
    sc = float(np.mean([x["scale_err"] for x in per]))
    mse = float(s["mse"][0])

    out = dict(ckpt=a.ckpt, round=rnd, pool=len(pool), leader_index=lead,
               leader_ewma=float(ewma[lead]),
               heldout_mse=mse, constant_baseline=T.constant_baseline(Yv),
               tau=float(s["tau"][0]), mean_abs_r=r, affine_ceiling=ceil,
               silent=float(s["silent"][0]), n_distinct=int(s["n_distinct"][0]),
               mse_action=float(s["mse_action"][0]),
               bias2=b2, scale_err=sc, resid=mse - b2 - sc,
               per_dim=per, genome_stats=T.genome_stats(pool[lead]),
               batch_champion_heldout=best.get("heldout_mse"))
    print(json.dumps(T.jsonable({k: v for k, v in out.items() if k != "per_dim"}), indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(T.jsonable(out), f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
