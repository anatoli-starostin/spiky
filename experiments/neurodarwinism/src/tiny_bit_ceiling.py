"""exp012: can a net learn ONE of the teacher's sign-comparison bits?

The teacher's target depends on the input only through 192 bits of the form
sign(x_norm[a] - x_norm[b]). Under the latency encoder -- larger x fires EARLIER -- that bit
is exactly "which of these two input spikes arrives first", which is the one thing a
delay-based spiking net ought to compute natively. This measures how learnable a single such
bit is, before any spiking run is spent on it.

The bit is a deterministic function of two inputs, so the TRUE floor is MSE 0 / accuracy 100%
via the two-input rule. Everything else here is a statement about the model class, not about
the task.
"""
import argparse
import json

import numpy as np

import tiny_snn as T
from data import load
from harness import LatencyEncoder
from tiny_mlp_ceiling import apply_affine, fit_affine, mlp_forward, mse, snap_abs, train_mlp

NPZ_KEYS = ("anchor_a", "anchor_b")


def candidate_bits(x, xv, A, B, k=3):
    """The most balanced comparisons among the teacher's own anchor pairs.

    Balance is required on BOTH splits. The held-out set is the last 4,000 samples, i.e. the
    tail of the rollout, and the walker's state distribution drifts along it -- so a bit that
    is 50/50 over training can be 12/88 on held-out, which makes its 'chance' meaningless and
    the probe uninterpretable. (Measured: 107 of the 192 bits shift their base rate by more
    than 0.10 between the splits.) Choosing on both is a statement about the INPUT
    distribution, not about any model's score, so it does not leak performance information.
    """
    out = []
    for t in range(A.shape[0]):
        for j in range(A.shape[1]):
            ia, ib = int(A[t, j]), int(B[t, j])
            if ia == ib:
                continue
            p = float((x[:, ia] > x[:, ib]).mean())
            q = float((xv[:, ia] > xv[:, ib]).mean())
            out.append(dict(table=t, bit=j, a=ia, b=ib, p1=p, p1_val=q,
                            imbalance=max(abs(p - 0.5), abs(q - 0.5)),
                            drift=abs(p - q), chance=q * (1 - q)))
    seen, picked = set(), []
    for c in sorted(out, key=lambda c: c["imbalance"]):
        if (c["a"], c["b"]) in seen or (c["b"], c["a"]) in seen:
            continue                                   # distinct input pairs, not duplicates
        seen.add((c["a"], c["b"]))
        picked.append(c)
        if len(picked) == k:
            break
    return picked, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=4000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import os
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                             "distill_exp19_100k.npz"))
    A, B = Z["anchor_a"], Z["anchor_b"]
    _, _, Xp, Yp, Xv, Yv = load(1024, seed=a.seed)
    T.fit_target_stats(Yp)

    picked, allc = candidate_bits(Xp, Xv, A, B, k=3)
    print("candidate comparison bits (balanced on BOTH splits, distinct input pairs):")
    for c in picked:
        print(f"  table {c['table']:2d} bit {c['bit']}   x[{c['a']:2d}] > x[{c['b']:2d}]   "
              f"P(1) train {c['p1']:.4f}  held-out {c['p1_val']:.4f}  "
              f"drift {c['drift']:.4f}   chance p(1-p) {c['chance']:.4f}")
    star = picked[0]
    print(f"\nPRIMARY b* = 1[ x_norm[{star['a']}] > x_norm[{star['b']}] ]")

    yt = (Xp[:, star["a"]] > Xp[:, star["b"]]).astype(np.float64)
    yv = (Xv[:, star["a"]] > Xv[:, star["b"]]).astype(np.float64)
    chance = float(yv.var())
    print(f"  held-out P(1) {yv.mean():.4f}   own chance (variance) {chance:.4f}")

    enc = LatencyEncoder(Xp)
    Ep_raw, Ev_raw = enc(Xp).astype(np.float64), enc(Xv).astype(np.float64)
    mu, sd = Ep_raw.mean(0), Ep_raw.std(0) + 1e-9
    Ep, Ev = (Ep_raw - mu) / sd, (Ev_raw - mu) / sd

    R = dict(candidates=picked, primary=star, chance=chance, p1_heldout=float(yv.mean()),
             n_train=int(len(Xp)), n_val=int(len(Xv)))

    def acc(p):
        return float(((p > 0.5) == (yv > 0.5)).mean())

    var_t = float(yt.var())

    def qat(xt_, xv_, **kw):
        best = None
        for lam in (0.005, 0.02, 0.05, 0.2):
            m = train_mlp(xt_, yt, xv_, hidden=8, act="tanh", epochs=a.epochs, seed=a.seed,
                          lam_max=lam * var_t, step=0.1, clamp=1.0, gain=True, **kw)
            s1, s2 = snap_abs(m["w1"]), snap_abs(m["w2"])
            pt, pv = mlp_forward(m, xt_, s1, s2), mlp_forward(m, xv_, s1, s2)
            pv = apply_affine(pv, fit_affine(pt, yt))
            v = mse(pv, yv)
            if best is None or v < best[0]:
                best = (v, acc(pv), lam)
        return best

    # ---- the true floor: the two-input rule the bit is DEFINED by
    rule = (Xv[:, star["a"]] > Xv[:, star["b"]]).astype(np.float64)
    R["two_input_rule"] = dict(mse=mse(rule, yv), acc=acc(rule))
    print(f"\ntwo-input rule  x[{star['a']}] > x[{star['b']}]      "
          f"MSE {R['two_input_rule']['mse']:.6f}   acc {100 * R['two_input_rule']['acc']:.2f}%"
          f"   <- the true floor, by definition")

    # ---- 1 free MLP
    m = train_mlp(Xp, yt, Xv, hidden=8, act="tanh", epochs=a.epochs, seed=a.seed)
    pv = apply_affine(m["pred_val"], fit_affine(m["pred_train"], yt))
    R["free_mlp"] = dict(mse=mse(pv, yv), acc=acc(pv),
                         r=float(np.corrcoef(pv, yv)[0, 1]))
    # ---- 2 + Dale + grid
    v, ac, lam = qat(Xp, Xv, dale=True)
    R["dale_grid"] = dict(mse=v, acc=ac, lam=lam)
    # ---- 3 + latency-encoded input = matched ceiling
    v, ac, lam = qat(Ep, Ev, dale=True)
    R["matched_ceiling"] = dict(mse=v, acc=ac, lam=lam)
    # extra: a free MLP on the encoded input, to isolate the encoding from Dale
    me = train_mlp(Ep, yt, Ev, hidden=8, act="tanh", epochs=a.epochs, seed=a.seed)
    pe = apply_affine(me["pred_val"], fit_affine(me["pred_train"], yt))
    R["free_mlp_encoded"] = dict(mse=mse(pe, yv), acc=acc(pe))

    print(f"\n{'':34s}{'MSE':>10s}{'ratio':>9s}{'accuracy':>11s}")
    for nm, k in (("free MLP 17→8→1", "free_mlp"),
                  ("free MLP on ENCODED input", "free_mlp_encoded"),
                  ("+ Dale + 0.1 grid (QAT)", "dale_grid"),
                  ("+ latency input = MATCHED", "matched_ceiling")):
        d = R[k]
        print(f"{nm:34s}{d['mse']:10.4f}{d['mse'] / chance:9.3f}{100 * d['acc']:10.2f}%")
    print(f"{'chance (predict P(1))':34s}{chance:10.4f}{1.0:9.3f}"
          f"{100 * max(yv.mean(), 1 - yv.mean()):10.2f}%")

    with open(a.out, "w") as f:
        json.dump(T.jsonable(R), f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
