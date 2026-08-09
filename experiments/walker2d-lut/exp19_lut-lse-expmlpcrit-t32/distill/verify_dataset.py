"""Standalone re-derivation of the distillation dataset's targets — numpy only, no torch.

The collector wrote the targets straight out of the live module, so a bug there would be
invisible to a check made inside the same process. This file re-derives `y_prelog` and
`y_action_mean` from `x` (the RAW observations) using ONLY what the .npz carries —
obs_mean/obs_var, anchor_a/anchor_b, weights, tau — and compares. If this passes, the
.npz is self-contained: everything needed to reproduce the teacher is inside it.

Usage:  python verify_dataset.py [--npz distill_exp19_100k.npz]
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(HERE, "distill_exp19_100k.npz"))
    a = ap.parse_args()
    Z = np.load(a.npz)
    print(f"{a.npz}  ({os.path.getsize(a.npz) / 1e6:.2f} MB)")
    print("keys:", ", ".join(f"{k}{list(Z[k].shape)}" for k in Z.files))

    x = Z["x"].astype(np.float64)
    tau = float(Z["tau"])
    tph = int(Z["tables_per_head"])
    clamp = float(Z["exp_clamp"])
    W = Z["weights"].astype(np.float64)                 # (32, 64, 6)
    A, B = Z["anchor_a"], Z["anchor_b"]                 # (32, 6) each
    T, table_dim, n_out = W.shape

    # 1. normalisation: x -> x_norm, exactly RunningNorm.norm
    xn = (x - Z["obs_mean"]) / np.sqrt(Z["obs_var"] + 1e-8)
    dn = np.abs(xn - Z["x_norm"].astype(np.float64)).max()
    print(f"\n1. x_norm re-derived from x + obs stats     max|diff| {dn:.3e}")

    # 2. address: sign-pack the anchor differences, MSB-first
    d = xn[:, A] - xn[:, B]                             # (N, 32, 6)
    pow2 = 1 << np.arange(A.shape[1] - 1, -1, -1)
    addr = ((d > 0).astype(np.int64) * pow2).sum(-1)    # (N, 32)

    # 3. the pre-final-log sum of exponentials
    sel = W[np.arange(T)[None, :], addr]                # (N, 32, 6)
    z = np.clip(sel / tau, -clamp, clamp)
    S = np.exp(z).sum(axis=1)                           # (N, 6)
    dp = np.abs(S - Z["y_prelog_f64"]).max()
    rp = (np.abs(S - Z["y_prelog_f64"]) / Z["y_prelog_f64"]).max()
    print(f"2. y_prelog re-derived from weights          max|diff| {dp:.3e}  "
          f"max rel {rp:.3e}")

    # 4. the outer readout
    act = tph * tau * (np.log(S) - np.log(tph))
    da = np.abs(act - Z["y_action_mean_f64"]).max()
    print(f"3. y_action_mean = tph*tau*log(y_prelog/tph) max|diff| {da:.3e}")

    # 4b. same, but from the STORED float32 y_prelog — this is the number a student that
    #     predicts y_prelog would actually be graded through.
    act32 = tph * tau * (np.log(Z["y_prelog"].astype(np.float64)) - np.log(tph))
    d32 = np.abs(act32 - Z["y_action_mean_f64"]).max()
    print(f"4. ... from the float32 y_prelog             max|diff| {d32:.3e}")

    ok = dn < 1e-5 and rp < 1e-6 and da < 1e-5 and d32 < 1e-4
    print(f"\n{'PASS' if ok else 'FAIL'} — the .npz is self-contained and internally consistent.")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
