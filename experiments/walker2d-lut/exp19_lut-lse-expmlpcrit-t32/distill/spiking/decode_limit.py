"""How much of the LIF gap is the DECODE, and how much is information the neuron destroyed?

A learning result is uninterpretable without this. If the output spike time `t_f` still
determines the teacher's action, then any residual error is a decode problem and no amount
of front-end training is *needed* to fix it. If `t_f` does NOT determine the action, there
is an information-theoretic floor that no decode and no front-end training can beat.

Measured non-parametrically and without training: bin `t_f` finely per output dimension and
report the within-bin spread of the teacher's action. That within-bin spread is the
irreducible error of the BEST POSSIBLE deterministic decode of `t_f` alone -- monotone,
affine, spline, neural, anything.

    python decode_limit.py [--bins 2000]
"""
import argparse
import os

import numpy as np
import torch

from student import SpikingStudent

HERE = os.path.dirname(os.path.abspath(__file__))
NPZ = os.path.join(HERE, "..", "distill_exp19_100k.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bins", type=int, default=2000)
    ap.add_argument("--n", type=int, default=100000)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Z = np.load(NPZ)
    X = torch.tensor(Z["x_norm"][: a.n], dtype=torch.float32, device=dev)
    A = torch.tensor(Z["y_action_mean"][: a.n], dtype=torch.float32, device=dev)
    astd = float(A.std())
    print(f"n={X.shape[0]:,}  bins={a.bins}  action std={astd:.4f}\n")
    print(f"{'neuron':6s} {'var':3s} {'decode':10s} {'mean|err|':>10s} {'norm':>8s} "
          f"{'FLOOR mean':>11s} {'floor norm':>11s}")

    for neuron in ("exact", "lif"):
        for var in ("D", "W"):
            for dm in (("affine",) if neuron == "exact" else ("affine", "corrected")):
                st = SpikingStudent(
                    torch.tensor(Z["weights"], device=dev),
                    torch.tensor(Z["anchor_a"], device=dev).long(),
                    torch.tensor(Z["anchor_b"], device=dev).long(),
                    float(Z["tau"]), neuron=neuron, variant=var, scope="races",
                    decode_mode=dm).to(dev)
                st.calibrate(X[:4096], A[:4096])
                ts, es = [], []
                for i in range(0, X.shape[0], 4096):
                    with torch.no_grad():
                        ah, t, _, _ = st(X[i:i + 4096])
                    ts.append(t)
                    es.append((ah - A[i:i + 4096]).abs())
                t_f = torch.cat(ts)
                err = torch.cat(es)
                # non-parametric floor: per-dim, bin t_f by rank, best constant per bin
                floor = []
                for o in range(A.shape[1]):
                    order = torch.argsort(t_f[:, o])
                    ao = A[order, o]
                    per = ao.numel() // a.bins
                    ao = ao[: per * a.bins].view(a.bins, per)
                    floor.append(float((ao - ao.median(dim=1, keepdim=True).values)
                                       .abs().mean()))
                fl = float(np.mean(floor))
                print(f"{neuron:6s} {var:3s} {dm:10s} {err.mean():>10.4e} "
                      f"{err.mean() / astd:>8.5f} {fl:>11.4e} {fl / astd:>11.5f}")

    print("\nFLOOR = mean |a - median(a | t_f bin)|, the best any deterministic decode of")
    print("t_f alone can do. If FLOOR << mean|err|, the gap is a DECODE problem. If they")
    print("are comparable, the neuron has destroyed information and no decode can recover it.")


if __name__ == "__main__":
    main()
