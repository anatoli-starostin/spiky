"""Distillation train loop STUB — shapes and gradients only, not a training run.

Loads distill_exp19_100k.npz, encodes the observations as input spike latencies, runs the
frozen LUT-derived TTFS readout (variant D or W), and matches the student's OUTPUT SPIKE
LATENCIES against the latencies the teacher's action targets correspond to.

The loss lives in latency space on purpose. The decode is exactly affine,
`a = -tph*t + bias`, so latency-MSE and action-MSE differ only by the constant tph^2 --
but latency is the quantity a spiking implementation actually produces, and keeping the
objective there means the same code works when the decode stops being exact.

Deliberately NOT a training run: `--steps` defaults to 5 and the point is to confirm
shapes, that the exact fixed point really is exact, and that gradients reach the front-end.

    python train_stub.py [--variant D|W] [--learnable-races] [--steps 5]
"""
import argparse
import os

import numpy as np
import torch

from lut_ttfs import DEFAULT_NPZ, LatencyEncoder, SpikingLutStudent, build_from_npz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=DEFAULT_NPZ)
    ap.add_argument("--variant", default="D", choices=["D", "W"])
    ap.add_argument("--learnable-races", action="store_true",
                    help="v2 front-end: learnable linear forms instead of fixed anchor pairs")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--soft", action="store_true",
                    help="soft gates everywhere (no straight-through)")
    a = ap.parse_args()

    torch.manual_seed(0)
    ro, Z = build_from_npz(a.npz, variant=a.variant)
    st = SpikingLutStudent(ro, LatencyEncoder(), learnable_races=a.learnable_races)

    X = torch.tensor(Z["x_norm"], dtype=torch.float64)
    Y = torch.tensor(Z["y_action_mean_f64"])
    # invert the exact affine decode to get the TARGET output latency
    T_tgt = (ro.decode_bias - Y) / ro.tph
    print(f"data     x_norm {tuple(X.shape)} {X.dtype}   y {tuple(Y.shape)}")
    print(f"targets  latency t* range [{T_tgt.min():.6f}, {T_tgt.max():.6f}]  "
          f"(action range [{Y.min():.3f}, {Y.max():.3f}])")

    trainable = [(n, p) for n, p in st.named_parameters() if p.requires_grad]
    print(f"variant  {a.variant}   learnable params: "
          + ", ".join(f"{n}{tuple(p.shape)}" for n, p in trainable)
          + f"   total {sum(p.numel() for _, p in trainable):,}")
    print(f"frozen   readout w{tuple(ro.w.shape)} = {ro.w.numel():,} scalars "
          f"= {ro.n_tables * ro.table_dim:,} cells x {ro.n_out} output synapses")

    # --- the exact fixed point: before any training, the student IS the teacher ---------
    with torch.no_grad():
        ahat, tout = st(X[:20000], hard=True, return_latency=True)
    print(f"\nfixed point (no training): |a_student - a_teacher| max "
          f"{(ahat - Y[:20000]).abs().max():.3e}")

    opt = torch.optim.Adam([p for _, p in trainable], lr=a.lr)
    n = X.shape[0]
    print(f"\n{'step':>4} {'latency MSE':>13} {'action RMSE':>12}   grad norms")
    for s in range(a.steps):
        idx = torch.randint(0, n, (a.batch,))
        ahat, tout = st(X[idx], hard=not a.soft, return_latency=True)
        loss = ((tout - T_tgt[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        gn = "  ".join(f"{nm}={0.0 if p.grad is None else p.grad.norm():.3e}"
                       for nm, p in trainable)
        opt.step()
        rmse = (ahat - Y[idx]).pow(2).mean().sqrt()
        print(f"{s:>4} {loss.item():>13.6e} {rmse.item():>12.6e}   {gn}")

    print("\nSTUB ONLY — no full training run was started.")
    print("Note: d(loss)/d(encoder.c) is identically zero. The offset cancels in every "
          "race difference t_b - t_a, so it is a gauge freedom, not a degree of freedom.")


if __name__ == "__main__":
    main()
