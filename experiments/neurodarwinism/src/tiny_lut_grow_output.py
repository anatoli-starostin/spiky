"""exp012: the LUT output stage with an ANTI-LEAKY (growing) membrane, on the real engine.

The decaying-membrane version failed structurally: V peaks at the last arrival in only 2.9%
of samples, so the crossing reports a partial sum. Flipping the sign of cf_1 fixes the
geometry -- and it is a parameter choice on the EXISTING kernel, not an engine change:

    v' = cf_1 * v + I      with cf_1 = +1/tau_m   (LIFNeuronMeta hardcodes -1/tau_m)

Then V(t) = A * sum_t e^{(t - a_t)/tau_eff} is MONOTONE INCREASING, every arrival is
amplified rather than decayed, and the unique crossing of a constant threshold gives

    T = -tau_eff * log( sum_t e^{-a_t/tau_eff} ) + const

With a_t = -(tau_eff/tau) * w_t + c  (bigger w -> EARLIER arrival, so it dominates):

    T = -tau_eff * log( sum_t e^{w_t/tau} ) + const'      -> out[o] is AFFINE in T.
"""
import argparse
import json

import numpy as np
import torch

from tiny_lut_output_stage import TAU, TPH, affine_fit, build_net, lut_targets, run_first_spike


def calibrate_growth(tau_m, n_euler=2, dt=0.5):
    """The kernel's ACTUAL per-tick growth factor -> the effective LSE temperature."""
    per_tick = (1.0 + dt / tau_m) ** n_euler
    return 1.0 / np.log(per_tick), per_tick


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau-m", type=float, default=10.0)
    ap.add_argument("--n", type=int, default=1024)
    ap.add_argument("--dims", default="0")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    import os
    Z = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                             "distill_exp19_100k.npz"))
    x = Z["x_norm"].astype(np.float64)
    ntr = len(x) - 4000
    w_sel, out_true, idx = lut_targets(Z, x[ntr:ntr + a.n])
    tau_eff, per_tick = calibrate_growth(a.tau_m)
    scale = tau_eff / TAU
    print(f"tau_m {a.tau_m} -> per-tick GROWTH {per_tick:.6f} -> CALIBRATED tau_eff "
          f"{tau_eff:.4f}   (naive +1/tau would say {a.tau_m:.4f})")
    print(f"scale tau_eff/tau = {scale:.3f} ticks per unit of w\n")

    R = {"tau_m": a.tau_m, "tau_eff": tau_eff, "per_tick_growth": per_tick,
         "scale": scale, "n": int(a.n), "dims": {}}
    for o in [int(v) for v in a.dims.split(",")]:
        ws = w_sel[:, :, o]
        # bigger w -> EARLIER; shift so the earliest arrival is at tick 1
        # The offset MUST be global. Shifting each sample by its own minimum adds a
        # per-sample constant to every a_t, which shifts T by exactly that constant and
        # destroys the signal -- that mistake cost an R2 of 0.097 with every structural
        # check passing, which is a good reminder that "the circuit fired correctly" and
        # "the circuit computes the right thing" are different claims.
        raw = -scale * ws
        C = float(np.ceil(-raw.min() + 1.0))
        arr = np.rint(raw + C).astype(np.int64)
        spread = float((arr.max(1) - arr.min(1)).mean())

        # theta must exceed V at the LAST arrival so the crossing is strictly after it.
        # V(a_max) = A * sum_t e^{(a_max - a_t)/tau_eff}; take the worst case over samples.
        Vlast = np.array([np.exp((r.max() - r) / tau_eff).sum() for r in arr])
        amp = 1.0 / (2.0 * Vlast.max())        # theta = 1.0, factor 2 of headroom
        Vmax_reached = amp * np.exp((arr.max() + 400 - arr.min()) / tau_eff).sum()
        n_ticks = int(arr.max() + 12 * a.tau_m + 20)

        net, ids = build_net(32, 1, np.ones(32, np.int64), np.arange(32), amp, a.tau_m,
                             threshold=1.0, grow=True)
        T = run_first_spike(net, ids, arr, 32, n_ticks)
        fired = T < n_ticks
        early = fired & (T <= arr.max(1) + 1)
        # participation at the crossing: how many terms are actually contributing
        part = np.array([(np.exp(-r / tau_eff).sum() ** 2) / (np.exp(-r / tau_eff) ** 2).sum()
                         for r in arr])

        y = out_true[:, o]
        half = fired & (np.arange(len(y)) < len(y) // 2)
        aa, bb = affine_fit(T.astype(float), y, half)
        ev = fired & (np.arange(len(y)) >= len(y) // 2)
        pred = aa * T.astype(float) + bb
        mse = float(((pred[ev] - y[ev]) ** 2).mean())
        mx = float(np.abs(pred[ev] - y[ev]).max())
        var = float(y[ev].var())
        d = dict(spread_ticks=spread, amp=float(amp), theta=1.0,
                 frac_fired=float(fired.mean()), frac_crossing_before_last=float(early.mean()),
                 live_terms_median=float(np.median(part)), max_V_headroom=float(Vmax_reached),
                 n_ticks=n_ticks, delay_max=int(arr.max()),
                 mse=mse, max_err=mx, target_var=var, r2=1 - mse / var, affine=[aa, bb])
        R["dims"][str(o)] = d
        print(f"dim {o}: spread {spread:.1f} ticks, amp {amp:.3e}, fired "
              f"{100 * fired.mean():.2f}%, crossing-before-last "
              f"{100 * early.mean():.2f}%, live terms {np.median(part):.2f}")
        print(f"   held-out MSE {mse:.6f}  max|err| {mx:.6f}  var {var:.6f}  "
              f"R2 {1 - mse / var:.6f}")
        del net
        torch.cuda.empty_cache()

    if a.out:
        json.dump(R, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
