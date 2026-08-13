"""exp_c21 follow-up — materialize the 4-bit checkpoint and verify it.

The sweep (`quant_sweep.py`) showed the 5647.5 policy survives 4-bit quantization of
BOTH halves at 101.8% retention. This writes that configuration out as a real,
self-describing artifact: integer codes plus per-table scales, not dequantized floats.

WHY THE CODES ARE VERIFIED AGAINST THE SWEEP, NOT JUST RECOMPUTED. It would be easy
to write a quantizer here that is "the same idea" as the sweep's and quietly differs
in a rounding rule or a scale axis, producing a file that scores differently from the
4/4 row it claims to be. So `quantize_codes` is asserted, array by array, to
dequantize bit-exactly to `quant_sweep.quantize`'s output. If that assert holds, the
file IS the 4/4 row by construction.

ON "w AND b SHARING A CONSISTENT PER-TABLE SCALE": the sweep gave w and b each their
OWN per-table max-abs scale (32 scales apiece), because they are separate arrays with
different magnitudes -- b's entries are small next to w's, and forcing them onto one
shared scale would crush b to zero and change the routing. This file reproduces the
sweep, so it stores `w_scale` (32,) and `b_scale` (32,) separately. Read "consistent"
as "the same per-table scheme applied to both", which is what was measured.

Writes `lut_sac_c21_seed4_20k_int4.npz`. Touches no `*_cpueval.json`.

Usage:
  python make_int4.py [--episodes 100]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import quant_sweep as Q                                    # noqa: E402
import perturb                                             # noqa: E402

SRC = "lut_sac_c21_seed4_20k_actor.npz"
DST = "lut_sac_c21_seed4_20k_int4.npz"
BITS = 4
QMAX = 2 ** (BITS - 1) - 1                                 # 7


def quantize_codes(arr, bits=BITS):
    """Symmetric per-table max-abs -> (int8 codes in [-qmax, qmax], float32 scales).

    Same arithmetic as quant_sweep.quantize, but returning the codes instead of
    throwing them away. An all-zero table gets scale 1.0 and codes 0 (there is no
    max-abs to divide by); that convention is inherited from the sweep so the two
    stay identical.
    """
    a = np.asarray(arr, np.float64)
    qmax = 2 ** (bits - 1) - 1
    scale = np.abs(a.reshape(a.shape[0], -1)).max(axis=1) / qmax
    scale = np.where(scale > 0, scale, 1.0)
    s = scale.reshape((-1,) + (1,) * (a.ndim - 1))
    codes = np.clip(np.rint(a / s), -qmax, qmax)
    return codes.astype(np.int8), scale, scale.astype(np.float32)


def dequant(codes, scale):
    s = scale.reshape((-1,) + (1,) * (codes.ndim - 1))
    return (codes.astype(np.float64) * s).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()

    z = np.load(os.path.join(HERE, SRC))
    fields = {}
    for name in ("weights", "w", "b"):
        codes, scale64, scale32 = quantize_codes(z[name])
        ref = Q.quantize(z[name], BITS)
        # The codes must be EXACTLY the sweep's. Checked by dequantizing with the
        # same float64 scale the sweep used: any rounding or clipping difference
        # shows up here as a mismatch.
        if not np.array_equal(dequant(codes, scale64), ref):
            bad = int((dequant(codes, scale64) != ref).sum())
            raise SystemExit(f"{name}: codes differ from quant_sweep in {bad} entries "
                             f"-- this file would NOT be the 4/4 row. Refusing to write.")
        # The FILE stores float32 scales (they are what a deployment would carry), so
        # its dequantized values sit within float32 rounding of the sweep's, not on
        # top of them. Measured, not assumed.
        drift = np.abs(dequant(codes, scale32) - ref)
        rel = float(drift.max() / (np.abs(ref).max() or 1.0))
        fields[name + "_q"] = codes
        fields[name + "_scale"] = scale32
        print(f"  {name:<8} {str(z[name].shape):<14} -> int8 codes in "
              f"[{codes.min():+d}, {codes.max():+d}], {len(scale32)} scales, "
              f"{int((codes != 0).sum())}/{codes.size} nonzero, "
              f"fp32-scale drift {rel:.2e} rel", flush=True)

    out = dict(fields,
               log_T_soft=z["log_T_soft"], log_T_sel=z["log_T_sel"],
               n_heads=z["n_heads"], tph=z["tph"],
               quant="int4 symmetric per-table, qmax=7, dequant=code*scale",
               source_actor=SRC, forward_mode="hard")
    path = os.path.join(HERE, DST)
    np.savez(path, **out)
    print(f"wrote {DST} ({os.path.getsize(path):,} bytes)", flush=True)

    # ---- verify by loading the file back from scratch and evaluating it --------
    q = np.load(path, allow_pickle=False)
    fn, _ = Q.build_actor(dequant(q["w_q"], q["w_scale"]),
                          dequant(q["b_q"], q["b_scale"]),
                          dequant(q["weights_q"], q["weights_scale"]),
                          q["log_T_soft"], q["log_T_sel"],
                          int(q["n_heads"]), int(q["tph"]))
    mean, sd, full = Q.eval_full(perturb.make_model(None, 1.0), fn,
                                 episodes=a.episodes)
    sweep = [r for r in json.load(open(os.path.join(HERE, Q.OUT)))["rows"]
             if r["part"] == "C_both" and r["table_bits"] == 4 and r["addr_bits"] == 4]
    print(f"verify: {mean:.1f} +/- {sd:.1f}  full {full}/{a.episodes}", flush=True)
    if sweep:
        s = sweep[0]
        print(f"sweep 4/4 row: {s['mean']:.1f} +/- {s['std']:.1f}  "
              f"full {s['full_length']}/{s['episodes']}  "
              f"(delta {mean - s['mean']:+.3f})", flush=True)


if __name__ == "__main__":
    main()
