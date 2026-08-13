"""Export the amplitude-encoded quantised spiking actor's artefact.

Sibling of the delay-based export path. Writes a NEW npz + meta (it does not touch the
existing spiking_lut_actor.npz). Everything the actor needs at inference:

  * the LUT tables, anchors and tau           (from the quantised PPO npz)
  * the frozen obs statistics                 (   "   )
  * `in_quant_edges`                          (   "   )  -- ordering only; the SNN never
                                                            needs `in_quant_dequant`
  * the output grid (`out_quant_levels/clip`) (   "   )
  * `beta` per dim + the decode affine        (from a validated pipeline build)
  * `gate_tick`, `t_in`, `n_ticks`, `tau_m_out`, `d_out`

Usage:
    python tiny_lut_quantised_export.py --tau-m-out 31.257 --n 512 --out <dir>
"""
import argparse
import json
import os

import numpy as np

import tiny_lut_quantised_pipeline as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=P.__dict__.get("_DEFAULT_NPZ") or
                    "/home/astarostin/projects/spiky/experiments/walker2d-lut/"
                    "exp19_lut-lse-expmlpcrit-t32/deploy/quantised/"
                    "walker2d_fastlut_lse_exp19_quantised.npz")
    ap.add_argument("--calib", required=True,
                    help="the JSON written by tiny_lut_quantised_pipeline.py --out "
                         "(supplies the validated beta + decode affine)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stem", default="spiking_lut_quantised_actor")
    a = ap.parse_args()

    Z = np.load(a.npz)
    C = json.load(open(a.calib))
    dims = sorted(int(k) for k in C["affine"])
    os.makedirs(a.out, exist_ok=True)

    Q = dict(
        weights=Z["weights"].astype(np.float32),
        anchor_a=Z["anchor_a"].astype(np.int64),
        anchor_b=Z["anchor_b"].astype(np.int64),
        tau=np.float64(float(Z["tau_actor"])),
        obs_mean=Z["obs_mean"].astype(np.float64),
        obs_var=Z["obs_var"].astype(np.float64),
        in_quant_edges=Z["in_quant_edges"].astype(np.float64),
        out_quant_levels=np.int64(int(Z["out_quant_levels"])),
        out_quant_clip=np.float64(float(Z["out_quant_clip"])),
        t_in=np.int64(P.T_IN),
        gate_tick=np.int64(P.GATE_TICK),
        emit=np.int64(P.EMIT),
        d_out=np.int64(P.D_OUT),
        n_ticks=np.int64(int(C["n_ticks"])),
        tau_m_out=np.float64(float(C["tau_m_out"])),
        beta=np.array([C["beta"][str(o)] for o in dims], np.float64),
        affine=np.array([C["affine"][str(o)] for o in dims], np.float64),  # (6,2) slope,off
    )
    npz = os.path.join(a.out, f"{a.stem}.npz")
    np.savez_compressed(npz, **Q)

    s3 = C["stage3"]
    meta = dict(
        source="exp012 amplitude-encoded spiking LUT (tiny_lut_quantised_pipeline.py)",
        policy=os.path.basename(a.npz),
        stage3_encoding="AMPLITUDE: every selected-cell->output synapse has delay=1 and "
                        "weight beta_o*exp(w/tau); all 32 land on one tick and the linear "
                        "anti-leak membrane integrates sum_t exp(w_t/tau) directly",
        readout="T = first output-spike tick; mu = slope*T + offset; silence -> -clip; "
                "then snap to the output grid",
        decode_slope_theory=-32.0 * float(Z["tau_actor"]) / P.calib(float(C["tau_m_out"])),
        tau_m_out=float(C["tau_m_out"]),
        n_ticks=int(C["n_ticks"]), dmax=int(C["dmax"]),
        neurons=int(C["neurons"]), synapses=int(C["synapses"]),
        levels=int(C["levels"]),
        verification=dict(
            stage1_bit_parity=C["stage1_bit_parity"],
            stage2_none=C["stage2_none"], stage2_multi=C["stage2_multi"],
            stage3_exact={k: v["exact"] for k, v in s3.items()},
            stage3_within_one_level={k: v["within1"] for k, v in s3.items()},
        ),
        note="Only in_quant_edges crosses into the SNN -- the network consumes tick "
             "ORDERING only, so in_quant_dequant is not needed at inference.",
    )
    json.dump(meta, open(os.path.join(a.out, f"{a.stem}_meta.json"), "w"), indent=2)
    print(f"wrote {npz} ({os.path.getsize(npz):,} bytes)")
    print(f"wrote {os.path.join(a.out, a.stem + '_meta.json')}")


if __name__ == "__main__":
    main()
