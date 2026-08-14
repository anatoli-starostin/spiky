"""Bake the big-data weight+offset PAIR into the actor artefact and verify end to end.

The pair is non-negotiable: the coordinate-descent fit re-derives the decode offset, and using
the shipped offset with the fitted weights mis-decodes by ~0.25 level (that error produced a
confidently wrong walker result once already).
"""
import argparse
import json
import os
import shutil
import sys
import time
import types

import numpy as np

BASE = "/home/astarostin/projects/spiky/experiments/walker2d-lut/walker2d-spiking"
ACT = f"{BASE}/deploy_quantised/spiking_lut_quantised_actor.npz"
WOPT = f"{BASE}/deploy_quantised/stage3_weights_bigdata.npy"
OOPT = f"{BASE}/deploy_quantised/stage3_offset_bigdata.npy"
POL = ("/home/astarostin/projects/spiky/experiments/walker2d-lut/"
       "exp19_lut-lse-expmlpcrit-t32/deploy/quantised/"
       "walker2d_fastlut_lse_exp19_quantised.npz")
DATA = f"{BASE}/analysis/software_teacher_io_dataset_100k.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--latency-n", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    # ---- 1. back up the current verified artefact ------------------------------------
    bak = f"{BASE}/deploy_quantised/spiking_lut_quantised_actor_GTSKEW_verified.npz.bak"
    if not os.path.exists(bak):
        shutil.copy2(ACT, bak)
    print(f"backup: {bak} ({os.path.getsize(bak):,} B)")

    # ---- 2. bake the PAIR -------------------------------------------------------------
    D = dict(np.load(ACT))
    Wo = np.load(WOPT).astype(np.float64)
    Oo = np.load(OOPT).astype(np.float64)
    old_W, old_off = D["weights"].astype(np.float64), D["affine"][:, 1].copy()
    aff = D["affine"].astype(np.float64).copy()
    aff[:, 1] = Oo
    D["weights"] = Wo.astype(np.float32)
    D["affine"] = aff
    D["stage3_fit"] = np.bytes_("coord-descent 8-bit log-domain, 153k teacher pairs")
    np.savez_compressed(ACT, **D)
    print(f"baked: {int((np.abs(Wo - old_W) > 1e-12).sum()):,} of {Wo.size:,} weights changed"
          f", max |dw| {np.abs(Wo - old_W).max():.6f}")
    print(f"       offset shift (level units): "
          f"{[round(float((Oo[o]-old_off[o])/(2/21)), 4) for o in range(6)]}")

    # ---- 2b. confirm the 8-bit grid ---------------------------------------------------
    Z = np.load(POL)
    tau = float(Z["tau_actor"])
    L0 = Z["weights"].astype(np.float64) / tau
    lo, hi = L0.min(), L0.max()
    step = (hi - lo) / 255.0
    Lg = (Wo / tau - lo) / step
    off_grid = np.abs(Lg - np.round(Lg)).max()
    print(f"8-bit grid check: max deviation from a grid point = {off_grid:.3e}  "
          f"{'ON GRID' if off_grid < 1e-9 else 'OFF GRID'}")

    # ---- 3/4. load the actor as the server would, verify + time -----------------------
    STAGE = "/tmp/_bake_check"
    shutil.rmtree(STAGE, ignore_errors=True)
    os.makedirs(STAGE + "/actors"); os.makedirs(STAGE + "/models")
    shutil.copy("/home/astarostin/projects/spiky/landing/walker2d-viz/server/actors/"
                "spiking_lut_quantised.py", STAGE + "/actors/")
    shutil.copy(ACT, STAGE + "/models/")
    open(STAGE + "/actors/__init__.py", "w").close()
    with open(STAGE + "/actors/base.py", "w") as f:
        f.write("class Actor:\n    def __init__(self, s):\n        self.action_space = s\n")
    sys.path.insert(0, STAGE)
    from actors.spiking_lut_quantised import SpikingLutQuantisedActor   # noqa: E402
    act = SpikingLutQuantisedActor(types.SimpleNamespace(shape=(6,)))
    print(f"\nactor loads: {act.name}  n_ticks {act.n_ticks}  levels {act.lv}")

    Dd = np.load(DATA)
    n_tr = 3 * 128 * 300
    raw = Dd["obs_raw"][n_tr:n_tr + a.n].astype(np.float64)      # held-out seed 3
    q_sw = Dd["action"][n_tr:n_tr + a.n].astype(np.float64)
    CL = float(Z["out_quant_clip"]); ST = 2 * CL / (int(Z["out_quant_levels"]) - 1)

    t0 = time.time()
    got = np.stack([act.act(o) for o in raw[:a.latency_n]]).astype(np.float64)
    lat = (time.time() - t0) / a.latency_n * 1000
    rest = np.stack([act.act(o) for o in raw[a.latency_n:]]).astype(np.float64)
    got = np.concatenate([got, rest])
    lev = np.rint((got - q_sw) / ST).astype(int)
    ex = [float((lev[:, o] == 0).mean()) for o in range(6)]
    w1 = [float((np.abs(lev[:, o]) <= 1).mean()) for o in range(6)]
    ms = [float(lev[:, o].mean()) for o in range(6)]
    lvv = np.linspace(-CL, CL, int(Z["out_quant_levels"]))
    print(f"\nBAKED ACTOR vs software, {len(raw):,} held-out states:")
    print(f"  per-dim exact  {[round(v*100,2) for v in ex]}   overall "
          f"{np.mean(ex)*100:.2f}%")
    print(f"  within-1-level {[round(v*100,3) for v in w1]}")
    print(f"  mean signed    {[round(v,4) for v in ms]}")
    print(f"  every value on the 22-level grid: "
          f"{bool(np.isclose(got[:, :, None], lvv[None, None, :], atol=1e-6).any(-1).all())}")
    print(f"  emitted range  [{got.min():.4f}, {got.max():.4f}]")
    print(f"  latency        {lat:.2f} ms/state on CPU")

    json.dump(dict(backup=bak, weights_changed=int((np.abs(Wo - old_W) > 1e-12).sum()),
                   offset_shift_levels=[float((Oo[o] - old_off[o]) / ST) for o in range(6)],
                   on_8bit_grid=bool(off_grid < 1e-9), n_states=int(len(raw)),
                   per_dim_exact=ex, per_dim_within1=w1, per_dim_mean_signed=ms,
                   overall_exact=float(np.mean(ex)), latency_ms=float(lat)),
              open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
