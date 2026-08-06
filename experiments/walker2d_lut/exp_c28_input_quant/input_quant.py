"""exp_c28 — quantize the 17 OBSERVATION INPUTS to 16 levels, closed loop.

exp_c26 rounded the policy's outputs; this rounds its inputs, which is the sensor side of
the same deployment question: if the walker's proprioception arrived as 4-bit readings,
would it still walk? Applied to the RAW observation each control step, before the teacher
normalizer, so the interpretation is a physical sensor and not a rescaled internal
feature. Closed loop, so the error feeds back through the physics.

RANGES: 0.5/99.5 percentile per channel, measured from an unquantized rollout, not
min/max. exp_c26 learned that the hard way -- min/max ranges put the grid endpoints at
whatever the single most extreme sample happened to be, which shifts every level slightly
and produced a spurious K=5 cliff there.

"Snap to nearest level, no clipping beyond the range ends" is implemented as nearest-of-16
on a fixed grid, which necessarily pins anything outside [lo, hi] to an endpoint level.
That is not an extra clipping step, it is what nearest-neighbour on a bounded grid means;
the fraction of samples affected is measured and reported rather than assumed to be the
1% the percentile rule implies (the rollout under quantization visits different states
than the one the ranges came from).

Writes input_quant.json. Reads the exp_c09 checkpoints read-only.

Usage:
  python input_quant.py [--levels 16] [--episodes 100]
"""
import argparse
import json
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac", "exp_c26_action_quant"):
    sys.path.insert(0, os.path.join(D, p))

import eval_cpu                                            # noqa: E402
import perturb                                             # noqa: E402
import action_quant as AQ                                  # noqa: E402

C09 = os.path.join(D, "exp_c09_lut_sac")
ACTORS = [("@10k", "lut_sac_c21_seed4_20k_at10000_actor.npz", 5286.557120404921),
          ("@20k", "lut_sac_c21_seed4_20k_actor.npz", 5647.482926605437)]

# Observation layout, from mjx_walker2d.observation = qpos[1:] ++ clip(qvel, -10, 10).
NAMES = ["z_height", "torso_angle", "r_hip", "r_knee", "r_ankle",
         "l_hip", "l_knee", "l_ankle",
         "vx", "vz", "torso_angvel",
         "r_hip_vel", "r_knee_vel", "r_ankle_vel",
         "l_hip_vel", "l_knee_vel", "l_ankle_vel"]


def quantize_obs(o, lo, hi, K):
    """Nearest of K uniformly spaced levels on [lo, hi], per channel."""
    step = np.where(hi - lo > 0, (hi - lo) / (K - 1), 1.0)
    return lo + np.clip(np.rint((o - lo) / step), 0, K - 1) * step


def verify_quantizer(lo, hi, K, samples):
    """16 distinct levels per channel, endpoints exactly at the range ends, and the
    jax implementation agreeing with the numpy one. Checked before any number is
    produced, because a wrong grid would silently move the whole result."""
    import jax.numpy as jnp

    def jq(o):
        step = jnp.where(hi - lo > 0, (hi - lo) / (K - 1), 1.0)
        return lo + jnp.clip(jnp.rint((o - lo) / step), 0, K - 1) * step

    # A bare linspace lands exactly on level midpoints, where rounding is a tie and
    # float32 and float64 legitimately disagree. Ties are measure-zero on real
    # observations, so the cross-implementation check runs on the REAL samples and the
    # probe is nudged off the midpoints for the level-count check.
    probe = np.linspace(lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo), 4001)
    probe = probe + 1e-4 * (hi - lo)
    q = quantize_obs(probe, lo, hi, K)
    for j in range(len(lo)):
        u = np.unique(q[:, j])
        assert len(u) == K, f"channel {j} produced {len(u)} levels, not {K}"
        # Tolerance is relative to the channel's span: lo/hi are percentiles of
        # float32 observations, and the top level is reconstructed as
        # lo + (K-1)*step, so it carries float32 rounding, not an off-by-one.
        tol = 1e-6 * max(float(hi[j] - lo[j]), 1e-6)
        assert abs(u[0] - lo[j]) < tol and abs(u[-1] - hi[j]) < tol, \
            f"channel {j} endpoints {u[0]}, {u[-1]} != {lo[j]}, {hi[j]}"
    # Cross-implementation check on real samples. jax runs float32 and numpy float64,
    # so a sample sitting within float32 epsilon of a level midpoint can legitimately
    # round the other way. What must NOT happen is a systematic difference: any
    # disagreement has to be exactly one step (a tie broken differently) and has to be
    # rare. The evaluation itself uses the numpy path only, so this is a cross-check,
    # not a load-bearing dependency.
    ref = quantize_obs(np.asarray(samples, np.float64), lo, hi, K)
    got = np.asarray(jq(jnp.asarray(samples)))
    step = (hi - lo) / (K - 1)
    diff = np.abs(got - ref)
    frac = float((diff > 1e-4).mean())
    worst = float((diff / step).max())
    assert worst < 1.01, f"jax/numpy differ by {worst:.2f} steps -- more than a tie"
    assert frac < 1e-3, f"{frac*100:.3f}% of samples disagree -- not tie-breaking"
    return frac


def rollout_record_obs(model, policy_fn, episodes, max_steps=1000, seed0=0):
    """perturb.eval_batched, recording the raw observation stream."""
    datas, alive = [], np.ones(episodes, bool)
    log = []
    for ep in range(episodes):
        d = mujoco.MjData(model)
        rng = np.random.default_rng(seed0 + ep)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, model.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, model.nv)
        mujoco.mj_forward(model, d)
        datas.append(d)
    for _ in range(max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        obs = np.stack([np.concatenate([datas[i].qpos[1:],
                                        np.clip(datas[i].qvel, -10, 10)])
                        for i in idx]).astype(np.float32)
        log.append(obs)
        act = np.clip(np.asarray(policy_fn(obs), np.float64), -1.0, 1.0)
        for j, i in enumerate(idx):
            d = datas[i]
            d.ctrl[:] = act[j]
            for _ in range(perturb.FRAME_SKIP):
                mujoco.mj_step(model, d)
            z, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
                alive[i] = False
    return np.concatenate(log, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", type=int, default=16)
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    K = a.levels
    m = perturb.make_model(None, 1.0)
    out = {}

    for label, ckpt, on_record in ACTORS:
        fn, n = eval_cpu.load_actor(os.path.join(C09, ckpt), forward_mode="hard")
        base = AQ.rollout(m, fn, episodes=a.episodes)
        if abs(base["mean"] - on_record) > 1.0:
            raise SystemExit(f"{label}: baseline {base['mean']:.1f} != {on_record:.1f}; "
                             f"retention against a drifted baseline would be meaningless")

        O = rollout_record_obs(m, fn, a.episodes)
        lo = np.percentile(O, 0.5, axis=0)
        hi = np.percentile(O, 99.5, axis=0)
        drift = verify_quantizer(lo, hi, K, O)

        counter = dict(n=0, out=0)

        def qpol(o, lo=lo, hi=hi):
            counter["n"] += o.size
            counter["out"] += int(((o < lo) | (o > hi)).sum())
            return fn(quantize_obs(np.asarray(o, np.float64), lo, hi, K)
                      .astype(np.float32))

        r = AQ.rollout(m, qpol, episodes=a.episodes)
        r["retention"] = 100.0 * r["mean"] / base["mean"]
        r["frac_outside_range"] = counter["out"] / max(counter["n"], 1)

        print(f"\n=== {label}  {ckpt} ({n:,} params) ===", flush=True)
        print(f"  quantizer verified: {K} levels/channel, endpoints exact, "
              f"jax-vs-numpy tie-break disagreement {drift*100:.4f}% of samples", flush=True)
        print(f"  baseline          {base['mean']:8.1f} +/- {base['sd']:6.1f}  "
              f"full {base['full']:>3}/{a.episodes}  vel {base['vel']:.3f}", flush=True)
        print(f"  {K}-level inputs   {r['mean']:8.1f} +/- {r['sd']:6.1f}  "
              f"full {r['full']:>3}/{a.episodes}  vel {r['vel']:.3f}  "
              f"ret {r['retention']:6.1f}%", flush=True)
        print(f"  {r['frac_outside_range']*100:.2f}% of quantized samples fell outside "
              f"[lo, hi] and were pinned to an endpoint level", flush=True)
        print(f"  {'ch':<14}{'lo':>9}{'hi':>9}{'span':>9}{'step':>9}", flush=True)
        for j, nm in enumerate(NAMES):
            print(f"  {nm:<14}{lo[j]:9.3f}{hi[j]:9.3f}{hi[j]-lo[j]:9.3f}"
                  f"{(hi[j]-lo[j])/(K-1):9.4f}", flush=True)

        out[label] = dict(checkpoint=ckpt, params=n, levels=K,
                          range_rule="0.5/99.5 percentile over an unquantized rollout",
                          channels=NAMES, lo=lo.tolist(), hi=hi.tolist(),
                          baseline=base, quantized=r,
                          jax_numpy_max_delta=drift)

    json.dump(dict(episodes=a.episodes, levels=K, actors=out),
              open(os.path.join(HERE, "input_quant.json"), "w"), indent=1)
    print("\nwrote input_quant.json", flush=True)


if __name__ == "__main__":
    main()
