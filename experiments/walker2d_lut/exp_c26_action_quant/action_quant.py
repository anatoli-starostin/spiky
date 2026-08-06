"""exp_c26 — how coarsely can the teacher LUT's TORQUE OUTPUTS be quantized? (#75)

This quantizes the ACTIONS, not the table. The distinction matters: exp_c21's int4 study
quantized stored parameters, an open-loop change to a function. Here the policy is exact
and its OUTPUT is rounded to K levels per joint at every control step, inside the loop, so
the error feeds back through the physics and can interact with the gait limit cycle. A
policy can be robust to one and fragile to the other.

TWO ACTORS, deliberately. The headline c21 checkpoint (@20k, 5647.5) has an episode sd of
592 because 3 of its 100 episodes fall -- so it is a blunt instrument for detecting a
degradation smaller than a few hundred. Its @10k sibling scores 5286.6 with sd 51 and
100/100 full-length. Sweeping both costs ~8 s per cell and makes the difference between
"we could not resolve an effect" and "there is none": the @10k arm is the sensitive
instrument, the @20k arm is continuity with the published number.

PER-JOINT RANGES, from an unquantized rollout. One shared grid would be wrong -- the six
joints do not use the same fraction of [-1, 1]. Both candidate range rules are computed:

  * observed min/max  -- spans everything the policy actually does
  * 0.5/99.5 percentile -- finer resolution where the mass is, at the cost of CLIPPING the
    tails

and min/max is used as primary, because this policy saturates: a large share of actions sit
at |a| ~ 1, and those saturated commands are not outliers to be trimmed, they are the
bang-bang part of the gait. Percentile ranges would quantize the tails away. The choice is
not left as an assertion -- `--range-rule percentile` re-runs the whole sweep the other way
so the sensitivity to it is measured, not argued.

QUANTIZER: midtread uniform on [lo_j, hi_j], K levels, so K=3 gives {lo, mid, hi} and every
grid includes both endpoints. Values are clipped into range before rounding.

Writes action_quant.json. Reads the exp_c09 checkpoints read-only; writes nothing there.

Usage:
  python action_quant.py [--episodes 100] [--range-rule minmax|percentile]
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
          "exp_c11_lut_sac_2x2", "exp_c09_lut_sac"):
    sys.path.insert(0, os.path.join(D, p))

import eval_cpu                                            # noqa: E402
import perturb                                             # noqa: E402

C09 = os.path.join(D, "exp_c09_lut_sac")
ACTORS = [("@10k", "lut_sac_c21_seed4_20k_at10000_actor.npz", 5286.557120404921),
          ("@20k", "lut_sac_c21_seed4_20k_actor.npz", 5647.482926605437)]
KS = [15, 11, 9, 7, 5, 3]
ACT = 6


def quantize_action(a, lo, hi, K):
    """Midtread uniform quantizer, per joint. a [B, 6] -> same shape, on the grid."""
    a = np.clip(a, lo, hi)
    span = np.where(hi - lo > 0, hi - lo, 1.0)
    step = span / (K - 1)
    return lo + np.rint((a - lo) / step) * step


def rollout(model, policy_fn, episodes=100, max_steps=1000, seed0=0, record=False):
    """perturb.eval_batched verbatim + alive mask, speed, and optionally the actions.

    `policy_fn` receives the observation and returns the FINAL command, so the quantizer
    lives inside it and therefore inside the closed loop.
    """
    dt = model.opt.timestep * perturb.FRAME_SKIP
    datas, alive, rets = [], np.ones(episodes, bool), np.zeros(episodes)
    length, x0 = np.zeros(episodes, int), np.zeros(episodes)
    acts = []
    for ep in range(episodes):
        d = mujoco.MjData(model)
        rng = np.random.default_rng(seed0 + ep)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, model.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, model.nv)
        mujoco.mj_forward(model, d)
        datas.append(d)
        x0[ep] = d.qpos[0]

    for _ in range(max_steps):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        obs = np.stack([np.concatenate([datas[i].qpos[1:],
                                        np.clip(datas[i].qvel, -10, 10)])
                        for i in idx]).astype(np.float32)
        act = np.clip(np.asarray(policy_fn(obs), np.float64), -1.0, 1.0)
        if record:
            acts.append(act)
        for j, i in enumerate(idx):
            d = datas[i]
            xb = d.qpos[0]
            d.ctrl[:] = act[j]
            for _ in range(perturb.FRAME_SKIP):
                mujoco.mj_step(model, d)
            rets[i] += 1.0 + (d.qpos[0] - xb) / dt - 1e-3 * float(act[j] @ act[j])
            length[i] += 1
            z, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
                alive[i] = False
    vel = np.array([(datas[i].qpos[0] - x0[i]) / max(length[i], 1) / dt
                    for i in range(episodes)])
    out = dict(mean=float(rets.mean()), sd=float(rets.std(ddof=1)),
               full=int(alive.sum()), vel=float(vel.mean()),
               len_mean=float(length.mean()))
    if record:
        out["actions"] = np.concatenate(acts, 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--range-rule", default="minmax", choices=["minmax", "percentile"])
    ap.add_argument("--out", default="action_quant.json")
    a = ap.parse_args()

    m = perturb.make_model(None, 1.0)
    results = {}
    for label, ckpt, on_record in ACTORS:
        fn, n = eval_cpu.load_actor(os.path.join(C09, ckpt), forward_mode="hard")

        base = rollout(m, fn, episodes=a.episodes, record=True)
        A = base.pop("actions")
        if abs(base["mean"] - on_record) > 1.0:
            raise SystemExit(
                f"{label}: baseline reproduced {base['mean']:.1f}, not {on_record:.1f}. "
                f"Every degradation below is measured against this number, so a drifting "
                f"baseline would make the whole table meaningless. Stopping.")

        mn, mx = A.min(0), A.max(0)
        p_lo, p_hi = np.percentile(A, 0.5, axis=0), np.percentile(A, 99.5, axis=0)
        lo, hi = (mn, mx) if a.range_rule == "minmax" else (p_lo, p_hi)
        sat = (np.abs(A) > 0.99).mean(0)

        print(f"\n=== {label}  {ckpt} ({n:,} params) ===", flush=True)
        print(f"  baseline {base['mean']:.1f} +/- {base['sd']:.1f}  "
              f"full {base['full']}/{a.episodes}  vel {base['vel']:.3f} m/s  "
              f"({len(A):,} action samples)", flush=True)
        print(f"  joint ranges ({a.range_rule}) and saturation:", flush=True)
        for j in range(ACT):
            print(f"    j{j}: min/max [{mn[j]:+.3f}, {mx[j]:+.3f}]  "
                  f"p0.5/99.5 [{p_lo[j]:+.3f}, {p_hi[j]:+.3f}]  "
                  f"|a|>0.99 in {sat[j]*100:5.1f}% of steps", flush=True)

        rows = [dict(K=None, **base, retention=100.0)]
        for K in KS:
            qfn = (lambda o, K=K: quantize_action(np.asarray(fn(o), np.float64),
                                                  lo, hi, K))
            r = rollout(m, qfn, episodes=a.episodes)
            r["retention"] = 100.0 * r["mean"] / base["mean"]
            r["K"] = K
            rows.append(r)
            print(f"  K={K:>3}: {r['mean']:8.1f} +/- {r['sd']:7.1f}  "
                  f"full {r['full']:>3}/{a.episodes}  vel {r['vel']:.3f}  "
                  f"ret {r['retention']:6.1f}%", flush=True)

        results[label] = dict(checkpoint=ckpt, params=n, on_record=on_record,
                              range_rule=a.range_rule,
                              lo=lo.tolist(), hi=hi.tolist(),
                              minmax=[mn.tolist(), mx.tolist()],
                              pct=[p_lo.tolist(), p_hi.tolist()],
                              saturated_frac=sat.tolist(), rows=rows)

    json.dump(dict(episodes=a.episodes, range_rule=a.range_rule,
                   quantizer="midtread uniform per joint, K levels on [lo, hi], "
                             "applied after tanh inside the control loop",
                   actors=results),
              open(os.path.join(HERE, a.out), "w"), indent=1)
    print(f"\nwrote {a.out}", flush=True)


if __name__ == "__main__":
    main()
