"""exp_c24 step 1 — re-score exp_c05's three saved ES winners at 100 episodes (#75).

exp_c05 evaluated its winners over **30** episodes, not the project-standard 100, and the
sep-CMA-ES MLP came out at 2996.7 +/- 913.8 -- a mean sitting exactly on the 3000 bar with
a spread wide enough that the 100-episode number could land on either side of it. The
whole CMA-ES pivot is premised on that number, so it gets measured properly first.

WHY THIS FILE EXISTS RATHER THAN JUST RUNNING exp_c05/eval_es_cpu.py:
that script writes `<mu>_cpueval.json` next to the weights, which would OVERWRITE the
30-episode results it took a 20-minute run to produce. Prior outputs are not overwritten
in this chapter, so the 100-episode numbers are written here instead.

The rollout below is a deliberate line-for-line mirror of `exp_c05_es/eval_es_cpu.py`:
same stock 100/50 CPU solver, same per-episode seeding (`default_rng(ep)`), same reward
reconstruction and termination test. The policies are built by importing exp_c05's own
`mlp_spec` / `lut_spec`, so nothing about the network is re-implemented here.

Usage (CPU only, no GPU):
  XLA_PYTHON_CLIENT_PREALLOCATE=false python rescore_c05.py
"""
import json, os, sys, time

import jax.numpy as jnp
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
C05 = os.path.join(BASE, "exp_c05_es")
for p in ("exp_c02_mjx_scaffold", "exp_c04_jax_lut", "exp_c05_es"):
    sys.path.insert(0, os.path.join(BASE, p))

import mjx_walker2d as W          # noqa: E402
import es_mjx                     # noqa: E402

RUNS = [
    ("es_mlp_sepcma_mu.npy", "mlp", {}, 2996.7, 913.8),
    ("es_mlp_openai_mu.npy", "mlp", {}, 2051.1, 157.7),
    ("es_lut_openai_mu.npy", "lut", dict(nap=6, tph=16), 904.0, 222.6),
]
EPISODES = 100


def evaluate(apply, flat, norm, m, episodes):
    dt = m.opt.timestep * W.FRAME_SKIP
    rets, lengths = [], []
    for ep in range(episodes):
        d = mujoco.MjData(m)
        rng = np.random.default_rng(ep)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, m.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, m.nv)
        mujoco.mj_forward(m, d)
        R, t = 0.0, 0
        for t in range(1, 1001):
            obs = np.concatenate([d.qpos[1:], np.clip(d.qvel, -10, 10)])
            act = np.asarray(apply(flat, jnp.asarray(obs, jnp.float32), norm))
            act = np.clip(act, -1, 1)
            x0 = d.qpos[0]
            d.ctrl[:] = act
            for _ in range(W.FRAME_SKIP):
                mujoco.mj_step(m, d)
            R += 1.0 + (d.qpos[0] - x0) / dt - 1e-3 * float(np.sum(act ** 2))
            z, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
                break
        rets.append(R)
        lengths.append(t)
    return rets, lengths


def main():
    st = json.load(open(os.path.join(BASE, "exp_c03_distillation", "dataset_stats.json")))
    norm = (jnp.asarray(st["obs_mean"], jnp.float32),
            jnp.asarray(st["obs_std"], jnp.float32))
    m = mujoco.MjModel.from_xml_path(W.XML)      # stock 100/50 -- the reference

    out = []
    for fn, policy, kw, old_mean, old_sd in RUNS:
        t0 = time.time()
        if policy == "mlp":
            _, apply, _ = es_mjx.mlp_spec(32)
        else:
            _, apply, _ = es_mjx.lut_spec(kw["nap"], kw["tph"])
        flat = jnp.asarray(np.load(os.path.join(C05, fn)))
        rets, lengths = evaluate(apply, flat, norm, m, EPISODES)
        mean, sd = float(np.mean(rets)), float(np.std(rets))
        full = int(sum(1 for L in lengths if L >= 1000))
        row = dict(file=fn, policy=policy, episodes=EPISODES,
                   cpu_reference_mean=mean, cpu_reference_std=sd,
                   full_length_episodes=full, mean_length=float(np.mean(lengths)),
                   old_30ep_mean=old_mean, old_30ep_std=old_sd,
                   delta_vs_30ep=mean - old_mean, wall_s=round(time.time() - t0, 1),
                   returns=rets, lengths=lengths)
        out.append(row)
        print(f"{fn:<26} {policy:<4} 100-ep {mean:7.1f} +/- {sd:6.1f}  "
              f"(30-ep was {old_mean:7.1f} +/- {old_sd:5.1f}, delta {mean-old_mean:+7.1f})  "
              f"full-length {full:>3}/100  median len {np.median(lengths):.0f}  "
              f"[{'SOLVED' if mean >= 3000 else 'below'} 3000]  {time.time()-t0:.0f}s",
              flush=True)

    json.dump(out, open(os.path.join(HERE, "rescore_c05_100ep.json"), "w"), indent=1)
    print("\nwrote rescore_c05_100ep.json")

    sep = out[0]
    if sep["cpu_reference_mean"] >= 3000:
        v = (f"exp_c05's best gradient-free MLP DOES clear the bar at 100 episodes "
             f"({sep['cpu_reference_mean']:.1f}). The 30-episode 2996.7 understated it.")
    else:
        v = (f"exp_c05's best gradient-free MLP does NOT clear the bar at 100 episodes "
             f"({sep['cpu_reference_mean']:.1f} vs 3000). The 30-episode 2996.7 was "
             f"optimistic, and 'basically solved' should not be repeated.")
    print(f"\nVERDICT: {v}")


if __name__ == "__main__":
    main()
