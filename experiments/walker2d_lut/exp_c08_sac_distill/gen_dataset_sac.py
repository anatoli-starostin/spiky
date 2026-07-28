"""exp_c08 1a — distillation dataset from the SAC teacher (#75).

Same recipe as exp_c03's PPO dataset — same size, same DAgger-style noise injection,
same CLIPPED-action labels — but the teacher is the SAC actor (exp_c01 seed 0, 5273.4).

The PPO teacher could be rolled out in batched MJX because it is JAX. SAC is a torch
model, so this collects in the CPU MuJoCo env instead, using the same lockstep trick as
the robustness harness: N environments stepped together with ONE batched policy call
per tick. A per-step single-sample forward would take about an hour for 4M pairs; this
takes minutes.

Environments auto-reset on termination so collection never stalls, and half of them are
driven with exploration noise while the LABEL is always the teacher's deterministic
clipped action — the student must learn the right action for states it will actually
visit once it is slightly wrong.
"""
import argparse, json, os, time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = ("/home/astarostin/projects/spiky/.venv/lib/python3.12/site-packages/"
       "gymnasium/envs/mujoco/assets/walker2d_v5.xml")
FRAME_SKIP = 4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        HERE, "..", "exp_c01_sac_baseline", "run_seed0", "sac_walker2d_final.zip"))
    ap.add_argument("--pairs", type=int, default=4_000_000)
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--noise-frac", type=float, default=0.5)
    ap.add_argument("--noise-std", type=float, default=0.1)
    a = ap.parse_args()

    from stable_baselines3 import SAC
    model = SAC.load(a.ckpt, device="cuda")
    print(f"teacher {os.path.basename(a.ckpt)} | "
          f"{sum(p.numel() for p in model.actor.parameters()):,} actor params",
          flush=True)

    m = mujoco.MjModel.from_xml_path(XML)
    rng = np.random.default_rng(0)
    noisy = np.arange(a.envs) < int(a.envs * a.noise_frac)

    def fresh(i):
        d = mujoco.MjData(m)
        r = np.random.default_rng(10_000 + i)
        d.qpos[:] += r.uniform(-5e-3, 5e-3, m.nq)
        d.qvel[:] += r.uniform(-5e-3, 5e-3, m.nv)
        mujoco.mj_forward(m, d)
        return d

    datas = [fresh(i) for i in range(a.envs)]
    steps_needed = int(np.ceil(a.pairs / a.envs))
    OBS = np.empty((steps_needed, a.envs, 17), np.float32)
    ACT = np.empty((steps_needed, a.envs, 6), np.float32)

    t0 = time.time()
    for t in range(steps_needed):
        obs = np.stack([np.concatenate([d.qpos[1:], np.clip(d.qvel, -10, 10)])
                        for d in datas]).astype(np.float32)
        act, _ = model.predict(obs, deterministic=True)
        label = np.clip(act, -1.0, 1.0)            # the teacher's BEHAVIOUR
        OBS[t] = obs
        ACT[t] = label
        drive = np.where(noisy[:, None],
                         label + a.noise_std * rng.standard_normal(label.shape),
                         label)
        drive = np.clip(drive, -1.0, 1.0)
        for i, d in enumerate(datas):
            d.ctrl[:] = drive[i]
            for _ in range(FRAME_SKIP):
                mujoco.mj_step(m, d)
            z, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
                datas[i] = fresh(t * a.envs + i)   # auto-reset
        if t % 500 == 0:
            el = time.time() - t0
            print(f"  step {t}/{steps_needed}  {((t+1)*a.envs)/max(el,1e-9):,.0f} "
                  f"pairs/s  ({el:.0f}s)", flush=True)

    obs = OBS.reshape(-1, 17)
    act = ACT.reshape(-1, 6)
    dt = time.time() - t0
    print(f"collected {len(obs):,} pairs in {dt:.1f}s ({len(obs)/dt:,.0f} pairs/s)",
          flush=True)

    np.save(os.path.join(HERE, "obs.npy"), obs)
    np.save(os.path.join(HERE, "act.npy"), act)
    json.dump(dict(pairs=int(len(obs)), teacher="SAC seed0", envs=a.envs,
                   noise_frac=a.noise_frac, noise_std=a.noise_std,
                   collect_s=round(dt, 1),
                   obs_mean=obs.mean(0).round(4).tolist(),
                   obs_std=obs.std(0).round(4).tolist(),
                   act_mean=act.mean(0).round(4).tolist(),
                   act_std=act.std(0).round(4).tolist(),
                   act_saturated_frac=float((np.abs(act) > 0.99).mean())),
              open(os.path.join(HERE, "dataset_stats.json"), "w"), indent=1)
    print(f"obs {obs.shape} act {act.shape} | action saturation "
          f"{float((np.abs(act) > 0.99).mean()):.1%}", flush=True)


if __name__ == "__main__":
    main()
