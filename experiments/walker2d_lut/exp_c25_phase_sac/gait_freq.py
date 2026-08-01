"""exp_c25 step 1 — what is the natural gait frequency of the c21 LUT-SAC walker?

The phase sweep needs a real number to centre on, not a guess. This rolls the trained
c21 seed-4 @ 20k actor deterministically on the CPU reference physics, records the six
joint angles (observation indices 2..7), and takes their FFT.

Sampling rate is the CONTROL rate, not the physics rate: one observation per policy
step, dt = FRAME_SKIP * timestep = 4 * 0.002 = 0.008 s -> 125 Hz. A 1000-step episode
is 8 s, so the frequency resolution is 1/8 = 0.125 Hz.

Two details that matter for getting an honest number:
  * The mean is removed before the transform. A joint that oscillates around a non-zero
    offset has a large DC component that would otherwise dominate every spectrum.
  * Only FULL-LENGTH episodes are used. A fallen walker's trace is a transient, not a
    gait, and averaging its spectrum in would smear the peak.

Writes gait_freq.json. Reads nothing it can damage; writes nothing outside this dir.

Usage:
  python gait_freq.py [--episodes 8]
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

CKPT = os.path.join(D, "exp_c09_lut_sac", "lut_sac_c21_seed4_20k_actor.npz")
JOINTS = ["r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle"]
JOINT_IDX = [2, 3, 4, 5, 6, 7]          # observation layout: qpos[1:] -> idx 2..7


def rollout_record(model, policy_fn, episodes, max_steps=1000, seed0=0):
    """perturb.eval_batched, recording the observation stream. Physics/reset/termination
    are copied line for line so the trajectories are the ones the eval scores."""
    dt = model.opt.timestep * perturb.FRAME_SKIP
    datas, alive = [], np.ones(episodes, bool)
    length = np.zeros(episodes, int)
    obs_log = [[] for _ in range(episodes)]
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
        for j, i in enumerate(idx):
            obs_log[i].append(obs[j])
        act = np.clip(np.asarray(policy_fn(obs), np.float64), -1.0, 1.0)
        for j, i in enumerate(idx):
            d = datas[i]
            d.ctrl[:] = act[j]
            for _ in range(perturb.FRAME_SKIP):
                mujoco.mj_step(model, d)
            length[i] += 1
            z, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
                alive[i] = False
    return [np.stack(o) for o in obs_log], length, dt


def dominant(sig, dt):
    """Peak of the mean-removed amplitude spectrum, ignoring DC. -> (freq_hz, spectrum)"""
    x = sig - sig.mean()
    n = len(x)
    amp = np.abs(np.fft.rfft(x)) / n
    freq = np.fft.rfftfreq(n, dt)
    k = int(np.argmax(amp[1:])) + 1              # skip bin 0 (DC)
    return float(freq[k]), freq, amp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=8)
    a = ap.parse_args()

    fn, n = eval_cpu.load_actor(CKPT, forward_mode="hard")
    m = perturb.make_model(None, 1.0)
    obs_log, length, dt = rollout_record(m, fn, a.episodes)
    print(f"control dt = {dt:.4f} s -> {1/dt:.1f} Hz sampling; episode lengths "
          f"{length.tolist()}", flush=True)

    full = [i for i in range(a.episodes) if length[i] >= 1000]
    if not full:
        raise SystemExit("no full-length episode: nothing here is a steady gait, so a "
                         "dominant frequency would be measuring a fall, not a stride.")
    print(f"using {len(full)}/{a.episodes} full-length episodes "
          f"(resolution {1.0/(1000*dt):.3f} Hz)", flush=True)

    per_joint = {}
    for name, ji in zip(JOINTS, JOINT_IDX):
        fs = [dominant(obs_log[i][:, ji], dt)[0] for i in full]
        per_joint[name] = dict(mean=float(np.mean(fs)), sd=float(np.std(fs)),
                               per_episode=[float(x) for x in fs])
        print(f"  {name:<8} {np.mean(fs):6.3f} Hz  +/- {np.std(fs):.3f}", flush=True)

    hips = per_joint["r_hip"]["per_episode"] + per_joint["l_hip"]["per_episode"]
    f_hip = float(np.mean(hips))
    allf = [v for j in per_joint.values() for v in j["per_episode"]]
    print(f"\nhip-based natural frequency: {f_hip:.3f} Hz "
          f"(stride period {1/f_hip:.3f} s)", flush=True)
    print(f"all-joint median: {np.median(allf):.3f} Hz", flush=True)

    json.dump(dict(checkpoint=os.path.basename(CKPT), params=n, dt=dt,
                   sampling_hz=1.0 / dt, episodes=a.episodes,
                   full_length_used=len(full), per_joint=per_joint,
                   f_hip_hz=f_hip, f_all_median_hz=float(np.median(allf)),
                   note="peak of the mean-removed amplitude spectrum, DC bin excluded"),
              open(os.path.join(HERE, "gait_freq.json"), "w"), indent=1)
    print("wrote gait_freq.json", flush=True)


if __name__ == "__main__":
    main()
