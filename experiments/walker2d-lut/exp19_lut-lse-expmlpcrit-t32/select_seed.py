"""Pick which exp19 seed to deploy — on DEPLOYED performance, not training return.

The training number (`final_ep_ret`) is measured with the STOCHASTIC policy (log_std floor
-1.897, so sigma >= 0.15) on MuJoCo-Warp physics with a reduced solver (iterations=10,
ls_iterations=8). The demo runs the DETERMINISTIC mean action on stock gymnasium
Walker2d-v5 physics. Those are different quantities, so the best training seed is not
necessarily the best seed to ship.

This evaluates every seed the way the server will actually run it and reports both numbers.

Usage:  python select_seed.py [--episodes 30]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
VIZ = os.path.join(HERE, "..", "..", "..", "landing", "walker2d-viz", "server")


def load_actor_params(npz_path):
    Q = np.load(npz_path)
    return dict(W=Q["weights"].astype(np.float64),
                a=Q["anchor_a"].astype(np.int64), b=Q["anchor_b"].astype(np.int64),
                tau=float(Q["tau_actor"]),
                mean=Q["obs_mean"].astype(np.float64),
                var=Q["obs_var"].astype(np.float64))


def act(P, obs):
    x = (np.asarray(obs, np.float64).reshape(-1)[: P["mean"].shape[0]] - P["mean"])
    x = x / np.sqrt(P["var"] + 1e-8)
    d = x[P["a"]] - x[P["b"]]
    nap = P["a"].shape[1]
    addr = ((d > 0).astype(np.int64) * (1 << np.arange(nap - 1, -1, -1))).sum(-1)
    T = P["W"].shape[0]
    sel = P["W"][np.arange(T), addr]
    z = sel / P["tau"]
    m = z.max(axis=0)
    lse = m + np.log(np.exp(z - m).sum(axis=0))
    return np.clip(T * P["tau"] * (lse - np.log(T)), -1.0, 1.0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--dir", default="/tmp/seedsel", help="folder holding seed{0,1,2}.npz")
    ap.add_argument("--train-dir", default="rerun_ckpt",
                    help="folder holding the ppo_s*.json for the TRAINING column. Must "
                         "match --dir's provenance: pointing it at the wrong run silently "
                         "prints another run's training numbers beside these deployed ones.")
    a = ap.parse_args()
    import gymnasium as gym
    import json

    env = gym.make("Walker2d-v5")
    print(f"{'seed':>5} {'train ep_ret':>13} {'deployed mean':>14} {'std':>8} "
          f"{'min':>8} {'max':>8} {'full/N':>8}")
    rows = []
    for s in (0, 1, 2):
        P = load_actor_params(os.path.join(a.dir, f"seed{s}.npz"))
        td = a.train_dir if os.path.isabs(a.train_dir) else os.path.join(HERE, a.train_dir)
        tr = json.load(open(os.path.join(td, f"ppo_s{s}.json")))["final_ep_ret"]
        rets, lens = [], []
        for ep in range(a.episodes):
            obs, _ = env.reset(seed=1000 + ep)      # SAME episode seeds for every model
            tot, n, done = 0.0, 0, False
            while not done:
                obs, r, term, trunc, _ = env.step(act(P, obs))
                tot += r
                n += 1
                done = term or trunc
            rets.append(tot)
            lens.append(n)
        rets, lens = np.array(rets), np.array(lens)
        rows.append((s, tr, rets.mean(), rets.std(), rets.min(), rets.max(),
                     int((lens >= 1000).sum())))
        print(f"{s:>5} {tr:>13.1f} {rets.mean():>14.1f} {rets.std():>8.1f} "
              f"{rets.min():>8.1f} {rets.max():>8.1f} {int((lens >= 1000).sum()):>4}/{a.episodes}")

    best = max(rows, key=lambda r: r[2])
    best_train = max(rows, key=lambda r: r[1])
    print(f"\nbest by DEPLOYED return : seed {best[0]}  ({best[2]:.1f})")
    print(f"best by TRAINING return : seed {best_train[0]}  ({best_train[1]:.1f})")
    if best[0] != best_train[0]:
        print("  -> they DISAGREE; ship the deployed-best, that is what viewers see.")
    else:
        print("  -> they agree.")


if __name__ == "__main__":
    main()
