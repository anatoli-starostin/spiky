"""Pick the deploy seed by DEPLOYED performance, not by training return.

exp19's deploy/README.md records why this matters: under mismatched physics "best training
seed" and "best deployed seed" *disagreed*; under matched physics they agree. Either way the
selection that counts is the one measured in the environment the server actually steps, so
this scores every checkpoint over N full episodes of gymnasium Walker2d-v5 with the
deterministic (mean) action — exactly what the deployed actor does.

torch is used here only to load and run the checkpoints; the shipped actor is pure numpy.

Usage:
    python select_seed.py --dir <ckpt dir> [--episodes 30]
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="directory holding actor_s*.pt")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seeds", default="0,1,2")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(SRC))
    from models import REGISTRY                                  # noqa: E402
    import gymnasium as gym

    env = gym.make("Walker2d-v5")
    print(f"env {env.spec.id}, {a.episodes} episodes/seed, deterministic mean action\n")
    print(f"{'seed':>4} {'mean':>9} {'std':>8} {'min':>9} {'max':>9} "
          f"{'len':>7} {'full-1000':>10} {'train-final':>12}")

    rows = []
    for s in [int(v) for v in a.seeds.split(",")]:
        p = os.path.join(a.dir, f"actor_s{s}.pt")
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ac = REGISTRY[ck["arch"]](ck["obs_dim"], ck["act_dim"])
        ac.load_state_dict(ck["state_dict"], strict=True)
        ac.eval()
        mean, var = ck["obs_mean"], ck["obs_var"]

        rets, lens = [], []
        for ep in range(a.episodes):
            obs, _ = env.reset(seed=1000 + ep)          # same episode seeds for every arm
            total, steps, done = 0.0, 0, False
            while not done:
                x = (torch.from_numpy(np.asarray(obs, np.float32)) - mean) / torch.sqrt(var + 1e-8)
                with torch.no_grad():
                    act = ac(x.unsqueeze(0))[0][0].numpy()
                obs, r, term, trunc, _ = env.step(np.clip(act, -1, 1).astype(np.float32))
                total += r
                steps += 1
                done = term or trunc
            rets.append(total)
            lens.append(steps)
        rets, lens = np.array(rets), np.array(lens)
        rows.append((s, rets, lens, ck["final_ep_ret"]))
        print(f"{s:>4} {rets.mean():9.1f} {rets.std():8.1f} {rets.min():9.1f} "
              f"{rets.max():9.1f} {lens.mean():7.1f} {int((lens>=1000).sum()):>6}/{a.episodes:<3} "
              f"{ck['final_ep_ret']:12.1f}")

    best = max(rows, key=lambda r: r[1].mean())
    print(f"\nbest DEPLOYED seed: {best[0]}  ({best[1].mean():.1f} +- {best[1].std():.1f})")
    best_train = max(rows, key=lambda r: r[3])[0]
    print(f"best TRAINING seed: {best_train}"
          + ("  — they agree" if best_train == best[0] else "  — they DISAGREE; deployed wins"))


if __name__ == "__main__":
    main()
