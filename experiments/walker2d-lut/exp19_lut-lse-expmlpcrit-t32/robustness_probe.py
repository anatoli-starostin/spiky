"""Is the deployed exp19 actor brittle to small physics changes?

The artifact verifies locally (4845.9 over 30 episodes under the server's exact harness) but
falls on the server. The plumbing checks out — same obs layout, same call pattern — so the
remaining candidate is that the policy sits close to a failure boundary and a slightly
different MuJoCo build (the image pins 3.11.0; here we have 3.10.0) tips it over.

This perturbs the solver the way a different build effectively would, and also emulates the
training env's much softer solver, to see how far the policy's competence extends.

Usage:  python robustness_probe.py [--episodes 10] [--npz PATH]
"""
import argparse
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def act_np(P, obs):
    x = (np.asarray(obs, np.float64).reshape(-1)[:17] - P["obs_mean"])
    x = x / np.sqrt(P["obs_var"] + 1e-8)
    d = x[P["anchor_a"]] - x[P["anchor_b"]]
    nap = P["anchor_a"].shape[1]
    addr = ((d > 0).astype(np.int64) * (1 << np.arange(nap - 1, -1, -1))).sum(-1)
    T = P["weights"].shape[0]
    sel = P["weights"][np.arange(T), addr]
    tau = float(P["tau_actor"])
    z = sel / tau
    m = z.max(axis=0)
    lse = m + np.log(np.exp(z - m).sum(axis=0))
    return np.clip(T * tau * (lse - np.log(T)), -1, 1).astype(np.float32)


def run(P, episodes, iters=None, ls=None):
    import gymnasium as gym
    env = gym.make("Walker2d-v5")
    m = env.unwrapped.model
    if iters is not None:
        m.opt.iterations = iters
    if ls is not None:
        m.opt.ls_iterations = ls
    rets, lens = [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=2000 + ep)
        tot, n, done = 0.0, 0, False
        while not done:
            obs, r, term, trunc, _ = env.step(act_np(P, obs))
            tot += r
            n += 1
            done = term or trunc
        rets.append(tot)
        lens.append(n)
    return np.array(rets), np.array(lens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--npz", default=os.path.join(HERE, "deploy",
                                                  "walker2d_fastlut_lse_exp19.npz"))
    a = ap.parse_args()
    Z = np.load(a.npz)
    P = {k: Z[k] for k in Z.files}

    import gymnasium as gym
    e = gym.make("Walker2d-v5")
    print(f"stock Walker2d-v5 solver: iterations={e.unwrapped.model.opt.iterations} "
          f"ls_iterations={e.unwrapped.model.opt.ls_iterations}")
    print(f"our training env used:    iterations=10 ls_iterations=8\n")
    print(f"{'solver setting':<34} {'mean ret':>10} {'std':>9} {'mean len':>9} {'full':>7}")
    grid = [("stock (deployment default)", None, None),
            ("iterations=10, ls=8 (training)", 10, 8),
            ("iterations=50, ls=25", 50, 25),
            ("iterations=200, ls=100", 200, 100)]
    for label, it, ls in grid:
        rets, lens = run(P, a.episodes, it, ls)
        print(f"{label:<34} {rets.mean():>10.1f} {rets.std():>9.1f} {lens.mean():>9.1f} "
              f"{int((lens >= 1000).sum()):>4}/{a.episodes}")
    print("\nA policy whose score swings wildly across these is brittle: a different MuJoCo")
    print("build differs from stock by far less than these perturbations do.")


if __name__ == "__main__":
    main()
