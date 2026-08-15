"""8-bit weight quantisation, measured in gymnasium — the half of the question that does
not need a spiking substrate.

Building the plain-sum SPIKING analogue is blocked: both spiking pipelines hardcode an
anti-leaky output neuron (`cf_1 = +1/TAU_M_OUT`) whose time-to-threshold supplies the
logarithm, and deliver the weight through a synapse that supplies the exponential. See
RESULTS.md. What *can* be measured without that substrate is the other half: does 8-bit
weight quantisation cost either pooling any return?

Each arm is quantised on ITS OWN natural grid, 256 levels:
    arm A (log-sum-exp) : uniform in L = W/tau, i.e. the log domain (stage3_cd_bigdata.py)
    arm B (plain sum)   : uniform in W, linear -- there is no tau to divide by

Evaluated in gymnasium Walker2d-v5, the deploy metric, with the same input (128-tick
Gaussian companding) and output (22-level uniform) quantisers the training used.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "src")
ARMS = {"A": "log-sum-exp (exp19)", "B": "plain sum (ablation)"}


def quantise_8bit(W, tau):
    """256 levels on the arm's own grid. Returns (Wq, description, step)."""
    if tau is not None:
        L = W / tau
        lo, hi = L.min(), L.max()
        step = (hi - lo) / 255.0
        Lq = lo + np.round((L - lo) / step) * step
        return Lq * tau, f"log domain L=W/tau (tau={tau:.6f})", float(step)
    lo, hi = W.min(), W.max()
    step = (hi - lo) / 255.0
    return lo + np.round((W - lo) / step) * step, "linear in W", float(step)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.abspath(SRC))

    from models import REGISTRY
    from obs_quant import GaussianCompandingQuantizer
    from act_quant import UniformActionQuantizer
    import gymnasium as gym

    env = gym.make("Walker2d-v5")
    iq = GaussianCompandingQuantizer(128, 1.0)
    oq = UniformActionQuantizer(22, 1.0, straight_through=False)
    print(f"env {env.spec.id}, {a.episodes} episodes, deterministic mean action, "
          f"input+output quantisers ON\n")
    print(f"{'arm':<24} {'seed':>4} {'weights':<26} {'mean':>9} {'std':>8} {'min':>9} "
          f"{'full-1000':>10}")

    report = {}
    for arm in ARMS:
        for p in sorted(glob.glob(os.path.join(a.dir, f"{arm}_s*.pt"))):
            seed = int(os.path.basename(p).split("_s")[1].split(".")[0])
            ck = torch.load(p, map_location="cpu", weights_only=False)
            W = ck["state_dict"]["actor_lut.weights"].numpy().astype(np.float64)
            tau = None
            if "actor_lut.exp_outputs_tau_raw" in ck["state_dict"]:
                r = float(ck["state_dict"]["actor_lut.exp_outputs_tau_raw"])
                tau = float(np.log1p(np.exp(-abs(r))) + max(r, 0.0))
            Wq, desc, step = quantise_8bit(W, tau)

            for tag, weights in (("float32 (reference)", W), (f"8-bit, {desc}", Wq)):
                sd = {k: v.clone() for k, v in ck["state_dict"].items()}
                sd["actor_lut.weights"] = torch.tensor(weights, dtype=torch.float32)
                ac = REGISTRY[ck["arch"]](ck["obs_dim"], ck["act_dim"],
                                          tables_per_head=ck["tables_per_head"])
                ac.load_state_dict(sd, strict=True)
                ac.eval()
                m, v = ck["obs_mean"], ck["obs_var"]

                rets, lens = [], []
                for ep in range(a.episodes):
                    obs, _ = env.reset(seed=1000 + ep)
                    tot, n, done = 0.0, 0, False
                    while not done:
                        x = (torch.from_numpy(np.asarray(obs, np.float32)) - m) \
                            / torch.sqrt(v + 1e-8)
                        with torch.no_grad():
                            mu = ac(iq(x.unsqueeze(0)))[0]
                            act = oq(mu)[0].numpy()
                        obs, r, term, trunc, _ = env.step(
                            np.clip(act, -1, 1).astype(np.float32))
                        tot += r
                        n += 1
                        done = term or trunc
                    rets.append(tot)
                    lens.append(n)
                rets, lens = np.array(rets), np.array(lens)
                print(f"{ARMS[arm]:<24} {seed:>4} {tag:<26} {rets.mean():>9.1f} "
                      f"{rets.std():>8.1f} {rets.min():>9.1f} "
                      f"{int((lens >= 1000).sum()):>7}/{a.episodes:<3}")
                report[f"{arm}_s{seed}_{'float' if weights is W else 'q8'}"] = dict(
                    mean=float(rets.mean()), std=float(rets.std()),
                    min=float(rets.min()), max=float(rets.max()),
                    full1000=int((lens >= 1000).sum()), grid=desc, step=step)
        print()

    json.dump(report, open(os.path.join(a.out, "quant_gym_eval.json"), "w"), indent=1)
    print("\n=== summary: cost of 8-bit weight quantisation, per arm ===")
    for arm in ARMS:
        f = [v["mean"] for k, v in report.items() if k.startswith(arm) and k.endswith("float")]
        q = [v["mean"] for k, v in report.items() if k.startswith(arm) and k.endswith("q8")]
        if f:
            print(f"  {ARMS[arm]:<24} float {np.mean(f):7.1f}   8-bit {np.mean(q):7.1f}   "
                  f"delta {np.mean(q) - np.mean(f):+7.1f}")


if __name__ == "__main__":
    main()
