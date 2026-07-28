"""exp_c07 — torch-side policies under perturbed dynamics (#75). SPIKY venv.

  * distilled LUT  (hyperplane nap4/tph32, 5,378 params — the Phase-1 winner)
  * SAC actor      (73,484 params — the second neural reference)

Both are frozen; the LUT's stored obs standardiser is applied unchanged.
"""
import argparse, json, os, sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c03_distillation"))

import perturb  # noqa: E402


def lut_policy_fn(ckpt, device="cuda"):
    from lut_policy import load
    m = load(ckpt, device=device)
    m.eval()

    @torch.no_grad()
    def f(obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device=device)
        return m(t).cpu().numpy()
    return f, m.describe()


def sac_policy_fn(zip_path, device="cuda"):
    from stable_baselines3 import SAC
    model = SAC.load(zip_path, device=device)

    def f(obs):
        act, _ = model.predict(obs, deterministic=True)
        return act
    n = sum(p.numel() for p in model.actor.parameters())
    return f, f"SAC actor {n:,} params"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--which", default="both", choices=["lut", "sac", "both"])
    a = ap.parse_args()

    jobs = []
    if a.which in ("lut", "both"):
        ck = os.path.join(HERE, "..", "exp_c03_distillation",
                          "lut_hyperplane_nap4_tph32_h1.pt")
        if os.path.exists(ck):
            fn, desc = lut_policy_fn(ck)
            jobs.append(("LUT-distilled", fn, desc))
        else:
            print(f"MISSING {ck}", flush=True)
    if a.which in ("sac", "both"):
        ck = os.path.join(HERE, "..", "exp_c01_sac_baseline", "run_seed0",
                          "sac_walker2d_final.zip")
        if os.path.exists(ck):
            fn, desc = sac_policy_fn(ck)
            jobs.append(("SAC-MLP", fn, desc))
        else:
            print(f"MISSING {ck}", flush=True)

    out = []
    for name, fn, desc in jobs:
        print(f"=== {name}: {desc} ===", flush=True)
        out += perturb.sweep(fn, name, episodes=a.episodes)
        json.dump(out, open(os.path.join(HERE, "results_torch.json"), "w"), indent=1)
    print("wrote results_torch.json", flush=True)


if __name__ == "__main__":
    main()
