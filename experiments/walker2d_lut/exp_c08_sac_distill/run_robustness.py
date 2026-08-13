"""exp_c08 — the SAC-distilled LUT through exp_c07's exact perturbation grid (#75).

Same harness, same axes, same points, same 100-episode deterministic protocol, same
frozen standardiser — so the resulting curve is directly comparable to the PPO-distilled
LUT and to both MLP teachers.
"""
import argparse, json, os, sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c07_robustness"))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c03_distillation"))

import perturb  # noqa: E402
from lut_policy import load  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(
        HERE, "lut_hyperplane_nap4_tph32_h1_sac.pt"))
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--name", default="LUT-SAC-distilled")
    # derive the output file from --name, so sweeping a second config cannot
    # silently clobber the first one's results
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_path = a.out or f"results_{a.name.lower().replace('-', '_')}.json"

    m = load(a.ckpt, device="cuda")
    m.eval()

    @torch.no_grad()
    def fn(obs):
        t = torch.as_tensor(obs, dtype=torch.float32, device="cuda")
        return m(t).cpu().numpy()

    print(f"=== {a.name}: {m.describe()} ===", flush=True)
    rows = perturb.sweep(fn, a.name, episodes=a.episodes)
    json.dump(rows, open(os.path.join(HERE, out_path), "w"), indent=1)
    print("wrote", out_path, flush=True)


if __name__ == "__main__":
    main()
