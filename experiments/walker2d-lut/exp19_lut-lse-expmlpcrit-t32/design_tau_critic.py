"""Choose tau_critic for exp19's exponential MLP-critic readout — on measurements, not taste.

The critic's final layer normally computes `value = sum_i w_i*h_i + b` over T = 256
penultimate units. exp19 replaces the sum with

    value = T * tau * log( (1/T) sum_i exp(w_i*h_i / tau) ) + b

which equals the plain sum as tau -> inf, and T*max_i(w_i*h_i) as tau -> 0. The Jensen gap
is ~ T * Var(u) / (2*tau), so tau trades "starts near plain-linear" against "the exponential
is actually live rather than inert".

This script measures, on REAL normalised Walker2d observations pushed through a real
initialised critic, the relative deviation of the exponential head from the plain linear
head across a tau grid, plus how peaked the pooling is (effective number of units carrying
the gradient, T = perfectly uniform/linear, 1 = winner-take-all).

Choice rule: the SMALLEST tau whose relative deviation from plain-linear is under 2% — i.e.
start as close to exp17's critic as makes no difference, while leaving tau as much room to
move downward (toward max) as possible, which is the effect exp19 is testing for.

Usage:  python design_tau_critic.py
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from models import REGISTRY, _mlp, _ortho                          # noqa: E402
from warp_env import WarpWalker2dVecEnv                            # noqa: E402
from ppo import RunningNorm                                        # noqa: E402

OBS, ACT, B = 17, 6, 8192


def main():
    torch.manual_seed(0)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Real observations, normalised the way ppo.py does it.
    env = WarpWalker2dVecEnv(num_envs=2048, seed=0)
    norm = RunningNorm(env.obs_dim, dev)
    obs = env.reset()
    norm.update(obs)
    for _ in range(60):                       # let the running norm settle on real data
        a = torch.rand(env.N, env.act_dim, device=dev) * 2 - 1
        obs, _, _, _ = env.step(a)
        norm.update(obs)
    x = norm.norm(obs)
    print(f"real normalised obs: {tuple(x.shape)}  mean {x.mean():+.4f}  std {x.std():.4f}")

    torch.manual_seed(0)
    vf = _ortho(_mlp([OBS, 256, 256, 1]), gain=1.0).to(dev)
    with torch.no_grad():
        h = vf[:-1](x)
        w = vf[-1].weight.view(1, -1)
        b = vf[-1].bias
        u = h * w
        plain = u.sum(-1) + b
        T = u.shape[-1]
        print(f"\nper-unit contributions u_i = w_i*h_i:  std {u.std():.5f}  "
              f"max|u| {u.abs().max():.5f}")
        print(f"plain linear value: mean {plain.mean():+.5f}  std {plain.std():.5f}")
        print(f"predicted Jensen gap T*Var(u)/(2*tau) = {T * u.var().item() / 2:.4f} / tau")

        # The raw deviation is dominated by the Jensen gap, which is very nearly a CONSTANT
        # offset -- and the critic's trainable bias `b` absorbs a constant within a few
        # updates. What actually matters is whether the SHAPE of the value function changes,
        # so the deviation below is measured after removing each head's own mean.
        print(f"\n{'tau':>8} {'value std':>11} {'raw dev':>9} {'shape dev':>10} "
              f"{'corr':>8} {'max softmax':>12} {'eff units':>10}")
        for tau in (0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 1.0, 4.0):
            z = torch.clamp(u / tau, min=-60.0, max=60.0)
            val = T * tau * (torch.logsumexp(z, dim=-1) - math.log(T)) + b
            raw = ((val - plain).abs().mean() / plain.abs().mean()).item()
            vc, pc = val - val.mean(), plain - plain.mean()
            shape = ((vc - pc).abs().mean() / pc.abs().mean()).item()
            corr = torch.corrcoef(torch.stack([vc, pc]))[0, 1].item()
            sm = torch.softmax(z, dim=-1)
            eff = (-(sm.clamp_min(1e-12).log() * sm).sum(-1)).exp().mean().item()
            print(f"{tau:>8.3f} {val.std():>11.5f} {raw:>8.1%} {shape:>9.1%} "
                  f"{corr:>8.4f} {sm.max():>12.4f} {eff:>10.1f}")
        print(f"\n(T = {T} units. eff units ~ {T} means the pooling is effectively LINEAR:")
        print(" the exponential is inert there, however large the raw offset looks.)")
        print(f"per-unit spread std(u) = {u.std():.5f} — the exponential only becomes live")
        print(" once tau approaches that scale.")


if __name__ == "__main__":
    main()
