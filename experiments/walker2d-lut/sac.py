"""GPU-resident SAC for the batched Warp Walker2d env — pure PyTorch, no JAX.

Data collection reuses the physics-CUDA-graph-captured rollout (--graph); the replay
buffer, twin critics, target critics, squashed-Gaussian actor and entropy temperature
all live on the GPU. The swappable actor comes from models.REGISTRY (forward(obs)->
(mean,_)); SAC squashes it with tanh + a state-independent log_std, so MLP/LUT/LIF
actors drop in unchanged. Twin Q-critics are standard MLP Q(obs,act) (models.QCritic),
independent of actor architecture.

── Regime caveat: why the branch's 5273 is NOT the expected number here ──────────
The hyperparameters below match the origin/walker2d-lut exp_c01 SAC baseline (SB3 SAC,
MLP[256,256] ReLU, lr 3e-4, buffer 1e6, batch 256, gamma 0.99, tau 0.005,
train_freq/gradient_steps 1, learning_starts 10k, auto-entropy target -6.0). That
baseline reached 5273 +/- 34 on Walker2d-v5 — BUT it ran a SINGLE CPU env with a 1:1
gradient-step-to-env-step ratio (UTD=1) over 1M sequential, highly-correlated env-steps.
This trainer is a *different regime*: thousands of parallel envs (default N=8192) feeding
one shared GPU replay buffer, with a MODERATE UTD (~4 grad steps per vec-step — far fewer
gradient updates per env-step than 1:1). In that massively-parallel regime SAC's
single-env sample-efficiency edge does NOT carry over. A fair equal-data head-to-head on
this framework (~82-100M env-steps, 3 seeds) had on-policy PPO decisively ahead
(best ~4750 +/- 140, stable) while this batched SAC was lower and high-variance
(best ~3000 +/- 1430, some seeds collapsing late). So do NOT expect ~5273 out of the box
here. To move batched SAC toward that number you would retune for the parallel regime:
higher UTD, critic LayerNorm / larger Q-ensemble, lower lr, reward scaling, and more
careful entropy / learning-starts tuning. As configured, PPO (ppo.py) is the better
default for GPU-massively-parallel Walker2d; SAC's niche is the low-sample / expensive-env
regime the exp_c01 baseline lived in.

Short-smoke usage (NOT to convergence):
    python sac.py --arch mlp --envs 8192 --graph --updates 300
"""
import os, math, time, argparse, json, copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from warp_env import WarpWalker2dVecEnv
from models import REGISTRY, QCritic
from ppo import RunningNorm

LOG2 = math.log(2.0)


class SquashedActor(nn.Module):
    """Wraps a REGISTRY actor module. Uses its mean head + a state-independent log_std,
    tanh-squashed with the exact log-prob correction. Interface unchanged for the arch."""

    def __init__(self, ac_module, log_std_bounds=(-5.0, 2.0)):
        super().__init__()
        self.ac = ac_module
        self.lo, self.hi = log_std_bounds

    def _mean_logstd(self, obs):
        mean, _ = self.ac(obs)
        log_std = self.ac.log_std.clamp(self.lo, self.hi)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self._mean_logstd(obs)
        dist = torch.distributions.Normal(mean, log_std.exp())
        u = dist.rsample()
        a = torch.tanh(u)
        # log(1 - tanh(u)^2) via the numerically-stable form
        logp = dist.log_prob(u).sum(-1) - (2 * (LOG2 - u - F.softplus(-2 * u))).sum(-1)
        return a, logp

    @torch.no_grad()
    def act_stoch(self, obs):
        return self.sample(obs)[0]

    @torch.no_grad()
    def act_det(self, obs):
        mean, _ = self._mean_logstd(obs)
        return torch.tanh(mean)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp", choices=list(REGISTRY))
    ap.add_argument("--envs", type=int, default=8192)
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--collect", type=int, default=1, help="vec-steps collected per iteration")
    ap.add_argument("--utd", type=int, default=4, help="gradient steps per iteration")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--capacity", type=int, default=1_000_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--learning-starts", type=int, default=10, help="warmup vec-steps (random)")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", action="store_true", help="throughput-only, skip episode logging")
    ap.add_argument("--out", default="sac_smoke.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)

    env = WarpWalker2dVecEnv(num_envs=a.envs, seed=a.seed)
    if a.graph:
        env.build_physics_graph()
    O, A, N = env.obs_dim, env.act_dim, a.envs

    actor = SquashedActor(REGISTRY[a.arch](O, A)).to(dev)
    q1, q2 = QCritic(O, A).to(dev), QCritic(O, A).to(dev)
    q1t, q2t = copy.deepcopy(q1), copy.deepcopy(q2)
    for p in list(q1t.parameters()) + list(q2t.parameters()):
        p.requires_grad_(False)
    log_alpha = torch.zeros(1, device=dev, requires_grad=True)
    target_entropy = -float(A)                       # -6.0, the branch convention

    opt_a = torch.optim.Adam(actor.parameters(), lr=a.lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=a.lr)
    opt_al = torch.optim.Adam([log_alpha], lr=a.lr)
    if a.compile:
        q1, q2 = torch.compile(q1), torch.compile(q2)

    buf = __import__("buffers").GPUReplayBuffer(a.capacity, O, A, dev)
    norm = RunningNorm(O, dev)

    obs = env.reset(); norm.update(obs)
    ep_ret = torch.zeros(N, device=dev); ep_len = torch.zeros(N, device=dev)
    acc = dict(ret=torch.zeros((), device=dev), ln=torch.zeros((), device=dev),
               cnt=torch.zeros((), device=dev), mx=torch.zeros((), device=dev))
    nparams = sum(p.numel() for p in actor.parameters())
    print(f"SAC arch={a.arch} actor_params={nparams:,} q_params={sum(p.numel() for p in q1.parameters()):,} "
          f"envs={N} utd={a.utd} batch={a.batch} collect={a.collect} graph={a.graph} compile={a.compile}",
          flush=True)

    # warmup: random transitions so the buffer/critics have data
    for _ in range(a.learning_starts):
        act = torch.rand(N, A, device=dev) * 2 - 1
        nobs, r, term, trunc = env.step(act)
        buf.add_batch(obs, act, r, nobs, term.float())
        obs = nobs; norm.update(obs)

    hist = []; t0 = time.time(); total = 0

    def collect_step(obs):
        act = actor.act_stoch(norm.norm(obs))
        nobs, r, term, trunc = env.step(act)
        buf.add_batch(obs, act, r, nobs, term.float())
        return nobs, r, term, trunc

    def grad_step():
        o, act, r, no, d = buf.sample(a.batch)
        on, non = norm.norm(o), norm.norm(no)
        alpha = log_alpha.exp().detach()
        with torch.no_grad():
            na, logp_n = actor.sample(non)
            qt = torch.min(q1t(non, na), q2t(non, na)) - alpha * logp_n
            y = r + a.gamma * (1 - d) * qt
        ql = F.mse_loss(q1(on, act), y) + F.mse_loss(q2(on, act), y)
        opt_q.zero_grad(set_to_none=True); ql.backward(); opt_q.step()
        ap, logp = actor.sample(on)
        qpi = torch.min(q1(on, ap), q2(on, ap))
        al = (alpha * logp - qpi).mean()
        opt_a.zero_grad(set_to_none=True); al.backward(); opt_a.step()
        alpha_loss = -(log_alpha * (logp + target_entropy).detach()).mean()
        opt_al.zero_grad(set_to_none=True); alpha_loss.backward(); opt_al.step()
        with torch.no_grad():
            for p, pt in zip(q1.parameters(), q1t.parameters()):
                pt.mul_(1 - a.tau).add_(a.tau * p)
            for p, pt in zip(q2.parameters(), q2t.parameters()):
                pt.mul_(1 - a.tau).add_(a.tau * p)
        return float(ql.detach()), float(al.detach()), float(log_alpha.exp().detach())

    for upd in range(1, a.updates + 1):
        for _ in range(a.collect):
            obs, r, term, trunc = collect_step(obs)
            norm.update(obs)
            if not a.bench:
                ep_ret += r; ep_len += 1
                done = (term | trunc).float()
                acc["ret"] += (ep_ret * done).sum(); acc["ln"] += (ep_len * done).sum()
                acc["cnt"] += done.sum(); acc["mx"] = torch.maximum(acc["mx"], (ep_ret * done).max())
                keep = 1 - done; ep_ret = ep_ret * keep; ep_len = ep_len * keep
        total += a.collect * N
        for _ in range(a.utd):
            qlv, alv, alpha = grad_step()

        if upd % 20 == 0 or upd == 1:
            el = time.time() - t0; sps = total / el
            cnt = float(acc["cnt"])
            er = float(acc["ret"]) / cnt if cnt > 0 else float("nan")
            row = dict(update=upd, env_steps=total, sps=round(sps), ep_ret=er,
                       ep_ret_max=float(acc["mx"]), alpha=alpha, qloss=qlv, aloss=alv,
                       utd_ratio=round(a.utd * a.batch / (a.collect * N), 2))
            hist.append(row)
            for k in ("ret", "ln", "cnt"):
                acc[k].zero_()
            acc["mx"].zero_()
            print(f"[upd {upd:>4}/{a.updates}] ep_ret {er:8.1f} (max {row['ep_ret_max']:7.1f}) | "
                  f"{sps:>9,.0f} env-steps/s | alpha {alpha:.3f} qL {qlv:8.1f}", flush=True)

    el = time.time() - t0
    summary = dict(algo="sac", arch=a.arch, envs=N, utd=a.utd, batch=a.batch, collect=a.collect,
                   utd_ratio=round(a.utd * a.batch / (a.collect * N), 3),
                   total_env_steps=total, wall_s=round(el, 1),
                   throughput_env_per_s=round(total / el), params=nparams,
                   final_ep_ret=hist[-1]["ep_ret"], first_ep_ret=hist[0]["ep_ret"], history=hist)
    json.dump(summary, open(os.path.join(os.path.dirname(__file__), a.out), "w"), indent=1)
    print(f"\nSAC throughput {summary['throughput_env_per_s']:,} env-steps/s | "
          f"UTD {summary['utd_ratio']} | ep_ret {summary['first_ep_ret']:.0f} -> "
          f"{summary['final_ep_ret']:.0f} ({total:,} env-steps, {el:.0f}s)")


if __name__ == "__main__":
    main()
