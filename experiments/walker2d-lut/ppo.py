"""GPU-resident PPO for the batched Warp Walker2d env — pure PyTorch, no JAX.

Rollouts and updates stay on-device (purejaxrl-style, but torch). Rollout buffers are
preallocated GPU tensors; the env state never leaves the GPU. Architecture is swappable
via models.REGISTRY (default: mlp).

Short-smoke usage (NOT to convergence):
    python ppo.py --arch mlp --envs 4096 --rollout 32 --updates 150
"""
import os, time, argparse, json
import numpy as np
import torch
import torch.nn as nn

from warp_env import WarpWalker2dVecEnv
from models import REGISTRY


class RunningNorm:
    """GPU running mean/std for observation normalization (Welford)."""

    def __init__(self, dim, device, eps=1e-8):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = eps

    @torch.no_grad()
    def update(self, x):
        bmean = x.mean(0); bvar = x.var(0, unbiased=False); bn = x.shape[0]
        delta = bmean - self.mean; tot = self.count + bn
        self.mean += delta * bn / tot
        m_a = self.var * self.count; m_b = bvar * bn
        self.var = (m_a + m_b + delta ** 2 * self.count * bn / tot) / tot
        self.count = tot

    def norm(self, x):
        return (x - self.mean) / torch.sqrt(self.var + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="mlp", choices=list(REGISTRY))
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", type=float, default=0.0)
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--graph", action="store_true", help="CUDA-graph-capture the physics in the rollout")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="smoke_results.json")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)

    env = WarpWalker2dVecEnv(num_envs=a.envs, seed=a.seed)
    if a.graph:
        env.build_physics_graph()
    N, T = a.envs, a.rollout
    ac = REGISTRY[a.arch](env.obs_dim, env.act_dim).to(dev)
    # torch.compile the PPO UPDATE (evaluate path), not the rollout act() — composes with
    # the physics CUDA-graph (separate mechanisms: Warp graph for physics, inductor for update)
    if a.compile:
        ac.evaluate = torch.compile(ac.evaluate)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr)
    norm = RunningNorm(env.obs_dim, dev)
    nparams = sum(p.numel() for p in ac.parameters())
    print(f"arch={a.arch} params={nparams:,} envs={N} rollout={T} "
          f"steps/update={N*T:,} device={dev} compile={a.compile}", flush=True)

    # preallocated GPU rollout buffers
    b_obs = torch.zeros(T, N, env.obs_dim, device=dev)
    b_act = torch.zeros(T, N, env.act_dim, device=dev)
    b_logp = torch.zeros(T, N, device=dev)
    b_val = torch.zeros(T, N, device=dev)
    b_rew = torch.zeros(T, N, device=dev)
    b_mask = torch.zeros(T, N, device=dev)   # 1 - terminated

    obs = env.reset()
    norm.update(obs)
    ep_ret = torch.zeros(N, device=dev)      # running per-env episodic return (raw reward)
    ep_len = torch.zeros(N, device=dev)
    # sync-free episode accumulators (kept on GPU; read only at log time)
    acc = dict(ret_sum=torch.zeros((), device=dev), len_sum=torch.zeros((), device=dev),
               cnt=torch.zeros((), device=dev), ret_max=torch.zeros((), device=dev))
    hist = []
    t_start = time.time()
    total_env_steps = 0

    for upd in range(1, a.updates + 1):
        for t in range(T):
            nobs = norm.norm(obs)
            a_t, logp_t, val_t = ac.act(nobs)
            nx_obs, rew, term, trunc = env.step(a_t)
            b_obs[t] = nobs; b_act[t] = a_t; b_logp[t] = logp_t
            b_val[t] = val_t; b_rew[t] = rew; b_mask[t] = (~term).float()
            # episode bookkeeping — GPU-only, NO host sync (no .item()/.tolist()/.any())
            ep_ret += rew; ep_len += 1
            done = (term | trunc).float()
            acc["ret_sum"] += (ep_ret * done).sum()
            acc["len_sum"] += (ep_len * done).sum()
            acc["cnt"] += done.sum()
            acc["ret_max"] = torch.maximum(acc["ret_max"], (ep_ret * done).max())
            keep = 1.0 - done
            ep_ret = ep_ret * keep
            ep_len = ep_len * keep
            obs = nx_obs
            norm.update(obs)
        total_env_steps += N * T

        # bootstrap + GAE (on GPU)
        with torch.no_grad():
            _, last_val = ac(norm.norm(obs))
            adv = torch.zeros(T, N, device=dev)
            gae = torch.zeros(N, device=dev)
            for t in reversed(range(T)):
                nextval = last_val if t == T - 1 else b_val[t + 1]
                delta = b_rew[t] + a.gamma * nextval * b_mask[t] - b_val[t]
                gae = delta + a.gamma * a.gae * b_mask[t] * gae
                adv[t] = gae
            ret = adv + b_val
        # flatten
        f_obs = b_obs.reshape(T * N, -1); f_act = b_act.reshape(T * N, -1)
        f_logp = b_logp.reshape(-1); f_adv = adv.reshape(-1); f_ret = ret.reshape(-1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        mb = (T * N) // a.minibatches
        last_info = {}
        for _ in range(a.epochs):
            perm = torch.randperm(T * N, device=dev)
            for s in range(0, T * N, mb):
                idx = perm[s:s + mb]
                nlogp, ent, val = ac.evaluate(f_obs[idx], f_act[idx])
                ratio = (nlogp - f_logp[idx]).exp()
                a1 = ratio * f_adv[idx]
                a2 = torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * f_adv[idx]
                pi_loss = -torch.min(a1, a2).mean()
                v_loss = 0.5 * (val - f_ret[idx]).pow(2).mean()
                ent_loss = ent.mean()
                loss = pi_loss + a.vf * v_loss - a.ent * ent_loss
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), a.max_grad)
                opt.step()
                last_info = dict(pi=float(pi_loss.detach()), v=float(v_loss.detach()),
                                 ent=float(ent_loss.detach()))

        if upd % 10 == 0 or upd == 1:
            el = time.time() - t_start
            sps = total_env_steps / el
            cnt = float(acc["cnt"])
            ep_ret_mean = float(acc["ret_sum"]) / cnt if cnt > 0 else float("nan")
            ep_len_mean = float(acc["len_sum"]) / cnt if cnt > 0 else float("nan")
            row = dict(update=upd, env_steps=total_env_steps, sps=round(sps, 0),
                       ep_ret_mean=ep_ret_mean, ep_ret_max=float(acc["ret_max"]),
                       ep_len_mean=ep_len_mean, n_done=int(cnt),
                       step_rew=float(b_rew.mean()), **last_info)
            hist.append(row)
            # reset window accumulators so each log reports the interval, not cumulative
            for k in ("ret_sum", "len_sum", "cnt"):
                acc[k].zero_()
            acc["ret_max"].zero_()
            print(f"[upd {upd:>4}/{a.updates}] ep_ret {row['ep_ret_mean']:8.1f} "
                  f"(max {row['ep_ret_max']:7.1f}, len {row['ep_len_mean']:5.0f}) | "
                  f"{sps:>9,.0f} env-steps/s | pi {last_info['pi']:+.3f} v {last_info['v']:.2f}",
                  flush=True)

    el = time.time() - t_start
    summary = dict(arch=a.arch, envs=N, rollout=T, updates=a.updates,
                   total_env_steps=total_env_steps, wall_s=round(el, 1),
                   throughput_env_per_s=round(total_env_steps / el, 0),
                   params=nparams, final_ep_ret=hist[-1]["ep_ret_mean"],
                   first_ep_ret=hist[0]["ep_ret_mean"], history=hist)
    json.dump(summary, open(os.path.join(os.path.dirname(__file__), a.out), "w"), indent=1)
    print(f"\nthroughput {summary['throughput_env_per_s']:,.0f} env-steps/s | "
          f"ep_ret {summary['first_ep_ret']:.0f} -> {summary['final_ep_ret']:.0f} "
          f"over {a.updates} updates ({total_env_steps:,} env-steps, {el:.0f}s)")


if __name__ == "__main__":
    main()
