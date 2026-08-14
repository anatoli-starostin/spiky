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
    ap.add_argument("--tables-per-head", type=int, default=None,
                    help="override the LUT arch's tables_per_head (LUT arches only; None=arch default)")
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--rollout", type=int, default=32)
    ap.add_argument("--updates", type=int, default=150)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatches", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lr-schedule", default="constant", choices=["constant", "cosine"],
                    help="cosine anneals lr from --lr down to --lr-min over --updates")
    ap.add_argument("--lr-min", type=float, default=0.0,
                    help="cosine floor (eta_min); default 0 preserves prior behavior")
    ap.add_argument("--logstd-min", type=float, default=None,
                    help="floor on the state-independent log_std (e.g. log(0.15)=-1.897) to "
                         "prevent policy std/entropy collapse; default None = no floor")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent", "--ent-coef", dest="ent", type=float, default=0.0)
    ap.add_argument("--target-kl", type=float, default=0.02,
                    help="KL early-stop: break epochs when approx_kl > 1.5*target_kl; <=0 disables")
    ap.add_argument("--norm-returns", action="store_true",
                    help="normalize rewards by a running discounted-return std (SB3 VecNormalize style)")
    ap.add_argument("--vf", type=float, default=0.5)
    ap.add_argument("--max-grad", type=float, default=0.5)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--graph", action="store_true", help="CUDA-graph-capture the physics in the rollout")
    ap.add_argument("--seed", type=int, default=0)
    # --- deployment-parity knobs (all default to the historical behaviour) ---
    ap.add_argument("--obs-clip-vel", type=float, default=None,
                    help="clip |qvel| in the observation, matching gymnasium's Walker2d "
                         "(which clips at 10). Default None = no clipping, the original "
                         "behaviour. Set 10.0 to train on the same observation a "
                         "gymnasium deployment will produce.")
    ap.add_argument("--solver-iters", type=int, default=10,
                    help="MuJoCo solver iterations. 10 is this env's historical value and "
                         "is far weaker than the MuJoCo default (100) that a stock "
                         "gymnasium deployment uses.")
    ap.add_argument("--ls-iters", type=int, default=8,
                    help="MuJoCo line-search iterations; historical value 8, stock "
                         "MuJoCo default 50.")
    ap.add_argument("--out", default="smoke_results.json")
    ap.add_argument("--save-model", default=None,
                    help="path to save the trained policy (torch .pt) at the end of training: "
                         "full state_dict PLUS the observation-normalisation statistics, which "
                         "the policy is useless without. Resolved like --out. Default None = "
                         "save nothing, preserving prior behaviour exactly.")
    a = ap.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)

    env = WarpWalker2dVecEnv(num_envs=a.envs, seed=a.seed,
                             solver_iters=a.solver_iters, ls_iters=a.ls_iters,
                             obs_clip_vel=a.obs_clip_vel)
    if a.graph:
        env.build_physics_graph()
    N, T = a.envs, a.rollout
    ac_kw = {} if a.tables_per_head is None else {"tables_per_head": a.tables_per_head}
    ac = REGISTRY[a.arch](env.obs_dim, env.act_dim, **ac_kw).to(dev)
    # torch.compile the PPO UPDATE (evaluate path), not the rollout act() — composes with
    # the physics CUDA-graph (separate mechanisms: Warp graph for physics, inductor for update)
    if a.compile:
        ac.evaluate = torch.compile(ac.evaluate)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr)
    # LR schedule: cosine anneals lr -> 0 over the full --updates (one step() per PPO update),
    # to stabilize late training; 'constant' preserves the prior behavior exactly.
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.updates, eta_min=a.lr_min)
             if a.lr_schedule == "cosine" else None)
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
    b_term = torch.zeros(T, N, device=dev)      # terminated (unhealthy) — zeroes the value bootstrap
    b_done = torch.zeros(T, N, device=dev)      # terminated OR truncated — cuts the GAE trace
    b_trueval = torch.zeros(T, N, device=dev)   # V(true next state) for exact bootstrap at boundaries
    # return normalization: reward scaling by running std of the discounted return (SB3-style)
    ret_rms = RunningNorm(1, dev)
    disc_ret = torch.zeros(N, device=dev)
    total_epochs = 0                            # for avg-epochs-per-update (KL early-stop) reporting

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
            b_obs[t] = nobs; b_act[t] = a_t; b_logp[t] = logp_t; b_val[t] = val_t
            done_f = (term | trunc).float()
            b_term[t] = term.float(); b_done[t] = done_f
            # exact truncation bootstrap: value of the TRUE next state (pre-reset)
            with torch.no_grad():
                _, b_trueval[t] = ac(norm.norm(env.true_next_obs))
            # return normalization: scale reward by running std of the discounted return
            if a.norm_returns:
                disc_ret = a.gamma * disc_ret + rew
                ret_rms.update(disc_ret[:, None])
                b_rew[t] = rew / torch.sqrt(ret_rms.var[0] + 1e-8)
                disc_ret = disc_ret * (1.0 - done_f)
            else:
                b_rew[t] = rew
            # episode bookkeeping on RAW reward — GPU-only, NO host sync
            ep_ret += rew; ep_len += 1
            acc["ret_sum"] += (ep_ret * done_f).sum()
            acc["len_sum"] += (ep_len * done_f).sum()
            acc["cnt"] += done_f.sum()
            acc["ret_max"] = torch.maximum(acc["ret_max"], (ep_ret * done_f).max())
            keep = 1.0 - done_f
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
                next_v_normal = last_val if t == T - 1 else b_val[t + 1]
                # boundary (term OR trunc): bootstrap V(true next); else V(actual next obs)
                nextval = torch.where(b_done[t].bool(), b_trueval[t], next_v_normal)
                nonterminal = 1.0 - b_term[t]      # zero the bootstrap ONLY on true termination
                delta = b_rew[t] + a.gamma * nonterminal * nextval - b_val[t]
                trace = 1.0 - b_done[t]            # cut the GAE trace on term OR trunc
                gae = delta + a.gamma * a.gae * trace * gae
                adv[t] = gae
            ret = adv + b_val
        # flatten
        f_obs = b_obs.reshape(T * N, -1); f_act = b_act.reshape(T * N, -1)
        f_logp = b_logp.reshape(-1); f_adv = adv.reshape(-1); f_ret = ret.reshape(-1)
        f_adv = (f_adv - f_adv.mean()) / (f_adv.std() + 1e-8)

        mb = (T * N) // a.minibatches
        last_info = {}
        epochs_done = 0
        stop_early = False
        for ep in range(a.epochs):
            perm = torch.randperm(T * N, device=dev)
            for s in range(0, T * N, mb):
                idx = perm[s:s + mb]
                nlogp, ent, val = ac.evaluate(f_obs[idx], f_act[idx])
                logratio = nlogp - f_logp[idx]
                ratio = logratio.exp()
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
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - logratio).mean()   # Schulman k3, unbiased & >=0
                last_info = dict(pi=float(pi_loss.detach()), v=float(v_loss.detach()),
                                 ent=float(ent_loss.detach()), kl=float(approx_kl))
                if a.target_kl > 0 and last_info["kl"] > 1.5 * a.target_kl:
                    stop_early = True
                    break
            epochs_done = ep + 1
            if stop_early:
                break
        total_epochs += epochs_done

        if sched is not None:
            sched.step()                       # one cosine step per PPO update
        if a.logstd_min is not None:           # project log_std back up to the floor
            with torch.no_grad():
                ac.log_std.clamp_(min=a.logstd_min)

        if upd % 10 == 0 or upd == 1:
            el = time.time() - t_start
            sps = total_env_steps / el
            cnt = float(acc["cnt"])
            ep_ret_mean = float(acc["ret_sum"]) / cnt if cnt > 0 else float("nan")
            ep_len_mean = float(acc["len_sum"]) / cnt if cnt > 0 else float("nan")
            row = dict(update=upd, env_steps=total_env_steps, sps=round(sps, 0),
                       ep_ret_mean=ep_ret_mean, ep_ret_max=float(acc["ret_max"]),
                       ep_len_mean=ep_len_mean, n_done=int(cnt), lr=opt.param_groups[0]["lr"],
                       logstd=float(ac.log_std.mean()), epochs_done=epochs_done,
                       step_rew=float(b_rew.mean()), **last_info)
            # architectures may expose extra scalars to log (e.g. fastlut_exp's c and t)
            if hasattr(ac, "extra_log"):
                row.update(ac.extra_log())
            hist.append(row)
            # reset window accumulators so each log reports the interval, not cumulative
            for k in ("ret_sum", "len_sum", "cnt"):
                acc[k].zero_()
            acc["ret_max"].zero_()
            print(f"[upd {upd:>4}/{a.updates}] ep_ret {row['ep_ret_mean']:8.1f} "
                  f"(max {row['ep_ret_max']:7.1f}, len {row['ep_len_mean']:5.0f}) | "
                  f"{sps:>9,.0f} env-steps/s | lr {row['lr']:.1e} | "
                  f"kl {last_info['kl']:.4f} ep{epochs_done}/{a.epochs} | "
                  f"pi {last_info['pi']:+.3f} v {last_info['v']:.2f}", flush=True)

    el = time.time() - t_start
    summary = dict(arch=a.arch, tables_per_head=a.tables_per_head, envs=N, rollout=T,
                   updates=a.updates, lr_schedule=a.lr_schedule,
                   lr_min=a.lr_min, logstd_min=a.logstd_min, ent_coef=a.ent,
                   target_kl=a.target_kl, norm_returns=a.norm_returns,
                   avg_epochs_per_update=round(total_epochs / a.updates, 2),
                   total_env_steps=total_env_steps, wall_s=round(el, 1),
                   throughput_env_per_s=round(total_env_steps / el, 0),
                   params=nparams, final_ep_ret=hist[-1]["ep_ret_mean"],
                   first_ep_ret=hist[0]["ep_ret_mean"], history=hist)
    json.dump(summary, open(os.path.join(os.path.dirname(__file__), a.out), "w"), indent=1)
    if a.save_model:
        # The obs-normalisation statistics are saved alongside the weights deliberately:
        # the policy is trained on norm.norm(obs), so weights WITHOUT these stats are not a
        # usable model. Everything is moved to CPU so the file loads without a GPU.
        ckpt = dict(
            arch=a.arch, tables_per_head=a.tables_per_head,
            obs_dim=env.obs_dim, act_dim=env.act_dim, seed=a.seed,
            state_dict={k: v.detach().cpu() for k, v in ac.state_dict().items()},
            obs_mean=norm.mean.detach().cpu(), obs_var=norm.var.detach().cpu(),
            obs_count=float(norm.count),
            final_ep_ret=summary["final_ep_ret"], config=vars(a),
        )
        mp = os.path.join(os.path.dirname(__file__), a.save_model)
        os.makedirs(os.path.dirname(mp), exist_ok=True)
        torch.save(ckpt, mp)
        print(f"saved model -> {mp}", flush=True)
    print(f"\nthroughput {summary['throughput_env_per_s']:,.0f} env-steps/s | "
          f"ep_ret {summary['first_ep_ret']:.0f} -> {summary['final_ep_ret']:.0f} "
          f"over {a.updates} updates ({total_env_steps:,} env-steps, {el:.0f}s)")


if __name__ == "__main__":
    main()
