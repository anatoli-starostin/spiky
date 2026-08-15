"""Weight-distribution comparison for the pooling ablation, plus each arm's return.

Reads the checkpoints run_ablation.sh wrote, and reports for each arm:
  * the LUT weight distribution — mean, std, min/max, dynamic range, kurtosis, mass near
    zero — pooled over seeds and per seed,
  * where those weights sit relative to the OUTPUT quantisation step (2/(L-1) over the
    clipped band), which is the thing that decides whether a level is reachable,
  * the Stage-3 delay span the spiking build would need, since that is set directly by the
    spread of the table weights (the same calculation probe_raw_readout.py does),
  * a deterministic closed-loop return under the training quantisers.

Usage: python analyse.py --dir <runs dir> [--out figures/]
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
TAU_M_OUT = 10.0


def calib(tau_m, n_euler=2, dt=0.5):
    return 1.0 / np.log((1.0 + dt / tau_m) ** n_euler)


def kurtosis(x):
    z = (x - x.mean()) / x.std()
    return float((z ** 4).mean() - 3.0)                     # excess kurtosis


def weight_stats(W, levels, clip, tau=None):
    """W: (T, K, n_out) LUT tables. Returns a dict of distribution statistics."""
    w = W.reshape(-1)
    step = 2.0 * clip / (levels - 1)                        # output quantisation step
    per_dim_range = [float(W[:, :, o].max() - W[:, :, o].min()) for o in range(W.shape[2])]
    # Stage-3 delay span: delays are rint(-scale*Wd + C), scale = tau_eff/tau_actor.
    spans = None
    if tau is not None and tau > 0:
        scale = calib(TAU_M_OUT) / tau
        spans = [int(np.ptp(np.rint(-scale * W[:, :, o]
                                    + np.ceil(scale * W[:, :, o].max() + 2))))
                 for o in range(W.shape[2])]
    return dict(
        n=int(w.size), mean=float(w.mean()), std=float(w.std()),
        min=float(w.min()), max=float(w.max()),
        absmax=float(np.abs(w).max()), range=float(w.max() - w.min()),
        kurtosis=kurtosis(w),
        frac_below_1e3=float((np.abs(w) < 1e-3).mean()),
        frac_below_1e2=float((np.abs(w) < 1e-2).mean()),
        p99_abs=float(np.percentile(np.abs(w), 99)),
        per_dim_range=per_dim_range,
        range_in_quant_steps=float((w.max() - w.min()) / step),
        # a single table's contribution, in units of the output step
        std_in_quant_steps=float(w.std() / step),
        summed_std_in_quant_steps=float(w.std() * np.sqrt(W.shape[0]) / step),
        delay_span_per_dim=spans,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    ap.add_argument("--levels", type=int, default=22)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--eval-envs", type=int, default=512)
    ap.add_argument("--eval-steps", type=int, default=2000)
    ap.add_argument("--no-eval", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.abspath(SRC))

    data = {}
    for arm in ARMS:
        for p in sorted(glob.glob(os.path.join(a.dir, f"{arm}_s*.pt"))):
            seed = int(os.path.basename(p).split("_s")[1].split(".")[0])
            ck = torch.load(p, map_location="cpu", weights_only=False)
            W = ck["state_dict"]["actor_lut.weights"].numpy().astype(np.float64)
            tau = None
            if "actor_lut.exp_outputs_tau_raw" in ck["state_dict"]:
                raw = float(ck["state_dict"]["actor_lut.exp_outputs_tau_raw"])
                tau = float(np.log1p(np.exp(-abs(raw))) + max(raw, 0.0))   # softplus
            data[(arm, seed)] = dict(ck=ck, W=W, tau=tau,
                                     final=ck["final_ep_ret"])

    print("=" * 96)
    print(f"{'arm':<26} {'seed':>4} {'final':>9} {'tau':>8} {'std':>10} {'absmax':>9} "
          f"{'range':>9} {'kurt':>8} {'|w|<1e-3':>9}")
    print("-" * 96)
    rows = {}
    for (arm, seed), d in sorted(data.items()):
        s = weight_stats(d["W"], a.levels, a.clip, d["tau"])
        rows[(arm, seed)] = s
        print(f"{ARMS[arm]:<26} {seed:>4} {d['final']:>9.1f} "
              f"{('%.5f' % d['tau']) if d['tau'] else '     —':>8} "
              f"{s['std']:>10.5f} {s['absmax']:>9.5f} {s['range']:>9.5f} "
              f"{s['kurtosis']:>8.2f} {s['frac_below_1e3']*100:>8.2f}%")

    # pooled per arm
    print("\n" + "=" * 96)
    pooled = {}
    for arm in ARMS:
        Ws = [d["W"] for (ar, _), d in data.items() if ar == arm]
        if not Ws:
            continue
        W = np.concatenate([w.reshape(-1) for w in Ws]).reshape(-1, 1, 1)
        s = weight_stats(np.concatenate(Ws, axis=0), a.levels, a.clip, None)
        pooled[arm] = s
        step = 2.0 * a.clip / (a.levels - 1)
        print(f"{ARMS[arm]} — pooled over {len(Ws)} seeds, {s['n']:,} weights")
        print(f"    mean {s['mean']:+.6f}   std {s['std']:.6f}   "
              f"min {s['min']:+.5f}   max {s['max']:+.5f}")
        print(f"    dynamic range {s['range']:.5f}   |w|max {s['absmax']:.5f}   "
              f"p99|w| {s['p99_abs']:.5f}   excess kurtosis {s['kurtosis']:+.2f}")
        print(f"    mass near zero: |w|<1e-3 {s['frac_below_1e3']*100:.2f}%   "
              f"|w|<1e-2 {s['frac_below_1e2']*100:.2f}%")
        print(f"    output quant step {step:.5f} ({a.levels} levels over ±{a.clip})")
        print(f"      one table's std = {s['std_in_quant_steps']:.3f} steps;  "
              f"32 tables summed ≈ {s['summed_std_in_quant_steps']:.2f} steps")
        print(f"      full weight range = {s['range_in_quant_steps']:.2f} steps")
        print()

    json.dump({f"{k[0]}_s{k[1]}": v for k, v in rows.items()}
              | {f"pooled_{k}": v for k, v in pooled.items()},
              open(os.path.join(a.out, "weight_stats.json"), "w"), indent=1)

    # ---- figure ---------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
    CA, CB = "#2f6feb", "#d1730a"
    step = 2.0 * a.clip / (a.levels - 1)

    fig, ax = plt.subplots(1, 3, figsize=(14.5, 4.3))
    for x in ax:
        x.set_facecolor("white")
        x.grid(True, color=GRID, lw=0.8, zorder=0)
        x.set_axisbelow(True)
        for side in ("top", "right"):
            x.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            x.spines[side].set_color(GRID)
        x.tick_params(colors=MUTED, labelsize=9)

    WA = np.concatenate([d["W"].reshape(-1) for (ar, _), d in data.items() if ar == "A"])
    WB = np.concatenate([d["W"].reshape(-1) for (ar, _), d in data.items() if ar == "B"])
    lim = max(np.abs(WA).max(), np.abs(WB).max()) * 1.02
    bins = np.linspace(-lim, lim, 121)
    ax[0].hist(WA, bins=bins, histtype="step", lw=2, color=CA, label=ARMS["A"], density=True)
    ax[0].hist(WB, bins=bins, histtype="step", lw=2, color=CB, label=ARMS["B"], density=True)
    ax[0].set_yscale("log")
    ax[0].set_title("LUT weight distribution", color=INK, fontsize=11)
    ax[0].set_xlabel("weight", color=MUTED)
    ax[0].set_ylabel("density (log)", color=MUTED)
    leg = ax[0].legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)

    # same, normalised to each arm's own std — shape rather than scale
    ax[1].hist(WA / WA.std(), bins=np.linspace(-6, 6, 121), histtype="step", lw=2,
               color=CA, density=True)
    ax[1].hist(WB / WB.std(), bins=np.linspace(-6, 6, 121), histtype="step", lw=2,
               color=CB, density=True)
    ax[1].set_yscale("log")
    ax[1].set_title("same, each normalised by its own std — shape only", color=INK, fontsize=11)
    ax[1].set_xlabel("weight / std", color=MUTED)

    # per-seed spread, in output-quantisation steps
    xs, ha, hb = [], [], []
    for (arm, seed), s in sorted(rows.items()):
        if arm == "A":
            xs.append(seed); ha.append(s["std"] / step)
        else:
            hb.append(s["std"] / step)
    w = 0.35
    ax[2].bar(np.array(xs) - w / 2, ha, w, color=CA, label=ARMS["A"])
    ax[2].bar(np.array(xs) + w / 2, hb, w, color=CB, label=ARMS["B"])
    ax[2].set_title("weight std, in output-quantisation steps", color=INK, fontsize=11)
    ax[2].set_xlabel("seed", color=MUTED)
    ax[2].set_xticks(xs)
    leg = ax[2].legend(frameon=False, fontsize=9)
    for t in leg.get_texts():
        t.set_color(INK)

    fig.suptitle("Pooling ablation — does log-sum-exp change where the weights end up?",
                 color=INK, fontsize=12.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(a.out, "weight_distribution.png")
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"wrote {p}")

    # ---- deterministic closed-loop return -------------------------------------------
    if a.no_eval:
        return
    from warp_env import WarpWalker2dVecEnv
    from models import REGISTRY
    from obs_quant import GaussianCompandingQuantizer
    from act_quant import UniformActionQuantizer
    dev = torch.device("cuda")
    env = WarpWalker2dVecEnv(num_envs=a.eval_envs, seed=0, obs_clip_vel=10.0,
                             solver_iters=100, ls_iters=50)
    iq = GaussianCompandingQuantizer(128, 1.0).to(dev)
    oq = UniformActionQuantizer(a.levels, a.clip).to(dev)
    print(f"\ndeterministic eval, {a.eval_envs} envs x {a.eval_steps} steps, "
          f"input+output quantisers ON:")
    for (arm, seed), d in sorted(data.items()):
        ck = d["ck"]
        ac = REGISTRY[ck["arch"]](ck["obs_dim"], ck["act_dim"],
                                  tables_per_head=ck["tables_per_head"]).to(dev)
        ac.load_state_dict({k: v.to(dev) for k, v in ck["state_dict"].items()}, strict=True)
        ac.eval()
        mean, var = ck["obs_mean"].to(dev).float(), ck["obs_var"].to(dev).float()
        obs = env.reset()
        ep = torch.zeros(env.N, device=dev)
        done_rets = []
        with torch.no_grad():
            for _ in range(a.eval_steps):
                n = iq((obs - mean) / torch.sqrt(var + 1e-8))
                act, _ = ac(n)
                obs, r, term, trunc = env.step(oq(act).clamp(-1, 1))
                ep += r
                dd = term | trunc
                if dd.any():
                    done_rets.append(ep[dd].clone())
                    ep = ep * (~dd).float()
        r = torch.cat(done_rets).cpu().numpy() if done_rets else np.zeros(0)
        print(f"  {ARMS[arm]:<26} seed {seed}: n={len(r):5d}  "
              f"mean {r.mean():7.1f} ± {r.std():6.1f}   median {np.median(r):7.1f}")


if __name__ == "__main__":
    main()
