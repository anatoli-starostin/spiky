"""Summarize the seed-swept PPO-vs-SAC benchmark: speed + quality tables + overlay chart."""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
B = os.path.join(HERE, "bench2")


def gpu_stats(tag):
    f = os.path.join(B, f"{tag}.gpu")
    if not os.path.exists(f):
        return float("nan"), float("nan")
    v = np.array([float(x) for x in open(f).read().split() if x.strip()])
    return (float(v.mean()), float(v.max())) if len(v) else (float("nan"), float("nan"))


def ret_series(hist):
    key = "ep_ret_mean" if hist and "ep_ret_mean" in hist[0] else "ep_ret"
    xs = np.array([h["env_steps"] for h in hist], float)
    ys = np.array([h[key] for h in hist], float)
    return xs, ys


def load(algo, seed):
    j = json.load(open(os.path.join(B, f"{algo}_s{seed}.json")))
    xs, ys = ret_series(j["history"])
    gm, gx = gpu_stats(f"{algo}_s{seed}")
    ok = ys[np.isfinite(ys)]
    return dict(algo=algo, seed=seed, wall=j["wall_s"], thr=j["throughput_env_per_s"],
                steps=j["total_env_steps"], final=float(ys[-1]), best=float(ok.max()) if len(ok) else float("nan"),
                gpu_mean=gm, gpu_max=gx, xs=xs, ys=ys)


runs = {a: [load(a, s) for s in (0, 1, 2)] for a in ("ppo", "sac")}

print("\n===== SPEED =====")
print(f"{'algo':4} {'wall(s)':>8} {'env-steps/s':>12} {'total env-steps':>16} {'GPU mean/max':>13}")
for a in ("ppo", "sac"):
    r = runs[a]
    print(f"{a:4} {np.mean([x['wall'] for x in r]):>8.0f} {np.mean([x['thr'] for x in r]):>12,.0f} "
          f"{np.mean([x['steps'] for x in r]):>16,.0f} "
          f"{np.nanmean([x['gpu_mean'] for x in r]):>5.0f}% /{np.nanmean([x['gpu_max'] for x in r]):>4.0f}%")

print("\n===== QUALITY (mean episodic return) =====")
print(f"{'algo':4} {'seed':>4} {'final':>9} {'best':>9}")
for a in ("ppo", "sac"):
    for x in runs[a]:
        print(f"{a:4} {x['seed']:>4} {x['final']:>9.1f} {x['best']:>9.1f}")
    f = [x['final'] for x in runs[a]]; b = [x['best'] for x in runs[a]]
    print(f"{a:4} {'agg':>4} {np.mean(f):>6.1f}±{np.std(f):<4.1f} {np.mean(b):>6.1f}±{np.std(b):<4.1f}")

# ---- overlay chart: return vs env-steps (left) and vs wall-clock proxy via steps/thr (right) ----
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
col = {"ppo": "#1f77b4", "sac": "#9467bd"}
for a in ("ppo", "sac"):
    for i, x in enumerate(runs[a]):
        lbl = f"{a.upper()} seed{x['seed']}"
        ax[0].plot(x["xs"] / 1e6, x["ys"], color=col[a], alpha=0.75, lw=1.3,
                   label=lbl if True else None)
        wall_x = x["xs"] / x["thr"]     # env-steps -> seconds (steady throughput)
        ax[1].plot(wall_x, x["ys"], color=col[a], alpha=0.75, lw=1.3, label=lbl)
ax[0].set_xlabel("env-steps (millions)"); ax[0].set_ylabel("mean episode return")
ax[0].set_title("Quality vs env-steps (sample-efficiency)"); ax[0].set_xscale("log")
ax[0].grid(alpha=.3, which="both"); ax[0].legend(fontsize=7, ncol=2)
ax[1].set_xlabel("wall-clock (s)"); ax[1].set_ylabel("mean episode return")
ax[1].set_title("Quality vs wall-clock (throughput)")
ax[1].grid(alpha=.3); ax[1].legend(fontsize=7, ncol=2)
fig.suptitle("PPO (384 upd ≈100M steps) vs SAC (10000 upd ≈82M steps), N=8192, physics-graph — 3 seeds each")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "ppo_vs_sac.png"), dpi=130)
print("\nsaved ppo_vs_sac.png")
