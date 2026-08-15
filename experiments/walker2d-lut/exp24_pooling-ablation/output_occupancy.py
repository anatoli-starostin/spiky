"""22-bucket occupancy of the ACTION MEAN — the thing the 22 levels actually quantise.

`UniformActionQuantizer` snaps the policy's action mean mu to `linspace(-1, +1, 22)`,
step 2/21 = 0.0952 (src/act_quant.py). So the occupancy question that bears on the post is
about the OUTPUT distribution, not the weights.

Two views, because they answer different questions:

  ON-POLICY   each arm rolls its own trajectory. This is the deployment-relevant
              distribution: what values does this policy actually emit? The arms visit
              different states, so it is not a controlled comparison.
  SHARED-STATE both arms are evaluated on the SAME state set — the union of states visited
              by all six runs, subsampled evenly so neither arm's own distribution
              dominates. This isolates the pooling operator's effect on the outputs given
              identical inputs.

Reports per arm: bucket counts, normalised entropy H/log(22), near-empty buckets,
busiest bucket, mass on the +-1 rails, out-of-band fraction before clipping, and the
per-action-dimension breakdown.

Usage: python output_occupancy.py --dir <ckpt dir>
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
LEVELS, CLIP = 22, 1.0
STEP = 2.0 * CLIP / (LEVELS - 1)


def bucketise(mu):
    """mu -> bucket index on linspace(-CLIP, +CLIP, LEVELS), exactly as the quantizer does."""
    mu_c = np.clip(mu, -CLIP, CLIP)
    return np.clip(np.rint((mu_c + CLIP) / STEP).astype(int), 0, LEVELS - 1)


def occ_stats(mu, empty_frac=0.001):
    idx = bucketise(mu)
    counts = np.bincount(idx.reshape(-1), minlength=LEVELS)
    p = counts / counts.sum()
    nz = p[p > 0]
    H = float(-(nz * np.log(nz)).sum() / np.log(LEVELS))
    return dict(
        counts=[int(c) for c in counts],
        entropy=H,
        near_empty=int((counts < empty_frac * counts.sum()).sum()),
        occupied=int((counts > 0).sum()),
        busiest_frac=float(counts.max() / counts.sum()),
        rail_frac=float((counts[0] + counts[-1]) / counts.sum()),
        rail_lo_frac=float(counts[0] / counts.sum()),
        rail_hi_frac=float(counts[-1] / counts.sum()),
        oob_frac=float((np.abs(mu) > CLIP).mean()),
        mu_std=float(mu.std()), mu_absmax=float(np.abs(mu).max()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "figures"))
    ap.add_argument("--envs", type=int, default=512)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--record-every", type=int, default=4)
    ap.add_argument("--shared-per-run", type=int, default=40000,
                    help="states contributed by each run to the shared set")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, os.path.abspath(SRC))

    from warp_env import WarpWalker2dVecEnv
    from models import REGISTRY
    from obs_quant import GaussianCompandingQuantizer
    from act_quant import UniformActionQuantizer

    dev = torch.device("cuda")
    env = WarpWalker2dVecEnv(num_envs=a.envs, seed=0, obs_clip_vel=10.0,
                             solver_iters=100, ls_iters=50)
    iq = GaussianCompandingQuantizer(128, 1.0).to(dev)
    oq = UniformActionQuantizer(LEVELS, CLIP).to(dev)

    runs = {}
    for arm in ARMS:
        for p in sorted(glob.glob(os.path.join(a.dir, f"{arm}_s*.pt"))):
            seed = int(os.path.basename(p).split("_s")[1].split(".")[0])
            runs[(arm, seed)] = p

    def load(p):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        ac = REGISTRY[ck["arch"]](ck["obs_dim"], ck["act_dim"],
                                  tables_per_head=ck["tables_per_head"]).to(dev)
        ac.load_state_dict({k: v.to(dev) for k, v in ck["state_dict"].items()}, strict=True)
        ac.eval()
        return ac, ck["obs_mean"].to(dev).float(), ck["obs_var"].to(dev).float()

    # ---- ON-POLICY: each arm rolls its own trajectory -------------------------------
    print("=== ON-POLICY rollouts (each arm drives its own states) ===")
    on_mu, state_pool = {}, []
    for (arm, seed), p in sorted(runs.items()):
        ac, m, v = load(p)
        obs = env.reset()
        mus, states = [], []
        with torch.no_grad():
            for t in range(a.steps):
                n = iq((obs - m) / torch.sqrt(v + 1e-8))
                mu, _ = ac(n)
                if t % a.record_every == 0:
                    mus.append(mu.detach().cpu())
                    states.append(obs.detach().cpu())
                obs, _, _, _ = env.step(oq(mu).clamp(-1, 1))
        on_mu[(arm, seed)] = torch.cat(mus).numpy().astype(np.float64)
        S = torch.cat(states)
        sel = torch.randperm(S.shape[0])[: a.shared_per_run]
        state_pool.append(S[sel])
        print(f"  {ARMS[arm]:<24} seed {seed}: {on_mu[(arm,seed)].shape[0]:,} samples")

    # ---- SHARED-STATE: every arm on the same states ---------------------------------
    shared = torch.cat(state_pool).to(dev)
    print(f"\n=== SHARED-STATE evaluation on {shared.shape[0]:,} states "
          f"(union of all six runs, evenly sampled) ===")
    sh_mu = {}
    for (arm, seed), p in sorted(runs.items()):
        ac, m, v = load(p)
        outs = []
        with torch.no_grad():
            for i in range(0, shared.shape[0], 8192):
                x = shared[i:i + 8192]
                n = iq((x - m) / torch.sqrt(v + 1e-8))
                outs.append(ac(n)[0].detach().cpu())
        sh_mu[(arm, seed)] = torch.cat(outs).numpy().astype(np.float64)

    report = {}
    for label, store in (("on_policy", on_mu), ("shared_state", sh_mu)):
        print(f"\n{'='*100}\n{label.upper().replace('_',' ')} — 22-bucket occupancy of the "
              f"action mean\n{'='*100}")
        print(f"{'arm':<24} {'entropy':>8} {'near-empty':>11} {'busiest':>9} "
              f"{'rails':>8} {'oob':>8} {'std(mu)':>9}")
        pooled = {}
        for arm in ARMS:
            mus = [store[k] for k in store if k[0] == arm]
            if not mus:
                continue
            allmu = np.concatenate(mus)
            s = occ_stats(allmu)
            pooled[arm] = s
            report[f"{label}_{arm}"] = s
            print(f"{ARMS[arm]:<24} {s['entropy']:>8.4f} {s['near_empty']:>8d}/22 "
                  f"{s['busiest_frac']*100:>8.2f}% {s['rail_frac']*100:>7.2f}% "
                  f"{s['oob_frac']*100:>7.2f}% {s['mu_std']:>9.4f}")
        for arm, s in pooled.items():
            print(f"\n  {ARMS[arm]} counts per bucket (level -1.0 ... +1.0):")
            print("   " + " ".join(f"{c:,}" for c in s["counts"]))
            print(f"   rails: low {s['rail_lo_frac']*100:.2f}%  high "
                  f"{s['rail_hi_frac']*100:.2f}%   |mu|max {s['mu_absmax']:.3f}")
        # per action dim, shared-state only
        if label == "shared_state":
            print("\n  per action dimension (shared-state), rail mass and entropy:")
            print(f"   {'dim':>3}  {'A rails':>9} {'A H':>7}  {'B rails':>9} {'B H':>7}")
            for o in range(6):
                row = []
                for arm in ARMS:
                    mus = np.concatenate([store[k][:, o] for k in store if k[0] == arm])
                    row.append(occ_stats(mus))
                print(f"   {o:>3}  {row[0]['rail_frac']*100:>8.2f}% {row[0]['entropy']:>7.4f}"
                      f"  {row[1]['rail_frac']*100:>8.2f}% {row[1]['entropy']:>7.4f}")
                report[f"perdim_{o}"] = dict(A=row[0], B=row[1])

    json.dump(report, open(os.path.join(a.out, "output_occupancy.json"), "w"), indent=1)

    # ---- figure ---------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRID = "#1f2328", "#6b7280", "#e5e7eb"
    CA, CB = "#2f6feb", "#d1730a"
    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.5))
    for x in ax:
        x.set_facecolor("white")
        x.grid(True, color=GRID, lw=0.8, zorder=0)
        x.set_axisbelow(True)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            x.spines[s].set_color(GRID)
        x.tick_params(colors=MUTED, labelsize=9)

    levels = np.linspace(-CLIP, CLIP, LEVELS)
    for i, (label, store, title) in enumerate((
            ("on_policy", on_mu, "on-policy — each arm's own states"),
            ("shared_state", sh_mu, "shared states — identical inputs to both arms"))):
        w = 0.4
        for arm, col, off in (("A", CA, -w / 2), ("B", CB, +w / 2)):
            mus = [store[k] for k in store if k[0] == arm]
            if not mus:
                continue
            c = np.array(occ_stats(np.concatenate(mus))["counts"], float)
            ax[i].bar(np.arange(LEVELS) + off, c / c.sum() * 100, w, color=col,
                      label=ARMS[arm])
        ax[i].set_title(title, color=INK, fontsize=11)
        ax[i].set_xlabel("action level (bucket 0 = −1.0 … bucket 21 = +1.0)", color=MUTED)
        ax[i].set_xticks(np.arange(0, LEVELS, 3))
        if i == 0:
            ax[i].set_ylabel("% of emitted action components", color=MUTED)
        leg = ax[i].legend(frameon=False, fontsize=9)
        for t in leg.get_texts():
            t.set_color(INK)

    fig.suptitle("22-level ACTION-MEAN occupancy — what the output quantiser actually snaps",
                 color=INK, fontsize=12.5, y=1.02)
    fig.tight_layout()
    p = os.path.join(a.out, "output_occupancy.png")
    fig.savefig(p, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
