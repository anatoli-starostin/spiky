"""exp_c29 — which added constants does the LUT actually key off? (#75)

const_lut_sac.py reports the STATIC half of this at startup: how many address bits are
wired to each constant. That is what the sampler drew, not what the policy uses. A
comparator wired to a threshold the walker never crosses is present in the architecture
and absent from the computation -- it emits the same bit in every state, costs a full
address bit, and halves that table's reachable rows.

So this measures the DYNAMIC half, on the states the trained policy actually visits:

  fire rate p   fraction of steps on which the bit is 1
  entropy H     binary entropy of p, in bits. H = 1 means the comparator splits the
                visited states evenly; H = 0 means it is stuck and carries nothing.
  dead          H < 0.01 bits, i.e. the bit is effectively constant in practice
  reachability  distinct rows each table actually addresses, out of 2^nap. This is the
                consequence that matters: it is where wasted comparators show up as
                lost capacity rather than as a diagnostic curiosity.

Reported per constant, per observation channel, and split obs-obs vs obs-const -- the
last of those is the one that answers whether the threshold tests earn their place
alongside the pairwise comparators they displaced.

Rolled out with the DETERMINISTIC policy (tanh of the row mean), 100 episodes, the same
CPU physics as the reference eval, so the state distribution here is the state
distribution the quoted return was earned on.

Writes <actor>_bitusage.json.

Usage:
  python bit_usage.py <actor.npz> [--episodes 100]
"""
import argparse
import json
import os
import sys

import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c26_action_quant", "exp_c28_input_quant"):
    sys.path.insert(0, os.path.join(D, p))
sys.path.insert(0, HERE)

import jax_lut_grad as L                                   # noqa: E402
import perturb                                             # noqa: E402
import input_quant as IQ                                   # noqa: E402
import eval_const_cpu as E                                 # noqa: E402
import const_lut_sac as C                                  # noqa: E402

OBS = 17
NAMES = C.NAMES


def entropy(p):
    """Binary entropy in bits, with the 0 log 0 limit taken as 0."""
    q = np.clip(p, 1e-12, 1 - 1e-12)
    h = -(q * np.log2(q) + (1 - q) * np.log2(1 - q))
    return np.where((p <= 0) | (p >= 1), 0.0, h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--dead-below", type=float, default=0.01,
                    help="entropy in bits under which a comparator counts as stuck")
    a = ap.parse_args()
    path = a.actor if os.path.isabs(a.actor) else os.path.join(HERE, a.actor)

    fn, n, meta = E.load_actor(path, forward_mode="hard")
    z = np.load(path)
    w, b = jnp.asarray(z["w"]), jnp.asarray(z["b"])
    nap = int(w.shape[1])
    T = int(w.shape[0])
    const = np.asarray(meta["constants"], np.float32)
    NC = len(const)
    have_pairs = "pair_a" in z.files
    if have_pairs:
        pa, pb = np.asarray(z["pair_a"]), np.asarray(z["pair_b"])
    else:
        raise SystemExit(f"{os.path.basename(path)} has no pair_a/pair_b; this "
                         f"checkpoint predates the wiring record and cannot be "
                         f"attributed to constants")

    # ---- the states the policy actually visits -----------------------------
    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = np.asarray(stats["obs_mean"], np.float32)
    osd = np.asarray(stats["obs_std"], np.float32)
    O = IQ.rollout_record_obs(perturb.make_model(None, 1.0), fn, a.episodes)
    Xs = (O - om) / (osd + 1e-6)
    if NC:
        Xs = np.concatenate([Xs, np.broadcast_to(const, (len(Xs), NC))], axis=1)

    proj = np.asarray(L._project(jnp.asarray(Xs, jnp.float32), w, b))   # [N, T, nap]
    bits = proj > 0
    p = bits.mean(0)                                                    # [T, nap]
    H = entropy(p)
    dead = H < a.dead_below

    rows = np.asarray(L._hard_index(jnp.asarray(proj), nap))            # [N, T]
    reach = np.array([len(np.unique(rows[:, t])) for t in range(T)])
    K = 2 ** nap

    a_is_c, b_is_c = pa >= OBS, pb >= OBS
    kind = np.where(a_is_c & b_is_c, 2, np.where(a_is_c ^ b_is_c, 1, 0))  # 0 oo,1 oc,2 cc

    print(f"{os.path.basename(path)} — constants={meta['const_set']} NC={NC}, "
          f"{len(Xs):,} visited states from {a.episodes} episodes", flush=True)
    print(f"  address bits {T * nap} in {T} tables x {nap}; "
          f"rows reached {reach.mean():.1f}/{K} mean, {reach.min()}-{reach.max()} range "
          f"({100 * reach.mean() / K:.1f}% of the table)", flush=True)

    out_kind = {}
    for k, nm in ((0, "obs-obs"), (1, "obs-const"), (2, "const-const")):
        sel = kind == k
        if not sel.any():
            continue
        out_kind[nm] = dict(n=int(sel.sum()), mean_H=float(H[sel].mean()),
                            median_H=float(np.median(H[sel])),
                            dead=int(dead[sel].sum()),
                            mean_fire=float(p[sel].mean()))
        print(f"  {nm:<11} {int(sel.sum()):>4} bits | mean H {H[sel].mean():.3f} | "
              f"median H {np.median(H[sel]):.3f} | dead {int(dead[sel].sum()):>3} "
              f"({100 * dead[sel].mean():.1f}%) | mean fire rate {p[sel].mean():.3f}",
              flush=True)

    per_const = []
    if NC:
        print(f"  {'const':<7}{'value':>8}{'bits':>6}{'meanH':>8}{'dead':>6}"
              f"{'fire':>7}", flush=True)
        for k in range(NC):
            sel = ((pa == OBS + k) | (pb == OBS + k))
            hh = H[sel]
            # Fire rate is reported ORIENTED as 1[x > c]: when the constant is the first
            # endpoint the encoded bit is 1[c > x], the complement, and averaging the two
            # orientations together would wash the threshold's meaning out.
            fr = np.where(pa[sel] == OBS + k, 1.0 - p[sel], p[sel])
            per_const.append(dict(index=k, value=float(const[k]), bits=int(sel.sum()),
                                  mean_H=float(hh.mean()) if sel.any() else 0.0,
                                  dead=int(dead[sel].sum()),
                                  mean_fire_above=float(fr.mean()) if sel.any() else 0.0))
            print(f"  c{k:<6}{const[k]:8.3f}{int(sel.sum()):6d}"
                  f"{hh.mean() if sel.any() else 0.0:8.3f}"
                  f"{int(dead[sel].sum()):6d}{fr.mean() if sel.any() else 0.0:7.3f}",
                  flush=True)

    per_obs = []
    for j in range(OBS):
        sel = (pa == j) | (pb == j)
        per_obs.append(dict(index=j, name=NAMES[j], bits=int(sel.sum()),
                            mean_H=float(H[sel].mean()) if sel.any() else 0.0,
                            dead=int(dead[sel].sum())))
    worst = sorted(per_obs, key=lambda d: d["mean_H"])[:4]
    print("  least informative observation channels: "
          + ", ".join(f"{d['name']} H={d['mean_H']:.3f}" for d in worst), flush=True)

    json.dump(dict(actor=os.path.basename(path), const_set=meta["const_set"],
                   n_const=NC, episodes=a.episodes, n_states=int(len(Xs)),
                   nap=nap, n_tables=T, rows_per_table=K,
                   rows_reached_mean=float(reach.mean()),
                   rows_reached_min=int(reach.min()),
                   rows_reached_max=int(reach.max()),
                   dead_below=a.dead_below,
                   by_kind=out_kind, per_const=per_const, per_obs=per_obs,
                   bit_entropy=H.tolist(), bit_fire=p.tolist()),
              open(path.replace("_actor.npz", "_bitusage.json"), "w"), indent=1)
    print(f"  wrote {os.path.basename(path).replace('_actor.npz', '_bitusage.json')}",
          flush=True)


if __name__ == "__main__":
    main()
