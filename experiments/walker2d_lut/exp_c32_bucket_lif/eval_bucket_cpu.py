"""exp_c32 — 100-episode deterministic CPU-reference eval of a Bucket-LIF actor.

The only number this experiment quotes. The MJX return printed during training is a
20-episode horizon-1000 proxy in perturbation-free MJX physics; it is for watching a run.

Forked from exp_c31/eval_pure_cpu.py, which cannot load these checkpoints: PureLIF stores
(delay, w, L, tau_raw, log_T_cross, log_temp_bit) with a per-DETECTOR bank of
n_tables*nap rows, whereas Bucket LIF stores (delay, w, tau_raw, log_T_cross, log_T_bkt,
beta_base, beta_raw) with one neuron PER TABLE. The shape assertions below matter for the
case that would otherwise pass silently: a checkpoint whose recorded n_buckets disagrees
with its beta_raw width would still run, addressing a table of the wrong height.

`eps` is read from the checkpoint and passed through for interface parity only -- the
module ignores it (run_parity.sh verifies 0.0 sensitivity). mode="hard" always.

Usage:
  python eval_bucket_cpu.py <actor.npz> [--episodes 100]
"""
import argparse
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
for p in ("exp_c02_mjx_scaffold", "exp_c07_robustness", "exp_c26_action_quant"):
    sys.path.insert(0, os.path.join(D, p))
sys.path.insert(0, HERE)

import perturb                                             # noqa: E402
import action_quant as AQ                                  # noqa: E402
import jax_bucket_lif as LIF                               # noqa: E402

ACT = 6
OBS = 17
PARAM_KEYS = ("delay", "w", "tau_raw", "log_T_cross", "log_T_bkt",
              "beta_base", "beta_raw", "table")


def load_actor(path):
    """Returns (policy_fn, n_params, meta). policy_fn takes RAW [B, 17] observations."""
    z = np.load(path, allow_pickle=False)
    p = {k: jnp.asarray(z[k]) for k in PARAM_KEYS}
    heads, tph = int(z["n_heads"]), int(z["tph"])
    nb, eps = int(z["n_buckets"]), float(z["eval_eps"])
    n_tables = heads * tph

    if p["delay"].shape != (n_tables, OBS):
        raise ValueError(f"{os.path.basename(path)}: delay is {p['delay'].shape} but "
                         f"heads={heads} tph={tph} imply {(n_tables, OBS)} — one LIF "
                         f"neuron PER TABLE, not per bit")
    if p["beta_raw"].shape != (n_tables, nb - 1):
        raise ValueError(f"{os.path.basename(path)}: beta_raw is {p['beta_raw'].shape}, "
                         f"expected {(n_tables, nb - 1)} for {nb} buckets")
    if p["table"].shape != (n_tables, nb, 2 * ACT):
        raise ValueError(f"{os.path.basename(path)}: table is {p['table'].shape}, "
                         f"expected {(n_tables, nb, 2 * ACT)}")
    for k in ("tau_raw", "log_T_cross", "log_T_bkt", "beta_base"):
        if p[k].shape != (n_tables,):
            raise ValueError(f"{os.path.basename(path)}: {k} is {p[k].shape}, expected "
                             f"{(n_tables,)} — per-LUT")

    # The invariant the softplus-cumsum parameterisation exists to guarantee. If a
    # checkpoint ever violated it the bucket index would be non-monotone in time and the
    # policy would be addressing nonsense, so it is asserted at load rather than assumed.
    b = np.asarray(LIF.boundaries(p))
    if not np.all(b[:, 1:] > b[:, :-1]):
        raise ValueError(f"{os.path.basename(path)}: bucket boundaries are not strictly "
                         f"increasing")

    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        y = LIF.apply(p, x, eps, heads, tph, nb, mode="hard").sum(1)
        return jnp.tanh(y[:, :ACT])          # deterministic: mean only

    n_det, n_tab = LIF.n_params(p)
    meta = dict(heads=heads, tph=tph, n_buckets=nb, eval_eps=eps,
                frontend_params=n_det, table_params=n_tab)
    return (lambda obs: np.asarray(act(jnp.asarray(obs)))), n_det + n_tab, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()
    path = a.actor if os.path.isabs(a.actor) else os.path.join(HERE, a.actor)
    fn, n, meta = load_actor(path)
    m = perturb.make_model(None, 1.0)

    t0 = time.time()
    r = AQ.rollout(m, fn, episodes=a.episodes)
    print(f"{os.path.basename(path)} ({n:,} params = {meta['frontend_params']:,} "
          f"front-end + {meta['table_params']:,} table) [hard] "
          f"{meta['n_buckets']} buckets / tph{meta['tph']}", flush=True)
    print(f"  CPU-reference {a.episodes}-ep deterministic: {r['mean']:.1f} +/- "
          f"{r['sd']:.1f}  full {r['full']}/{a.episodes}  vel {r['vel']:.3f} m/s  "
          f"len {r['len_mean']:.0f}  [{time.time() - t0:.0f}s]", flush=True)

    out = dict(actor=os.path.basename(path), params=n, episodes=a.episodes,
               forward_mode="hard", cpu_reference_mean=r["mean"],
               cpu_reference_std=r["sd"], full_length=r["full"],
               velocity=r["vel"], length_mean=r["len_mean"], **meta)
    json.dump(out, open(path.replace("_actor.npz", "_cpueval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
