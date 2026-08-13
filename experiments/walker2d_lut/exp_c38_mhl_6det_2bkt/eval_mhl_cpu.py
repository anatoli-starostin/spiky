"""exp_c38 — 100-episode deterministic CPU-reference eval of an MHL-LIF actor.

THE ONLY NUMBER THIS EXPERIMENT QUOTES. The MJX return printed during training is a
20-episode horizon-1000 proxy in perturbation-free MJX physics; it is for watching a run,
not for comparing runs.

Forked from exp_c37/eval_bucket_cpu.py, which cannot load these checkpoints: the bucket
variants store (delay, w_raw, tau_raw, log_T_cross, log_T_bkt, beta_base, beta_raw) with
ONE LIF per table and per-table shapes, whereas LIFMultiHeadLUT carries a detector axis on
delay/w_raw/tau_raw/beta_base/beta_raw and a table of n_buckets**n_det cells. The shape
assertions below matter for the case that would otherwise pass silently: a checkpoint whose
recorded (n_det, n_buckets) disagree with its table height would still run, addressing a
table of the wrong shape through a mixed-radix index that happens to stay in range.

mode="eval" always -- the reference's module.eval() path, which parity_check asserts
equals the training (straight-through) forward value to 1.8e-07. Note the reference has no
mode kwarg at all any more: `forward(x)` branches on `self.training`, and the retired
st/hard/soft vocabulary of the c30/c31/c32b modules is gone. Our `mode=` argument names
that branch and takes only "train"/"eval".

Usage:
  python eval_mhl_cpu.py <actor.npz> [--episodes 100]
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
import jax_mhl_lut as LIF                                  # noqa: E402

ACT = 6
OBS = 17
PARAM_KEYS = ("delay", "w_raw", "tau_raw", "beta_base", "beta_raw",
              "log_T_cross", "log_T_bkt", "table")


def load_actor(path):
    """Returns (policy_fn, n_params, meta). policy_fn takes RAW [B, 17] observations."""
    z = np.load(path, allow_pickle=False)
    p = {k: jnp.asarray(z[k]) for k in PARAM_KEYS}
    heads, tph = int(z["n_heads"]), int(z["tph"])
    nb, nd = int(z["n_buckets"]), int(z["n_det"])
    frozen = bool(int(z["freeze_temperature"]))
    n_tables, cells = heads * tph, nb ** nd
    base = os.path.basename(path)

    for k in ("delay", "w_raw"):
        if p[k].shape != (n_tables, nd, OBS):
            raise ValueError(f"{base}: {k} is {p[k].shape} but heads={heads} tph={tph} "
                             f"n_det={nd} imply {(n_tables, nd, OBS)} — one LIF per "
                             f"(table, detector)")
    if p["tau_raw"].shape != (n_tables, nd):
        raise ValueError(f"{base}: tau_raw is {p['tau_raw'].shape}, expected "
                         f"{(n_tables, nd)}")
    if p["beta_base"].shape != (n_tables, nd, 1):
        raise ValueError(f"{base}: beta_base is {p['beta_base'].shape}, expected "
                         f"{(n_tables, nd, 1)} — per DETECTOR, not per table")
    if p["beta_raw"].shape != (n_tables, nd, nb - 1):
        raise ValueError(f"{base}: beta_raw is {p['beta_raw'].shape}, expected "
                         f"{(n_tables, nd, nb - 1)} for {nb} buckets")
    if p["table"].shape != (n_tables, cells, 2 * ACT):
        raise ValueError(f"{base}: table is {p['table'].shape}, expected "
                         f"{(n_tables, cells, 2 * ACT)} = n_buckets**n_det cells")
    for k in ("log_T_cross", "log_T_bkt"):
        if p[k].shape != (n_tables,):
            raise ValueError(f"{base}: {k} is {p[k].shape}, expected {(n_tables,)} — "
                             f"the temperatures are per TABLE, shared across detectors")

    # The invariant the softplus-cumsum parameterisation exists to guarantee. Vacuous at
    # n_buckets=2 (one boundary per detector), asserted anyway so a later config change
    # that breaks it fails at load rather than silently addressing nonsense.
    b = np.asarray(LIF.boundaries(p))
    if b.shape[-1] > 1 and not np.all(b[..., 1:] > b[..., :-1]):
        raise ValueError(f"{base}: bucket boundaries are not strictly increasing")
    # The freeze, verified from the checkpoint rather than from the config that claimed it.
    if frozen and (float(np.abs(np.asarray(p["log_T_cross"])).max()) != 0.0
                   or float(np.abs(np.asarray(p["log_T_bkt"])).max()) != 0.0):
        raise ValueError(f"{base}: freeze_temperature was set but the temperatures MOVED "
                         f"— the gradient mask did not hold")

    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        y = LIF.apply(p, x, heads, tph, nb, nd, mode="eval").sum(1)
        return jnp.tanh(y[:, :ACT])          # deterministic: mean only

    n_fe, n_tab = LIF.n_params(p)
    meta = dict(heads=heads, tph=tph, n_buckets=nb, n_det=nd, cells=cells,
                freeze_temperature=frozen, frontend_params=n_fe, table_params=n_tab)
    return (lambda obs: np.asarray(act(jnp.asarray(obs)))), n_fe + n_tab, meta


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
          f"front-end + {meta['table_params']:,} table) [eval] "
          f"{meta['n_det']} det x {meta['n_buckets']} bkt = {meta['cells']} cells / "
          f"tph{meta['tph']}", flush=True)
    print(f"  CPU-reference {a.episodes}-ep deterministic: {r['mean']:.1f} +/- "
          f"{r['sd']:.1f}  full {r['full']}/{a.episodes}  vel {r['vel']:.3f} m/s  "
          f"len {r['len_mean']:.0f}  [{time.time() - t0:.0f}s]", flush=True)

    out = dict(actor=os.path.basename(path), params=n, episodes=a.episodes,
               forward_mode="eval", cpu_reference_mean=r["mean"],
               cpu_reference_std=r["sd"], full_length=r["full"],
               velocity=r["vel"], length_mean=r["len_mean"], **meta)
    json.dump(out, open(path.replace("_actor.npz", "_cpueval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
