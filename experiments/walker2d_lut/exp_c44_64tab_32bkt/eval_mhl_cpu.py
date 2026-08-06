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

    # THE MONOTONICITY INVARIANT, and a correction to how this chapter has stated it.
    #
    # `boundaries = beta_base + cumsum(softplus(beta_raw))` is STRICTLY increasing in exact
    # arithmetic -- softplus is strictly positive, so the cumulative sum strictly grows and
    # there is no way for the optimiser to produce a crossed pair. Every README in this
    # line has asserted that, and this loader used to enforce `>` accordingly.
    #
    # In float32 it is not true, and exp_c43 is where it first showed. With 64 buckets the
    # boundaries span out to ~51, where the float32 spacing is ~3.8e-06; a trained
    # increment of softplus(beta_raw) ~ 4.9e-07 is then BELOW the representable step and
    # the cumsum returns the same value twice. Seed 1 has 5 such ties out of 1,984 gaps.
    # (Measured: it is cancellation, not softplus underflow -- the smallest softplus is
    # 4.9e-07, comfortably normal.) At the 4- and 16-bucket configs the boundaries never
    # grew far enough for this to bite, which is why it went unseen until now.
    #
    # A tie is BENIGN: two equal boundaries mean an empty bucket, i.e. that digit value is
    # never emitted. The addressing stays monotone and well defined. What would be a real
    # fault is a DECREASING pair, which would make the bucket index non-monotone in spike
    # time and the policy would be addressing nonsense. So the check enforces
    # non-decreasing and reports ties rather than rejecting them.
    b = np.asarray(LIF.boundaries(p))
    if b.shape[-1] > 1:
        gaps = np.diff(b, axis=-1)
        if (gaps < 0).any():
            raise ValueError(f"{base}: bucket boundaries DECREASE at "
                             f"{int((gaps < 0).sum())} of {gaps.size} gaps — the "
                             f"addressing is non-monotone in spike time")
        n_tied = int((gaps == 0).sum())
        if n_tied:
            print(f"  note: {n_tied} of {gaps.size} boundary gaps are exactly 0 "
                  f"(float32 cumsum resolution at boundary magnitudes up to "
                  f"{b.max():.1f}) — {n_tied} empty buckets, addressing still monotone",
                  flush=True)
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
