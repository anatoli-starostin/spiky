"""exp_c30 — 100-episode deterministic CPU-reference eval of a LIF-detector actor.

The only number this experiment quotes. The MJX return printed during training is a
20-episode horizon-1000 proxy in perturbation-free MJX physics; it is for watching a run
and is not comparable to anything in RESULTS.md.

Forked from exp_c29/eval_const_cpu.py, which cannot load these checkpoints: it rebuilds a
hyperplane LUT from (w, b, weights), whereas a LIF actor's addressing is a detector bank.
The published exp_c09 evaluator is left untouched so it keeps producing the numbers it has
always produced.

`eps` COMES FROM THE CHECKPOINT, never from a flag default. The gate sharpness is part of
the function: a policy scored at an eps it was not annealed to is not a degraded version
of itself, it is a different policy, and the failure would look like an ordinary bad
result rather than a mistake. mode="hard" always — that is the deployed regime, and it is
the same forward VALUE as the mode="st" used in training.

Usage:
  python eval_lif_cpu.py <actor.npz> [--episodes 100]
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
import jax_lif_lowrank as LIF                              # noqa: E402

ACT = 6
OBS = 17
PARAM_KEYS = ("d", "w", "r", "tau_s_raw", "Pu", "Pv", "Pb", "tau_p_raw", "theta",
              "log_temp_bit", "table")


def load_actor(path):
    """Returns (policy_fn, n_params, meta). policy_fn takes RAW [B, 17] observations."""
    z = np.load(path, allow_pickle=False)
    p = {k: jnp.asarray(z[k]) for k in PARAM_KEYS}
    heads, tph, nap = int(z["n_heads"]), int(z["tph"]), int(z["nap"])
    eps = float(z["eval_eps"])

    # The checkpoint records the shape it was trained at; the detector bank's own shape
    # records what will actually be consumed. If those disagree the actor would still
    # run -- on a silently misaligned input -- so they are compared.
    m_expect = heads * tph * nap
    if p["d"].shape != (m_expect, OBS):
        raise ValueError(f"{os.path.basename(path)}: detector bank is {p['d'].shape} "
                         f"but heads={heads} tph={tph} nap={nap} imply "
                         f"{(m_expect, OBS)}")
    if p["table"].shape[:2] != (heads * tph, 1 << nap):
        raise ValueError(f"{os.path.basename(path)}: table is {p['table'].shape} but "
                         f"the config implies {(heads * tph, 1 << nap)}")

    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        y = LIF.apply(p, x, eps, heads, tph, nap, mode="hard").sum(1)
        return jnp.tanh(y[:, :ACT])          # deterministic: mean only

    n_det, n_tab = LIF.n_params(p)
    meta = dict(heads=heads, tph=tph, nap=nap, eval_eps=eps,
                detector_params=n_det, table_params=n_tab)
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
    print(f"{os.path.basename(path)} ({n:,} params = {meta['detector_params']:,} "
          f"detectors + {meta['table_params']:,} table) [hard] "
          f"nap{meta['nap']}/tph{meta['tph']} eps={meta['eval_eps']}", flush=True)
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
