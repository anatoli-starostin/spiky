"""exp_c31 — 100-episode deterministic CPU-reference eval of a PureLIF (TTFS) actor.

The only number this experiment quotes. The MJX return printed during training is a
20-episode horizon-1000 proxy in perturbation-free MJX physics; it is for watching a run
and is not comparable to anything in RESULTS.md.

Forked from exp_c30/eval_lif_cpu.py, which cannot load these checkpoints: it expects the
(d, w, r, tau_s_raw, P, tau_p_raw, theta) detector bank of LIFDetectorsMHL, whereas PureLIF
stores (delay, w, L, tau_raw, log_T_cross). Loading one with the other would KeyError rather
than silently mis-evaluate, but the shape assertions below are kept for the case that
matters more: a checkpoint whose recorded heads/tph/nap disagree with the tensors it
carries would still run, on a silently misaligned input.

`eps` is read from the checkpoint and passed through for interface parity ONLY -- PureLIF
ignores it entirely (run_parity.sh verifies 0.0 sensitivity). There is therefore no
train/eval regime to match here, which was the one delicate thing about exp_c30's evaluator.
mode="hard" always: the deployed regime, and byte-for-byte the same forward VALUE as the
mode="st" used in training.

Usage:
  python eval_pure_cpu.py <actor.npz> [--episodes 100]
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
import jax_pure_lif as LIF                                 # noqa: E402

ACT = 6
OBS = 17
PARAM_KEYS = ("delay", "w", "L", "tau_raw", "log_T_cross", "log_temp_bit", "table")


def load_actor(path):
    """Returns (policy_fn, n_params, meta). policy_fn takes RAW [B, 17] observations."""
    z = np.load(path, allow_pickle=False)
    p = {k: jnp.asarray(z[k]) for k in PARAM_KEYS}
    heads, tph, nap = int(z["n_heads"]), int(z["tph"]), int(z["nap"])
    eps = float(z["eval_eps"])

    m_expect = heads * tph * nap
    if p["delay"].shape != (m_expect, OBS):
        raise ValueError(f"{os.path.basename(path)}: detector bank is "
                         f"{p['delay'].shape} but heads={heads} tph={tph} nap={nap} "
                         f"imply {(m_expect, OBS)}")
    if p["L"].shape != (m_expect,):
        raise ValueError(f"{os.path.basename(path)}: deadline L is {p['L'].shape}, "
                         f"expected {(m_expect,)}")
    # The per-LUT block is the one PureLIF changed shape on (log_temp_bit became a vector).
    for k in ("tau_raw", "log_T_cross", "log_temp_bit"):
        if p[k].shape != (heads * tph,):
            raise ValueError(f"{os.path.basename(path)}: {k} is {p[k].shape}, expected "
                             f"{(heads * tph,)} — per-LUT, not per-detector, not scalar")
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
          f"nap{meta['nap']}/tph{meta['tph']}  (eps {meta['eval_eps']}, inert)",
          flush=True)
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
