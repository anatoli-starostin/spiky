"""exp_c29 — 100-episode deterministic CPU-reference eval of a constant-augmented actor.

exp_c09's eval_cpu.py cannot load these checkpoints: it standardises a 17-dim
observation and hands it straight to the LUT, whereas an exp_c29 actor's addressing
expects 17 + NC dimensions with the constant block appended. Rather than teach the
published evaluator a new shape, the loader is forked here -- exp_c09's file keeps
producing the numbers it has always produced.

The constants come from THE CHECKPOINT, never from constants.json. A policy scored
against thresholds it was not trained on is not a degraded version of itself, it is a
different function; and the failure would look like an ordinary bad result rather than a
mistake. The set name is carried too, and printed, so a mislabelled run is visible.

This is the only number this experiment quotes. The MJX return printed during training
is a 20-episode horizon-1000 proxy in perturbation-free MJX physics; it is useful for
watching a run and it is not comparable to anything in RESULTS.md.

Usage:
  python eval_const_cpu.py <actor.npz> [--episodes 100]
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
for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c07_robustness",
          "exp_c11_lut_sac_2x2", "exp_c26_action_quant"):
    sys.path.insert(0, os.path.join(D, p))

import jax_lut_ext as X                                    # noqa: E402
import perturb                                             # noqa: E402
import action_quant as AQ                                  # noqa: E402

ACT = 6
OBS = 17


def load_actor(path, forward_mode="hard"):
    """Returns (policy_fn, n_params, meta). policy_fn takes RAW [B, 17] observations."""
    z = np.load(path, allow_pickle=False)
    p = dict(w=jnp.asarray(z["w"]), b=jnp.asarray(z["b"]),
             weights=jnp.asarray(z["weights"]),
             log_T_soft=jnp.asarray(z["log_T_soft"]),
             log_T_sel=jnp.asarray(z["log_T_sel"]))
    heads, tph = int(z["n_heads"]), int(z["tph"])
    const = np.asarray(z["constants"], np.float32) if "constants" in z.files \
        else np.zeros(0, np.float32)
    cset = str(z["const_set"]) if "const_set" in z.files else "none"
    aobs = OBS + len(const)
    # The checkpoint records the input width it was trained at; w's own shape records
    # what the addressing will actually consume. If those two ever disagree the actor
    # would still run -- on a silently misaligned input -- so they are compared.
    if "obs_dim" in z.files and int(z["obs_dim"]) != aobs:
        raise ValueError(f"{os.path.basename(path)}: obs_dim {int(z['obs_dim'])} but "
                         f"{len(const)} constants imply {aobs}")
    if p["w"].shape[-1] != aobs:
        raise ValueError(f"{os.path.basename(path)}: addressing expects "
                         f"{p['w'].shape[-1]} inputs, constants imply {aobs}")

    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    cj = jnp.asarray(const)[None, :]

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        if len(const):
            x = jnp.concatenate(
                [x, jnp.broadcast_to(cj, (x.shape[0], len(const)))], axis=-1)
        y = X.apply(forward_mode)(x, p["w"], p["b"], p["weights"],
                                  p["log_T_soft"], p["log_T_sel"],
                                  heads, tph).sum(1)
        return jnp.tanh(y[:, :ACT])          # deterministic: mean only
    # Parameter count follows exp_c09's convention (table + addressing). The constants
    # are NOT counted: they are frozen buffers, like the anchor indices they are
    # compared against, and counting them would flatter the augmented arms by 16.
    n = int(np.prod(z["weights"].shape) + np.prod(z["w"].shape) + np.prod(z["b"].shape))
    meta = dict(const_set=cset, n_const=len(const), obs_dim=aobs,
                constants=const.tolist(), heads=heads, tph=tph)
    return (lambda obs: np.asarray(act(jnp.asarray(obs)))), n, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--forward-mode", default="hard",
                    choices=["hard", "hybrid_smooth"])
    a = ap.parse_args()
    path = a.actor if os.path.isabs(a.actor) else os.path.join(HERE, a.actor)
    fn, n, meta = load_actor(path, forward_mode=a.forward_mode)
    m = perturb.make_model(None, 1.0)

    t0 = time.time()
    r = AQ.rollout(m, fn, episodes=a.episodes)
    print(f"{os.path.basename(path)} ({n:,} params) [{a.forward_mode}] "
          f"constants={meta['const_set']} NC={meta['n_const']} "
          f"input={meta['obs_dim']}", flush=True)
    print(f"  CPU-reference {a.episodes}-ep deterministic: {r['mean']:.1f} +/- "
          f"{r['sd']:.1f}  full {r['full']}/{a.episodes}  vel {r['vel']:.3f} m/s  "
          f"len {r['len_mean']:.0f}  [{time.time() - t0:.0f}s]", flush=True)

    out = dict(actor=os.path.basename(path), params=n, episodes=a.episodes,
               forward_mode=a.forward_mode, cpu_reference_mean=r["mean"],
               cpu_reference_std=r["sd"], full_length=r["full"],
               velocity=r["vel"], length_mean=r["len_mean"], **meta)
    json.dump(out, open(path.replace("_actor.npz", "_cpueval.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
