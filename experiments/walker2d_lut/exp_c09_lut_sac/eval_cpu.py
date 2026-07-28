"""exp_c09 — CPU-reference eval of a LUT-SAC actor (#75).

Deterministic (tanh of the row's mean, ignoring the per-row sigma), 100 episodes,
gymnasium Walker2d-v5 on CPU MuJoCo — the only number comparable to the
4407 / 5512 / 5277 anchors. The MJX return printed during training is a horizon-1000
proxy in perturbation-free MJX physics and is NOT that number.
"""
import argparse, json, os, sys, time

import jax, jax.numpy as jnp
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c06_jax_backprop"))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c07_robustness"))

import mjx_walker2d as W          # noqa: E402
import jax_lut_grad as L         # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "exp_c11_lut_sac_2x2"))
import jax_lut_ext as X          # noqa: E402
import perturb                   # noqa: E402

ACT = 6


def mode_from_name(name):
    """Infer the training forward mode from the run tag in the filename."""
    return "hybrid_smooth" if "hybrid_smooth" in name else "hard"


def load_actor(path, forward_mode="hard"):
    """forward_mode MUST match the mode the actor was TRAINED in. Evaluating a
    smooth-trained table with the hard forward (or vice versa) is not a small
    approximation -- the distillation 2x2 measured that cross-mode swap at
    5520 -> 462. The checkpoint does not record the mode, so the caller must
    supply it;  infers it from the run tag as a convenience."""
    z = np.load(path)
    p = dict(w=jnp.asarray(z["w"]), b=jnp.asarray(z["b"]),
             weights=jnp.asarray(z["weights"]),
             log_T_soft=jnp.asarray(z["log_T_soft"]),
             log_T_sel=jnp.asarray(z["log_T_sel"]))
    heads, tph = int(z["n_heads"]), int(z["tph"])
    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        y = X.apply(forward_mode)(x, p["w"], p["b"], p["weights"],
                                  p["log_T_soft"], p["log_T_sel"],
                                  heads, tph).sum(1)
        return jnp.tanh(y[:, :ACT])          # deterministic: mean only
    n = int(np.prod(z["weights"].shape) + np.prod(z["w"].shape)
            + np.prod(z["b"].shape))
    return (lambda obs: np.asarray(act(jnp.asarray(obs)))), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--forward-mode", default=None,
                    choices=["hard", "hybrid_smooth"],
                    help="default: inferred from the checkpoint filename")
    ap.add_argument("--progress", metavar="LABEL", default=None,
                    help="stream a live timestamped progress line under LABEL")
    a = ap.parse_args()
    mode = a.forward_mode or mode_from_name(a.actor)
    fn, n = load_actor(os.path.join(HERE, a.actor), forward_mode=mode)
    m = perturb.make_model(None, 1.0)

    prog = None
    if a.progress:
        def prog(step, max_steps, done, eps, mean_so_far):
            ts = time.strftime("%H:%M:%S", time.gmtime())
            bar = int(24 * step / max_steps)
            print(f"    [{ts}] {a.progress:<12} [{'#' * bar}{'.' * (24 - bar)}] "
                  f"step {step:>4}/{max_steps}  fallen {done:>3}/{eps}  "
                  f"mean-so-far {mean_so_far:7.1f}", flush=True)

    mean, sd, _ = perturb.eval_batched(m, fn, episodes=a.episodes, progress=prog)
    print(f"{a.actor} ({n:,} params) [{mode}] | CPU-reference {a.episodes}-ep deterministic: "
          f"{mean:.1f} +/- {sd:.1f}  [anchors: PPO-scratch 4407 | SAC 5277 | "
          f"distill 5512]", flush=True)
    json.dump(dict(actor=a.actor, params=n, episodes=a.episodes, forward_mode=mode,
                   cpu_reference_mean=mean, cpu_reference_std=sd),
              open(os.path.join(HERE, a.actor.replace("_actor.npz", "_cpueval.json")),
                   "w"), indent=1)


if __name__ == "__main__":
    main()
