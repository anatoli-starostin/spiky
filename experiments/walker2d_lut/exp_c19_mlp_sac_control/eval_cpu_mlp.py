"""exp_c19 — CPU-reference eval of an MLP-SAC actor (#75).

A deliberate mirror of exp_c09/eval_cpu.py: the same perturbation-free CPU MuJoCo model,
the same perturb.eval_batched, the same 100 episodes from the same episode seeds, the same
deterministic action rule (tanh of mu, sigma ignored), and the same output JSON schema.
Only the network differs. If these two scripts ever disagree on anything but the actor,
the LUT-vs-MLP comparison stops being like-for-like -- which is why this is a mirror
rather than a generalisation with a flag.
"""
import argparse, json, os, sys, time

import jax, jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c07_robustness"))
import perturb                   # noqa: E402

ACT = 6


def load_actor(path):
    z = np.load(path)
    p = {k: jnp.asarray(z[k]) for k in ("w1", "b1", "w2", "b2", "w3", "b3")}
    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        h = jax.nn.relu(x @ p["w1"] + p["b1"])
        h = jax.nn.relu(h @ p["w2"] + p["b2"])
        y = h @ p["w3"] + p["b3"]
        return jnp.tanh(y[:, :ACT])          # deterministic: mean only
    n = int(sum(np.prod(z[k].shape) for k in ("w1", "b1", "w2", "b2", "w3", "b3")))
    return (lambda obs: np.asarray(act(jnp.asarray(obs)))), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("actor")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--progress", metavar="LABEL", default=None)
    a = ap.parse_args()
    fn, n = load_actor(os.path.join(HERE, a.actor))
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
    print(f"{a.actor} ({n:,} params) [mlp] | CPU-reference {a.episodes}-ep deterministic: "
          f"{mean:.1f} +/- {sd:.1f}  [anchors: PPO-scratch 4407 | SAC 5277 | "
          f"distill 5512]", flush=True)
    json.dump(dict(actor=a.actor, params=n, episodes=a.episodes, forward_mode="mlp",
                   cpu_reference_mean=mean, cpu_reference_std=sd),
              open(os.path.join(HERE, a.actor.replace("_actor.npz", "_cpueval.json")),
                   "w"), indent=1)


if __name__ == "__main__":
    main()
