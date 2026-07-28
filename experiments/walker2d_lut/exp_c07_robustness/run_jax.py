"""exp_c07 — JAX-side policies under perturbed dynamics (#75). WALKER2D_MJX venv.

  * PPO teacher      (71,948-param actor — the primary neural reference, 5555.5)
  * LUT-from-scratch (26,880 params, exp_c06 — PPO-trained, no teacher)

Both frozen; the LUT's stored obs standardiser is applied unchanged.
"""
import argparse, json, os, sys

import jax, jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c04_jax_lut"))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c06_jax_backprop"))

import perturb  # noqa: E402


def ppo_policy_fn(params_path):
    from cross_check import load_policy
    net, params = load_policy(params_path)

    @jax.jit
    def act(o):
        mean, _, _ = net.apply(params, o)
        return mean
    return (lambda obs: np.asarray(act(jnp.asarray(obs)))), "PPO actor 71,948 params"


def scratch_lut_policy_fn(npz_path):
    import jax_lut_grad as L
    z = np.load(npz_path)
    p = dict(w=jnp.asarray(z["w"]), b=jnp.asarray(z["b"]),
             weights=jnp.asarray(z["weights"]),
             log_T_soft=jnp.asarray(z["log_T_soft"]),
             log_T_sel=jnp.asarray(z["log_T_sel"]),
             n_heads=int(z["n_heads"]), tph=int(z["tph"]),
             obs_mean=jnp.asarray(z["obs_mean"]), obs_std=jnp.asarray(z["obs_std"]))
    f = jax.jit(lambda o: L.policy(p, o))
    n = int(np.prod(z["weights"].shape) + np.prod(z["w"].shape) + np.prod(z["b"].shape))
    return (lambda obs: np.asarray(f(jnp.asarray(obs)))), f"LUT-scratch {n:,} params"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()

    jobs = []
    ck = os.path.join(HERE, "..", "exp_c02_mjx_scaffold", "ppo_policy_full.msgpack")
    if os.path.exists(ck):
        fn, desc = ppo_policy_fn(ck)
        jobs.append(("PPO-MLP", fn, desc))
    else:
        print(f"MISSING {ck}", flush=True)
    ck = os.path.join(HERE, "..", "exp_c06_jax_backprop", "lut_scratch_params.npz")
    if os.path.exists(ck):
        fn, desc = scratch_lut_policy_fn(ck)
        jobs.append(("LUT-scratch", fn, desc))
    else:
        print(f"MISSING {ck}", flush=True)

    out = []
    for name, fn, desc in jobs:
        print(f"=== {name}: {desc} ===", flush=True)
        out += perturb.sweep(fn, name, episodes=a.episodes)
        json.dump(out, open(os.path.join(HERE, "results_jax.json"), "w"), indent=1)
    print("wrote results_jax.json", flush=True)


if __name__ == "__main__":
    main()
