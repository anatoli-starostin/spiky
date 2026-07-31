"""exp_c24 — score a sep-CMA-ES winner in the CPU reference env, 100 episodes (#75).

Reuses `rescore_c05.evaluate` verbatim rather than mirroring it again, so this number and
the exp_c05 re-scores in `rescore_c05_100ep.json` come from the *same* code path and are
directly comparable. The MJX fitness the run optimises is a proxy and is never quoted.

Usage:
  XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cpu python eval_sepcma.py <name>_mu.npy
"""
import argparse, json, os, sys

import jax.numpy as jnp
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
for p in ("exp_c02_mjx_scaffold", "exp_c04_jax_lut", "exp_c05_es"):
    sys.path.insert(0, os.path.join(BASE, p))

import mjx_walker2d as W          # noqa: E402
import es_mjx                     # noqa: E402
import rescore_c05                # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mu")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()

    st = json.load(open(os.path.join(BASE, "exp_c03_distillation", "dataset_stats.json")))
    norm = (jnp.asarray(st["obs_mean"], jnp.float32),
            jnp.asarray(st["obs_std"], jnp.float32))
    _, apply, _ = es_mjx.mlp_spec(a.hidden)
    flat = jnp.asarray(np.load(os.path.join(HERE, a.mu)))
    m = mujoco.MjModel.from_xml_path(W.XML)

    rets, lengths = rescore_c05.evaluate(apply, flat, norm, m, a.episodes)
    mean, sd = float(np.mean(rets)), float(np.std(rets))
    full = int(sum(1 for L in lengths if L >= 1000))
    print(f"{a.mu} CPU-reference {a.episodes}-ep deterministic: {mean:.1f} +/- {sd:.1f}"
          f"  | {full}/{a.episodes} full-length  | median len {np.median(lengths):.0f}"
          f"  | bar 3000 -> {'SOLVED' if mean >= 3000 else 'below'}", flush=True)
    print(f"  exp_c05 baselines at the same 100 episodes: MLP 'sepcma' 3022.3 +/- 889.9, "
          f"MLP OpenAI-ES 2058.1 +/- 143.2")
    json.dump(dict(mu=a.mu, episodes=a.episodes, cpu_reference_mean=mean,
                   cpu_reference_std=sd, full_length_episodes=full,
                   mean_length=float(np.mean(lengths)), returns=rets, lengths=lengths,
                   solved=bool(mean >= 3000)),
              open(os.path.join(HERE, a.mu.replace("_mu.npy", "_cpueval.json")), "w"),
              indent=1)


if __name__ == "__main__":
    main()
