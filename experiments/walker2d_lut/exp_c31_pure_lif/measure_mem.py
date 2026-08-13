"""exp_c31 — peak GPU memory of each actor front-end, measured under ONE harness.

Answers the concrete question: does PureLIF cost more VRAM than the earlier LIF variants
and the hyperplane LUT actor, and do 3 seeds fit on the 5090 at once.

METHOD, and why it is shaped this way:

  * ONE PROCESS PER MODEL. `peak_bytes_in_use` is a high-water mark that never comes back
    down, so measuring two models in one process reports the max of the two for whichever
    ran second. The driver (`--all`) re-executes this file once per model.
  * `XLA_PYTHON_CLIENT_PREALLOCATE=false` is MANDATORY and set by the driver. With JAX's
    default preallocation the process grabs 75% of the card up front and every model
    "uses" ~24 GB, which is exactly the confusion this measurement exists to remove.
  * The measured region is the ACTOR LOSS BACKWARD at the training batch (512), which is
    what actually dominates: PureLIF materialises a (batch, 192, 17, 17) tensor and c30
    materialises two of them. The critic is a 256-wide MLP, identical across all four, and
    is included so the number is a whole-update figure rather than a front-end-only one.
  * The REPLAY BUFFER and the MJX env state are model-INDEPENDENT and are reported
    separately rather than folded in, because they do not scale with the front-end and
    would blur the comparison. The buffer is 1e6 x 42 float32 = 168 MB by arithmetic; it
    is allocated here too so the reported total is honest.

The measurement runs the real `jax.value_and_grad` of a SAC-shaped actor loss, not a bare
forward: the backward is where the (B,M,N,N) intermediates are held live, and a
forward-only number would understate every LIF variant by roughly half.

Usage:
  python measure_mem.py --all                 # driver: runs each model in a subprocess
  python measure_mem.py --model c31           # one model, in this process
"""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
OBS, ACT = 17, 6
NAP, TPH, HEADS, BATCH = 6, 32, 1, 512
MODELS = ("hyperplane", "c30", "c30b", "c31")


def _paths():
    for p in ("exp_c02_mjx_scaffold", "exp_c06_jax_backprop", "exp_c11_lut_sac_2x2",
              "exp_c30_lif_detectors", "exp_c30b_lif_pmatched"):
        sys.path.insert(0, os.path.join(D, p))
    sys.path.insert(0, HERE)


def build(model, key):
    """-> (params, actor_fn(params, x) -> [B, 12]). x is ALREADY standardised."""
    import jax.numpy as jnp
    _paths()

    if model == "hyperplane":
        import jax_lut_grad as L
        import jax_lut_ext as X
        om, osd = jnp.zeros(OBS), jnp.ones(OBS)
        p = L.init(key, NAP, TPH, HEADS, OBS, 2 * ACT, om, osd, table_std=0.05)
        p = {k: v for k, v in p.items()
             if k not in ("n_heads", "tph", "obs_mean", "obs_std")}
        ap = X.apply("hard")

        def fn(p, x):
            return ap(x, p["w"], p["b"], p["weights"], p["log_T_soft"],
                      p["log_T_sel"], HEADS, TPH).sum(1)
        return p, fn

    if model == "c30":
        import jax_lif_mhl as M
    elif model == "c30b":
        import jax_lif_lowrank as M
    elif model == "c31":
        import jax_pure_lif as M
    else:
        raise ValueError(model)

    p = M.init(key, NAP, TPH, HEADS, OBS, 2 * ACT)

    def fn(p, x):
        return M.apply(p, x, 0.7, HEADS, TPH, NAP, mode="st").sum(1)
    return p, fn


def measure(model):
    import jax
    import jax.numpy as jnp
    _paths()

    dev = jax.local_devices()[0]
    k = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(k, 3)
    p, fn = build(model, k1)
    n_par = sum(int(v.size) for v in jax.tree.leaves(p))

    # Critic, identical across all four front-ends.
    def q_init(kk, hidden=256):
        a, b, c = jax.random.split(kk, 3)
        return dict(w1=jax.random.normal(a, (OBS + ACT, hidden)) * 0.1,
                    b1=jnp.zeros(hidden),
                    w2=jax.random.normal(b, (hidden, hidden)) * 0.1,
                    b2=jnp.zeros(hidden),
                    w3=jax.random.normal(c, (hidden, 1)) * 0.01, b3=jnp.zeros(1))
    qp = q_init(k2)

    def q_apply(qp, s, a):
        h = jnp.concatenate([s, a], -1)
        h = jax.nn.relu(h @ qp["w1"] + qp["b1"])
        h = jax.nn.relu(h @ qp["w2"] + qp["b2"])
        return jnp.squeeze(h @ qp["w3"] + qp["b3"], -1)

    x = jax.random.normal(k3, (BATCH, OBS))

    # The replay buffer, allocated exactly as the trainers allocate it, so the reported
    # total is a real process footprint and not a front-end microbenchmark.
    N = 1_000_000
    buf = dict(s=jnp.zeros((N, OBS)), a=jnp.zeros((N, ACT)), r=jnp.zeros(N),
               s2=jnp.zeros((N, OBS)), d=jnp.zeros(N))
    buf_bytes = sum(int(v.size) * 4 for v in buf.values())
    jax.block_until_ready(buf["s"])
    base = dev.memory_stats().get("bytes_in_use", 0)

    @jax.jit
    def step(p, qp, x):
        def a_loss(p):
            y = fn(p, x)
            mu, log_std = y[:, :ACT], jnp.clip(y[:, ACT:], -5.0, 2.0)
            a = jnp.tanh(mu + jnp.exp(log_std))
            q = jnp.minimum(q_apply(qp["q1"], x, a), q_apply(qp["q2"], x, a))
            return (-q).mean()
        return jax.value_and_grad(a_loss)(p)

    qp2 = dict(q1=qp, q2=qp)
    for _ in range(3):                      # compile, then two live iterations
        v, g = step(p, qp2, x)
        jax.block_until_ready(g)

    ms = dev.memory_stats()
    peak = int(ms.get("peak_bytes_in_use", 0))
    return dict(model=model, params=n_par, peak_bytes=peak,
                buffer_bytes=buf_bytes, base_bytes=int(base),
                peak_gb=round(peak / 2**30, 3),
                activation_gb=round((peak - base) / 2**30, 3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    if a.model:
        print("RESULT " + json.dumps(measure(a.model)), flush=True)
        return

    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false",
               XLA_FLAGS="--xla_gpu_deterministic_ops=true",
               CUBLAS_WORKSPACE_CONFIG=":4096:8")
    out = []
    for mdl in MODELS:
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "--model", mdl],
                           capture_output=True, text=True, env=env, cwd=HERE)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith("RESULT ")]
        if not line:
            print(f"  {mdl:<12} FAILED\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
            out.append(dict(model=mdl, error=(r.stderr or r.stdout)[-400:]))
            continue
        d = json.loads(line[0][len("RESULT "):])
        out.append(d)
        print(f"  {d['model']:<12} params {d['params']:>8,}  peak "
              f"{d['peak_gb']:>6.3f} GB  (of which activations "
              f"{d['activation_gb']:.3f} GB, replay buffer "
              f"{d['buffer_bytes']/2**30:.3f} GB)", flush=True)
    json.dump(out, open(os.path.join(HERE, "memory_profile.json"), "w"), indent=1)
    print("\nwrote memory_profile.json")


if __name__ == "__main__":
    main()
