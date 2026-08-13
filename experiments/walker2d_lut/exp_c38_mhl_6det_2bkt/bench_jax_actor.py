"""exp_c38 — time the JAX port at the exact shipped config, for the head-to-head.

Runs in the MJX venv (jax, no torch). Prints the same three numbers as
`bench_torch_ref.py` at the same shapes, so the ratio is meaningful:

    eval forward       mode="eval"  — the hard path, no softmax, no temperatures
    train forward      mode="train" — the straight-through path
    train fwd+bwd      value_and_grad, what training pays twice per SAC update

Batch 512, CUDA, fp32, steady state with compile excluded. Sweeps SORT_FORM so the head-to-
head also shows what the arrival-ordering spelling is worth; pass a comma-separated list.

Usage (from run_headtohead.sh):
  python bench_jax_actor.py [reps] [forms]
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax_mhl_lut as LIF                                   # noqa: E402

HEADS, TPH, NDET, NB, OBS, NOUT = 1, 32, 6, 2, 17, 12
BATCH = 512


def timeit(fn, args, reps):
    out = jax.block_until_ready(fn(*args))
    t0 = time.time()
    for _ in range(reps):
        out = fn(*args)
    jax.block_until_ready(out)
    return (time.time() - t0) / reps * 1e3


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    forms = sys.argv[2].split(",") if len(sys.argv) > 2 else [LIF.SORT_FORM]
    key = jax.random.PRNGKey(0)
    kp, kx = jax.random.split(key)
    p = LIF.init(kp, NB, NDET, TPH, HEADS, OBS, NOUT, delay_init_std=4.0)
    x = jax.random.normal(kx, (BATCH, OBS))
    gout = jax.random.normal(kx, (BATCH, HEADS, NOUT))
    n_fe, n_tab = LIF.n_params(p)
    print(f"jax {jax.__version__}  {jax.devices()[0].device_kind}  batch={BATCH}  "
          f"{TPH} tables x {NDET} det x {NB} bkt = {NB**NDET} cells  "
          f"params {n_fe + n_tab:,}   XLA_FLAGS={os.environ.get('XLA_FLAGS', '(none)')}",
          flush=True)

    all_res = {}
    for form in forms:
        LIF.SORT_FORM = form
        f_eval = jax.jit(lambda p, x: LIF.apply(p, x, HEADS, TPH, NB, NDET, mode="eval"))
        f_train = jax.jit(lambda p, x: LIF.apply(p, x, HEADS, TPH, NB, NDET,
                                                 mode="train"))

        def loss(p, x, gout):
            return (LIF.apply(p, x, HEADS, TPH, NB, NDET, mode="train") * gout).sum()
        f_bwd = jax.jit(jax.value_and_grad(loss))

        res = dict(eval_fwd=timeit(f_eval, (p, x), reps),
                   train_fwd=timeit(f_train, (p, x), reps),
                   train_fwd_bwd=timeit(f_bwd, (p, x, gout), reps))
        for k in ("eval_fwd", "train_fwd", "train_fwd_bwd"):
            print(f"  jax  SORT_FORM={form:<9} {k:<16} {res[k]:9.3f} ms", flush=True)
        all_res[form] = res

    all_res["batch"] = BATCH
    all_res["xla_flags"] = os.environ.get("XLA_FLAGS", "")
    json.dump(all_res, open(os.path.join(HERE, "bench_jax_actor.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
