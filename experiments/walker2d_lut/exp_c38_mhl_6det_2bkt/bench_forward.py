"""exp_c38 — where the MHL actor's time goes, at the shipped shape.

The smoke run came in around 1.0-1.2 s/iter against exp_c37's 0.23, so the detector axis
costs something and it is worth knowing WHAT before spending three seeds on it. Each stage
is timed jitted and block_until_ready'd, batch 512 (the training batch), 32 tables x 6
detectors x 2 buckets.

Usage:
  python bench_forward.py
"""
import json
import os
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import jax_mhl_lut as LIF                                   # noqa: E402

B, OBS, ACT = 512, 17, 12
HEADS, TPH, NDET, NB = 1, 32, 6, 2
REPS = 50


def bench(name, fn, *args):
    out = jax.block_until_ready(fn(*args))                  # compile
    t0 = time.time()
    for _ in range(REPS):
        out = fn(*args)
    jax.block_until_ready(out)
    ms = (time.time() - t0) / REPS * 1e3
    print(f"  {name:<34} {ms:8.3f} ms")
    return ms


def main():
    key = jax.random.PRNGKey(0)
    kp, kx = jax.random.split(key)
    p = LIF.init(kp, NB, NDET, TPH, HEADS, OBS, ACT, delay_init_std=4.0)
    x = jax.random.normal(kx, (B, OBS))
    gout = jax.random.normal(kx, (B, HEADS, ACT))
    print(f"batch {B}  {HEADS}x{TPH} tables  {NDET} det  {NB} bkt  "
          f"{NB**NDET} cells  device {jax.devices()[0]}")

    res = {}
    res["membrane"] = bench("membrane (sort + scan)",
                            jax.jit(lambda p, x: LIF.membrane(p, x)[1]), p, x)
    res["first_spike"] = bench("first_spike (+ soft crossing)",
                               jax.jit(lambda p, x: LIF.first_spike(p, x)), p, x)

    @jax.jit
    def bkt(p, x):
        th, ts = LIF.first_spike(p, x)
        return LIF.bucket(p, th, ts)
    res["to_bucket"] = bench("... + bucket", bkt, p, x)

    @jax.jit
    def hard_only(p, x):
        th, ts = LIF.first_spike(p, x)
        b, _ = LIF.bucket(p, th, ts)
        return LIF.hard_read(p, b, NDET, NB)
    res["hard_read"] = bench("... + hard_read (= mode=eval)", hard_only, p, x)

    @jax.jit
    def soft_only(p, x):
        th, ts = LIF.first_spike(p, x)
        _, g = LIF.bucket(p, th, ts)
        return LIF.soft_read(p, g, NDET, NB)
    res["soft_read"] = bench("... + soft_read (6 einsums)", soft_only, p, x)

    res["fwd_train"] = bench("full forward mode=train",
                             jax.jit(lambda p, x: LIF.apply(p, x, HEADS, TPH, NB, NDET)),
                             p, x)
    res["fwd_eval"] = bench("full forward mode=eval",
                            jax.jit(lambda p, x: LIF.apply(p, x, HEADS, TPH, NB, NDET,
                                                           mode="eval")), p, x)
    res["address"] = bench("address (coverage diagnostic)",
                           jax.jit(lambda p, x: LIF.address(p, x, NDET, NB)), p, x)

    def loss(p, x, gout):
        return (LIF.apply(p, x, HEADS, TPH, NB, NDET) * gout).sum()
    res["grad"] = bench("value_and_grad(train)",
                        jax.jit(jax.value_and_grad(loss)), p, x, gout)

    print(f"\n  one SAC update calls the st forward+grad twice (actor loss, and the "
          f"target's\n  actor_sample) plus `address` once for coverage, so the actor's "
          f"share of an update\n  is roughly 2 x grad + address = "
          f"{2 * res['grad'] + res['address']:.1f} ms; x32 updates = "
          f"{(2 * res['grad'] + res['address']) * 32 / 1e3:.2f} s/iter.")
    json.dump(res, open(os.path.join(HERE, "bench_forward.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
