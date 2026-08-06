"""Where exactly do the baseline and fused trainers diverge?

The fused update block was verified bit-exact in exp_profiling, but that harness held the
replay-buffer `size` CONSTANT. In a real run `size` grows every iteration until the buffer
fills. This runs both inner loops side by side, in one process, from an identical state,
and reports the first stage at which they differ — so the cause is located rather than
guessed at.

Usage:
  python localise_divergence.py
"""
import json
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)
import jax_bucket_lif as LIF                               # noqa: E402

OBS, ACT, TPH, NB, BATCH, UPD = 17, 6, 64, 32, 512, 32
EPS = 0.3


def main():
    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)
    k0, ka, kd = jax.random.split(jax.random.PRNGKey(0), 3)
    p0 = LIF.init(ka, NB, TPH, 1, OBS, 2 * ACT)

    NBUF = 100000
    kb1, kb2 = jax.random.split(kd)
    buf = dict(s=jax.random.normal(kb1, (NBUF, OBS)),
               a=jnp.tanh(jax.random.normal(kb2, (NBUF, ACT))),
               r=jax.random.uniform(kb1, (NBUF,)),
               s2=jax.random.normal(kb2, (NBUF, OBS)),
               d=jnp.zeros(NBUF))

    def sample_loop(key, size):
        """The baseline's batch-index stream: size as a concrete Python int."""
        out = []
        for _ in range(UPD):
            key, kb = jax.random.split(key)
            out.append(jax.random.randint(kb, (BATCH,), 0, size))
        return key, jnp.stack(out)

    @jax.jit
    def sample_scan(key, size):
        """The fused version's stream: size as a traced argument inside lax.scan."""
        def one(key, _):
            key, kb = jax.random.split(key)
            return key, jax.random.randint(kb, (BATCH,), 0, size)
        return jax.lax.scan(one, key, None, length=UPD)

    print("=== batch-index stream: Python loop vs lax.scan, as `size` grows ===")
    print(f"  {'size':>8}{'identical?':>13}{'first mismatch':>17}")
    key = k0
    firstbad = None
    for it in range(8):
        size = 320 + 64 * it
        _, a_idx = sample_loop(key, size)
        _, b_idx = sample_scan(key, size)
        a_idx, b_idx = np.asarray(a_idx), np.asarray(b_idx)
        same = bool(np.array_equal(a_idx, b_idx))
        pos = "-" if same else str(int(np.argmax((a_idx != b_idx).any(-1))))
        if not same and firstbad is None:
            firstbad = (size, a_idx, b_idx)
        print(f"  {size:>8}{str(same):>13}{pos:>17}")

    if firstbad is None:
        print("\n  Batch-index streams AGREE for every size tested.")
    else:
        size, A, B = firstbad
        u = int(np.argmax((A != B).any(-1)))
        print(f"\n  DIVERGES at size={size}, update #{u} of {UPD}")
        print(f"    loop : {A[u][:8]}")
        print(f"    scan : {B[u][:8]}")
        print(f"    identical updates before it: {u}")

    # Is the *key* stream itself the same? If the keys agree but the draws differ, the
    # cause is randint; if the keys differ, it is the split sequence.
    ka_, _ = sample_loop(k0, 512)
    kb_, _ = sample_scan(k0, 512)
    print(f"\n  final key after {UPD} splits identical: "
          f"{bool(np.array_equal(np.asarray(ka_), np.asarray(kb_)))}")


if __name__ == "__main__":
    main()
