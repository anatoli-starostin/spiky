"""exp_c18 — dump each seed's INITIAL table weights (#75). MJX venv (needs jax).

The snapshots record (w, b) but not the table, because the table is 24,576 floats per
snapshot and the movement question only concerns addressing. The *init* table is still
needed to answer "how does seed 4's starting point differ", and it is exactly
reconstructible: it is a deterministic function of PRNGKey(seed) through the same call
chain lut_sac.py uses. Reconstructing rather than re-recording keeps the training runs
untouched; mirroring the call chain exactly (rather than re-deriving the draw) is what
makes the reconstruction trustworthy.
"""
import json, os, sys

import jax, jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(D, "exp_c06_jax_backprop"))
sys.path.insert(0, os.path.join(D, "exp_c09_lut_sac"))
import jax_lut_grad as L  # noqa: E402

OBS, ACT = 17, 6
SEEDS = (0, 1, 2, 3, 4, 5)
NAP, TPH, HEADS = 6, 32, 1
OUT = "/tmp/c18_table_init.npz"


def main():
    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)

    out = {}
    for s in SEEDS:
        # lut_sac.py: PRNGKey(seed) -> split 4 -> actor key -> actor_init -> L.init,
        # then the log-sigma half of every row is biased by -1/(heads*tph).
        key = jax.random.PRNGKey(s)
        _, ka, _, _ = jax.random.split(key, 4)
        p = L.init(ka, NAP, TPH, HEADS, OBS, 2 * ACT, om, osd, table_std=0.05)
        w0 = np.array(p["weights"])          # copy: np.asarray of a jax array is read-only
        w0[:, :, ACT:] += -1.0 / (HEADS * TPH)
        out[f"weights_{s}"] = w0
        print(f"seed {s}: table init {w0.shape} std {w0.std():.6f} "
              f"|W| {np.linalg.norm(w0):.4f}")
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
