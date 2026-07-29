"""exp_c14 — do the learned hyperplanes actually MOVE in hard mode? (#75). MJX venv.

Two independent checks, because they can disagree and the disagreement is informative:

  1. INIT-vs-FINAL. The init is deterministic given the seed, so it can be reconstructed
     exactly and compared with the trained checkpoint. Tensor drift is the weak measure;
     the one that matters is HOW MANY SIGN BITS FLIP on real observations, because that is
     what addressing actually is. w can drift a few percent and address identically.

  2. GRADIENT FLOW. One forward+backward in hard mode, reporting ||grad_w||, ||grad_b||
     against ||grad_weights||. If w and b get ~0 gradient, "learned addressing" in hard
     mode is a misnomer and the random init IS the addressing.

Read-only: loads checkpoints, trains nothing, writes one JSON.
"""
import json, os, sys

import jax, jax.numpy as jnp
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(D, "exp_c06_jax_backprop"))
import jax_lut_grad as L  # noqa: E402

C09 = os.path.join(D, "exp_c09_lut_sac")
OBS, ACT = 17, 6
N_OBS = 2000


def init_wb(seed, nap, tph, heads):
    """Reconstruct the init EXACTLY as lut_sac.py does it: PRNGKey(seed) -> split 4 ->
    the actor key -> L.init -> split 3. Any drift in this reconstruction would show up
    as a spurious delta, so it mirrors the real call chain rather than re-deriving it."""
    key = jax.random.PRNGKey(seed)
    _, ka, _, _ = jax.random.split(key, 4)
    p = L.init(ka, nap, tph, heads, OBS, 2 * ACT, jnp.zeros(OBS), jnp.ones(OBS))
    return np.asarray(p["w"]), np.asarray(p["b"])


def bits(x, w, b):
    a = np.einsum("bd,tnd->btn", x, w) + b[None]
    return a > 0


def main():
    stats = json.load(open(os.path.join(D, "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = np.asarray(stats["obs_mean"], np.float32)
    osd = np.asarray(stats["obs_std"], np.float32)
    obs = np.load(os.path.join(D, "exp_c03_distillation", "obs.npy"),
                  mmap_mode="r")[:N_OBS]
    x = ((np.asarray(obs, np.float32) - om) / (osd + 1e-6)).astype(np.float32)
    print(f"{x.shape[0]} standardised observations from the distillation dataset\n")

    targets = [(f"lut_sac_c14_hyperplane_hard_s{s}_actor.npz", s, f"exp_c14 seed {s}")
               for s in (0, 1, 2)]
    targets.append(("lut_sac_c11_hyperplane_hard_actor.npz", 0,
                    "exp_c11 (PRNGKey(0), pre-flag)"))

    rows = []
    print(f"{'checkpoint':<32}{'|Δw|/|w|':>10}{'cos':>8}{'angle°':>8}"
          f"{'|Δb|/|b|':>10}{'mean|Δb|':>10}{'BITS FLIPPED':>14}")
    for fname, seed, label in targets:
        path = os.path.join(C09, fname)
        if not os.path.exists(path):
            print(f"  (missing {fname})")
            continue
        z = np.load(path)
        wf, bf = np.asarray(z["w"]), np.asarray(z["b"])
        nap, tph, heads = wf.shape[1], int(z["tph"]), int(z["n_heads"])
        wi, bi = init_wb(seed, nap, tph, heads)
        if wi.shape != wf.shape:
            print(f"  (shape mismatch for {label}: {wi.shape} vs {wf.shape})")
            continue

        dw = np.linalg.norm(wf - wi) / np.linalg.norm(wi)
        db = np.linalg.norm(bf - bi) / np.linalg.norm(bi)
        # per-row cosine: rows are the objects that define a bit, so rotate-per-row is
        # the meaningful geometry, not a global tensor angle.
        a_, b_ = wi.reshape(-1, OBS), wf.reshape(-1, OBS)
        cos = ((a_ * b_).sum(-1)
               / (np.linalg.norm(a_, axis=-1) * np.linalg.norm(b_, axis=-1) + 1e-12))
        ang = np.degrees(np.arccos(np.clip(cos, -1, 1)))
        flip = float((bits(x, wi, bi) != bits(x, wf, bf)).mean())

        print(f"{label:<32}{dw:>10.4f}{cos.mean():>8.4f}{ang.mean():>8.2f}"
              f"{db:>10.4f}{np.abs(bf - bi).mean():>10.4f}{100 * flip:>13.2f}%")
        rows.append(dict(checkpoint=fname, label=label, seed=seed,
                         rel_dw=float(dw), mean_cos=float(cos.mean()),
                         mean_angle_deg=float(ang.mean()), rel_db=float(db),
                         mean_abs_db=float(np.abs(bf - bi).mean()),
                         bit_flip_fraction=flip))

    # ---- 2. gradient flow in HARD mode -------------------------------------
    print("\ngradient flow through the hard-mode custom_vjp (one forward+backward):")
    z = np.load(os.path.join(C09, "lut_sac_c14_hyperplane_hard_s0_actor.npz"))
    p = {k: jnp.asarray(z[k]) for k in
         ("w", "b", "weights", "log_T_soft", "log_T_sel")}
    heads, tph = int(z["n_heads"]), int(z["tph"])
    xb = jnp.asarray(x[:512])

    def loss(w, b, weights, ts, tl):
        y = L.lut_apply(xb, w, b, weights, ts, tl, heads, tph).sum(1)
        return jnp.square(jnp.tanh(y[:, :ACT])).mean()

    g = jax.grad(loss, argnums=(0, 1, 2, 3, 4))(
        p["w"], p["b"], p["weights"], p["log_T_soft"], p["log_T_sel"])
    names = ("grad_w", "grad_b", "grad_weights", "grad_log_T_soft", "grad_log_T_sel")
    gn = {}
    for n, gi in zip(names, g):
        v = np.asarray(gi)
        gn[n] = dict(norm=float(np.linalg.norm(v)),
                     max_abs=float(np.abs(v).max()),
                     frac_nonzero=float((v != 0).mean()))
        print(f"  {n:<18} ||g|| {gn[n]['norm']:11.4e}   max|g| "
              f"{gn[n]['max_abs']:11.4e}   nonzero {100 * gn[n]['frac_nonzero']:6.2f}%")
    ratio = gn["grad_w"]["norm"] / (gn["grad_weights"]["norm"] + 1e-30)
    print(f"  ||grad_w|| / ||grad_weights|| = {ratio:.4f}")

    json.dump(dict(deltas=rows, grads=gn, grad_w_over_grad_table=float(ratio),
                   n_obs=int(x.shape[0])),
              open(os.path.join(HERE, "hyperplane_movement.json"), "w"), indent=1)
    print("\nwrote hyperplane_movement.json")


if __name__ == "__main__":
    main()
