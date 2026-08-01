"""exp_c21 follow-up — post-training quantization sweep of the best LUT-SAC policy.

The checkpoint is `lut_sac_c21_seed4_20k_actor.npz` (seed 4 @ 20,000 iters,
hyperplane x hard, CPU-reference 5647.5 +/- 592.2 over 100 deterministic episodes).
A LUT policy has two structurally different halves and this sweep quantizes them
INDEPENDENTLY, because they fail in different ways:

  * TABLE CONTENT (`weights`, [32, 64, 12]) is read out and summed. Error here is
    additive and averages down across the 32 tables -- a graceful-degradation path.
  * ADDRESSING (`w` [32, 6, 17], `b` [32, 6]) only ever enters through
    index = sum_i 2^i * 1[<w_i, x> + b_i > 0]. Its magnitudes are discarded; only
    the SIGN survives. Error here is not additive, it is a discrete routing change:
    one flipped bit jumps to a different row of 64, and the read-out value has no
    reason to be close. So the honest diagnostic for part B is not the return alone
    but the fraction of (state, table, hyperplane) sign decisions that flipped.

Protocol: the same deterministic 100-episode CPU harness that produced 5647.5
(`perturb.eval_batched`, seed0=0, nominal model, hard forward). `eval_full` below is
that function verbatim with one extra return value -- the alive mask, so we can also
report how many episodes ran the full 1000 steps. The full-precision row is run as a
baseline and must reproduce 5647.5; that reproduction is what certifies the protocol
match, and the script refuses to continue if it drifts.

Writes `lut_sac_c21_seed4_20k_quant_sweep.json`. Deliberately does NOT touch any
existing `*_cpueval.json`.

Usage:
  python quant_sweep.py [--episodes 100]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_cpu                                            # noqa: E402  (sets up paths)
import jax                                                 # noqa: E402
import jax.numpy as jnp                                    # noqa: E402
import jax_lut_ext as X                                    # noqa: E402
import jax_lut_grad as L                                   # noqa: E402
import perturb                                             # noqa: E402

ACT = 6
CKPT = "lut_sac_c21_seed4_20k_actor.npz"
OUT = "lut_sac_c21_seed4_20k_quant_sweep.json"
BASELINE_ON_RECORD = 5647.482926605437


# ---------------------------------------------------------------- quantization
def quantize(arr, bits):
    """Symmetric per-table linear quantization, max-abs scale, then dequantize.

    Per-table means one scale per index along axis 0 -- 32 scales for `weights`
    [32, 64, 12], 32 for `w` [32, 6, 17], 32 for `b` [32, 6]. Signed symmetric with
    qmax = 2^(bits-1) - 1, so 8 bits -> +/-127 and 2 bits -> {-1, 0, +1}. (2-bit
    symmetric is genuinely ternary; the fourth code of a 2-bit word is unused by the
    symmetric scheme. That is the standard convention and is stated here so the
    2-bit row is not read as 4 levels.)

    A table that is exactly all-zero has no scale; it is passed through untouched.
    """
    a = np.asarray(arr, np.float64)
    qmax = 2 ** (bits - 1) - 1
    flat = a.reshape(a.shape[0], -1)
    scale = np.abs(flat).max(axis=1) / qmax                 # [T]
    scale = np.where(scale > 0, scale, 1.0)
    s = scale.reshape((-1,) + (1,) * (a.ndim - 1))
    q = np.clip(np.rint(a / s), -qmax, qmax)
    return (q * s).astype(np.float32)


# ------------------------------------------------------------------ the policy
def build_actor(w, b, weights, log_T_soft, log_T_sel, heads, tph,
                w_ref=None, b_ref=None):
    """Deterministic hard-forward actor, identical in form to eval_cpu.load_actor.

    If w_ref/b_ref are given, every call also recomputes the routing bits with the
    REFERENCE addressing on the same normalized state and accumulates how many of the
    32*6 sign decisions differ. Counting happens along the QUANTIZED policy's own
    trajectory, which is the right conditioning: we want the flip rate in the states
    this policy actually visits, not in states it never reaches.
    """
    p = dict(w=jnp.asarray(w), b=jnp.asarray(b), weights=jnp.asarray(weights),
             log_T_soft=jnp.asarray(log_T_soft), log_T_sel=jnp.asarray(log_T_sel))
    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    om = jnp.asarray(stats["obs_mean"], jnp.float32)
    osd = jnp.asarray(stats["obs_std"], jnp.float32)

    @jax.jit
    def act(obs):
        x = (obs - om) / (osd + 1e-6)
        y = X.apply("hard")(x, p["w"], p["b"], p["weights"],
                            p["log_T_soft"], p["log_T_sel"], heads, tph).sum(1)
        return jnp.tanh(y[:, :ACT])

    if w_ref is None:
        return (lambda obs: np.asarray(act(jnp.asarray(obs)))), None

    wr, br = jnp.asarray(w_ref), jnp.asarray(b_ref)

    @jax.jit
    def flips(obs):
        x = (obs - om) / (osd + 1e-6)
        return jnp.sum(((L._project(x, p["w"], p["b"]) > 0)
                        != (L._project(x, wr, br) > 0))).astype(jnp.int32)

    tally = dict(flipped=0, total=0)
    nbits = int(np.asarray(w).shape[0] * np.asarray(w).shape[1])

    def fn(obs):
        o = jnp.asarray(obs)
        tally["flipped"] += int(flips(o))
        tally["total"] += obs.shape[0] * nbits
        return np.asarray(act(o))

    return fn, tally


# ------------------------------------------------------------------- the eval
def eval_full(model, policy_fn, episodes=100, max_steps=1000, seed0=0):
    """perturb.eval_batched verbatim, plus the alive mask.

    Copied rather than called so we can report full-length-episode counts; keeping it
    a copy (not a reimplementation) is what keeps it comparable to the 5647.5 number.
    """
    dt = model.opt.timestep * perturb.FRAME_SKIP
    datas, alive, rets = [], np.ones(episodes, bool), np.zeros(episodes)
    for ep in range(episodes):
        d = mujoco.MjData(model)
        rng = np.random.default_rng(seed0 + ep)
        d.qpos[:] += rng.uniform(-5e-3, 5e-3, model.nq)
        d.qvel[:] += rng.uniform(-5e-3, 5e-3, model.nv)
        mujoco.mj_forward(model, d)
        datas.append(d)

    for step in range(1, max_steps + 1):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        obs = np.stack([np.concatenate([datas[i].qpos[1:],
                                        np.clip(datas[i].qvel, -10, 10)])
                        for i in idx]).astype(np.float32)
        act = np.clip(np.asarray(policy_fn(obs), np.float64), -1.0, 1.0)
        for j, i in enumerate(idx):
            d = datas[i]
            x0 = d.qpos[0]
            d.ctrl[:] = act[j]
            for _ in range(perturb.FRAME_SKIP):
                mujoco.mj_step(model, d)
            rets[i] += 1.0 + (d.qpos[0] - x0) / dt - 1e-3 * float(act[j] @ act[j])
            z, ang = d.qpos[1], d.qpos[2]
            if not (0.8 < z < 2.0 and -1.0 < ang < 1.0):
                alive[i] = False
    return float(rets.mean()), float(rets.std()), int(alive.sum())


# -------------------------------------------------------------------- the plan
BITS = [8, 6, 4, 3, 2]
PLAN = ([dict(part="baseline", table=None, addr=None)]
        + [dict(part="A_table", table=n, addr=None) for n in BITS]
        + [dict(part="B_addr", table=None, addr=n) for n in BITS]
        + [dict(part="C_both", table=4, addr=8),
           dict(part="C_both", table=8, addr=4),
           dict(part="C_both", table=4, addr=4)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100)
    a = ap.parse_args()

    z = np.load(os.path.join(HERE, CKPT))
    w0, b0, W0 = z["w"], z["b"], z["weights"]
    heads, tph = int(z["n_heads"]), int(z["tph"])
    lts, lsel = z["log_T_soft"], z["log_T_sel"]
    model = perturb.make_model(None, 1.0)

    rows, t0 = [], time.time()
    for cfg in PLAN:
        w = w0 if cfg["addr"] is None else quantize(w0, cfg["addr"])
        b = b0 if cfg["addr"] is None else quantize(b0, cfg["addr"])
        W = W0 if cfg["table"] is None else quantize(W0, cfg["table"])
        track = cfg["addr"] is not None
        fn, tally = build_actor(w, b, W, lts, lsel, heads, tph,
                                w_ref=w0 if track else None,
                                b_ref=b0 if track else None)
        mean, sd, full = eval_full(model, fn, episodes=a.episodes)
        row = dict(part=cfg["part"], table_bits=cfg["table"], addr_bits=cfg["addr"],
                   mean=mean, std=sd, full_length=full, episodes=a.episodes)
        if tally is not None and tally["total"]:
            row["sign_flip_frac"] = tally["flipped"] / tally["total"]
            row["sign_decisions"] = tally["total"]
        rows.append(row)

        if cfg["part"] == "baseline":
            row["on_record"] = BASELINE_ON_RECORD
            if a.episodes == 100 and abs(mean - BASELINE_ON_RECORD) > 1.0:
                raise SystemExit(
                    f"baseline reproduced {mean:.1f}, not {BASELINE_ON_RECORD:.1f} -- "
                    f"the protocol does not match the number on record, so no "
                    f"retention percentage computed from it would be trustworthy. "
                    f"Stopping rather than publishing incomparable rows.")
        base = rows[0]["mean"]
        row["retention_pct"] = 100.0 * mean / base
        tag = (f"{cfg['part']:<9} table={str(cfg['table']):>4} addr={str(cfg['addr']):>4}")
        extra = (f"  flips {row['sign_flip_frac']*100:6.3f}%"
                 if "sign_flip_frac" in row else "")
        print(f"[{time.time()-t0:7.1f}s] {tag} -> {mean:7.1f} +/- {sd:6.1f}  "
              f"full {full:>3}/{a.episodes}  ret {row['retention_pct']:6.1f}%{extra}",
              flush=True)
        json.dump(dict(checkpoint=CKPT, episodes=a.episodes,
                       protocol="perturb.eval_batched, seed0=0, nominal model, "
                                "hard forward, deterministic (tanh of row mean)",
                       quantization="symmetric per-table max-abs, "
                                    "qmax = 2^(bits-1) - 1, dequantized to float32",
                       rows=rows),
                  open(os.path.join(HERE, OUT), "w"), indent=1)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
