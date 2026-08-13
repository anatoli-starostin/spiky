"""exp_c39 — LIFMultiHeadLUT SAC: 3 LIF detectors x 4 buckets, the width-vs-count control.

A fork of exp_c38/mhl_sac.py with only the detector/bucket split changed. Everything else
is deliberately untouched -- twin-Q MLP critic, replay, learned per-cell log-sigma, the
trust region on row deltas, learning rates, target entropy, tau, gamma, batch, warmup --
so any difference is attributable to the front-end. It carries the exp_profiling
optimisations (on-device coverage, jitted buffer insert, fused 32-update lax.scan) and the
sort-free `rank` arrival ordering that made exp_c38 faster than the torch reference.

THE CONFIGURATION: 1 head, 32 tables, 3 LIF detectors per table, 4 buckets per detector,
frozen temperatures, delay_init_std=4.

PARAMETER COUNT: 28,384 total = 3,808 front-end + 24,576 table, of which 28,320 are
trainable (the 64 frozen temperature scalars are counted but never move). That is
**101.3% of the hyperplane baseline** -- the closest parameter match of any model in this
entire chapter, and it arrives there without being tuned for it.

    model                       front-end   table    total   vs c18
    exp_c32b bucket 16x32           1,696   6,144    7,840    0.28x
  > exp_c39 mhl 3det x 4bkt x 32    3,808  24,576   28,384    1.01x
    exp_c18 hyperplane              3,456  24,576   28,032    1.00x
    exp_c37 bucket 32bkt x 64tab    4,416  24,576   28,992    1.03x
    exp_c36 bucket 16bkt x 128tab   6,784  24,576   31,360    1.12x
    exp_c31 PureLIF                 6,816  24,576   31,392    1.12x
    exp_c38 mhl 6det x 2bkt x 32    7,168  24,576   31,744    1.13x

WHAT THIS ISOLATES. 4**3 = 64 cells per table over 32 tables -- the SAME row count, the
SAME 24,576-entry table and the SAME table count as exp_c38 and exp_c31. The three
experiments now differ only in how those 64 rows are addressed:

    c31  ONE LIF, its single spike time compared against 6 learned deadlines. The 6 bits
         are 6 views of one scalar and cannot be independent.        -> 2951 +/- 2109
    c38  SIX independent LIFs, one binary test each.  Digit COUNT 6, digit WIDTH 2.
                                                                     -> 3214 +/- 1526
    c39  THREE independent LIFs, a 4-way ordered quantisation each.
         Digit COUNT 3, digit WIDTH 4.                               -> this run

So c38 vs c39 is a clean width-against-count trade at fixed capacity, and it also halves
the front-end (7,168 -> 3,808) because each detector carries its own 17 delays and 17
synapses -- three detectors cost half of six. If addressing capacity is what matters the
two should tie; if the number of INDEPENDENT scalars matters, c39 should fall back toward
c31; if the ORDERED structure within a digit matters (bucket indices are monotone in spike
time, bits are not), c39 could beat c38 with fewer parameters.

The c38 result makes the third reading worth testing. c38 was the first configuration in
the chapter to break the addressing-entropy plateau -- effective cells per table went from
the 1.7-2.5 that EVERY bucket configuration c32b-c37 converged to, up to 7.6-10.8 -- and
yet it did not separate from c31 in return (|t| 0.17). Whether c39's ordered digits land
nearer c38's diversity or the old plateau is the diagnostic to watch.

WHAT IS NEW MECHANICALLY, beyond the detector axis:

  DELAYS START SPREAD. delay_init_std=4 seeds every synapse's delay from a half-normal
  with scale 4 on a 32-wide window, instead of the zeros every previous run in this
  chapter used. At zero init all 17 synapses of a detector arrive in latency order and the
  6 detectors of a table are distinguished ONLY by their weights -- which start i.i.d., so
  the 6 detectors of a table start nearly interchangeable and the 64 cells collapse toward
  a diagonal. A spread delay breaks that symmetry at initialisation: each detector sees a
  different arrival ORDER, which is the one thing this model family is built to read.

  TEMPERATURES ARE FROZEN AT 1.0. T_cross and T_bkt are non-trainable. In c32b-c37 both
  were free and the run was reported as having no sharpening schedule and therefore no
  systematic terminal dip. Freezing removes the remaining degree of freedom by which the
  soft surrogate could drift away from the hard forward it is a surrogate for. JAX has no
  requires_grad, so the freeze is a gradient mask applied below; `Tbkt`/`Tcr` are logged
  every eval so a broken mask is visible immediately rather than at the autopsy.

  THERE IS NO `eps`. The reference dropped it from the signature. Earlier runs in this
  chapter carried an inert eps through for interface parity; nothing here does.

COLD START. At init the three boundaries per detector sit evenly at 8, 16 and 24 across
the 32-wide window, so each detector starts as an unbiased 4-way quantiser -- but the
membrane must still cross theta_mem = 1.0 to spike at all, and a non-firing detector folds
into the LAST bucket by construction, which at 4 buckets means digit 3 rather than c38's
digit 1. `cell-cov`, `eff-cells`, `digit` and `nosp` are logged so a run that never leaves
that corner is visible at the first eval rather than at the autopsy.
"""
import argparse, json, os, sys, time
from functools import partial

import jax, jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, HERE)

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx           # noqa: E402
import jax_mhl_lut as LIF        # noqa: E402

OBS, ACT = 17, 6
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0
FREEZE_KEYS = ("log_T_cross", "log_T_bkt")

CFG = {}


def actor_init(key, n_buckets, n_det, tph, heads, delay_init_std, boundary_offset=0.0,
               table_init_std=0.1, share_betas=False):
    p = LIF.init(key, n_buckets, n_det, tph, heads, OBS, 2 * ACT,
                 delay_init_std=delay_init_std, boundary_offset=boundary_offset,
                 table_init_std=table_init_std, share_betas=share_betas)
    # Same modest, uniform starting spread as exp_c09: bias the log-sigma half of every
    # cell so the tph tables sum to -1.0 rather than to 0.
    p["table"] = p["table"].at[:, :, ACT:].add(-1.0 / (heads * tph))
    return p


def actor_out(p, obs, mode="train"):
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    y = LIF.apply(p, x, CFG["n_heads"], CFG["tph"], CFG["n_buckets"], CFG["n_det"],
                  mode=mode).sum(1)                                  # [B, 12]
    mu, log_std = y[:, :ACT], jnp.clip(y[:, ACT:], LOGSTD_MIN, LOGSTD_MAX)
    return mu, log_std


def actor_sample(p, obs, key):
    """Reparameterised tanh-Gaussian: returns (action, logp, mu_tanh)."""
    mu, log_std = actor_out(p, obs, mode="train")
    std = jnp.exp(log_std)
    e = jax.random.normal(key, mu.shape)
    pre = mu + std * e
    a = jnp.tanh(pre)
    logp = (-0.5 * jnp.square(e) - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    logp -= jnp.log(1.0 - jnp.square(a) + 1e-6).sum(-1)
    return a, logp, jnp.tanh(mu)


def actor_rows(p, obs):
    """The mixed-radix CELL each table addresses. [B, n_tables] in [0, n_buckets**n_det)."""
    x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
    return LIF.address(p, x, CFG["n_det"], CFG["n_buckets"])


def q_init(key, hidden=256):
    def one(k):
        k1, k2, k3 = jax.random.split(k, 3)
        return dict(w1=jax.random.normal(k1, (OBS + ACT, hidden)) * 0.1,
                    b1=jnp.zeros(hidden),
                    w2=jax.random.normal(k2, (hidden, hidden)) * 0.1,
                    b2=jnp.zeros(hidden),
                    w3=jax.random.normal(k3, (hidden, 1)) * 0.01, b3=jnp.zeros(1))
    k1, k2 = jax.random.split(key)
    return dict(q1=one(k1), q2=one(k2))


def q_apply(qp, s, a):
    h = jnp.concatenate([s, a], -1)
    h = jax.nn.relu(h @ qp["w1"] + qp["b1"])
    h = jax.nn.relu(h @ qp["w2"] + qp["b2"])
    return jnp.squeeze(h @ qp["w3"] + qp["b3"], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--envs", type=int, default=64)
    ap.add_argument("--rollout", type=int, default=1)
    ap.add_argument("--updates", type=int, default=32)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--buckets", type=int, default=16)
    ap.add_argument("--ndet", type=int, default=1)
    ap.add_argument("--tph", type=int, default=128)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--delay-init-std", type=float, default=0.0)
    ap.add_argument("--share-betas", type=int, default=0,
                    help="exp_c45: 1 = ONE bucket ladder (a beta_base scalar plus a "
                         "beta_raw vector of length n_buckets-1) shared by EVERY "
                         "(table, detector); 0 = the stock per-(table, detector) "
                         "ladders. Sharing removes (T*D - 1) * n_buckets front-end "
                         "parameters and asks whether the per-table ladder carries real "
                         "capacity or is dead weight.")
    ap.add_argument("--table-init-std", type=float, default=0.1,
                    help="exp_c42: std of the random table draws, replacing the "
                         "reference hard-coded 0.1. Default is 0.1/sqrt(tph), the "
                         "fan-in correction that makes the SUMMED mu-head output std "
                         "~0.1 at any table count. The trainer log-sigma bias is "
                         "unchanged, so initial sigma stays exp(-1).")
    ap.add_argument("--boundary-offset", type=float, default=0.0,
                    help="exp_c41: per-detector additive BOUNDARY offset (detector d gets "
                         "d*offset added to its beta_base, sliding its whole [8,16,24] "
                         "ladder along the time axis). Unlike exp_c40's delay offset this "
                         "cannot change whether a detector fires. 0.0 reproduces the "
                         "stock exp_c39 init byte-for-byte.")
    ap.add_argument("--freeze-temperature", type=int, default=0,
                    help="1 = log_T_cross/log_T_bkt held at 0 (T=1.0), as the reference's "
                         "freeze_temperature=True does with requires_grad=False")
    ap.add_argument("--target-entropy", type=float, default=-6.0)
    ap.add_argument("--row-clip", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out_name = a.out or f"mhl_sac{a.tag}"          # exp_c39 tags are _c39_s{seed}
    freeze = bool(a.freeze_temperature)

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    obs_mean = jnp.asarray(stats["obs_mean"], jnp.float32)
    obs_std = jnp.asarray(stats["obs_std"], jnp.float32)

    key = jax.random.PRNGKey(a.seed)
    key, ka, kq, kr = jax.random.split(key, 4)
    CFG.update(n_heads=a.heads, tph=a.tph, n_buckets=a.buckets, n_det=a.ndet,
               obs_mean=obs_mean, obs_std=obs_std)
    ap_ = actor_init(ka, a.buckets, a.ndet, a.tph, a.heads, a.delay_init_std,
                     a.boundary_offset, a.table_init_std, bool(a.share_betas))

    qp = q_init(kq)
    qt = jax.tree.map(lambda x: x, qp)
    log_alpha = jnp.log(jnp.asarray(0.2))

    n_tables = a.heads * a.tph
    K = a.buckets ** a.ndet                      # cells per table
    n_det_p, n_tab = LIF.n_params(ap_)
    n_frozen = 2 * n_tables if freeze else 0
    print(f"MHL-LIF SAC actor heads={a.heads} tph={a.tph} n_det={a.ndet} "
          f"buckets={a.buckets} seed={a.seed} | 12 outputs/cell (6 mu + 6 log-sigma) | "
          f"front-end {n_det_p:,} + table {n_tab:,} = {n_det_p + n_tab:,} params "
          f"({100*(n_det_p+n_tab)/28032:.1f}% of the 28,032 hyperplane baseline; "
          f"{n_det_p + n_tab - n_frozen:,} trainable) | cells {n_tables}x{K} = "
          f"{n_tables*K:,} | delay_init_std={a.delay_init_std} "
          f"table_init_std={a.table_init_std:.5f} "
          f"share_betas={bool(a.share_betas)} | temperatures "
          f"{'FROZEN at 1.0' if freeze else 'trainable'} | no eps", flush=True)

    tx_a = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.actor_lr))
    tx_q = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    tx_al = optax.adam(a.lr)
    os_a, os_q, os_al = tx_a.init(ap_), tx_q.init(qp), tx_al.init(log_alpha)

    m = W.make_model()
    mx = mjx.put_model(m)
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(kr, a.envs))

    N = a.buffer
    buf = dict(s=jnp.zeros((N, OBS)), a=jnp.zeros((N, ACT)), r=jnp.zeros(N),
               s2=jnp.zeros((N, OBS)), d=jnp.zeros(N))
    ptr, size = 0, 0
    row_updates = jnp.zeros((n_tables, K))

    def norm(s):
        return (s - obs_mean) / (obs_std + 1e-6)

    @jax.jit
    def rollout(ap_, st, key, random_actions):
        def one(carry, _):
            st, key = carry
            key, k1 = jax.random.split(key)
            a_rand = jax.random.uniform(k1, (a.envs, ACT), minval=-1.0, maxval=1.0)
            a_pol, _, _ = actor_sample(ap_, st.obs, k1)
            act = jnp.where(random_actions, a_rand, a_pol)
            nst = v_step(st, act)
            return (nst, key), (st.obs, act, nst.reward, nst.obs, nst.done)
        (st, key), tr = jax.lax.scan(one, (st, key), None, length=a.rollout)
        return st, key, tr

    @partial(jax.jit, static_argnums=())
    def update(ap_, qp, qt, log_alpha, os_a, os_q, os_al, batch, key):
        s, act, r, s2, d = batch
        ns, ns2 = norm(s), norm(s2)
        alpha = jnp.exp(log_alpha)

        key, k1 = jax.random.split(key)
        a2, logp2, _ = actor_sample(ap_, s2, k1)
        q1t = q_apply(qt["q1"], ns2, a2)
        q2t = q_apply(qt["q2"], ns2, a2)
        target = r + a.gamma * (1 - d) * (jnp.minimum(q1t, q2t) - alpha * logp2)
        target = jax.lax.stop_gradient(target)

        def q_loss(qp):
            e1 = q_apply(qp["q1"], ns, act) - target
            e2 = q_apply(qp["q2"], ns, act) - target
            return (jnp.square(e1) + jnp.square(e2)).mean(), jnp.abs(e1)
        (ql, td), gq = jax.value_and_grad(q_loss, has_aux=True)(qp)
        uq, os_q = tx_q.update(gq, os_q, qp)
        qp = optax.apply_updates(qp, uq)

        key, k2 = jax.random.split(key)

        def a_loss(ap_):
            an, logp, _ = actor_sample(ap_, s, k2)
            q = jnp.minimum(q_apply(qp["q1"], ns, an), q_apply(qp["q2"], ns, an))
            return (alpha * logp - q).mean(), logp
        (al, logp), ga = jax.value_and_grad(a_loss, has_aux=True)(ap_)

        # THE TEMPERATURE FREEZE. torch does this with requires_grad=False; JAX has no
        # such flag, so the gradient is masked before the optimiser sees it. Adam of an
        # identically-zero gradient leaves both moments at zero and the update at exactly
        # zero, so the parameters are frozen in the strict sense, not merely slowed. The
        # same device exp_c35 used to freeze the bucket boundaries.
        if freeze:
            ga = dict(ga, **{k: jnp.zeros_like(ga[k]) for k in FREEZE_KEYS})

        # Per-row trust region, as in exp_c09. Applies to the table only; the LIF
        # front-end is continuous and has no row semantics.
        if a.row_clip > 0:
            gw = ga["table"]
            nrm = jnp.linalg.norm(gw, axis=-1, keepdims=True)
            ga = dict(ga, table=gw * jnp.minimum(1.0, a.row_clip / (nrm + 1e-8)))
        ua, os_a = tx_a.update(ga, os_a, ap_)
        ap_ = optax.apply_updates(ap_, ua)

        def al_loss(log_alpha):
            return (-jnp.exp(log_alpha)
                    * (jax.lax.stop_gradient(logp) + a.target_entropy)).mean()
        gal = jax.grad(al_loss)(log_alpha)
        ual, os_al = tx_al.update(gal, os_al, log_alpha)
        log_alpha = optax.apply_updates(log_alpha, ual)

        qt = jax.tree.map(lambda t, s_: (1 - a.tau) * t + a.tau * s_, qt, qp)
        rows = actor_rows(ap_, s)
        return (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key,
                dict(q_loss=ql, a_loss=al, alpha=alpha, logp=logp.mean(),
                     td=td.mean(), rows=rows))

    # =========================================================================
    # The exp_profiling optimisations (measured 1.45x on the harness, ~1.56x projected
    # on a full run). Introduced in exp_c37; see that experiment's README for the
    # measurement and for the CORRECTION about bit-exactness -- the fused block is
    # semantically identical to the Python loop but XLA is free to reassociate float
    # reductions differently between the two, so runs across the change are not
    # bit-diffable. Against this chapter's seed-to-seed sd of ~1,200 that is irrelevant,
    # but it is why c38 is run entirely on the fused trainer rather than half and half.
    # =========================================================================

    @jax.jit
    def insert(buf, idx, s_, a_, r_, s2_, d_):
        return dict(s=buf["s"].at[idx].set(s_), a=buf["a"].at[idx].set(a_),
                    r=buf["r"].at[idx].set(r_), s2=buf["s2"].at[idx].set(s2_),
                    d=buf["d"].at[idx].set(d_))

    @jax.jit
    def update_block(ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, cov, buf, size):
        def one(carry, _):
            ap_, qp, qt, la, os_a, os_q, os_al, key, cov = carry
            key, kb = jax.random.split(key)
            bi = jax.random.randint(kb, (a.batch,), 0, size)
            batch = (buf["s"][bi], buf["a"][bi], buf["r"][bi],
                     buf["s2"][bi], buf["d"][bi])
            (ap_, qp, qt, la, os_a, os_q, os_al, key, info) = update(
                ap_, qp, qt, la, os_a, os_q, os_al, batch, key)
            flat = (jnp.arange(n_tables)[None, :] * K + info["rows"]).ravel()
            cov = cov + jnp.zeros(n_tables * K).at[flat].add(1.0).reshape(n_tables, K)
            return (ap_, qp, qt, la, os_a, os_q, os_al, key, cov), info["alpha"]
        carry = (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, cov)
        carry, alphas = jax.lax.scan(one, carry, None, length=a.updates)
        return carry + (alphas[-1],)

    @jax.jit
    def det_action(ap_, obs):
        mu, _ = actor_out(ap_, obs, mode="eval")
        return jnp.tanh(mu)

    @jax.jit
    def addr_stats(ap_, obs):
        """(mean digit, no-spike rate, mean effective cells per table).

        digit    mean over (sample, table, detector) of the hard bucket digit, which at
                 n_buckets=4 ranges over 0..3 (NOT the 0/1 bit rate exp_c38 logged at
                 2 buckets -- the two are not comparable and are deliberately named
                 differently). A detector that never spikes folds into the LAST bucket, so
                 this starts near 3 and should FALL. Pinned at 0 or at n_buckets-1 means
                 the detectors have collapsed to a constant and the table is 1 row.
        nospike  fraction of detectors whose membrane never crossed theta_mem. A detector
                 that never spikes folds into bucket 1 by construction, which is the c32
                 failure mode; if this sits near 1.0 the model is not using its input.
        eff      2**entropy of the per-table cell-occupancy distribution, averaged over
                 tables -- the chapter's standard addressing diagnostic, so this number is
                 directly comparable to the 1.7-2.5 bits every bucket configuration
                 converged to in c32b-c37.
        """
        x = (obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6)
        t_hard, t_soft = LIF.first_spike(ap_, x)
        b, _ = LIF.bucket(ap_, t_hard, t_soft)
        idx = LIF.cell_index(b, CFG["n_det"], CFG["n_buckets"])          # (B,T)
        oh = jax.nn.one_hot(idx, K)                                      # (B,T,K)
        q = oh.sum(0) / oh.shape[0]                                      # (T,K)
        ent = -(q * jnp.log2(jnp.where(q > 0, q, 1.0))).sum(-1)          # (T,)
        return (b.mean(), (t_hard >= LIF.T_WINDOW).mean(),
                jnp.exp2(ent).mean())

    def eval_mjx(ap_, episodes=20, horizon=1000, seed=0):
        """mode="eval" — the reference's module.eval() path, and byte-for-byte the
        training forward VALUE (parity_check asserts train == eval to 1.8e-07)."""
        stx = v_reset(jax.random.split(jax.random.PRNGKey(seed), episodes))

        @jax.jit
        def run(stx):
            def one(c, _):
                stx, ret, alive = c
                act = det_action(ap_, stx.obs)
                nst = v_step(stx, act)
                return (nst, ret + nst.reward * alive, alive * (1 - nst.done)), None
            (stx, ret, alive), _ = jax.lax.scan(
                one, (stx, jnp.zeros(episodes), jnp.ones(episodes)), None,
                length=horizon)
            return ret
        return float(np.asarray(run(stx)).mean())

    rows_log, t0, best = [], time.time(), -1e9
    last_alpha = None
    total_steps = 0

    for it in range(a.iters):
        key, kro = jax.random.split(key)
        st, kro, tr = rollout(ap_, st, kro, it < a.warmup)
        s_, a_, r_, s2_, d_ = [np.asarray(x).reshape((-1,) + x.shape[2:]) for x in tr]
        n = len(s_)
        idx = jnp.asarray((ptr + np.arange(n)) % N)
        buf = insert(buf, idx, s_, a_, r_, s2_, d_)
        ptr = int((ptr + n) % N)
        size = min(size + n, N)
        total_steps += n

        if it >= a.warmup:
            (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key, row_updates,
             last_alpha) = update_block(ap_, qp, qt, log_alpha, os_a, os_q, os_al,
                                        key, row_updates, buf, size)

        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            ret = eval_mjx(ap_, episodes=a.eval_episodes)
            cov = float((row_updates > 0).mean())
            digit, nosp, eff = [float(v) for v in addr_stats(ap_, st.obs)]
            tbkt = float(np.exp(np.asarray(ap_["log_T_bkt"])).mean())
            tcr = float(np.exp(np.asarray(ap_["log_T_cross"])).mean())
            best = max(best, ret)
            el = time.time() - t0
            rows_log.append(dict(iter=it + 1, env_steps=total_steps, mjx_return=ret,
                                 row_coverage=cov, digit=digit, nospike=nosp,
                                 eff_cells=eff, t_bkt=tbkt, t_cross=tcr,
                                 alpha=(float(last_alpha) if last_alpha is not None
                                        else None),
                                 elapsed_s=round(el, 1)))
            print(f"[{it+1:>5}/{a.iters}] steps {total_steps:>9,} | MJX ret {ret:8.1f} "
                  f"| cell-cov {cov*100:5.1f}% | digit {digit:5.3f} | nosp {nosp:5.3f} "
                  f"| eff {eff:5.2f}/{K} | Tbkt {tbkt:5.3f} | Tcr {tcr:5.3f} "
                  f"| best {best:8.1f} | {el/60:5.1f}m", flush=True)
            json.dump(dict(iter=it + 1, iters=a.iters, env_steps=total_steps,
                           mjx_return=ret, best=best, row_coverage=cov,
                           eta_s=(a.iters - it - 1) * el / (it + 1), done=False),
                      open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
            ck = dict({k: np.asarray(v) for k, v in ap_.items()},
                      n_heads=np.int32(a.heads), tph=np.int32(a.tph),
                      n_buckets=np.int32(a.buckets), n_det=np.int32(a.ndet),
                      freeze_temperature=np.int32(int(freeze)),
                      delay_init_std=np.float32(a.delay_init_std),
                      boundary_offset=np.float32(a.boundary_offset),
                      table_init_std=np.float32(a.table_init_std),
                      share_betas=np.int32(int(a.share_betas)))
            np.savez(os.path.join(HERE, f"{out_name}_actor.npz"), **ck)
            if ret >= best:
                np.savez(os.path.join(HERE, f"{out_name}_best_actor.npz"),
                         **ck, best_iter=np.int32(it + 1), best_mjx=np.float32(ret))

    json.dump(dict(config=vars(a), frontend_params=n_det_p, table_params=n_tab,
                   total_params=n_det_p + n_tab,
                   trainable_params=n_det_p + n_tab - n_frozen,
                   cells_per_table=K, total_env_steps=total_steps,
                   wall_s=round(time.time() - t0, 1), best_mjx=best, history=rows_log),
              open(os.path.join(HERE, out_name + ".json"), "w"), indent=1)
    json.dump(dict(iter=a.iters, iters=a.iters, env_steps=total_steps,
                   mjx_return=rows_log[-1]["mjx_return"] if rows_log else 0.0,
                   best=best,
                   row_coverage=rows_log[-1]["row_coverage"] if rows_log else 0.0,
                   eta_s=0.0, done=True),
              open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
    print(f"done: best MJX {best:.1f} over {total_steps:,} env-steps in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
