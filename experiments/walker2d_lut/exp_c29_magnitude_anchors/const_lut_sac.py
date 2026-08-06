"""exp_c29 — do fixed CONSTANTS in the LUT input cure anchor magnitude blindness? (#75)

FORK of exp_c09_lut_sac/lut_sac.py. Everything below the constant machinery is that file
unchanged; it is forked rather than flagged because exp_c09's trainer produced published
numbers (c11-c22) and must keep running bit-identically.

THE DEFECT UNDER TEST. Anchor addressing computes bit_i = 1[x[a] - x[b] > 0] -- a
pairwise comparison between two coordinates of the standardised observation. Two things
follow, and both are limitations a hyperplane bit does not have:

  * it is invariant to adding any constant to the whole observation vector, so it can
    read RELATIVE order but never absolute level;
  * it can never express "is this joint's velocity above 3 sigma", because there is
    nothing constant to compare against.

Call that magnitude blindness. The intervention is the cheapest one imaginable: append
NC fixed numbers to the LUT's input. A pair (obs j, const k) is then 1[x_j > c_k] -- a
threshold test -- and it costs no parameters, no new forward, and no gradient (the
addressing is frozen in anchors mode anyway). If magnitude blindness is what has been
costing the anchors, this recovers it; if it is not, nothing moves.

WHAT IS AND IS NOT AUGMENTED. The constants go to the ACTOR'S ADDRESSING ONLY. The
critic still sees the 17-dim observation, and the replay buffer still stores 17 dims.
That is deliberate and it matters: a constant is by definition uninformative to a
function approximator that can already fit a bias, so feeding it to the critic would add
17 -> 33 input weights of pure dead capacity and make the arms differ in a second,
uninteresting way. Only the LUT's comparators can do anything with a constant, so only
they get it. It also keeps the buffer, the critic and the SAC machinery bit-identical to
the baseline arm, so a difference in outcome has one possible cause.

ARMS (--constants). The values are built and verified by make_constants.py; see its
header for how each set is derived and why there are three.
  none     17-dim input, ordinary frozen anchors. The magnitude-blind baseline.
  grid     33-dim; 16 evenly tiled thresholds -- exp_c28's "16 levels are enough" set.
  random   33-dim; 16 irregular thresholds over the same range.
  clumped  33-dim; 16 thresholds inside the central fifth of the range only.

DEGENERATE BITS. lutorch's samplers know nothing about the obs/const split, so at
input_dim 33 with 16 constants roughly 23% of the drawn pairs have BOTH endpoints in the
constant block. Such a bit is 1[c_p > c_q]: the same value in every state, contributing
one permanently-stuck address bit and halving that table's reachable rows. Left alone it
would cost the augmented arms about a quarter of their capacity and the experiment would
be measuring that, not magnitude. They are repaired -- see repair_const_const -- and the
count is reported.

--- original exp_c09 header follows ---

exp_c09 — LUT-SAC: an off-policy actor-critic whose ACTOR is a lookup table (#75).

Diagnosis this is built on (established in Phases 1-4, not re-derived here): the
from-scratch gap is an OPTIMIZATION problem, not a capacity one. Distillation puts a
5,378-param LUT at 5512, so the table can represent the gait; PPO-from-scratch reaches
only 4407 because (a) on-policy data is a narrow state distribution while the LUT's
backward scatters into a SINGLE addressed row per sample, so most rows barely train, and
(b) a global fixed Gaussian is the wrong exploration for a rugged, piecewise-constant
landscape.

The design answers both:

  * **Actor = LUT with 12 outputs per cell**: 6 action means AND 6 log-stds. Exploration
    spread becomes state-local and *learned*, stored in the table itself:
    a = tanh(mu_row + sigma_row * eps). Where the critic is flat the entropy term widens
    that cell; where it is sharp, it narrows.
  * **Off-policy replay** decouples the update distribution from the current policy, so
    rows keep receiving gradient long after the policy stops visiting them.
  * **Coverage-prioritised sampling** additionally up-weights transitions addressing
    rarely-UPDATED rows — aimed directly at the sparse-scatter problem.
  * **Per-row trust region**: a row update is a step change in the policy for every state
    in that cell (unlike an MLP's smeared update). Row deltas are norm-clipped.

Critic is an MLP twin-Q (Variant A) so the question stays "can a LUT be the actor?".
A LUT critic is a separate research branch and is deliberately NOT built here.

Diagnostic: `row_coverage` = fraction of table rows that have received an update. The
diagnosis predicts it should RISE with return; both are logged every iteration.
"""
import argparse, json, os, sys, time
from functools import partial

import jax, jax.numpy as jnp
import numpy as np
import optax

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c02_mjx_scaffold"))
sys.path.insert(0, os.path.join(HERE, "..", "exp_c06_jax_backprop"))

import mjx_walker2d as W          # noqa: E402
from mujoco import mjx           # noqa: E402
import jax_lut_grad as L         # noqa: E402
sys.path.insert(0, os.path.join(HERE, "..", "exp_c11_lut_sac_2x2"))
import jax_lut_ext as X           # noqa: E402

OBS, ACT = 17, 6
# mjx_walker2d.observation = qpos[1:] ++ clip(qvel, -10, 10). Named here so the wiring
# report can say WHICH channel a comparator is attached to; a bare index would make the
# instrumentation unreadable at exactly the moment it has to be read.
NAMES = ["z_height", "torso_angle", "r_hip", "r_knee", "r_ankle",
         "l_hip", "l_knee", "l_ankle",
         "vx", "vz", "torso_angvel",
         "r_hip_vel", "r_knee_vel", "r_ankle_vel",
         "l_hip_vel", "l_knee_vel", "l_ankle_vel"]
LOGSTD_MIN, LOGSTD_MAX = -5.0, 2.0
# torch HyperplaneMultiHeadLUT's `initial_weights_noise` default, the std used by its
# hyperplane_init="random" mode (hyperplane_multi_head_lut.py:682).
TORCH_HP_NOISE = 0.001


# =============================================================================
# Actor: LUT with 12 outputs per cell (6 mu + 6 log sigma)
# =============================================================================

# Static/buffer state kept OUT of the differentiable pytree: jax.grad rejects the
# int fields (n_heads/tph), and the observation standardiser is a frozen buffer that
# must not receive gradient. Set once in main().
CFG = {}


# -----------------------------------------------------------------------------
# The constant block
# -----------------------------------------------------------------------------

def augment(x):
    """[B, 17] standardised observation -> [B, 17 + NC], the LUT's addressing input.

    The constants are appended AFTER standardisation, not before, because they are
    already expressed in standardised units (make_constants.py measures the percentile
    range in that space). Running them through the normaliser would rescale a threshold
    by a channel statistic it has nothing to do with.
    """
    if CFG["nconst"] == 0:
        return x
    return jnp.concatenate(
        [x, jnp.broadcast_to(CFG["const"], (x.shape[0], CFG["nconst"]))], axis=-1)


def anchor_pairs(n_tables, nap, input_dim, seed, policy, heads):
    """lutorch's own draw, returned as INDEX PAIRS rather than as the (w, b) encoding.

    jax_lut_ext exposes only w = e_a - e_b, and this experiment needs the pairs
    themselves twice over: to repair constant-constant bits, and to report which
    constant each surviving comparator is wired to. The indices could be recovered from
    the sparse w, but that would be a second derivation of the same fact and a place for
    the two to disagree; the sampler already caches the indices, so they are read from
    there. The wb call is made first purely to populate that cache.
    """
    X.anchor_pair_wb_lutorch(n_tables, nap, input_dim, seed=seed, policy=policy,
                             heads=heads)
    path = os.path.join(os.path.expanduser("~/.cache/spiky_anchors"),
                        f"anchors_{policy}_t{n_tables}_nap{nap}_d{input_dim}"
                        f"_h{heads}_s{seed}_cpu.npz")
    z = np.load(path)
    return z["anchor_a"].copy(), z["anchor_b"].copy()


def repair_const_const(ia, ib, n_obs, seed):
    """Redraw any bit whose BOTH endpoints are constants.

    1[c_p > c_q] is the same value in every state. The bit is not merely useless: it is
    STUCK, so that table addresses only half its rows, and with ~23% of bits affected at
    d=33 an augmented arm would lose about a quarter of its capacity for a reason that
    has nothing to do with the hypothesis.

    The repair moves the SECOND endpoint to an observation coordinate, choosing the
    least-used one so the balance lutorch's sampler was aiming for is preserved rather
    than destroyed by piling the repairs onto whichever coordinate came first. Ties are
    broken by a generator seeded from the run seed, so the repair is deterministic and
    travels with the seed exactly as the draw does.

    Note what is NOT repaired: a pair of two observation coordinates is left alone, and
    so is an obs-const pair. Only the provably-constant bit is touched.
    """
    ia, ib = ia.copy(), ib.copy()
    bad = (ia >= n_obs) & (ib >= n_obs)
    if not bad.any():
        return ia, ib, 0
    rng = np.random.default_rng(seed + 90_000)
    use = np.bincount(np.concatenate([ia[ia < n_obs].ravel(), ib[ib < n_obs].ravel()]),
                      minlength=n_obs).astype(np.int64)
    for t, i in zip(*np.nonzero(bad)):
        cand = np.flatnonzero(use == use.min())
        j = int(rng.choice(cand))
        ib[t, i] = j
        use[j] += 1
    return ia, ib, int(bad.sum())


def wiring_report(ia, ib, n_obs, nconst, names, n_repaired):
    """What the addressing is actually made of, printed before a single step is taken.

    A bit is classified by its two endpoints: obs-obs is an ordinary anchor comparator,
    obs-const is a threshold test (the thing this experiment adds), const-const is dead
    and should be zero after the repair. The per-constant counts are the STATIC half of
    the instrumentation the task asks for -- how often each added constant is wired to
    anything at all. The DYNAMIC half, whether those comparators actually carry
    information once the policy is running, is bit_usage.py.
    """
    a_is_c, b_is_c = ia >= n_obs, ib >= n_obs
    oo = int((~a_is_c & ~b_is_c).sum())
    oc = int((a_is_c ^ b_is_c).sum())
    cc = int((a_is_c & b_is_c).sum())
    tot = ia.size
    print(f"  addressing wiring: {tot} bits = {oo} obs-obs ({100*oo/tot:.1f}%) + "
          f"{oc} obs-const ({100*oc/tot:.1f}%) + {cc} const-const "
          f"({100*cc/tot:.1f}%); {n_repaired} const-const bits repaired", flush=True)
    if cc:
        raise ValueError(f"{cc} const-const bits survived the repair -- those are dead "
                         f"address bits and would silently cost this arm capacity")
    if nconst:
        cnt = np.bincount(np.concatenate([ia[a_is_c].ravel(), ib[b_is_c].ravel()])
                          - n_obs, minlength=nconst)
        print("  bits wired to each constant: "
              + " ".join(f"c{k}:{c}" for k, c in enumerate(cnt)), flush=True)
    ocnt = np.bincount(np.concatenate([ia[~a_is_c].ravel(), ib[~b_is_c].ravel()]),
                       minlength=n_obs)
    print("  bits wired to each observation channel: "
          + " ".join(f"{names[j][:6]}:{c}" for j, c in enumerate(ocnt)), flush=True)
    return dict(obs_obs=oo, obs_const=oc, const_const=cc, repaired=n_repaired,
                per_const=(cnt.tolist() if nconst else []), per_obs=ocnt.tolist())


def actor_init(key, nap, tph, heads, obs_mean, obs_std, aobs, table_std=0.05):
    p = L.init(key, nap, tph, heads, aobs, 2 * ACT, obs_mean, obs_std,
               table_std=table_std)
    # start with a modest, uniform spread: bias the log-sigma half of every row
    p["weights"] = p["weights"].at[:, :, ACT:].add(-1.0 / (heads * tph))
    return {k: v for k, v in p.items()
            if k not in ("n_heads", "tph", "obs_mean", "obs_std")}


def actor_out(p, obs):
    x = augment((obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6))
    y = CFG["apply"](x, p["w"], p["b"], p["weights"], p["log_T_soft"],
                     p["log_T_sel"], CFG["n_heads"], CFG["tph"]).sum(1)   # [B, 12]
    mu, log_std = y[:, :ACT], jnp.clip(y[:, ACT:], LOGSTD_MIN, LOGSTD_MAX)
    return mu, log_std


def actor_sample(p, obs, key):
    """Reparameterised tanh-Gaussian: returns (action, logp, mu_tanh)."""
    mu, log_std = actor_out(p, obs)
    std = jnp.exp(log_std)
    eps = jax.random.normal(key, mu.shape)
    pre = mu + std * eps
    a = jnp.tanh(pre)
    logp = (-0.5 * jnp.square(eps) - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)
    logp -= jnp.log(1.0 - jnp.square(a) + 1e-6).sum(-1)      # tanh correction
    return a, logp, jnp.tanh(mu)


def actor_rows(p, obs):
    """The row index each table addresses — the coverage diagnostic. [B, n_tables]"""
    x = augment((obs - CFG["obs_mean"]) / (CFG["obs_std"] + 1e-6))
    return L._hard_index(L._project(x, p["w"], p["b"]), p["w"].shape[1])


# =============================================================================
# Critic: twin Q MLP
# =============================================================================

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


# =============================================================================
# Training
# =============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=4)
    ap.add_argument("--updates", type=int, default=8, help="grad steps per iteration")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--buffer", type=int, default=1_000_000)
    ap.add_argument("--warmup", type=int, default=100, help="random-action iterations")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--tau", type=float, default=0.005)
    ap.add_argument("--nap", type=int, default=6)
    ap.add_argument("--tph", type=int, default=64)
    ap.add_argument("--heads", type=int, default=1)
    ap.add_argument("--target-entropy", type=float, default=-6.0)
    ap.add_argument("--row-clip", type=float, default=0.0,
                    help="per-row trust region: max L2 norm of a row delta (0=off)")
    ap.add_argument("--coverage-bonus", type=float, default=0.0,
                    help="weight of the rarely-updated-row bonus in sampling (0=off)")
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--addressing", default="hyperplane",
                    choices=["hyperplane", "anchors"],
                    help="anchors = frozen w = e_a - e_b (FastMHL semantics)")
    ap.add_argument("--seed", type=int, default=0,
                    help="drives BOTH the jax PRNGKey and the anchor draw")
    ap.add_argument("--anchor-policy", default="balanced",
                    choices=["balanced", "connected", "canonical_distinct",
                             "canonical_full_coverage", "legacy_jax"],
                    help="lutorch AnchorSamplingPolicy for --addressing anchors. "
                         "NOTE FastMultiHeadLut itself uses canonical_full_coverage "
                         "and REJECTS balanced. legacy_jax = the home-grown draw that "
                         "produced the exp_c11/exp_c12 numbers.")
    ap.add_argument("--hyperplane-init", default="anchor_pairs",
                    choices=["anchor_pairs", "random_torch", "legacy_jax"],
                    help="init for --addressing hyperplane. anchor_pairs (default) is "
                         "TORCH'S DEFAULT: w = e_a - e_b, b = 0, so the model starts as "
                         "a bit-exact FastMultiHeadLut and learns away. random_torch = "
                         "torch's other mode, N(0, 0.001) with b = 0. legacy_jax = the "
                         "old dense N(0,0.5^2)/N(0,0.1^2) draw that produced exp_c11 "
                         "and exp_c14.")
    ap.add_argument("--hyperplane-anchor-policy", default="canonical_full_coverage",
                    choices=["canonical_full_coverage", "canonical_distinct"],
                    help="sampler for the anchor_pairs hyperplane init. Only these two "
                         "are offered because HyperplaneMultiHeadLUT itself REJECTS "
                         "balanced/connected for this init (ValueError), so 'balanced' "
                         "here would not be torch-faithful.")
    ap.add_argument("--forward-mode", default="hard",
                    choices=["hard", "hybrid_smooth"])
    ap.add_argument("--constants", default="none",
                    choices=["none", "grid", "random", "clumped"],
                    help="which fixed-constant block to append to the LUT's addressing "
                         "input. `none` is the magnitude-blind baseline and leaves the "
                         "17-dim path untouched. The other three are built and verified "
                         "by make_constants.py; see its header. The set NAME and the "
                         "VALUES both go into the checkpoint, so an eval cannot be run "
                         "against the wrong block.")
    ap.add_argument("--const-file", default=None,
                    help="default: constants.json next to this script")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    ap.add_argument("--snap-every", type=int, default=0,
                    help="if >0, record the addressing (w, b) and the row-update "
                         "histogram every N iters into <out>_snaps.npz. Purely "
                         "diagnostic -- it reads state and consumes no randomness, so "
                         "a run with snapshots on is bit-identical to one without. "
                         "Needed to ask whether addressing is still MOVING late in "
                         "training, which a single final checkpoint cannot answer.")
    ap.add_argument("--checkpoint-at", default="", metavar="ITERS",
                    help="comma-separated iteration counts at which to save an EXTRA, "
                         "never-overwritten checkpoint <out>_at<N>_actor.npz. The normal "
                         "<out>_actor.npz is rewritten at every eval, so without this a "
                         "longer run cannot be compared against a shorter one at the "
                         "shorter one's horizon.")
    ap.add_argument("--freeze-wb-from", metavar="ACTOR_NPZ", default=None,
                    help="load (w, b) from a trained checkpoint and FREEZE them: the "
                         "addressing receives no gradient, exactly as in --addressing "
                         "anchors, and only the table content (plus the critic and the "
                         "temperature) learns. Overrides whatever the init branches "
                         "produced. This transplants one run's learned routing into "
                         "another run to ask whether the routing carries the result.")
    a = ap.parse_args()
    out_name = a.out or f"lut_sac{a.tag}"

    stats = json.load(open(os.path.join(HERE, "..", "exp_c03_distillation",
                                        "dataset_stats.json")))
    obs_mean = jnp.asarray(stats["obs_mean"], jnp.float32)
    obs_std = jnp.asarray(stats["obs_std"], jnp.float32)

    # ---- the constant block ------------------------------------------------
    cfile = a.const_file or os.path.join(HERE, "constants.json")
    if a.constants == "none":
        const = np.zeros(0, np.float32)
    else:
        cj = json.load(open(cfile))
        const = np.asarray(cj["sets"][a.constants], np.float32)
        # The block is what defines this arm, so a silent shape or content surprise is
        # the one failure that must not reach a 40-minute run. Re-checked here rather
        # than trusted from the builder: this process is the one that will train on it.
        if const.ndim != 1 or len(const) != int(cj["levels"]):
            raise ValueError(f"{a.constants}: expected {cj['levels']} constants, "
                             f"got shape {const.shape}")
        if not np.all(np.diff(const) > 0):
            raise ValueError(f"{a.constants} constants are not strictly increasing; "
                             f"duplicates would give bit-identical input dimensions")
    NC = len(const)
    AOBS = OBS + NC

    key = jax.random.PRNGKey(a.seed)
    key, ka, kq, kr = jax.random.split(key, 4)
    CFG.update(n_heads=a.heads, tph=a.tph, obs_mean=obs_mean, obs_std=obs_std,
               apply=X.apply(a.forward_mode), nconst=NC,
               const=jnp.asarray(const)[None, :] if NC else None)
    ap_ = actor_init(ka, a.nap, a.tph, a.heads, obs_mean, obs_std, AOBS)
    pair_a = pair_b = None
    wiring = None
    if a.addressing == "anchors":
        # anchor pairs written as a FROZEN hyperplane: w = e_a - e_b, b = 0.
        # The ENCODING is verified bit-exact against FastMultiHeadLut
        # (exp_c11/verify_ext.py check B); the DRAW now comes from lutorch's own
        # sampler (check C), instead of the home-grown one that only matched its
        # spirit. --anchor-policy legacy_jax restores that older draw, which is what
        # the exp_c11 2x2 and the exp_c12 sweep were run with.
        if a.anchor_policy == "legacy_jax":
            w0, b0 = X.anchor_pair_wb(np.random.default_rng(a.seed),
                                      a.heads * a.tph, a.nap, AOBS)
        else:
            # Drawn over the FULL augmented dimension, so the sampler's own balancing
            # decides how much of the wiring is threshold tests -- it is not imposed
            # here. What IS imposed is that no bit ends up comparing two constants.
            pair_a, pair_b = anchor_pairs(a.heads * a.tph, a.nap, AOBS, a.seed,
                                          a.anchor_policy, a.heads)
            pair_a, pair_b, n_rep = repair_const_const(pair_a, pair_b, OBS, a.seed)
            w0, b0 = X.pairs_to_wb(pair_a, pair_b, AOBS)
        ap_ = dict(ap_, w=w0, b=b0)
    elif a.hyperplane_init != "legacy_jax":
        # Hyperplane addressing, initialised the way torch does it. w and b stay
        # LEARNABLE either way -- only the starting point changes.
        if a.hyperplane_init == "anchor_pairs":
            # Torch's DEFAULT: w = e_a - e_b, b = 0, exact and unperturbed, so the
            # model starts bit-for-bit as a FastMultiHeadLut and learns away from it.
            w0, b0 = X.anchor_pair_wb_lutorch(a.heads * a.tph, a.nap, AOBS,
                                              seed=a.seed,
                                              policy=a.hyperplane_anchor_policy,
                                              heads=a.heads)
        else:   # random_torch — torch's other mode: N(0, initial_weights_noise), b = 0
            w0 = jax.random.normal(jax.random.fold_in(ka, 1),
                                   (a.heads * a.tph, a.nap, AOBS)) * TORCH_HP_NOISE
            b0 = jnp.zeros((a.heads * a.tph, a.nap), jnp.float32)
        ap_ = dict(ap_, w=w0, b=b0)
    # Transplanted, frozen addressing. Applied LAST so it overrides every init branch,
    # and shape-checked rather than broadcast: a silent shape mismatch here would train a
    # different architecture than the one being tested.
    if a.freeze_wb_from:
        zf = np.load(a.freeze_wb_from)
        w0, b0 = jnp.asarray(zf["w"]), jnp.asarray(zf["b"])
        if w0.shape != ap_["w"].shape or b0.shape != ap_["b"].shape:
            raise ValueError(
                f"--freeze-wb-from shape mismatch: checkpoint has w{tuple(w0.shape)} "
                f"b{tuple(b0.shape)}, this config needs w{tuple(ap_['w'].shape)} "
                f"b{tuple(ap_['b'].shape)}")
        ap_ = dict(ap_, w=w0, b=b0)

    # The addressing is frozen in anchors mode by design, and in transplant mode by
    # request. One flag, so the gradient-zeroing below cannot disagree with the init.
    freeze_addr = (a.addressing == "anchors") or (a.freeze_wb_from is not None)

    qp = q_init(kq)
    qt = jax.tree.map(lambda x: x, qp)
    log_alpha = jnp.log(jnp.asarray(0.2))

    init_tag = (a.anchor_policy if a.addressing == "anchors"
                else a.hyperplane_init)
    if a.addressing == "hyperplane" and a.hyperplane_init == "anchor_pairs":
        init_tag += f"({a.hyperplane_anchor_policy})"
    if a.freeze_wb_from:
        init_tag = f"FROZEN-wb<-{os.path.basename(a.freeze_wb_from)}"

    n_tables = a.heads * a.tph
    K = 2 ** a.nap
    n_table_params = int(np.prod(ap_["weights"].shape))
    n_idx = int(np.prod(ap_["w"].shape) + np.prod(ap_["b"].shape))
    print(f"LUT-SAC actor nap={a.nap} tph={a.tph} heads={a.heads} "
          f"seed={a.seed} addressing={a.addressing}/{init_tag} | "
          f"12 outputs/cell (6 mu + 6 log-sigma) | table {n_table_params:,} + "
          f"addressing {n_idx:,} = {n_table_params + n_idx:,} params | "
          f"rows {n_tables}x{K} = {n_tables*K:,}", flush=True)
    print(f"  constants: {a.constants} | NC = {NC} | LUT input {OBS} -> {AOBS} "
          f"(actor addressing ONLY; critic and replay stay {OBS}-dim)"
          + ("   [BASELINE: magnitude-blind, no thresholds available]"
             if NC == 0 else ""), flush=True)
    if NC:
        print("  values: " + " ".join(f"{c:+.3f}" for c in const), flush=True)
    if pair_a is not None:
        wiring = wiring_report(pair_a, pair_b, OBS, NC, NAMES, n_rep)

    tx_a = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.actor_lr))
    tx_q = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(a.lr))
    tx_al = optax.adam(a.lr)
    os_a, os_q, os_al = tx_a.init(ap_), tx_q.init(qp), tx_al.init(log_alpha)

    m = W.make_model()
    mx = mjx.put_model(m)
    reset, step = W.make_env(mx)
    v_reset, v_step = jax.vmap(reset), jax.vmap(step)
    st = v_reset(jax.random.split(kr, a.envs))

    # ---- replay buffer (GPU-resident, circular) --------------------------
    N = a.buffer
    buf = dict(s=jnp.zeros((N, OBS)), a=jnp.zeros((N, ACT)), r=jnp.zeros(N),
               s2=jnp.zeros((N, OBS)), d=jnp.zeros(N))
    ptr, size = 0, 0
    row_updates = jnp.zeros((n_tables, K))     # how often each row got a gradient

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

        # --- critic ---
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

        # --- actor (reparameterised, through the verified LUT surrogate) ---
        key, k2 = jax.random.split(key)

        def a_loss(ap_):
            an, logp, _ = actor_sample(ap_, s, k2)
            q = jnp.minimum(q_apply(qp["q1"], ns, an), q_apply(qp["q2"], ns, an))
            return (alpha * logp - q).mean(), logp
        (al, logp), ga = jax.value_and_grad(a_loss, has_aux=True)(ap_)

        # per-row trust region: a row update is a STEP change for every state in
        # that cell, so bound its L2 norm (an MLP has no analogue of this).
        if freeze_addr:
            # no gradient to the addressing — fixed anchor pairs as in FastMHL, or a
            # transplanted routing held fixed on purpose (--freeze-wb-from)
            ga = dict(ga, w=jnp.zeros_like(ga["w"]), b=jnp.zeros_like(ga["b"]))
        if a.row_clip > 0:
            gw = ga["weights"]
            nrm = jnp.linalg.norm(gw, axis=-1, keepdims=True)
            ga = dict(ga, weights=gw * jnp.minimum(1.0, a.row_clip / (nrm + 1e-8)))
        ua, os_a = tx_a.update(ga, os_a, ap_)
        ap_ = optax.apply_updates(ap_, ua)

        # --- temperature ---
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

    @jax.jit
    def det_action(ap_, obs):
        mu, _ = actor_out(ap_, obs)
        return jnp.tanh(mu)

    def eval_mjx(ap_, episodes=20, horizon=1000, seed=0):
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

    extra_ckpts = {int(t) for t in a.checkpoint_at.split(",") if t.strip()}
    if extra_ckpts and any(t % a.eval_every for t in extra_ckpts):
        raise ValueError(f"--checkpoint-at {sorted(extra_ckpts)} must be multiples of "
                         f"--eval-every {a.eval_every}; checkpoints are written on the "
                         f"eval cadence, so any other iteration would silently never fire")

    # Everything an eval needs to reconstruct this actor's input. The VALUES travel with
    # the checkpoint, not just the set name: constants.json could be regenerated with a
    # different rollout or a different seed, and a policy scored against thresholds it
    # was not trained on would look broken for a reason nobody would find.
    ckpt_extra = dict(constants=const.astype(np.float32),
                      const_set=np.str_(a.constants), obs_dim=np.int32(AOBS))
    if pair_a is not None:
        ckpt_extra.update(pair_a=pair_a.astype(np.int32),
                          pair_b=pair_b.astype(np.int32))

    rows_log, t0, best = [], time.time(), -1e9
    total_steps = 0

    # Addressing snapshots. iter 0 is the INIT, recorded before any update, so the
    # diagnostic never has to reconstruct it and cannot drift from what actually ran.
    snaps = []

    def snap(it_done):
        snaps.append(dict(iter=it_done,
                          w=np.asarray(ap_["w"]), b=np.asarray(ap_["b"]),
                          row_updates=np.asarray(row_updates)))
        np.savez_compressed(
            os.path.join(HERE, f"{out_name}_snaps.npz"),
            iters=np.asarray([s["iter"] for s in snaps], np.int32),
            w=np.stack([s["w"] for s in snaps]),
            b=np.stack([s["b"] for s in snaps]),
            row_updates=np.stack([s["row_updates"] for s in snaps]).astype(np.float32))

    if a.snap_every:
        snap(0)

    for it in range(a.iters):
        key, kro = jax.random.split(key)
        st, kro, tr = rollout(ap_, st, kro, it < a.warmup)
        s_, a_, r_, s2_, d_ = [np.asarray(x).reshape((-1,) + x.shape[2:]) for x in tr]
        n = len(s_)
        idx = (ptr + np.arange(n)) % N
        buf = dict(s=buf["s"].at[idx].set(s_), a=buf["a"].at[idx].set(a_),
                   r=buf["r"].at[idx].set(r_), s2=buf["s2"].at[idx].set(s2_),
                   d=buf["d"].at[idx].set(d_))
        ptr = int((ptr + n) % N)
        size = min(size + n, N)
        total_steps += n

        if it >= a.warmup:
            for _ in range(a.updates):
                key, kb = jax.random.split(key)
                bi = jax.random.randint(kb, (a.batch,), 0, size)
                batch = (buf["s"][bi], buf["a"][bi], buf["r"][bi],
                         buf["s2"][bi], buf["d"][bi])
                (ap_, qp, qt, log_alpha, os_a, os_q, os_al, key,
                 info) = update(ap_, qp, qt, log_alpha, os_a, os_q, os_al, batch, key)
                rr = np.asarray(info["rows"])
                flat = (np.arange(n_tables)[None, :] * K + rr).ravel()
                cnt = np.bincount(flat, minlength=n_tables * K).reshape(n_tables, K)
                row_updates = row_updates + jnp.asarray(cnt)

        if a.snap_every and ((it + 1) % a.snap_every == 0 or it == a.iters - 1):
            snap(it + 1)

        if (it + 1) % a.eval_every == 0 or it == a.iters - 1:
            ret = eval_mjx(ap_, episodes=a.eval_episodes)
            cov = float((row_updates > 0).mean())
            best = max(best, ret)
            el = time.time() - t0
            rows_log.append(dict(iter=it + 1, env_steps=total_steps, mjx_return=ret,
                                 row_coverage=cov, alpha=float(info["alpha"]) if it >= a.warmup else None,
                                 elapsed_s=round(el, 1)))
            print(f"[{it+1:>5}/{a.iters}] steps {total_steps:>9,} | MJX ret {ret:8.1f} "
                  f"| row-cov {cov*100:5.1f}% | best {best:8.1f} | {el/60:5.1f}m",
                  flush=True)
            json.dump(dict(iter=it + 1, iters=a.iters, env_steps=total_steps,
                           mjx_return=ret, best=best, row_coverage=cov,
                           eta_s=(a.iters - it - 1) * el / (it + 1), done=False),
                      open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
            np.savez(os.path.join(HERE, f"{out_name}_actor.npz"),
                     **{k: np.asarray(v) for k, v in ap_.items()
                        if k not in ("n_heads", "tph")},
                     n_heads=np.int32(a.heads), tph=np.int32(a.tph), **ckpt_extra)
            if (it + 1) in extra_ckpts:
                np.savez(os.path.join(HERE, f"{out_name}_at{it + 1}_actor.npz"),
                         **{k: np.asarray(v) for k, v in ap_.items()
                            if k not in ("n_heads", "tph")},
                         n_heads=np.int32(a.heads), tph=np.int32(a.tph), **ckpt_extra)
                print(f"  [checkpoint kept at iter {it + 1}]", flush=True)

    json.dump(dict(config=vars(a), constants=const.tolist(), obs_dim=AOBS,
                   wiring=wiring,
                   table_params=n_table_params, index_params=n_idx,
                   total_params=n_table_params + n_idx, total_env_steps=total_steps,
                   wall_s=round(time.time() - t0, 1), best_mjx=best, history=rows_log),
              open(os.path.join(HERE, out_name + ".json"), "w"), indent=1)
    json.dump(dict(iter=a.iters, iters=a.iters, env_steps=total_steps,
                   mjx_return=rows_log[-1]["mjx_return"] if rows_log else 0.0,
                   best=best, row_coverage=rows_log[-1]["row_coverage"] if rows_log else 0.0,
                   eta_s=0.0, done=True),
              open(os.path.join(HERE, out_name + ".partial"), "w"), indent=1)
    print(f"done: best MJX {best:.1f} over {total_steps:,} env-steps in "
          f"{(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
