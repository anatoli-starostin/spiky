"""exp011 substrate: a FastMultiHeadLut candidate, trained by plain backprop.

The chapter's evolutionary loop has only ever been run on SPNet, where the inner learning step
is STDP and every negative result is confounded by "did the substrate even train?". This module
swaps the substrate for one we trust: the REAL `spiky.lutorch.FastMultiHeadLut` -- the
anchor-pair LUT that IS the distillation teacher -- with the inner step replaced by Adam on
random minibatches of the same dataset.

The genome is the LUT's CAPACITY HYPERPARAMETERS, not its weights. Weights are trained from
scratch for every candidate, so selection sees "how well does this ARCHITECTURE fit, once
trained", which is exactly the question "what is the minimal anchor-pair LUT that fits this
dataset".

    param_count = n_heads * tables_per_head * 2^n_anchor_pairs * n_outputs   (+2 learnable temps)

NOTE ON n_heads. The head axis and the tables_per_head axis are the SAME capacity axis here:
the module sums over tables within a head, and a 6-dim target forces us to reduce the head axis
too, so `n_heads=h, tables_per_head=t` is the same function class as `n_heads=1,
tables_per_head=h*t` with the same parameter count. The genome therefore pins n_heads at 1 by
default and moves capacity on tables_per_head; --evolve-heads re-opens it if you want the
redundancy in the search space.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import load                                       # noqa: E402

N_IN, N_OUT = 17, 6

# HARD FORWARD ONLY, FOR TRAINING AND EVALUATION ALIKE.
#
# The hard forward selects one row per table by sign-packing the pairwise differences, which is
# piecewise constant and has a zero-a.e. true derivative. It is still trainable because
# FastMultiHeadLut ships a SOFT SURROGATE backward: the input and temperature gradients come
# from a full K-row softmax surrogate pinned to the chosen row, while the weight gradient is a
# 1-row scatter reflecting the actual forward. So backprop rides the hard forward pass and the
# surrogate gradient -- there is no need for, and no use of, a smooth forward path here.
#
# Keeping it hard everywhere also means TRAINING AND EVALUATION ARE THE SAME FUNCTION. A
# hybrid_smooth-trained, hard-evaluated candidate would be scored on a network it never
# optimised, and the whole point of exp011 is a substrate whose training we do not have to
# reason about. `lut_grad_check.py` verifies all of this empirically rather than assuming it.
FORWARD_MODE = "hard"

# THE ANCHORING AXIS -- enumerated from the implementation, not invented.
#
# `AnchorSamplingPolicy` in spiky.lutorch.lut_helpers defines FOUR members, but
# FastMultiHeadLut.__init__ explicitly REJECTS two of them:
#
#     if policy not in (CANONICAL_FULL_COVERAGE, CANONICAL_DISTINCT):
#         raise ValueError("anchor_sampling_policy must be CANONICAL_FULL_COVERAGE or "
#                          "CANONICAL_DISTINCT, got ...")
#
# so BALANCED and CONNECTED are not options here at all -- they remain in lut_helpers for
# main-branch consumers and there is a test asserting FastMultiHeadLut turns them down. The
# real, supported set is therefore exactly these two:
#
#   canonical_full_coverage  canonical-pool tiled-randperm. Guarantees full coverage of all
#                            C(input_dim, 2) pairs whenever n_tables * NAP >= C(input_dim, 2),
#                            plus a greedy swap-repair pass keeping pairs distinct within a
#                            table across permutation boundaries. (the module default)
#   canonical_distinct       each table independently draws NAP canonical (a<b) pairs without
#                            replacement. Distinct within a table, NO cross-table coverage
#                            guarantee -- so tables can overlap or leave inputs unused.
#
# Both require NAP <= C(input_dim, 2) = C(17, 2) = 136, which NAP_RANGE (max 12) never
# approaches, so every (policy, NAP) combination in the search space is constructible.
#
# The second half of "how anchors are configured" is WHICH pairs get drawn, which is set by
# `random_seed`. That is already an evolvable gene (`anchor_seed`), so the anchoring axis is
# really two genes: the policy that decides the drawing RULE, and the seed that decides the
# DRAW. Both mutate.
#
# NOT included, and worth flagging: soft_score_temp / select_temp / learnable_temps change how
# anchor DIFFERENCES are scored, not how anchors are placed. I read those as a separate
# (temperature) axis rather than an anchoring scheme; say the word if they should be folded in.
ANCHOR_POLICIES = ("canonical_full_coverage", "canonical_distinct")


def _policy(name):
    from spiky.lutorch.lut_helpers import AnchorSamplingPolicy
    return AnchorSamplingPolicy(name)


# ----------------------------------------------------------------- literal anchor pairs
# HOW FastMultiHeadLut STORES ANCHORS. Two int64 BUFFERS (not Parameters), both
# [n_tables, n_anchor_pairs] with a < b in every entry, where n_tables = n_heads *
# tables_per_head:
#
#     self.register_buffer("soft_anchor_a_long", anchor_a_long.contiguous())
#     self.register_buffer("soft_anchor_b_long", anchor_b_long.contiguous())
#
# Every forward path reads them off `self` -- the compiled hard body, the eval fast path and
# the soft surrogate backward all take `self.soft_anchor_a_long, self.soft_anchor_b_long` as
# arguments. Nothing else is derived from the pair VALUES: `soft_bit_matrix` and `soft_powers`
# depend only on NAP. So overwriting the two buffers fully redefines the anchoring.
#
# THE CONSTRUCTOR HAS NO EXPLICIT-PAIRS ARGUMENT -- it only accepts (policy, random_seed) and
# calls get_balanced_anchor_pairs itself. The smallest clean injection is therefore: construct
# normally, then copy_ the evolved pairs into those two buffers. set_anchor_pairs() does that
# and asserts the shapes match, so a genome/shape mismatch is loud rather than silent.
#
# Because they are BUFFERS, they are not in model.parameters() and the optimiser cannot touch
# them: backprop trains the row weights (and temps if learnable) while the evolved anchoring is
# held fixed for the whole of a candidate's training, exactly as the gradient-free/gradient
# split intends. lut_metrics_check.py verifies that the pairs come back bit-identical after
# training rather than assuming it.


# ----------------------------------------------------------------- warm start (Lamarckian)
# THE WEIGHT LAYOUT, confirmed against the module: self.weights is an nn.Parameter of shape
#     [n_tables, table_dim, n_outputs] = [n_heads * tables_per_head, 2^NAP, n_outputs]
#
# THE ROW-DESCENT MAPPING, confirmed empirically (see lut_warmstart_check.py check 1).
# `_msb_powers` is MSB-FIRST -- powers[i] = 2^(NAP-1-i) -- so anchor pair 0 is the most
# significant bit and the LAST pair is the least significant. `resize_pairs` APPENDS new pairs
# at the end of the pair axis, so an added anchor becomes the new LSB and therefore
#
#     old row k  ->  new rows {2k, 2k+1}          i.e. np.repeat(w, 2, axis=1)
#
# Duplicating each parent row into its two children makes the added anchor a PERFECTLY NEUTRAL
# split: whichever way the new bit falls, the row read holds the same values, so the child
# reproduces the parent's function EXACTLY before any training (measured max |diff| = 0.0).
# The alternative mapping (np.tile, k -> {k, k+2^NAP}) does NOT, and is measurably different
# (max |diff| 1.34), so the choice is not ambiguous.
#
# WHY LOW STD FOR GENUINELY NEW CELLS. A new table starts near zero so it contributes almost
# nothing to the summed output: it has to EARN its weight through training rather than arriving
# with a random head start. That matters here because fitness charges for size -- a new table
# that is initialised large would perturb a working function and be charged for it before it
# ever had a chance to help, biasing the search against growth for the wrong reason.


def remap_weights(w, nap_from, nap_to, tables_to, std=1e-4, rng=None):
    """Remap a trained [n_tables, 2^nap_from, n_outputs] weight array into the child's shape.

    NAP is handled first, then the table axis, mirroring the order resize_pairs uses so the
    weights and the anchor pairs stay in step.

      NAP GROWS    each row is repeated 2^g times -> the added anchors are neutral splits and
                   the function is preserved exactly.
      NAP SHRINKS  groups of 2^s consecutive rows are AVERAGED -- the exact inverse of the
                   duplication above, so a grow-then-shrink round trip is the identity when the
                   duplicated rows have not yet diverged.
      tph GROWS    surviving tables keep their weights; new tables start at `std` (near-zero).
      tph SHRINKS  the tail is dropped, survivors keep their weights. This DOES change the
                   function -- the forward sums over tables -- and nothing can prevent that;
                   backprop re-fits the survivors.

    NOTE ON ANCHOR EDITS. When mutation rewires an anchor pair, the row -> input routing
    changes underneath these weights, so an inherited row no longer means quite what it meant
    in the parent. The weights are therefore only APPROXIMATELY valid after a pair edit. That
    is accepted by design: it is a warm start, not an exact transplant, and training refines it.
    """
    rng = np.random.default_rng() if rng is None else rng
    w = np.asarray(w, np.float32)
    if nap_to > nap_from:
        w = np.repeat(w, 1 << (nap_to - nap_from), axis=1)
    elif nap_to < nap_from:
        s = 1 << (nap_from - nap_to)
        w = w.reshape(w.shape[0], w.shape[1] // s, s, w.shape[2]).mean(axis=2)
    n_have = w.shape[0]
    if tables_to < n_have:
        w = w[:tables_to]
    elif tables_to > n_have:
        new = rng.normal(0.0, std, (tables_to - n_have, w.shape[1], w.shape[2]))
        w = np.concatenate([w, new.astype(np.float32)], 0)
    return np.ascontiguousarray(w, np.float32)


def set_weights(model, w):
    with torch.no_grad():
        model.weights.copy_(torch.tensor(np.asarray(w, np.float32),
                                         dtype=model.weights.dtype,
                                         device=model.weights.device))
    return model


def get_weights(model):
    return model.weights.detach().float().cpu().numpy()


def canonical_pool(input_dim=N_IN):
    """All C(input_dim, 2) canonical (a<b) pairs, as an [P, 2] int array."""
    a, b = np.triu_indices(input_dim, 1)
    return np.stack([a, b], 1).astype(np.int64)


def initial_pairs(g, input_dim=N_IN):
    """Seed the genome's pairs using the module's OWN sampler, not a reimplementation.

    The policy gene keeps exactly one job from here on: deciding how the INITIAL pairs are
    drawn. After that, evolution edits pairs directly and the policy is inert -- which is why
    mutate() stops perturbing it once pairs are evolvable.
    """
    from spiky.lutorch.lut_helpers import get_balanced_anchor_pairs
    a, b = get_balanced_anchor_pairs(
        n_tables=n_tables(g), n_anchor_pairs=int(g["n_anchor_pairs"]), input_dim=input_dim,
        device=torch.device("cpu"), random_seed=int(g["anchor_seed"]),
        policy=_policy(g.get("anchor_policy", ANCHOR_POLICIES[0])), n_heads=int(g["n_heads"]))
    return np.stack([a.cpu().numpy(), b.cpu().numpy()], -1).astype(np.int64)


def pairs_valid(pairs, input_dim=N_IN):
    """(a < b), in range, and distinct within each table -- the invariants both samplers hold.

    -> (ok, reason). Within-table distinctness is what CANONICAL_DISTINCT guarantees by
    construction and what CANONICAL_FULL_COVERAGE's swap-repair pass restores, so mutation has
    to preserve it or evolved genomes would be outside the space the module ever produces.
    """
    p = np.asarray(pairs)
    if p.ndim != 3 or p.shape[-1] != 2:
        return False, f"shape {p.shape}, expected [n_tables, nap, 2]"
    if p.min() < 0 or p.max() >= input_dim:
        return False, f"index range {p.min()}..{p.max()} outside [0, {input_dim})"
    if not (p[..., 0] < p[..., 1]).all():
        return False, "not all pairs satisfy a < b"
    key = p[..., 0] * input_dim + p[..., 1]
    for t in range(key.shape[0]):
        if len(np.unique(key[t])) != key.shape[1]:
            return False, f"table {t} has duplicate pairs"
    return True, "ok"


def set_anchor_pairs(model, pairs):
    """Overwrite the module's anchor buffers with evolved pairs. Shapes must already match."""
    p = np.asarray(pairs, np.int64)
    want = tuple(model.soft_anchor_a_long.shape)
    assert p.shape[:2] == want, f"pairs {p.shape[:2]} do not match module anchors {want}"
    dev = model.soft_anchor_a_long.device
    model.soft_anchor_a_long.copy_(torch.tensor(p[..., 0], dtype=torch.int64, device=dev))
    model.soft_anchor_b_long.copy_(torch.tensor(p[..., 1], dtype=torch.int64, device=dev))
    return model


def get_anchor_pairs(model):
    """Read the pairs back out, as an [n_tables, nap, 2] int array."""
    return np.stack([model.soft_anchor_a_long.cpu().numpy(),
                     model.soft_anchor_b_long.cpu().numpy()], -1).astype(np.int64)

# The capacity/optimisation knobs the genome carries. Values here are the reference mid-size
# config: the teacher's own shape (NAP 6 -> 64 rows, 32 tables, 1 head).
DEFAULT_GENOME = dict(
    n_anchor_pairs=6,          # NAP in [1, 15]; each table has 2^NAP rows
    tables_per_head=32,        # tables summed per head
    n_heads=1,                 # see the module docstring: redundant with tables_per_head
    forward_mode="hard",       # THE ONLY MODE USED -- see FORWARD_MODE below
    learnable_temps=False,
    soft_score_temp=0.5,
    select_temp=0.5,
    lr=3e-3,
    anchor_policy="canonical_full_coverage",   # the anchoring axis; see ANCHOR_POLICIES
    anchor_seed=1,             # which anchor-pair draw this candidate got
)

# NAP is capped at 15 by the module, but 2^NAP * tables_per_head * 6 floats has to fit and be
# trainable in a few thousand steps. 12 gives 4096 rows/table, already 1.5M params at 64 tables.
NAP_RANGE = (1, 12)
TPH_RANGE = (1, 128)
HEADS_RANGE = (1, 8)
LR_RANGE = (3e-4, 3e-2)


def param_count(g):
    n = g["n_heads"] * g["tables_per_head"] * (1 << g["n_anchor_pairs"]) * N_OUT
    # +1 for the trainable readout temperature, counted honestly even though it is one scalar
    # against ~10^4 weights and cannot move the size story.
    return int(n + (2 if g["learnable_temps"] else 0) + (1 if g.get("train_tau", True) else 0))


def n_tables(g):
    return int(g["n_heads"] * g["tables_per_head"])


def max_tph_within(n_heads, max_throughput):
    """Largest tables_per_head whose throughput stays inside a budget.

    THE CONSTRAINT IS EXACTLY A BOUND ON tph. throughput = n_heads * tph * n_outputs is
    deterministic, monotone and depends on NOTHING else -- not NAP, not the anchor pairs, not
    the weights. So "reject any candidate over budget" and "never propose a tph above
    max_throughput / (n_heads * n_outputs)" accept precisely the same set of genomes. Bounding
    at draw time is the cheaper of the two: a rejected candidate would otherwise cost 2000
    backprop steps before being thrown away.

    Genes are CLAMPED into the bound rather than redrawn, matching how NAP_RANGE and TPH_RANGE
    are already handled everywhere else in this chapter. That does concentrate members at the
    boundary -- but selection already wants maximum tables (the full run pinned 47/48 members at
    the cap), so the clamp is not what creates that, it just moves where it happens.
    """
    if max_throughput is None:
        return TPH_RANGE[1]
    return max(1, min(TPH_RANGE[1], int(max_throughput) // (int(n_heads) * N_OUT)))


def throughput(g):
    """Weight ENTRIES read per forward pass, per sample.

        throughput = n_heads * tables_per_head * n_outputs        -- INDEPENDENT OF NAP

    Why: the hard forward sign-packs each table's NAP differences into ONE row index, then
    F.embedding_bag gathers exactly that one row (n_outputs floats) from each of the
    n_heads*tables_per_head tables and sums. NAP sets how many rows EXIST per table (2^NAP,
    which is what makes param_count exponential) but not how many are READ, which is always 1.

    So the two costs diverge sharply: doubling NAP doubles nothing in throughput while
    doubling the parameter count. This is verified empirically in lut_metrics_check.py by
    counting non-zero weight-gradient entries at batch size 1 -- in hard mode the weight
    gradient is a 1-row scatter at the chosen row, so its support IS the set of entries the
    forward read.
    """
    return int(g["n_heads"] * g["tables_per_head"] * N_OUT)


# The distillation teacher's own shape, the reference point every evolved member is compared
# against: FastMultiHeadLut(exp_outputs=True), NAP 6 (64 rows) x 32 tables x 1 head.
TEACHER_SHAPE = dict(n_anchor_pairs=6, tables_per_head=32, n_heads=1)


def teacher_anchor_pairs():
    """The teacher's ACTUAL anchor pairs, read from the dataset the teacher generated.

    WHY NOT JUST SEED IT. The teacher's anchors are canonical_full_coverage with random_seed=1
    -- but `get_balanced_anchor_pairs` seeds a `torch.Generator(device=...)`, and the CUDA and
    CPU streams differ. exp011 builds on CUDA, so passing anchor_seed=1 produced a DIFFERENT
    draw, and every teacher comparison before this was against the teacher's SHAPE with random
    anchors rather than against the teacher. Measured: the CPU draw is bit-identical to the
    checkpoint's `soft_anchor_a_long`/`soft_anchor_b_long`, the CUDA draw is not.

    Reading the pairs out of the .npz removes the device question entirely -- these are the
    arrays the collector wrote straight from the live module, so they are the teacher's by
    construction, whatever device anything is built on later.
    """
    from data import NPZ
    Z = np.load(NPZ)
    return np.stack([Z["anchor_a"], Z["anchor_b"]], -1).astype(np.int64)


def teacher_genome():
    """The reference: the teacher's shape AND its actual anchors AND its tau."""
    return dict(DEFAULT_GENOME, **TEACHER_SHAPE,
                anchor_pairs=teacher_anchor_pairs(), tau=TEACHER_TAU)


def dominates_teacher(mse, params, tput, t_mse, t_params, t_tput, tol=0.0):
    """Does this member Pareto-dominate the teacher?

    Fit no worse than the teacher (within `tol` of its MSE), strictly better on at least one of
    (params, throughput), and no worse on the other. Returned as a dict so the log can say
    WHICH way it wins rather than just yes/no.
    """
    fit_ok = mse <= t_mse + tol
    p_better, p_worse = params < t_params, params > t_params
    t_better, t_worse = tput < t_tput, tput > t_tput
    return dict(fit_ok=bool(fit_ok),
                params_better=bool(p_better), throughput_better=bool(t_better),
                dominates=bool(fit_ok and (p_better or t_better)
                               and not p_worse and not t_worse),
                both_better=bool(fit_ok and p_better and t_better))


def genome_str(g):
    pol = g.get("anchor_policy", ANCHOR_POLICIES[0]).replace("canonical_", "")
    return (f"NAP {g['n_anchor_pairs']:2d} (2^{g['n_anchor_pairs']}={1 << g['n_anchor_pairs']:5d} "
            f"rows) x tph {g['tables_per_head']:3d} x heads {g['n_heads']} "
            f"anchors {pol:13s} lr {g['lr']:.4g}  -> {param_count(g):,} params")


def build(g, device="cuda", init_weights=None):
    """The REAL FastMultiHeadLut. Nothing here is a reimplementation.

    `init_weights` (warm start) must already be in THIS genome's shape -- call remap_weights
    first. Absent, the engine's own random init stands, which is the cold-start default.
    """
    from spiky.lutorch.fast_multi_head_lut import FastMultiHeadLut
    assert g.get("forward_mode", FORWARD_MODE) == FORWARD_MODE, (
        f"exp011 is hard-forward only (see FORWARD_MODE); got {g.get('forward_mode')!r}")
    m = FastMultiHeadLut(
        input_dim=N_IN, n_heads=g["n_heads"], n_outputs=N_OUT,
        n_anchor_pairs=g["n_anchor_pairs"], tables_per_head=g["tables_per_head"],
        forward_mode=FORWARD_MODE,
        weight_dtype=torch.float32,
        use_bf16=bool(g.get("use_bf16", True)),
        anchor_sampling_policy=_policy(g.get("anchor_policy", ANCHOR_POLICIES[0])),
        soft_score_temp=g["soft_score_temp"], select_temp=g["select_temp"],
        learnable_temps=g["learnable_temps"],
        random_seed=int(g["anchor_seed"]),
        device=torch.device(device))
    # LITERAL ANCHORING: explicit evolved pairs replace whatever the policy+seed drew. Absent,
    # the module's own sampling stands, which is the previous behaviour exactly.
    if g.get("anchor_pairs") is not None:
        set_anchor_pairs(m, g["anchor_pairs"])
    # WARM START: inherited weights, already remapped to this genome's shape by the caller.
    if init_weights is not None:
        assert tuple(np.asarray(init_weights).shape) == tuple(m.weights.shape), (
            f"inherited weights {np.asarray(init_weights).shape} do not match "
            f"{tuple(m.weights.shape)} -- remap_weights was not applied or used a wrong shape")
        set_weights(m, init_weights)
    # THE READOUT TEMPERATURE. Defaults to the teacher's tau and is TRAINABLE by default
    # (parameterised as log-tau so it stays positive); --fixed-tau freezes it.
    #
    # Trainable is the deliberate choice, not an oversight. tau is ONE scalar against 10^4
    # weights, so it cannot meaningfully change the size story, and freezing it at a value the
    # teacher fitted for ITS anchors would hand every differently-anchored candidate an
    # arbitrary handicap -- the whole point of the experiment is that anchors differ.
    lt = torch.tensor(float(np.log(g.get("tau", TEACHER_TAU))), device=torch.device(device))
    if g.get("train_tau", True):
        m.lse_log_tau = torch.nn.Parameter(lt)
    else:
        m.register_buffer("lse_log_tau", lt)
    return m


# ----------------------------------------------------------------- the readout
# THE TARGET IS A LOG-SUM-EXP, NOT A SUM. The exp19 teacher computes
#
#     y = tph * tau * ( logsumexp_t( clip(W[t, addr_t] / tau, -60, 60) ) - log(tph) )
#
# while the engine's only reduction is F.embedding_bag(mode='sum'). Students trained under a
# sum were fitting a different function class from the one that generated the data.
#
# THE ENGINE HAS NO LSE PATH IN THIS CHECKOUT. `_FORWARD_MODES` is ("hard", "hybrid_smooth"),
# every reduction in the file is embedding_bag(mode='sum'), and `git log -S exp_outputs --
# src/spiky/lutorch/` is empty -- the `_exp_outputs_fwd` the dataset README quotes belongs to
# the exp19-era file, which this repo replaced in af82d49e. So the LSE cannot be delegated to
# the engine; it is applied here, on top of the engine's OWN addressing.
#
# WHY THAT IS STILL FAITHFUL, AND NOT A REIMPLEMENTATION OF THE LUT. We take the row index
# straight from the engine's compiled hard forward (`_soft_lut_fwd_body`), so the sign-pack,
# the MSB-first packing and the anchor routing are bit-identical to the engine's. Only the
# reduction over tables changes. And the gradient is right: the only trainable tensors are the
# row weights (and log-tau), the addresses are discrete and carry no gradient to x anyway, and
# gathering the selected rows gives autograd exactly the 1-row-per-table scatter that hard mode
# produces. With the LSE applied outside, the weight gradient is in fact EXACT given the
# addresses rather than a surrogate.
EXP_CLAMP = 60.0
TEACHER_TAU = 0.09036567807197571


def _row_index(model, x):
    """The engine's own sign-pack row index, [B, n_tables]. No reimplementation."""
    from spiky.lutorch.fast_multi_head_lut import _soft_lut_fwd_body
    _, idx = _soft_lut_fwd_body(x, model.weights, model.soft_anchor_a_long,
                                model.soft_anchor_b_long, model.soft_powers,
                                model.n_heads, model.tables_per_head, model.table_dim)
    return idx


def _fwd(model, x):
    """[B, n_outputs] under the TEACHER'S readout.

    `model.lse_log_tau` is attached by build(); tph * tau * (logsumexp - log tph) reduces over
    the table axis exactly as the teacher does. n_heads is pinned at 1 in this experiment, so
    the table axis and the head axis coincide.
    """
    idx = _row_index(model, x)                                   # [B, n_tables]
    T = idx.shape[1]
    w_sel = model.weights[torch.arange(T, device=idx.device)[None, :], idx]   # [B, T, n_out]
    tau = model.lse_log_tau.exp()
    z = torch.clamp(w_sel.float() / tau, -EXP_CLAMP, EXP_CLAMP)
    n = float(model.tables_per_head)
    return (n * tau * (torch.logsumexp(z, dim=1) - math.log(n))).float()


@torch.no_grad()
def evaluate(model, X, Y, batch=4096):
    se, n = 0.0, 0
    for i in range(0, X.shape[0], batch):
        p = _fwd(model, X[i:i + batch])
        se += float(((p - Y[i:i + batch]) ** 2).sum())
        n += p.numel()
    return se / n


def train_eval(g, Xtr, Ytr, Xte, Yte, steps=2000, batch=512, seed=0, device="cuda",
               eval_every=0, log=None, init_weights=None, return_weights=False):
    """Train ONE candidate from scratch and return its held-out MSE and size.

    Weights are always freshly initialised: the genome is the architecture, so a candidate's
    score has to be "what this architecture reaches when trained", not "what these particular
    weights happen to be worth". That is also what makes the score comparable across the pool
    when the pool contains different shapes.
    """
    torch.manual_seed(seed)
    model = build(g, device, init_weights=init_weights)
    # pre-training loss: the number that shows whether a warm start actually carried anything
    with torch.no_grad():
        pre = evaluate(model, Xte, Yte)
    opt = torch.optim.Adam(model.parameters(), lr=float(g["lr"]))
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    t0 = time.time()
    curve = []
    for s in range(steps):
        idx = torch.from_numpy(rng.integers(0, n, batch)).to(device)
        loss = torch.nn.functional.mse_loss(_fwd(model, Xtr[idx]), Ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if eval_every and ((s + 1) % eval_every == 0 or s == 0):
            m = evaluate(model, Xte, Yte)
            lv = float(loss.detach())
            curve.append(dict(step=s + 1, train_batch_mse=lv, heldout_mse=m))
            if log:
                print(f"      step {s + 1:5d}  batch {lv:.5f}  held-out {m:.5f}", flush=True)
    # The returned genome is JSON-safe: the anchor-pair ARRAY is replaced by its shape plus a
    # stable digest. Carrying the full array for every candidate every round would bloat the
    # history by orders of magnitude (a 128x12 genome is 1,536 pairs), while the digest is
    # enough to tell whether two candidates share an anchoring. The live genome keeps the array.
    gj = {k: (float(v) if isinstance(v, float) else v)
          for k, v in g.items() if k != "anchor_pairs"}
    if g.get("anchor_pairs") is not None:
        ap = np.asarray(g["anchor_pairs"], np.int64)
        gj["anchor_pairs_shape"] = list(ap.shape)
        gj["anchor_pairs_digest"] = int(hash(ap.tobytes()) & 0xFFFFFFFF)
    out = dict(genome=gj,
               params=param_count(g), steps=steps, batch=batch,
               heldout_mse=evaluate(model, Xte, Yte),
               train_mse=evaluate(model, Xtr[:20000], Ytr[:20000]),
               pretrain_heldout_mse=pre, warm_started=init_weights is not None,
               seconds=round(time.time() - t0, 1), curve=curve)
    if return_weights:
        out["weights"] = get_weights(model)
    del model, opt
    torch.cuda.empty_cache()
    return out


def to_device(seed, n_val, device="cuda"):
    """The chapter's split, moved to the GPU once. x_norm -> y_action_mean."""
    _, _, Xp, Yp, Xv, Yv = load(64, seed, n_val)
    t = lambda a: torch.tensor(np.asarray(a, np.float32), device=device)   # noqa: E731
    return t(Xp), t(Yp), t(Xv), t(Yv)


def baselines(Ytr, Yte):
    """What "fits" has to beat: predict the training mean for everything."""
    mu = Ytr.mean(0, keepdim=True)
    return dict(constant_predictor_mse=float(((Yte - mu) ** 2).mean()),
                target_var=float(Yte.var()),
                target_sd=float(Yte.std()))


def main():
    ap = argparse.ArgumentParser(description="exp011 sanity: train single LUT configs")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--n-val", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--nap", type=int, nargs="+", default=None)
    ap.add_argument("--tph", type=int, nargs="+", default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xtr, Ytr, Xte, Yte = to_device(a.seed, a.n_val, dev)
    base = baselines(Ytr, Yte)
    print(f"exp011 substrate check: {Xtr.shape[0]:,} train / {Xte.shape[0]:,} held-out states, "
          f"target sd {base['target_sd']:.4f}, dev {dev}")
    print(f"  BASELINE constant predictor held-out MSE {base['constant_predictor_mse']:.5f}\n")

    if a.nap and a.tph:
        grid = list(zip(a.nap, a.tph))
    else:
        # small -> large, spanning three orders of magnitude in parameter count
        grid = [(4, 4), (5, 8), (6, 32), (8, 32), (10, 32)]
    rows = []
    for nap, tph in grid:
        g = dict(DEFAULT_GENOME, n_anchor_pairs=nap, tables_per_head=tph,
                 use_bf16=not a.no_bf16)
        if a.lr:
            g["lr"] = a.lr
        print(f"  {genome_str(g)}")
        r = train_eval(g, Xtr, Ytr, Xte, Yte, a.steps, a.batch, a.seed, dev,
                       eval_every=a.eval_every, log=True)
        r["baselines"] = base
        rows.append(r)
        print(f"    -> held-out MSE {r['heldout_mse']:.5f}  train {r['train_mse']:.5f}  "
              f"({r['params']:,} params, {r['seconds']}s)\n", flush=True)

    print(f"{'NAP':>4} {'tph':>4} {'params':>10} {'held-out MSE':>13} {'vs constant':>12}")
    for r in rows:
        g = r["genome"]
        print(f"{g['n_anchor_pairs']:4d} {g['tables_per_head']:4d} {r['params']:10,} "
              f"{r['heldout_mse']:13.5f} "
              f"{r['heldout_mse'] / base['constant_predictor_mse']:11.3f}x")
    if a.out:
        json.dump(dict(baselines=base, rows=rows), open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
