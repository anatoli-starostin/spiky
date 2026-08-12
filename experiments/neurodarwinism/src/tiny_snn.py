"""exp012 substrate: a TINY spnet whose every synapse is a gene. No STDP anywhere.

WHAT IS DIFFERENT FROM THE REST OF THE CHAPTER. Every earlier experiment evolved a
*scaffold* (which input wires where, with what delay) and let STDP settle the weights
inside an 800+200 reservoir. Here there is no reservoir, no settling and no plasticity:

    17 input  ->  8 excitatory + 2 inhibitory hidden  ->  6 output          (33 neurons)

and the genome IS the network -- topology, per-synapse delay and per-synapse weight, all
mutated directly. `learning_rate=0` on every SynapseMeta, `do_train=False` on every
episode, so the weights the genome ships are the weights that run. Classic evolutionary
optimisation of the weights themselves.

DENSE GENOME. 27 possible sources (17 in + 8 exc + 2 inh) x 16 possible targets
(8 exc + 2 inh + 6 out) = 432 cells, of which 330 are LEGAL: outputs are sinks (they source
nothing), inputs project only to hidden units (no one-hop input->output shortcut), and
self-loops on hidden units are allowed. That is small enough to carry the WHOLE adjacency
matrix as three [27, 16] arrays -- a presence mask, a delay and a weight -- so "evolve the
topology" is literally a bitmask mutation and no bookkeeping of edge lists is needed.

DALE'S LAW IS STRUCTURAL, NOT POLICED. The sign of a synapse is a property of its SOURCE
ROW and is never itself a gene: rows 0..24 (inputs and excitatory hidden) are clipped to
[0, +w_ceiling], rows 25..26 (inhibitory hidden) to [-w_ceiling, 0]. A mutation cannot move
a weight across zero, so the law holds for every genome ever constructed rather than being
restored by a repair pass.

TIMING, 96 ticks: [0,32) latency-coded input, [32,64) computation, [64,96) readout, and the
readout is `--readout-window 32` -- each output's first spike inside [64,96) re-based to
0..31, a silent output reading 32.

ONE UNIFORM DELAY RANGE, [1, 64], for every synapse regardless of what it connects. No output
gate and no per-class ranges. Reaching the readout window is still something selection has to
discover -- the silent fraction at round 0 versus at the end of a run measures whether it does
-- but unlike [1, 32] the window is now reachable about equally often from every input tick,
so the latency code keeps its monotonicity. See the reachability table below.

OBJECTIVE. -MSE against exp009's quantised centred target (`steady_state.target_offsets`,
imported so it is bit-identical): centre and scale per dimension on the TRAINING POOL, clip
at +-2.5 sigma, quantise to 32 levels, `1-u` so the largest action targets offset 0. The
reference points that target comes with:  constant predictor 39.19,  exp009's best
MSE-trained 800-excitatory STDP net 37.52.
"""
import numpy as np
import torch

from harness import (GROUP_SIZE, LatencyEncoder, N_IN, N_OUT, N_TICKS, T_IN,
                     kendall_tau_b, run_episode)
import steady_state
from steady_state import fit_target_stats, target_offsets   # noqa: F401
# NB: TARGET_STATS is deliberately NOT imported by name. fit_target_stats() rebinds it as a
# module global, and a `from ... import` would freeze the None it holds at import time.

# ------------------------------------------------------------------ geometry
N_EXC, N_INH = 8, 2
N_SRC = N_IN + N_EXC + N_INH            # 27 -- inputs and both hidden pools may project
N_TGT = N_EXC + N_INH + N_OUT           # 16 -- hidden and outputs may be projected onto
ROW_IN = slice(0, N_IN)                 # 0..16   excitatory sign
ROW_EXC = slice(N_IN, N_IN + N_EXC)     # 17..24  excitatory sign
ROW_INH = slice(N_IN + N_EXC, N_SRC)    # 25..26  INHIBITORY sign
COL_EXC = slice(0, N_EXC)               # 0..7
COL_INH = slice(N_EXC, N_EXC + N_INH)   # 8..9
COL_OUT = slice(N_EXC + N_INH, N_TGT)   # 10..15
READOUT_WINDOW = 32

# ---- DELAY RANGE. ONE uniform range for every synapse, whatever it connects.
# No output gate, no per-class ranges, no architectural help of any kind: landing an output
# spike inside the readout window [64, 96) is left entirely to evolution.
#
# WHY [1, 64] AND NOT [1, 32]. A two-hop input->output path fires at input_tick + d1 + d2, so
# what matters is not the range's width but whether every input tick can reach the readout
# window [64, 96) about equally often. Counting the (d1, d2) pairs that land in-window:
#
#     input tick        0      8     16     24     31
#     [1,32]  (1024)    1     45    153    325    528     <- a 528x bias
#     [1,64]  (4096) 1582   1719   1790   1751   1520     <- flat to within 1.18x
#
# The chapter's convention is EARLIER SPIKE = LARGER VALUE, so under [1,32] the observations
# the code puts first were exactly the ones the readout could almost never see, and the
# latency code lost its monotonicity. [1,64] removes that bias without adding any gate: the
# sum of two draws now spans the window from either end of the input phase.
#
# exp012 defines this locally rather than rebinding harness's D_MIN/D_MAX. Those two names
# are shared by exp001-exp011 and by steady_state.py, and changing them there would silently
# alter every other experiment in the chapter; harness.py is left untouched and nothing here
# reads anything delay-related from it.
D_LO, D_HI = 1, 64
N_METAS = D_HI - D_LO + 1               # 64

# a source row's sign, +1 or -1, fixed for the lifetime of every genome
SIGN = np.ones(N_SRC, np.float64)
SIGN[ROW_INH] = -1.0
# per-target-column delay bounds. Uniform now, but kept as arrays because init, mutation and
# delays_ok all index through them -- so reintroducing a per-class range is a two-line change.
DLY_LO = np.full(N_TGT, D_LO, np.int64)
DLY_HI = np.full(N_TGT, D_HI, np.int64)

# WHICH EDGES MAY EXIST AT ALL, and WHICH DELAYS ARE FIXED.
#
# The two inhibitory units are constrained into a PURE FAST LATERAL-INHIBITION POOL:
#   * inputs reach them not at all      (in->inh illegal)
#   * they reach the outputs not at all (inh->out illegal)
#   * they reach excitatory units at delay EXACTLY 1, always     (inh->exc pinned)
# so the only thing they can do is see what the excitatory pool just did (via exc->inh) and
# push back on it essentially immediately. exc->inh, inh->inh and inh->exc all stay legal;
# the weights on the pinned synapses still evolve normally, Dale-negative.
#
# Inputs also still cannot reach the outputs directly -- that would let the readout bypass the
# hidden layer entirely.
#
# All three are OPERATOR INVARIANTS, in the same sense as Dale's law: enforced at
# initialisation, after every mutation and after every crossover, so no genome that can be
# constructed violates them. They are not repaired ad hoc and they are not merely asserted.
LEGAL = np.ones((N_SRC, N_TGT), bool)
LEGAL[ROW_IN, COL_OUT] = False
LEGAL[ROW_IN, COL_INH] = False
LEGAL[ROW_INH, COL_OUT] = False

# cells whose delay is pinned, and the value it is pinned to
PIN_DELAY = np.zeros((N_SRC, N_TGT), bool)
PIN_DELAY[ROW_INH, COL_EXC] = True
PIN_VALUE = 1


def enforce(g):
    """Re-impose the structural invariants on a genome, in place. -> the same genome.

    Called at the end of random_genome, mutate and crossover -- the three places a genome can
    come into existence. Anything that produces a genome any other way is a bug.
    """
    g["mask"] &= LEGAL
    g["delay"][PIN_DELAY] = PIN_VALUE
    return g

def metas(w_ceiling):
    """One SynapseMeta per discrete delay in [1, 64]. 64 metas, index d - 1.

    A meta in this experiment IS just a delay -- every other field is identical across the
    bank -- so there is exactly one per delay and no piecewise indexing anywhere. (Worth
    knowing if a per-class range ever comes back: `register_synapse_meta` DEDUPLICATES metas
    that agree on every field and returns the existing id, so two banks whose ranges overlap
    trip `assert m_id == i` at spnet.py:105.)

    learning_rate 0 everywhere -- this experiment has no plasticity at all. The weight bounds
    are wide and SIGNED (unlike harness.delay_metas' [0, 45]) because inhibitory synapses
    carry negative weights and a tiny net needs single-synapse weights an order of magnitude
    larger than a 100-fan-in reservoir does.
    """
    from spiky.spnet.spnet import SynapseMeta
    return [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d,
                        initial_weight=0.0, min_weight=-w_ceiling, max_weight=w_ceiling,
                        initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                        _forward_group_size=GROUP_SIZE, _backward_group_size=GROUP_SIZE)
            for d in range(D_LO, D_HI + 1)]


# ------------------------------------------------------------------ genome
def random_genome(rng, w_max, p_init=0.5):
    """Dense [27,16] adjacency, masked to LEGAL. Sign comes from the row and is never drawn."""
    mask = (rng.random((N_SRC, N_TGT)) < p_init) & LEGAL
    dly = rng.integers(DLY_LO[None, :], DLY_HI[None, :] + 1, (N_SRC, N_TGT))
    w = rng.uniform(0.0, w_max, (N_SRC, N_TGT)) * SIGN[:, None]
    return enforce(dict(mask=mask, delay=dly.astype(np.int64), weight=w))


def mutate(g, rng, w_max, p_add=0.02, p_prune=0.02, p_delay=0.08, p_weight=0.25,
           w_sigma=0.15, w_ceiling=None):
    """Topology, delay and weight all mutate; the sign of a row never does.

    A new edge starts at 0.10..0.60 of w_max rather than at zero, so that switching it on is
    a change selection can actually see in the same round it happens.
    """
    ceil = 1.5 * w_max if w_ceiling is None else w_ceiling
    h = {k: v.copy() for k, v in g.items()}

    off = (~h["mask"]) & LEGAL          # an illegal edge is never a candidate for switching on
    add = off & (rng.random((N_SRC, N_TGT)) < p_add)
    if add.any():
        n = int(add.sum())
        h["mask"][add] = True
        h["weight"][add] = rng.uniform(0.10, 0.60, n) * w_max * np.broadcast_to(
            SIGN[:, None], (N_SRC, N_TGT))[add]
        h["delay"][add] = rng.integers(np.broadcast_to(DLY_LO[None, :], (N_SRC, N_TGT))[add],
                                       np.broadcast_to(DLY_HI[None, :] + 1, (N_SRC, N_TGT))[add])

    prune = h["mask"] & (rng.random((N_SRC, N_TGT)) < p_prune)
    h["mask"][prune] = False

    # pinned cells are excluded from the delay walk outright, rather than walked and then
    # snapped back -- so a pinned synapse never even spends a mutation draw
    m = h["mask"] & (~PIN_DELAY) & (rng.random((N_SRC, N_TGT)) < p_delay)
    if m.any():
        step = rng.choice([-1, 1], int(m.sum()))
        lo = np.broadcast_to(DLY_LO[None, :], (N_SRC, N_TGT))[m]
        hi = np.broadcast_to(DLY_HI[None, :], (N_SRC, N_TGT))[m]
        h["delay"][m] = np.clip(h["delay"][m] + step, lo, hi)

    m = h["mask"] & (rng.random((N_SRC, N_TGT)) < p_weight)
    h["weight"][m] += rng.normal(0.0, w_sigma * w_max, int(m.sum()))
    # DALE: clip each row into its own half-line. A weight can reach 0 but never cross it.
    h["weight"][:N_IN + N_EXC] = np.clip(h["weight"][:N_IN + N_EXC], 0.0, ceil)
    h["weight"][N_IN + N_EXC:] = np.clip(h["weight"][N_IN + N_EXC:], -ceil, 0.0)
    return enforce(h)


def crossover(g1, g2, rng):
    """UNIFORM per-cell recombination. Each (src,tgt) cell is one coherent gene bundle.

    For every cell independently, a fair coin picks a parent and the child takes that parent's
    (mask, weight, delay) for the cell TOGETHER. Presence, strength and timing of one synapse
    therefore never come from different parents -- splitting them would let the child inherit
    a weight that was tuned against a delay it does not have, which is not recombination of
    traits but destruction of them.

    BOTH INVARIANTS SURVIVE FOR FREE, which is why the dense representation was worth having:
      * Dale -- the sign of a weight is a function of its SOURCE ROW, and both parents obey
        it, so whichever parent a cell comes from the sign is already correct for that row.
      * legality -- a cell absent from LEGAL is masked off in both parents, so it is masked
        off in the child whichever coin lands.
    Both are asserted rather than assumed, because "for free" is exactly the kind of claim
    that quietly stops being true.

    Parents are not modified.
    """
    pick = rng.random((N_SRC, N_TGT)) < 0.5          # True -> take g1, False -> take g2
    child = dict(mask=np.where(pick, g1["mask"], g2["mask"]),
                 delay=np.where(pick, g1["delay"], g2["delay"]),
                 weight=np.where(pick, g1["weight"], g2["weight"]))
    enforce(child)
    assert dale_ok(child)[0], "crossover produced a Dale violation"
    assert legal_ok(child)[0], "crossover produced an illegal edge"
    assert pins_ok(child)[0], "crossover produced an unpinned inh->exc delay"
    return child


def bundle_coherent(child, g1, g2):
    """Every child cell's (mask, weight, delay) came from ONE parent, not a mix.

    -> (ok, n_incoherent). The check that makes `crossover` more than a docstring: a cell is
    coherent if the whole triple matches g1's or the whole triple matches g2's.
    """
    def same(a, b):
        return ((a["mask"] == b["mask"]) & (a["delay"] == b["delay"])
                & (a["weight"] == b["weight"]))
    bad = int((~(same(child, g1) | same(child, g2))).sum())
    return bad == 0, bad


def dale_ok(g):
    """Structural invariant, asserted rather than assumed. -> (ok, n_violations)."""
    w = g["weight"]
    bad = int((w[:N_IN + N_EXC] < 0).sum() + (w[N_IN + N_EXC:] > 0).sum())
    return bad == 0, bad


def delays_ok(g):
    """Every PRESENT synapse's delay inside its own target column's range."""
    d, m = g["delay"], g["mask"]
    lo = np.broadcast_to(DLY_LO[None, :], (N_SRC, N_TGT))
    hi = np.broadcast_to(DLY_HI[None, :], (N_SRC, N_TGT))
    bad = int(((d < lo) | (d > hi))[m].sum())
    return bad == 0, bad


def legal_ok(g):
    """No genome may carry an edge outside LEGAL. -> (ok, n_violations)."""
    bad = int((g["mask"] & ~LEGAL).sum())
    return bad == 0, bad


def pins_ok(g):
    """Every PRESENT pinned synapse sits at exactly PIN_VALUE. -> (ok, n_violations)."""
    bad = int(((g["delay"] != PIN_VALUE) & PIN_DELAY & g["mask"]).sum())
    return bad == 0, bad


def structure_counts(g):
    """The three counts the invariants are about, for a genome. All should be as stated."""
    m = g["mask"]
    pin = m & PIN_DELAY
    return dict(n_in_to_inh=int(m[ROW_IN, COL_INH].sum()),          # must be 0
                n_inh_to_out=int(m[ROW_INH, COL_OUT].sum()),        # must be 0
                n_in_to_out=int(m[ROW_IN, COL_OUT].sum()),          # must be 0
                n_inh_to_exc=int(pin.sum()),
                inh_to_exc_delays=sorted(set(g["delay"][pin].tolist())),
                n_exc_to_inh=int(m[ROW_EXC, COL_INH].sum()),
                n_inh_to_inh=int(m[ROW_INH, COL_INH].sum()))


def genome_stats(g):
    m = g["mask"]
    w = np.abs(g["weight"])[m]
    return dict(n_syn=int(m.sum()),
                n_in_hidden=int(m[ROW_IN].sum()),
                n_rec=int(m[N_IN:, COL_EXC].sum() + m[N_IN:, COL_INH].sum()),
                n_out=int(m[:, COL_OUT].sum()),
                n_inh=int(m[ROW_INH].sum()),
                w_mean=float(w.mean()) if w.size else 0.0,
                w_max=float(w.max()) if w.size else 0.0)


# ------------------------------------------------------------------ build
def build(genomes, device="cuda", seed=1, w_ceiling=200.0):
    """Pack P tiny candidates into ONE SpikingNet, disjoint id blocks per candidate.

    Same one-chunk / one-add_connections / shuffle-off recipe as harness.build (see its note
    on why the reservoir and the I/O cannot be split into separate chunks); the difference is
    that here there is no reservoir to keep frozen -- every synapse comes from the genome.
    """
    from spiky.spnet.spnet import SpikingNet, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine

    P = len(genomes)
    neuron_metas = [NeuronMeta(neuron_type=0, a=0.02, d=8.0),      # excitatory hidden, RS
                    NeuronMeta(neuron_type=1, a=0.1, d=2.0),       # inhibitory hidden, FS
                    NeuronMeta(neuron_type=2, a=0.02, d=8.0),      # input
                    NeuronMeta(neuron_type=3, a=0.02, d=8.0)]      # output
    counts = [P * N_EXC, P * N_INH, P * N_IN, P * N_OUT]
    spnet = SpikingNet(synapse_metas=metas(w_ceiling), neuron_metas=neuron_metas,
                       neuron_counts=counts, initial_synapse_capacity=1 << 20,
                       summation_dtype=torch.float32)
    spnet.to_device(device)
    ids = [spnet.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(4)]

    tri, wts = [], []
    for c, g in enumerate(genomes):
        E = ids[0][c * N_EXC:(c + 1) * N_EXC]
        I = ids[1][c * N_INH:(c + 1) * N_INH]
        In = ids[2][c * N_IN:(c + 1) * N_IN]
        Ou = ids[3][c * N_OUT:(c + 1) * N_OUT]
        src_ids = np.concatenate([In, E, I])          # rows 0..26
        tgt_ids = np.concatenate([E, I, Ou])          # cols 0..15
        r, col = np.nonzero(g["mask"])
        if r.size == 0:
            continue
        d = g["delay"][r, col]
        meta = d - D_LO                       # one bank, so the mapping is no longer piecewise
        tri.append(np.stack([meta, src_ids[r], tgt_ids[col]], 1))
        wts.append(g["weight"][r, col].astype(np.float64))

    triples = np.concatenate(tri, 0) if tri else np.zeros((0, 3), np.int64)
    weights = np.concatenate(wts, 0) if wts else np.zeros(0, np.float64)
    # (source, target) must be unique. The dense genome cannot produce a duplicate within a
    # candidate and the id blocks are disjoint across candidates, so this is an assertion.
    key = triples[:, 1].astype(np.int64) * (1 << 22) + triples[:, 2]
    assert len(np.unique(key)) == len(key), "duplicate (src,tgt) -- genome packing is wrong"

    total_neurons = sum(counts)
    ge = SynapseGrowthEngine(
        device=device, synapse_group_size=GROUP_SIZE,
        max_groups_in_buffer=max(4096, 8 * (len(triples) + total_neurons)))
    for i in range(4):
        ge.register_neuron_type(max_synapses=4 * N_TGT, growth_command_list=[])
    for i in range(4):
        t = torch.tensor(ids[i], dtype=torch.int32)
        n = t.numel()
        ge.add_neurons(neuron_type_index=i, identifiers=t,
                       coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                                torch.full((n,), float(i))], 1))

    chunk = ge._grow_explicit(torch.tensor(triples, dtype=torch.int32, device=device), seed,
                              weights=torch.tensor(weights, dtype=torch.float32, device=device))
    spnet.add_connections(chunk, seed)
    chunk.recycle()
    spnet.compile(shuffle_synapses_random_seed=None)
    return dict(spnet=spnet, ids=ids, P=P, triples=triples, weights=weights, device=device)


def verify_round_trip(H):
    """Every synapse's weight and delay straight back out of the compiled net.

    A local copy of harness.verify_round_trip because the delay decode differs: this net's
    metas span [1, 64] rather than harness's [1, 20], so `meta + harness.D_MIN` would report
    every delay above 20 as wrong.
    """
    from spiky.spnet.spnet import NeuronDataType     # noqa: F401  (import parity w/ harness)
    sp, ids, dev = H["spnet"], H["ids"], H["device"]
    all_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32, device=dev)
    n = sp.count_synapses(all_ids, True)
    bufs = [torch.zeros(n, dtype=t, device=dev) for t in
            (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], True)
    s, _m, w, d, t = (x.cpu().numpy() for x in bufs)
    got_w = {(int(a), int(b)): float(c) for a, b, c in zip(s, t, w)}
    got_d = {(int(a), int(b)): int(e) for a, b, e in zip(s, t, d)}
    okw = okd = miss = 0
    neg_ok = 0
    for (mm, ss, tt), ww in zip(H["triples"], H["weights"]):
        k = (int(ss), int(tt))
        if k not in got_w:
            miss += 1
            continue
        want_d = int(mm) + D_LO
        okw += abs(got_w[k] - ww) < 1e-3
        okd += got_d[k] == want_d
        neg_ok += (ww < 0) and abs(got_w[k] - ww) < 1e-3
    n_neg = int((H["weights"] < 0).sum())
    return dict(n_exported=int(n), n_requested=len(H["triples"]), weights_ok=okw,
                delays_ok=okd, missing=miss, n_negative=n_neg, negative_ok=neg_ok)


# ------------------------------------------------------------------ score
def decode_actions(first):
    """Offsets 0..31 -> actions, the exact inverse of target_offsets' affine+quantise.

    Only for REPORTING: it puts the tiny net's error on the same scale as exp011's LUT MSE
    (teacher 0.000084) so the two chapters are comparable. Selection uses the offset MSE.
    A silent output decodes at offset 32, i.e. just past the -C clip.
    """
    mu, sd, C, n = steady_state.TARGET_STATS
    u = 1.0 - np.asarray(first, np.float64) / (n - 1)
    return mu + sd * (u * 2.0 * C - C)


def score(H, X, Y, enc, current=200.0, fast=None):
    """-> dict of per-candidate arrays. Fitness is -MSE, so selection still maximises.

    fast=None uses the module default (FAST_EPISODE); pass False to force the shared
    harness.run_episode. tiny_equiv_check.py runs both and asserts they agree exactly.
    """
    use_fast = FAST_EPISODE if fast is None else fast
    if use_fast:
        first, R = fast_run_episode(H, X, enc, current, readout_window=READOUT_WINDOW)
    else:
        first, R = run_episode(H, X, enc, current, readout_window=READOUT_WINDOW)
    tgt = target_offsets(Y)[:, None, :]                        # [B, 1, N_OUT]
    mse = ((first - tgt) ** 2).mean(axis=(0, 2))               # [P]
    P = first.shape[1]
    tau = np.array([kendall_tau_b(-first[:, p, :], Y).mean() for p in range(P)])
    silent = (first >= READOUT_WINDOW).mean(axis=(0, 2))
    n_distinct = np.array([len(np.unique(first[:, p, :])) for p in range(P)])
    mse_act = np.array([((decode_actions(first[:, p, :]) - Y) ** 2).mean() for p in range(P)])
    return dict(fitness=-mse, mse=mse, tau=tau, silent=silent, n_distinct=n_distinct,
                mse_action=mse_act, first=first, raster=R)


# ------------------------------------------------------------------ fast episode runner
# A tiny-LOCAL replacement for harness.run_episode. harness.py is shared by exp001-exp011 and
# is deliberately left alone; this path is used only by exp012 and is asserted bit-identical
# to the shared one by tiny_equiv_check.py.
#
# The profile that motivated it (pool 512, batch 256, 962 ms/round):
#     build sparse ids (285 MB)   620 ms  64.5 %     <- constant across rounds, rebuilt anyway
#     D2H .cpu().numpy() (302 MB)  92 ms   9.6 %     <- 302 MB moved to compute a 3 MB answer
#     H2D copy of both (570 MB)    89 ms   9.3 %
#     numpy readout (argmax etc)   74 ms   7.7 %
#     process_ticks (GPU sim)      49 ms   5.0 %     <- the only part doing real work
FAST_EPISODE = True
_SPARSE_CACHE = {}


def _sparse_ids(cols, B, n_in, device):
    """(a) The constant sparse-input id tensor, built ONCE on the GPU and reused.

    Its contents are the same `cols` vector tiled B*n_in times and depend only on the batch
    size and the neuron ids -- both fixed for a whole run. harness.py rebuilds it every round
    via `np.broadcast_to(...).astype(int32)` followed by `.copy()`, which materialises 285 MB
    twice per round on the host. Here `expand(...).contiguous()` materialises it once, on the
    device, for the lifetime of the process.

    Keyed on the ids themselves, not just their shape: the net is rebuilt every round, and if
    the engine ever handed back different ids a shape-only key would silently serve a stale
    tensor. Cheap to check, catastrophic to get wrong.
    """
    key = (int(B), int(n_in), int(cols.size), str(device))
    hit = _SPARSE_CACHE.get(key)
    if hit is not None and np.array_equal(hit[0], cols):
        return hit[1]
    t = torch.as_tensor(np.ascontiguousarray(cols, dtype=np.int32), device=device)
    t = t.reshape(1, 1, -1).expand(B, n_in, -1).contiguous()
    _SPARSE_CACHE[key] = (np.array(cols, copy=True), t)
    return t


def fast_run_episode(H, X, enc, current=200.0, readout_window=READOUT_WINDOW):
    """-> first-spike tick [B, P, N_OUT] (readout_window if silent), window raster.

    Identical semantics to harness.run_episode(..., train=False, readout_window=W): the first
    spike inside the final W ticks, re-based to 0..W-1, silence reading exactly W.

    (b) the readout runs ON DEVICE and only the [B, P, N_OUT] answer crosses the bus, instead
        of the whole [B, P, N_OUT, 96] raster.

    (c) -- exporting ONLY the readout window -- is NOT done, because the engine cannot do it.
    `export_neuron_data`'s Python signature takes `first_tick`, but any non-zero value faults
    with cudaErrorIllegalAddress at spnet_runtime.cu:71; verified in isolation, `first_tick=0`
    exports 96 ticks fine and `first_tick=64` dies. So the full raster is still materialised
    ON THE DEVICE and the window is sliced there. That costs VRAM, not bus traffic, and
    export_neuron_data measured 8-15 ms of a 962 ms round, so little is lost.

    WHY NOT torch.argmax FOR THE FIRST SPIKE. numpy's argmax guarantees the FIRST maximal
    index; torch's does not promise it. A ranked-weight max does, exactly and deterministically:
    give tick k the weight W-k, so among the ticks that spiked the earliest one carries the
    largest weight, and an all-silent row maxes at 0 which decodes to W -- the null. That is
    the same value harness.readout_null(W) returns.
    """
    from spiky.spnet.spnet import NeuronDataType
    sp, ids, P, dev = H["spnet"], H["ids"], H["P"], H["device"]
    B = X.shape[0]
    W = int(readout_window)
    ticks = enc(X)

    cols = ids[2]
    va = np.zeros((B, T_IN, cols.size), np.float32)
    for b in range(B):
        for j in range(N_IN):
            va[b, ticks[b, j], j::N_IN] = current
    sp.process_ticks(n_ticks_to_process=N_TICKS, batch_size=B, n_input_ticks=T_IN,
                     input_values=torch.as_tensor(va, device=dev),
                     sparse_input=_sparse_ids(cols, B, T_IN, dev),
                     do_train=False, do_record_voltage=False,
                     do_reset_context=True, _stdp_period=32)

    out_ids = torch.as_tensor(np.ascontiguousarray(ids[3], dtype=np.int32), device=dev)
    R = sp.export_neuron_data(out_ids, B, NeuronDataType.Spike, 0, N_TICKS - 1)
    R = R.reshape(B, P, N_OUT, N_TICKS)[..., N_TICKS - W:]
    wgt = torch.arange(W, 0, -1, device=R.device, dtype=R.dtype)
    first = W - (R.ne(0) * wgt).amax(-1)
    return first.double().cpu().numpy(), R


def affine_ceiling_and_r(pred, tgt):
    """-> (mean |r| over dimensions, MSE after the BEST per-dimension affine rescaling).

    REPORTING ONLY. Neither number ever touches selection -- the genome carries no decode and
    fitness is the raw offset MSE. The affine ceiling is `var(t) * (1 - r^2)` per dimension,
    i.e. what the SAME predictions would score if a per-dimension `alpha*p + beta` were fitted
    on them. It is the useful upper bound on how much of the residual MSE is a coding artefact
    rather than genuine error, which is exactly the quantity the [1,32] substrate destroyed.
    """
    pred = np.asarray(pred, np.float64)
    tgt = np.asarray(tgt, np.float64)
    rs, ceil = [], []
    for d in range(pred.shape[1]):
        p, t = pred[:, d], tgt[:, d]
        r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 else 0.0
        r = 0.0 if not np.isfinite(r) else r
        rs.append(abs(r))
        ceil.append(float(t.var() * (1.0 - r ** 2)))
    return float(np.mean(rs)), float(np.mean(ceil))


def jsonable(o):
    """numpy scalars/arrays -> plain python, so every report file dumps cleanly."""
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


def constant_baseline(Y):
    """MSE of the best constant predictor on the offset scale -- the number to beat.

    exp009 measured 39.19 on its pool; recomputing it here keeps the comparison honest if the
    batch or the split ever changes.
    """
    tgt = target_offsets(Y)
    return float(((tgt - tgt.mean(0)) ** 2).mean())
