"""Core primitives for the neurodarwinism chapter: spnet reservoir + latency-coded I/O.

(Historically named `es_harness.py`. It is NOT an ES driver -- it is the shared substrate
every experiment in this chapter is built on: the network geometry constants, the group-
aligned weight builder, the delay meta table, the latency encoder, genome construction
and mutation, the episode runner, and the scoring functions. Renamed on the move into
`src/`; the dataset loader moved out to `data.py`.)

CORRECTED RECIPE (supersedes the weight-palette version):
  * DELAYS   one SynapseMeta per discrete delay, range [d, d]. ~20 metas total.
  * WEIGHTS  CONTINUOUS, set explicitly via `_grow_explicit(weights=)`. The metas'
             initial_weight is discarded.
             (This file used to carry its own chain-following aligner, because the stock
             helper recovered each group's owning source by scanning memory in order and
             landed only 86/516 weights. The engine's aligner follows the chain as of
             PR #94, so the local copy is gone and the plain `weights=` path is exact.)
  * ONE _grow_explicit chunk, ONE add_connections, group_size 2,
    compile(shuffle_synapses_random_seed=None)  (shuffle OFF).

WHY ONE CHUNK. connections_manager.cu:76-79 rejects a chunk touching a SOURCE neuron an
earlier chunk configured. Output synapses are sourced FROM excitatory reservoir neurons —
the reservoir's own sources — so reservoir and I/O cannot be split. The reservoir is
therefore generated as explicit triples too.

LAYOUT. Candidate c owns disjoint id blocks in every neuron meta:
    meta 0 excitatory 800/cand   meta 1 inhibitory 200/cand
    meta 2 input       17/cand   meta 3 output       6/cand
Explicit triples only wire within a candidate's blocks. That isolation is ASSERTED by the
packed-vs-solo gate, not assumed.

TIMING, 96 ticks: [0,32) latency-coded input, [32,64) computation, [64,96) readout.
Phases are not muted; the split is a propagation budget.
"""
import os

import numpy as np
import torch


N_EXC, N_INH, N_IN, N_OUT = 800, 200, 17, 6
FANOUT_IN, FANIN_OUT = 100, 100
EXC_FANOUT_E, EXC_FANOUT_I, INH_FANOUT = 80, 20, 100
RES_W_EXC, RES_W_INH = 6.0, -5.0
D_MIN, D_MAX = 1, 20
N_TICKS, T_IN = 96, 32
GROUP_SIZE = 2


def delay_metas(group_size=GROUP_SIZE, d_min=D_MIN, d_max=D_MAX):
    """One SynapseMeta per discrete delay. initial_weight is irrelevant (set explicitly)."""
    from spiky.spnet.spnet import SynapseMeta
    return [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d,
                        initial_weight=0.0, min_weight=0.0, max_weight=45.0,
                        initial_noise_level=0.0, weight_decay=0.9, weight_scaling_cf=0.0,
                        _forward_group_size=group_size, _backward_group_size=group_size)
            for d in range(d_min, d_max + 1)]


# ------------------------------------------------------- latency encoder
class LatencyEncoder:
    """Spread x_norm across the FULL [0, T_IN) window; larger x fires EARLIER.

    The previous encoder reused the LUT's t = c - m*x fit for a [0,1] window, which put
    every observation into ticks 12-22 — 11 distinct values out of 32. Calibrating on
    dataset percentiles recovers the whole window and ~3x the timing resolution.
    """

    def __init__(self, X, lo_pct=0.5, hi_pct=99.5):
        self.lo = float(np.percentile(X, lo_pct))
        self.hi = float(np.percentile(X, hi_pct))

    def __call__(self, x):
        u = (np.asarray(x, np.float64) - self.lo) / max(self.hi - self.lo, 1e-9)
        t = (1.0 - np.clip(u, 0.0, 1.0)) * (T_IN - 1)
        return np.clip(t.round().astype(np.int64), 0, T_IN - 1)


# ------------------------------------------------------- genome / reservoir
def random_genome(rng, w_max):
    """DALE'S LAW: both evolved synapse sets are sourced from EXCITATORY neurons —
    in_w from the input units (neuron meta 2, RS) and out_w from excitatory reservoir
    neurons — so their weights must be >= 0. Drawing them signed let a third of the
    genome wire excitatory cells through inhibitory synapses."""
    return dict(
        in_tgt=rng.integers(0, N_EXC, (N_IN, FANOUT_IN)),
        in_dly=rng.integers(D_MIN, D_MAX + 1, (N_IN, FANOUT_IN)),
        in_w=rng.uniform(0.0, w_max, (N_IN, FANOUT_IN)),
        out_src=rng.integers(0, N_EXC, (N_OUT, FANIN_OUT)),
        out_dly=rng.integers(D_MIN, D_MAX + 1, (N_OUT, FANIN_OUT)),
        out_w=rng.uniform(0.0, w_max, (N_OUT, FANIN_OUT)),
    )


def mutate(g, rng, w_max, p_rewire=0.02, p_delay=0.05, p_weight=0.20, w_sigma=0.15):
    h = {k: v.copy() for k, v in g.items()}
    for tgt, dly, w in (("in_tgt", "in_dly", "in_w"), ("out_src", "out_dly", "out_w")):
        m = rng.random(h[tgt].shape) < p_rewire
        h[tgt][m] = rng.integers(0, N_EXC, int(m.sum()))
        m = rng.random(h[dly].shape) < p_delay
        h[dly][m] = np.clip(h[dly][m] + rng.choice([-1, 1], int(m.sum())), D_MIN, D_MAX)
        m = rng.random(h[w].shape) < p_weight
        h[w][m] += rng.normal(0.0, w_sigma * w_max, int(m.sum()))
        h[w][:] = np.clip(h[w], 0.0, 1.5 * w_max)      # Dale's law: floor at 0, not -w_max
    return h


def reservoir_wiring(rng):
    """FROZEN and identical across candidates (learning_rate 0 on every meta)."""
    return dict(
        e_e=rng.integers(0, N_EXC, (N_EXC, EXC_FANOUT_E)),
        e_e_d=rng.integers(D_MIN, D_MAX + 1, (N_EXC, EXC_FANOUT_E)),
        e_i=rng.integers(0, N_INH, (N_EXC, EXC_FANOUT_I)),
        e_i_d=rng.integers(D_MIN, D_MAX + 1, (N_EXC, EXC_FANOUT_I)),
        i_e=rng.integers(0, N_EXC, (N_INH, INH_FANOUT)),
    )


# ------------------------------------------------------- build
def build(genomes, res, device="cuda", seed=1, res_scale=1.0, res_edges=None,
          out_gate=None):
    """Pack candidates into ONE SpikingNet with continuous explicit weights.

    res_edges: optional (src_local, tgt_local, weight, delay, is_inh) arrays overriding the
    generated reservoir — used by arm B to place an edge-exact imported reservoir.

    out_gate: optional (lo, hi). Appends a second meta bank holding delays [lo, hi] and routes
    every OUTPUT synapse through it, so an output neuron cannot fire before `lo` ticks after
    its presynaptic spike. `out_dly` is clamped into the range. None (default) = unchanged.
    (Mirrors steady_state's --out-delay-gate; see its note on why a delay bank rather than an
    inhibitory current, which diverges in the Izhikevich quadratic.)
    """
    from spiky.spnet.spnet import SpikingNet, NeuronMeta
    from spiky.util.synapse_growth import SynapseGrowthEngine

    P = len(genomes)
    metas = delay_metas()
    n_base_metas = len(metas)
    if out_gate is not None:
        lo, hi = int(out_gate[0]), int(out_gate[1])
        assert 1 <= lo <= hi < N_TICKS, f"out_gate {out_gate} must satisfy 1 <= lo <= hi < {N_TICKS}"
        metas = metas + delay_metas(d_min=lo, d_max=hi)
    neuron_metas = [NeuronMeta(neuron_type=0, a=0.02, d=8.0),
                    NeuronMeta(neuron_type=1, a=0.1, d=2.0),
                    NeuronMeta(neuron_type=2, a=0.02, d=8.0),
                    NeuronMeta(neuron_type=3, a=0.02, d=8.0)]
    counts = [P * N_EXC, P * N_INH, P * N_IN, P * N_OUT]
    spnet = SpikingNet(synapse_metas=metas, neuron_metas=neuron_metas,
                       neuron_counts=counts, initial_synapse_capacity=1 << 22,
                       summation_dtype=torch.float32)
    spnet.to_device(device)
    ids = [spnet.get_neuron_ids_by_meta(i).cpu().numpy() for i in range(4)]

    tri, wts = [], []

    def add(src_ids, tgt_ids, tgt_idx, dly, w):
        s = np.repeat(src_ids, tgt_idx.shape[1])
        t = tgt_ids[np.asarray(tgt_idx).ravel()]
        d = np.asarray(dly).ravel()
        tri.append(np.stack([d - D_MIN, s, t], 1))
        wts.append(np.broadcast_to(np.asarray(w, np.float64).ravel(), s.shape).copy())

    for c, g in enumerate(genomes):
        E = ids[0][c * N_EXC:(c + 1) * N_EXC]
        I = ids[1][c * N_INH:(c + 1) * N_INH]
        In = ids[2][c * N_IN:(c + 1) * N_IN]
        Ou = ids[3][c * N_OUT:(c + 1) * N_OUT]
        if res_edges is None:
            add(E, E, res["e_e"], res["e_e_d"], RES_W_EXC * res_scale)
            add(E, I, res["e_i"], res["e_i_d"], RES_W_EXC * res_scale)
            add(I, E, res["i_e"], np.ones_like(res["i_e"]), RES_W_INH * res_scale)
        else:
            # Both ENDS may be inhibitory. Mapping only excitatory targets silently dropped
            # every synapse onto the 200 inhibitory units (14,709 of 91,424), so the import
            # was faithful only over the excitatory-targeted subset.
            sl, s_inh, tl, t_inh, ww, dd = res_edges
            s = np.where(s_inh, I[np.clip(sl, 0, N_INH - 1)], E[np.clip(sl, 0, N_EXC - 1)])
            t = np.where(t_inh, I[np.clip(tl, 0, N_INH - 1)], E[np.clip(tl, 0, N_EXC - 1)])
            tri.append(np.stack([dd - D_MIN, s, t], 1))
            wts.append(ww.astype(np.float64))
        add(In, E, g["in_tgt"], g["in_dly"], g["in_w"])
        s = E[g["out_src"].ravel()]
        t = np.repeat(Ou, FANIN_OUT)
        od = np.asarray(g["out_dly"]).ravel()
        if out_gate is None:
            om = od - D_MIN
        else:
            om = n_base_metas + (np.clip(od, lo, hi) - lo)
        tri.append(np.stack([om, s, t], 1))
        wts.append(np.asarray(g["out_w"], np.float64).ravel())

    triples = np.concatenate(tri, 0)
    weights = np.concatenate(wts, 0)
    # (source, target) must be unique; keep the first occurrence
    key = triples[:, 1].astype(np.int64) * (1 << 22) + triples[:, 2]
    _, uniq = np.unique(key, return_index=True)
    uniq = np.sort(uniq)
    triples, weights = triples[uniq], weights[uniq]

    total_neurons = sum(counts)
    ge = SynapseGrowthEngine(
        device=device, synapse_group_size=GROUP_SIZE,
        max_groups_in_buffer=max(4096, 8 * (len(triples) + total_neurons)))
    for i in range(4):
        ge.register_neuron_type(max_synapses=8 * (N_EXC + N_INH), growth_command_list=[])
    for i in range(4):
        t = torch.tensor(ids[i], dtype=torch.int32)
        n = t.numel()
        ge.add_neurons(neuron_type_index=i, identifiers=t,
                       coordinates=torch.stack([torch.arange(n).float(), torch.zeros(n),
                                                torch.full((n,), float(i))], 1))

    tri_t = torch.tensor(triples, dtype=torch.int32, device=device)
    w_t = torch.tensor(weights, dtype=torch.float32, device=device)
    chunk = ge._grow_explicit(tri_t, seed, weights=w_t)
    spnet.add_connections(chunk, seed)
    chunk.recycle()
    spnet.compile(shuffle_synapses_random_seed=None)          # shuffle OFF
    return dict(spnet=spnet, ids=ids, P=P, triples=triples, weights=weights,
                device=device)


def verify_round_trip(H):
    """Every synapse's weight and delay, straight back out of the compiled net."""
    sp, ids, dev = H["spnet"], H["ids"], H["device"]
    all_ids = torch.tensor(np.concatenate(ids), dtype=torch.int32, device=dev)
    n = sp.count_synapses(all_ids, True)
    bufs = [torch.zeros(n, dtype=t, device=dev) for t in
            (torch.int32, torch.int32, torch.float32, torch.int32, torch.int32)]
    sp.export_synapses(all_ids, bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], True)
    s, m, w, d, t = (x.cpu().numpy() for x in bufs)
    got_w = {}
    got_d = {}
    for a_, b_, c_, e_ in zip(s, t, w, d):
        got_w[(int(a_), int(b_))] = float(c_)
        got_d[(int(a_), int(b_))] = int(e_)
    tri, wants = H["triples"], H["weights"]
    okw = okd = miss = 0
    for (mm, ss, tt), ww in zip(tri, wants):
        k = (int(ss), int(tt))
        if k not in got_w:
            miss += 1
            continue
        okw += abs(got_w[k] - ww) < 1e-3
        okd += got_d[k] == int(mm) + D_MIN
    return dict(n_exported=int(n), n_requested=len(tri), weights_ok=okw,
                delays_ok=okd, missing=miss)


# ------------------------------------------------------- run / fitness
def readout_null(readout_window=None):
    """The value a silent output takes, i.e. 'later than any real spike'.

    Downstream code tests for silence against this, so never hard-code N_TICKS once a
    windowed readout is in play: under a W-tick window the ticks run 0..W-1 and the null is W.
    """
    return N_TICKS if readout_window is None else int(readout_window)


def run_episode(H, X, enc, current=200.0, train=False, readout_window=None):
    """-> first-spike tick [B, P, 6] (readout_null() if silent), raw raster.

    readout_window=None (DEFAULT, unchanged): first spike anywhere in the 96-tick simulation,
    null = 96.

    readout_window=W: only the FINAL W ticks count. The value is the first spike inside
    [N_TICKS - W, N_TICKS), re-based to 0..W-1, and null = W. Spikes before the window are
    discarded outright, even from a neuron that never fires again -- such an output reads as
    silent. The raster R is returned whole either way, so nothing downstream loses information.

    WHY THIS MIGHT MATTER. The declared timing is [0,32) input, [32,64) computation,
    [64,96) readout -- but the phases are only a propagation budget, never enforced, and the
    default readout ignores them: an output neuron driven straight through by the input volley
    lands its first spike in the input phase, and its rank is then decided before the network
    has computed anything. Restricting to the final third makes the readout read the phase it
    was designed to read, and re-basing means the six outputs are ranked on a 32-tick scale
    rather than a 96-tick one.
    """
    from spiky.spnet.spnet import NeuronDataType
    sp, ids, P, dev = H["spnet"], H["ids"], H["P"], H["device"]
    B = X.shape[0]
    ticks = enc(X)
    K = P * N_IN
    va = np.zeros((B, T_IN, K), np.float32)
    sp_ids = np.broadcast_to(ids[2].reshape(1, 1, K), (B, T_IN, K)).astype(np.int32)
    for b in range(B):
        for j in range(N_IN):
            va[b, ticks[b, j], j::N_IN] = current
    out_ids = torch.tensor(ids[3], dtype=torch.int32, device=dev)
    sp.process_ticks(n_ticks_to_process=N_TICKS, batch_size=B, n_input_ticks=T_IN,
                     input_values=torch.tensor(va, device=dev),
                     sparse_input=torch.tensor(sp_ids.copy(), device=dev),
                     do_train=train, do_record_voltage=False,
                     do_reset_context=True, _stdp_period=32)
    R = sp.export_neuron_data(out_ids, B, NeuronDataType.Spike, 0, N_TICKS - 1)
    R = R.cpu().numpy().reshape(B, P, N_OUT, N_TICKS)
    if readout_window is None:
        window = R
        null = N_TICKS
    else:
        W = int(readout_window)
        if not 1 <= W <= N_TICKS:
            raise ValueError(f"readout_window must be in [1, {N_TICKS}], got {W}")
        window = R[..., N_TICKS - W:]        # the FINAL W ticks; earlier spikes discarded
        null = W
    has = window.any(-1)
    first = np.where(has, window.argmax(-1), null)
    return first.astype(np.float64), R


def kendall_tau_b(pred, targ):
    """tau-b over the 6 outputs, per sample. pred/targ [B,6] -> [B]."""
    n = pred.shape[1]
    iu, ju = np.triu_indices(n, 1)
    dp = np.sign(pred[:, iu] - pred[:, ju])
    dt = np.sign(targ[:, iu] - targ[:, ju])
    n0 = len(iu)
    n1 = (dp == 0).sum(1)
    n2 = (dt == 0).sum(1)
    den = np.sqrt(np.maximum((n0 - n1) * (n0 - n2), 1e-12))
    return (dp * dt).sum(1) / den


def fitness(H, X, Y, enc, current=200.0):
    """Earlier spike = larger action, so predicted value is -t (rank-invariant)."""
    first, R = run_episode(H, X, enc, current)
    P = first.shape[1]
    taus = np.stack([kendall_tau_b(-first[:, p, :], Y) for p in range(P)], 1)
    return taus.mean(0), first, R


def own_null(first, Y, n_shuf=200):
    """Chance level for THESE predictions: shuffle Y against this model's own orderings.

    The previous metric took one null from a gen-0 random-wiring net and subtracted it from
    every evolved model. But the label-shuffle null is a property of the MODEL's predicted
    orderings (a near-constant ordering scores well against the average target ordering
    regardless of input), so it must be recomputed for whatever predictions are being
    scored. Measured cost of getting this wrong: per-seed nulls of +0.18/-0.10/-0.15 on
    networks whose raw tau agreed to +-0.05.

    (Lifted from the retired `es_sweep.py` on the move into `src/`: it is a scoring
    function and belongs beside `kendall_tau_b`, not in a sweep driver.)
    """
    pred = -first[:, 0, :]
    return float(np.mean([kendall_tau_b(pred, Y[np.random.default_rng(k).permutation(Y.shape[0])]).mean()
                          for k in range(n_shuf)]))
