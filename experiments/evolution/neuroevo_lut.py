"""neuroevo_lut.py — neuroevolution harness for the toy LUT on the REAL spiky spnet
CPU kernel. v2: random real-valued minibatches + held-out validation + NEAT-style
crossover + diversity (random immigrants) + self-adaptive mutation.

FIXED CONTRACT (every individual):
  * 6 INPUT neurons forced to spike at latency ticks t_i = round(A - B*x_i) (earlier
    = larger); real-valued x. * 4 OUTPUT neurons read by FIRST-SPIKE latency (silent
    -> ranked last). Everything else is GENOME: hidden neurons (each with its own
    spnet meta: leak cf_1, threshold, reset c, recovery d) + a directed synapse graph
    (src, dst, weight, delay) with NEAT innovation numbers.

EVALUATION (v2):
  * TRAIN: each generation draw a FRESH minibatch x ~ U[-1,1]^6 (size NE_TRAIN_BATCH),
    label on the fly with the oracle LUT; RE-EVALUATE the whole population on it (no
    individual hides behind a lucky batch).
  * VALIDATION: a FIXED held-out set of NE_VAL real x (fixed seed, never used for
    selection); the best individual's validation fitness each gen is the TRUE curve.
  * All batched into the real spnet CPU kernel (batch_size = minibatch, chunked).

FITNESS (unchanged form): fitness = strict_frac + 0.25*ordering_frac, over the batch.
  Observed only through the 4 output first-spike latencies -> a ranking.
    strict   = exact match of the net ranking to the oracle row's value ranking.
    ordering = graded pairwise Kendall concordance (the smooth shaping term).

PLATEAU FIXES (the v1 smoke converged at gen 2):
  * NEAT-style CROSSOVER: synapse genes carry global innovation numbers; matching
    genes inherit randomly, disjoint/excess from the fitter parent; hidden nodes are
    globally identified so metas align.
  * DIVERSITY = RANDOM IMMIGRANTS (default 15% fresh genomes/gen). Chosen over
    speciation/novelty because it's the simplest mechanism that directly attacks the
    observed premature convergence (continuously injects new genetic material with no
    distance-metric/among-species bookkeeping) — a clean fit for a prototype→real run.
  * SELF-ADAPTIVE mutation: each genome carries a sigma (weight-step) that mutates
    log-normally and drives the Gaussian weight perturbations.
Degenerate genomes (build error / no valid synapse) are guarded -> fitness 0.
"""
import math
import os
import random
import sys

import torch
import spiky_cuda  # noqa: F401
from lutmodel import build_model, lut_bits, bits_to_row
from spnet_harness import build_packed, build_packed_ref
from spiky.spnet.spnet import NeuronMeta, NeuronDataType

PACKED = bool(os.environ.get("NE_PACKED"))     # population-parallel packed eval (one build per chunk)
DEVICE = os.environ.get("NE_DEVICE", "cpu")     # 'cpu' (default) or 'cuda' — Phase 2 GPU target
if DEVICE == "cpu":
    torch.set_num_threads(1)

m = build_model(0)
W, b, V, K, D, Dout = m["W"], m["b"], m["V"], m["K"], m["D"], m["Dout"]

A_LAT, B_LAT = 8.0, 5.0          # input latency: t_i = round(A - B*x_i), x in [-1,1] -> [3,13]
N_TICKS = 44
IN_THR, OUT_THR = 0.5, 1.0
MAX_HIDDEN, MAX_SYN, CHUNK = 12, 60, 512
MAX_TYPES = 2           # hidden neuron-TYPE palette cap per genome (start small; raise later)
                        # (biological: few cell types, many cells) -> registers <= MAX_TYPES+2 metas,
                        # making the native MAX_NEURON_METAS=512 cap irrelevant regardless of hidden count

IN_LABELS = ["i%d" % i for i in range(D)]
OUT_LABELS = ["o%d" % d for d in range(Dout)]

# ----- global NEAT bookkeeping -----
INNOV, INNOV_CTR, NODE_CTR = {}, [0], [0]


def innov_of(s, t):
    key = (s, t)
    if key not in INNOV:
        INNOV_CTR[0] += 1
        INNOV[key] = INNOV_CTR[0]
    return INNOV[key]


def new_hidden():
    NODE_CTR[0] += 1
    return "h%d" % NODE_CTR[0]


def oracle_order(x):
    row = bits_to_row(lut_bits(m, x))
    return tuple(sorted(range(Dout), key=lambda d: -V[row][d]))


# ----- genome (type-palette encoding) -----
#   "types": [ {leak, thr, d}, ... ]           the neuron-TYPE palette (<= MAX_TYPES)
#   "hid":   {label: type_index}               each hidden neuron -> a type in the palette
#   "syn":   {innov: [src, tgt, weight, delay]} (NEAT innovation numbers)
#   "sigma": float                             self-adaptive mutation step
# Input/output neurons keep their fixed leak-free metas (the 2 special metas), unchanged.
def rnd_type(rng):
    return {"leak": max(0.0, rng.gauss(0.1, 0.1)), "thr": abs(rng.gauss(1.0, 0.6)) + 0.2,
            "d": abs(rng.gauss(0.0, 30.0))}


def default_out_type():
    """Legacy fallback for the shared output-neuron meta == the old fixed leakfree(OUT_THR):
    leak 0, threshold OUT_THR, no reset kick. Fresh genomes evolve their own (rnd_type)."""
    return {"leak": 0.0, "thr": OUT_THR, "d": 0.0}


def add_syn(g, s, t, w, dl):
    g["syn"][innov_of(s, t)] = [s, t, float(w), int(max(1, dl))]


def one_edge_per_pair(syn):
    """Collapse a syn dict to a simple digraph: at most one synapse per (src,tgt).
    Deterministic — keeps the first entry (in dict order) for each pair. Parallel edges
    arise only from merging genomes numbered by different per-process innovation registries."""
    out, seen = {}, set()
    for k, v in syn.items():
        st = (v[0], v[1])
        if st in seen:
            continue
        seen.add(st)
        out[k] = v
    return out


def random_genome(rng):
    g = {"types": [rnd_type(rng) for _ in range(rng.randint(1, MAX_TYPES))],   # hidden palette (<= MAX_TYPES)
         "hid": {}, "syn": {}, "sigma": 0.6, "out_type": rnd_type(rng)}        # one evolved output meta
    for _ in range(rng.randint(1, 4)):
        g["hid"][new_hidden()] = rng.randrange(len(g["types"]))         # assign a random type index
    for o in OUT_LABELS:
        for _ in range(rng.randint(1, 3)):
            s = rng.choice(IN_LABELS + list(g["hid"].keys()))
            add_syn(g, s, o, rng.uniform(-1.0, 3.0), rng.randint(1, 5))
    for h in list(g["hid"].keys()):
        add_syn(g, rng.choice(IN_LABELS), h, rng.uniform(-1.0, 3.0), rng.randint(1, 5))
    return g


def clone(g):
    return {"types": [dict(t) for t in g["types"]], "hid": dict(g["hid"]),
            "syn": {k: list(v) for k, v in g["syn"].items()}, "sigma": g["sigma"],
            "out_type": dict(g.get("out_type") or default_out_type())}


# ----- build genome -> real spnet -----
def leakfree(thr, c=0.0, d=0.0):
    return NeuronMeta(neuron_type=0, cf_2=0.0, cf_1=0.0, cf_0=0.0, a=0.0, b=0.0,
                      c=float(c), d=float(d), spike_threshold=float(thr))


def meta_of(t):
    """t = a palette type {leak, thr, d}; c is fixed at 0 (as before)."""
    return NeuronMeta(neuron_type=0, cf_2=0.0, cf_1=-float(t["leak"]), cf_0=0.0, a=0.0, b=0.0,
                      c=0.0, d=float(t["d"]), spike_threshold=float(t["thr"]))


def type_meta(g, h):
    """The NeuronMeta for hidden neuron h, resolved through its type_index into the palette."""
    return meta_of(g["types"][g["hid"][h]])


def build_individual(g):
    hids = list(g["hid"].keys())
    nodes = IN_LABELS + OUT_LABELS + hids
    local = {n: i for i, n in enumerate(nodes)}
    neuron_metas = [leakfree(IN_THR), leakfree(OUT_THR)] + [type_meta(g, h) for h in hids]
    node_meta = [0] * D + [1] * Dout + [2 + i for i in range(len(hids))]
    syn = [(local[s], local[t], float(w), int(max(1, dl)))
           for (s, t, w, dl) in g["syn"].values() if s in local and t in local]
    if not syn:
        return None
    packed = build_packed([{"n_nodes": len(nodes), "node_meta": node_meta, "synapses": syn}], neuron_metas)
    return packed, local


def _score_chunk(sp, gid, xs, true_orders):
    in_gid = [gid("i%d" % i) for i in range(D)]
    Oids = torch.tensor([gid("o%d" % d) for d in range(Dout)], dtype=torch.int32)
    nb = len(xs)
    S = torch.zeros((nb, N_TICKS, D), dtype=torch.int32)
    Vv = torch.zeros((nb, N_TICKS, D), dtype=torch.float32)
    for bidx, x in enumerate(xs):
        for i in range(D):
            t = int(round(A_LAT - B_LAT * x[i]))
            t = min(max(t, 1), N_TICKS - 1)
            S[bidx, t, i] = in_gid[i]
            Vv[bidx, t, i] = 50.0
    sp.process_ticks(n_ticks_to_process=N_TICKS, batch_size=nb, n_input_ticks=N_TICKS,
                     input_values=Vv, do_train=False, sparse_input=S, do_reset_context=True)
    spk = sp.export_neuron_data(Oids, nb, NeuronDataType.Spike, 0, N_TICKS - 1).view(nb, Dout, N_TICKS)
    strict = 0.0
    order_sum = 0.0
    for bidx in range(nb):
        first = []
        for d in range(Dout):
            nz = torch.nonzero(spk[bidx, d] > 0.5).flatten()
            first.append(int(nz[0].item()) if nz.numel() else N_TICKS + 1)
        if any(f >= N_TICKS for f in first):     # HARD CONSTRAINT: a silent output -> corner scores 0
            continue
        net_order = tuple(sorted(range(Dout), key=lambda d: (first[d], d)))
        true_order = true_orders[bidx]
        if net_order == true_order:
            strict += 1
        conc = 0
        for a in range(Dout):
            for c2 in range(a + 1, Dout):
                if first[true_order[a]] <= first[true_order[c2]]:
                    conc += 1
        order_sum += conc / 6.0
    return strict, order_sum


def eval_batch(g, xs, true_orders):
    try:
        built = build_individual(g)
    except Exception:
        return 0.0, 0.0, 0.0
    if built is None:
        return 0.0, 0.0, 0.0
    packed, local = built
    sp = packed["spnet"]
    gid = lambda lbl: packed["gid"](0, local[lbl])
    strict_tot = order_tot = 0.0
    try:
        for i in range(0, len(xs), CHUNK):
            s, o = _score_chunk(sp, gid, xs[i:i + CHUNK], true_orders[i:i + CHUNK])
            strict_tot += s
            order_tot += o
    except Exception:
        return 0.0, 0.0, 0.0
    n = len(xs)
    sf, of = strict_tot / n, order_tot / n
    return sf + 0.25 * of, sf, of


# ----- POPULATION-PARALLEL packed eval (one build + one process_ticks for MANY genomes) -----
# Instead of build+teardown a fresh SpikingNet per genome, pack a whole chunk of genomes
# as DISJOINT sub-networks into ONE device-resident net (reusing build_packed's native
# multi-candidate packing with a combined neuron-meta list: shared input/output metas at
# indices 0/1, each genome's hidden metas appended). One process_ticks over the corner
# batch then evaluates every genome at once. On CPU this yields the SAME fitness as the
# per-genome eval_batch (disjoint subnets never interact); on GPU it's the path that makes
# per-genome build/teardown overhead vanish. device='cpu' (default) or 'cuda'.
def build_population(genomes, device='cpu', ref=False):
    """Pack genomes (each assumed to have >=1 synapse) into one build_packed net.
    ref=True uses the reference build (build_packed_ref) for equivalence checks.
    NeuronMetas must be DE-DUPLICATED before SpikingNet (its register_neuron_meta
    dedupes identical metas, which would break its `assert m_id == i`); different
    genomes routinely share identical hidden metas (crossover copies them), so we
    dedup by parameter signature and remap every node's meta index accordingly."""
    metas = []
    key2idx = {}

    def meta_index(nm):
        k = (round(nm.cf_2, 6), round(nm.cf_1, 6), round(nm.cf_0, 6), round(nm.a, 6),
             round(nm.b, 6), round(nm.c, 6), round(nm.d, 6), round(nm.spike_threshold, 6),
             int(nm.neuron_type))
        if k not in key2idx:
            key2idx[k] = len(metas)
            metas.append(nm)
        return key2idx[k]

    in_idx = meta_index(leakfree(IN_THR))
    cands = []
    for g in genomes:
        hids = list(g["hid"].keys())
        nodes = IN_LABELS + OUT_LABELS + hids            # local order: 0..5 in, 6..9 out, 10.. hidden
        loc = {n: i for i, n in enumerate(nodes)}
        out_idx = meta_index(meta_of(g.get("out_type") or default_out_type()))  # ONE evolved meta, all outputs
        node_meta = [in_idx] * D + [out_idx] * Dout + [meta_index(type_meta(g, h)) for h in hids]
        # Enforce a simple digraph: at most ONE synapse per (src,tgt). Parallel edges only
        # arise from merging genomes numbered by different per-process innovation registries;
        # the native net has no meaningful place for them. Deterministic: keep the first seen.
        syn, seen_edges = [], set()
        for (s, t, w, dl) in g["syn"].values():
            if s in loc and t in loc and (loc[s], loc[t]) not in seen_edges:
                seen_edges.add((loc[s], loc[t]))
                syn.append((loc[s], loc[t], float(w), int(max(1, dl))))
        cands.append({"n_nodes": len(nodes), "node_meta": node_meta, "synapses": syn})
    return (build_packed_ref if ref else build_packed)(cands, metas, device=device)


def _score_population_ref(packed, ncand, xs, true_orders, device='cpu'):
    """REFERENCE (loop-based) scoring — kept as the equivalence baseline for the
    vectorized _score_population below. Same result, just Python-loop O(chunk)."""
    gid = packed["gid"]
    sp = packed["spnet"]
    in_gid = [[gid(c, i) for i in range(D)] for c in range(ncand)]           # input local idx = i
    Oids = torch.tensor([gid(c, D + d) for c in range(ncand) for d in range(Dout)],
                        dtype=torch.int32)                                    # output local idx = D+d
    nb = len(xs)
    kdim = ncand * D
    S = torch.zeros((nb, N_TICKS, kdim), dtype=torch.int32)
    Vv = torch.zeros((nb, N_TICKS, kdim), dtype=torch.float32)
    for b, x in enumerate(xs):
        for i in range(D):
            t = min(max(int(round(A_LAT - B_LAT * x[i])), 1), N_TICKS - 1)
            for c in range(ncand):
                j = c * D + i
                S[b, t, j] = in_gid[c][i]
                Vv[b, t, j] = 50.0
    if device != 'cpu':
        S, Vv, Oids = S.to(device), Vv.to(device), Oids.to(device)
    sp.process_ticks(n_ticks_to_process=N_TICKS, batch_size=nb, n_input_ticks=N_TICKS,
                     input_values=Vv, do_train=False, sparse_input=S, do_reset_context=True)
    spk = sp.export_neuron_data(Oids, nb, NeuronDataType.Spike, 0, N_TICKS - 1)
    spk = spk.view(nb, ncand, Dout, N_TICKS).cpu()
    res = []
    for c in range(ncand):
        strict = order_sum = 0.0
        for b in range(nb):
            first = []
            for d in range(Dout):
                nz = torch.nonzero(spk[b, c, d] > 0.5).flatten()
                first.append(int(nz[0].item()) if nz.numel() else N_TICKS + 1)
            if any(f >= N_TICKS for f in first):     # HARD CONSTRAINT: a silent output -> corner scores 0
                continue
            net_order = tuple(sorted(range(Dout), key=lambda d: (first[d], d)))
            to = true_orders[b]
            if net_order == to:
                strict += 1
            conc = 0
            for a in range(Dout):
                for c2 in range(a + 1, Dout):
                    if first[to[a]] <= first[to[c2]]:
                        conc += 1
            order_sum += conc / 6.0
        sf, of = strict / nb, order_sum / nb
        res.append((sf + 0.25 * of, sf, of))
    return res


_ORD_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]   # the 6 output pairs (Dout=4)


def _first_spike(spk, big):
    """Vectorized first-spike tick per element: earliest tick where spk>0.5, else `big`.
    spk shape (..., N_TICKS) -> (...) long. Matches nz[0] / sentinel of the ref loop."""
    n_ticks = spk.shape[-1]
    mask = spk > 0.5
    tr = torch.arange(n_ticks, device=spk.device).view(*([1] * (spk.dim() - 1)), n_ticks)
    filled = torch.where(mask, tr.expand_as(mask), torch.full_like(mask, big, dtype=torch.long))
    return filled.amin(dim=-1)


def _strict_conc(first, true_orders):
    """first: (nb, ncand, Dout) first-spike ticks; true_orders: list of nb Dout-perms.
    Returns (strict, conc) each (nb, ncand): strict = exact ranking match (net ranking =
    STABLE argsort of first, ties broken by output index = the ref's (first[d], d) key);
    conc = # concordant pairs of the true ranking. NOT averaged (caller divides)."""
    nb, ncand, Dout_ = first.shape
    dev = first.device
    true_t = torch.tensor(true_orders, dtype=torch.long, device=dev)          # (nb, Dout)
    net_order = torch.argsort(first, dim=2, stable=True)                       # (nb, ncand, Dout)
    strict = (net_order == true_t.view(nb, 1, Dout_)).all(dim=2).float()       # (nb, ncand)
    fo = torch.gather(first, 2, true_t.view(nb, 1, Dout_).expand(nb, ncand, Dout_))  # first in true-rank order
    conc = torch.zeros((nb, ncand), dtype=torch.float32, device=dev)
    for a, b2 in _ORD_PAIRS:                                                   # 6 fixed comparisons
        conc += (fo[:, :, a] <= fo[:, :, b2]).float()
    return strict, conc


def _score_from_first(first, true_orders):
    """Per-genome (fitness, strict_frac, ordering_frac), averaged over the nb corners.
    HARD CONSTRAINT — all outputs must fire: on any corner where some output stays silent
    (its first-spike is the sentinel), that corner scores 0 for BOTH strict and ordering.
    A net earns fitness only by firing all Dout outputs across the eval set."""
    ncand = first.shape[1]
    strict, conc = _strict_conc(first, true_orders)
    all_fire = (first < N_TICKS).all(dim=2).float()               # (nb, ncand): every output fired
    sf = (strict * all_fire).mean(dim=0)
    of = (conc / len(_ORD_PAIRS) * all_fire).mean(dim=0)
    fit = sf + 0.25 * of
    return [(float(fit[c]), float(sf[c]), float(of[c])) for c in range(ncand)]


def _score_population(packed, ncand, xs, true_orders, device='cpu'):
    """VECTORIZED packed scoring — same result as _score_population_ref, no per-genome
    Python loops. Input tensor built with tensor ops (subnet-offset placement); readout
    is a vectorized first-spike + argsort ranking + pairwise-concordance fitness."""
    gid = packed["gid"]
    sp = packed["spnet"]
    dev = torch.device(device)
    nb = len(xs)
    in_gid = torch.tensor([[gid(c, i) for i in range(D)] for c in range(ncand)],
                          dtype=torch.int32, device=dev)                        # (ncand, D)
    Oids = torch.tensor([gid(c, D + d) for c in range(ncand) for d in range(Dout)],
                        dtype=torch.int32, device=dev)
    xs_t = torch.tensor(xs, dtype=torch.float32, device=dev)                    # (nb, D)
    ticks = (A_LAT - B_LAT * xs_t).round().long().clamp(1, N_TICKS - 1)         # (nb, D) same tick for all genomes
    kdim = ncand * D
    S = torch.zeros((nb, N_TICKS, kdim), dtype=torch.int32, device=dev)
    Vv = torch.zeros((nb, N_TICKS, kdim), dtype=torch.float32, device=dev)
    bb = torch.arange(nb, device=dev).view(nb, 1, 1).expand(nb, D, ncand)
    ii = torch.arange(D, device=dev).view(1, D, 1).expand(nb, D, ncand)
    cc = torch.arange(ncand, device=dev).view(1, 1, ncand).expand(nb, D, ncand)
    tt = ticks.view(nb, D, 1).expand(nb, D, ncand)
    col = cc * D + ii                                                          # column c*D+i
    val = in_gid.t().reshape(1, D, ncand).expand(nb, D, ncand)                 # val[b,i,c] = in_gid[c,i]
    S[bb.reshape(-1), tt.reshape(-1), col.reshape(-1)] = val.reshape(-1)
    Vv[bb.reshape(-1), tt.reshape(-1), col.reshape(-1)] = 50.0
    sp.process_ticks(n_ticks_to_process=N_TICKS, batch_size=nb, n_input_ticks=N_TICKS,
                     input_values=Vv, do_train=False, sparse_input=S, do_reset_context=True)
    spk = sp.export_neuron_data(Oids, nb, NeuronDataType.Spike, 0, N_TICKS - 1).view(nb, ncand, Dout, N_TICKS)
    first = _first_spike(spk, N_TICKS + 1)                                     # (nb, ncand, Dout)
    return _score_from_first(first, true_orders)


def eval_population(genomes, xs, true_orders, device='cpu', chunk=128):
    """Fitness for every genome, packed in chunks. Matches [eval_batch(g,...) for g] on CPU.
    chunk packs more genomes per net (amortizing native add_connections/_grow_explicit/
    process_ticks fixed costs). Per-genome metas now = 1 shared input + 1 EVOLVED output
    (unique per genome) + <= MAX_TYPES hidden, so the packed net holds ~ (1 + MAXTYPES)*chunk
    distinct metas; keep chunk modest (the run packs 32 -> ~100 metas) to stay under the 512
    neuron-meta cap. Larger chunks risk approaching it as evolved output metas rarely dedup.
    Genomes with no synapses score 0.0 (as build_individual->None in eval_batch)."""
    out = [(0.0, 0.0, 0.0)] * len(genomes)
    for s in range(0, len(genomes), chunk):
        idx = [i for i in range(s, min(s + chunk, len(genomes))) if genomes[i]["syn"]]
        if not idx:
            continue
        try:
            packed = build_population([genomes[i] for i in idx], device=device)
            scores = _score_population(packed, len(idx), xs, true_orders, device=device)
        except Exception:
            scores = [(0.0, 0.0, 0.0)] * len(idx)
        for k, i in enumerate(idx):
            out[i] = scores[k]
    return out


# ----- temporal STREAMING evaluation -----
# K real-valued x presented at t=0,T,2T on one timeline; each read within its own
# window. Spacing T is ANNEALED wide->tight over the run so evolution earns the hard
# (residual-ringing) regime. A single-shot volley here settles in <~20 ticks (inputs
# fire at ticks 3..13, outputs a few hops later), so WIN=20; T_WIDE=30 (near-independent)
# -> T_TIGHT=16 (next presentation starts while the previous is still settling).
STREAM_K = int(os.environ.get("NE_STREAM_K", 3))
STREAM_WIN = int(os.environ.get("NE_STREAM_WIN", 20))
T_WIDE, T_TIGHT = 30, 16


def anneal_T(gen, gens):
    if gens <= 1:
        return T_TIGHT
    return int(round(T_WIDE - (T_WIDE - T_TIGHT) * (gen / (gens - 1))))


def _score_stream_chunk(sp, gid, trials, T, W):
    in_gid = [gid("i%d" % i) for i in range(D)]
    Oids = torch.tensor([gid("o%d" % d) for d in range(Dout)], dtype=torch.int32)
    K = STREAM_K
    L = (K - 1) * T + W + 6
    nb = len(trials)
    kdim = K * D
    S = torch.zeros((nb, L, kdim), dtype=torch.int32)
    Vv = torch.zeros((nb, L, kdim), dtype=torch.float32)
    for b, (xs, _) in enumerate(trials):
        for p in range(K):
            for i in range(D):
                t = p * T + int(round(A_LAT - B_LAT * xs[p][i]))
                t = min(max(t, 1), L - 1)
                j = p * D + i
                S[b, t, j] = in_gid[i]
                Vv[b, t, j] = 50.0
    sp.process_ticks(n_ticks_to_process=L, batch_size=nb, n_input_ticks=L,
                     input_values=Vv, do_train=False, sparse_input=S, do_reset_context=True)
    spk = sp.export_neuron_data(Oids, nb, NeuronDataType.Spike, 0, L - 1).view(nb, Dout, L)
    strict = order_sum = 0.0
    for b, (xs, tos) in enumerate(trials):
        for p in range(K):
            lo = p * T
            first = []
            for d in range(Dout):
                nz = torch.nonzero(spk[b, d, lo:lo + W] > 0.5).flatten()
                first.append(int(nz[0].item()) if nz.numel() else W + 1)   # relative to window
            net_order = tuple(sorted(range(Dout), key=lambda d: (first[d], d)))
            to = tos[p]
            if net_order == to:
                strict += 1
            conc = 0
            for a in range(Dout):
                for c2 in range(a + 1, Dout):
                    if first[to[a]] <= first[to[c2]]:
                        conc += 1
            order_sum += conc / 6.0
    denom = nb * K
    return strict / denom, order_sum / denom


def eval_stream(g, trials, T):
    try:
        built = build_individual(g)
    except Exception:
        return 0.0, 0.0, 0.0
    if built is None:
        return 0.0, 0.0, 0.0
    packed, local = built
    sp = packed["spnet"]
    gid = lambda lbl: packed["gid"](0, local[lbl])
    sf = of = 0.0
    n = 0
    try:
        for i in range(0, len(trials), 128):
            ch = trials[i:i + 128]
            s, o = _score_stream_chunk(sp, gid, ch, T, STREAM_WIN)
            sf += s * len(ch); of += o * len(ch); n += len(ch)
    except Exception:
        return 0.0, 0.0, 0.0
    sf /= n; of /= n
    return sf + 0.25 * of, sf, of


def make_stream_trials(rng, n):
    trials = []
    for _ in range(n):
        xs = [[rng.uniform(-1, 1) for _ in range(D)] for _ in range(STREAM_K)]
        trials.append((xs, [oracle_order(x) for x in xs]))
    return trials


# ----- POPULATION-PARALLEL packed STREAM eval (matches eval_stream on CPU) -----
def _score_population_stream_ref(packed, ncand, trials, T, device='cpu'):
    """REFERENCE (loop-based) streamed scoring — equivalence baseline for the vectorized
    _score_population_stream below."""
    W, K = STREAM_WIN, STREAM_K
    L = (K - 1) * T + W + 6
    gid, sp = packed["gid"], packed["spnet"]
    in_gid = [[gid(c, i) for i in range(D)] for c in range(ncand)]
    Oids = torch.tensor([gid(c, D + d) for c in range(ncand) for d in range(Dout)], dtype=torch.int32)
    nb, kdim = len(trials), ncand * K * D
    S = torch.zeros((nb, L, kdim), dtype=torch.int32)
    Vv = torch.zeros((nb, L, kdim), dtype=torch.float32)
    for b, (xs, _) in enumerate(trials):
        for p in range(K):
            for i in range(D):
                t = min(max(p * T + int(round(A_LAT - B_LAT * xs[p][i])), 1), L - 1)
                for c in range(ncand):
                    j = (c * K + p) * D + i
                    S[b, t, j] = in_gid[c][i]
                    Vv[b, t, j] = 50.0
    if device != 'cpu':
        S, Vv, Oids = S.to(device), Vv.to(device), Oids.to(device)
    sp.process_ticks(n_ticks_to_process=L, batch_size=nb, n_input_ticks=L,
                     input_values=Vv, do_train=False, sparse_input=S, do_reset_context=True)
    spk = sp.export_neuron_data(Oids, nb, NeuronDataType.Spike, 0, L - 1).view(nb, ncand, Dout, L).cpu()
    res = []
    for c in range(ncand):
        strict = order_sum = 0.0
        for b, (xs, tos) in enumerate(trials):
            for p in range(K):
                lo = p * T
                first = []
                for d in range(Dout):
                    nz = torch.nonzero(spk[b, c, d, lo:lo + W] > 0.5).flatten()
                    first.append(int(nz[0].item()) if nz.numel() else W + 1)
                net_order = tuple(sorted(range(Dout), key=lambda d: (first[d], d)))
                to = tos[p]
                if net_order == to:
                    strict += 1
                conc = 0
                for a in range(Dout):
                    for c2 in range(a + 1, Dout):
                        if first[to[a]] <= first[to[c2]]:
                            conc += 1
                order_sum += conc / 6.0
        denom = nb * K
        sf, of = strict / denom, order_sum / denom
        res.append((sf + 0.25 * of, sf, of))
    return res


def _score_population_stream(packed, ncand, trials, T, device='cpu'):
    """VECTORIZED streamed scoring — same result as _score_population_stream_ref, no
    per-genome Python loops (only a K-presentation loop, K=3). Inputs placed with tensor
    ops; per-window first-spike read + concordance accumulated across presentations."""
    gid = packed["gid"]
    sp = packed["spnet"]
    dev = torch.device(device)
    W, K = STREAM_WIN, STREAM_K
    L = (K - 1) * T + W + 6
    nb = len(trials)
    in_gid = torch.tensor([[gid(c, i) for i in range(D)] for c in range(ncand)],
                          dtype=torch.int32, device=dev)                        # (ncand, D)
    Oids = torch.tensor([gid(c, D + d) for c in range(ncand) for d in range(Dout)],
                        dtype=torch.int32, device=dev)
    xs_t = torch.tensor([[[trials[b][0][p][i] for i in range(D)] for p in range(K)]
                         for b in range(nb)], dtype=torch.float32, device=dev)  # (nb, K, D)
    base = torch.arange(K, device=dev).view(1, K, 1) * T
    ticks = (base + (A_LAT - B_LAT * xs_t).round().long()).clamp(1, L - 1)      # (nb, K, D)
    kdim = ncand * K * D
    S = torch.zeros((nb, L, kdim), dtype=torch.int32, device=dev)
    Vv = torch.zeros((nb, L, kdim), dtype=torch.float32, device=dev)
    bb = torch.arange(nb, device=dev).view(nb, 1, 1, 1).expand(nb, K, D, ncand)
    pp = torch.arange(K, device=dev).view(1, K, 1, 1).expand(nb, K, D, ncand)
    ii = torch.arange(D, device=dev).view(1, 1, D, 1).expand(nb, K, D, ncand)
    cc = torch.arange(ncand, device=dev).view(1, 1, 1, ncand).expand(nb, K, D, ncand)
    tt = ticks.view(nb, K, D, 1).expand(nb, K, D, ncand)
    col = (cc * K + pp) * D + ii                                               # column (c*K+p)*D+i
    val = in_gid.t().reshape(1, 1, D, ncand).expand(nb, K, D, ncand)           # val[b,p,i,c] = in_gid[c,i]
    S[bb.reshape(-1), tt.reshape(-1), col.reshape(-1)] = val.reshape(-1)
    Vv[bb.reshape(-1), tt.reshape(-1), col.reshape(-1)] = 50.0
    sp.process_ticks(n_ticks_to_process=L, batch_size=nb, n_input_ticks=L,
                     input_values=Vv, do_train=False, sparse_input=S, do_reset_context=True)
    spk = sp.export_neuron_data(Oids, nb, NeuronDataType.Spike, 0, L - 1).view(nb, ncand, Dout, L)
    strict_acc = torch.zeros((nb, ncand), dtype=torch.float32, device=spk.device)
    conc_acc = torch.zeros((nb, ncand), dtype=torch.float32, device=spk.device)
    for p in range(K):                                                        # per presentation window
        lo = p * T
        fp = _first_spike(spk[:, :, :, lo:lo + W], W + 1)                      # (nb, ncand, Dout), rel to window
        s, c = _strict_conc(fp, [trials[b][1][p] for b in range(nb)])
        strict_acc += s
        conc_acc += c
    denom = nb * K
    sf = strict_acc.sum(dim=0) / denom
    of = (conc_acc / len(_ORD_PAIRS)).sum(dim=0) / denom
    fit = sf + 0.25 * of
    return [(float(fit[c]), float(sf[c]), float(of[c])) for c in range(ncand)]


def eval_population_stream(genomes, trials, T, device='cpu', chunk=64):
    out = [(0.0, 0.0, 0.0)] * len(genomes)
    for s in range(0, len(genomes), chunk):
        idx = [i for i in range(s, min(s + chunk, len(genomes))) if genomes[i]["syn"]]
        if not idx:
            continue
        try:
            packed = build_population([genomes[i] for i in idx], device=device)
            scores = _score_population_stream(packed, len(idx), trials, T, device=device)
        except Exception:
            scores = [(0.0, 0.0, 0.0)] * len(idx)
        for k, i in enumerate(idx):
            out[i] = scores[k]
    return out


# ----- mutation (self-adaptive sigma) -----
def src_pool(g):
    return IN_LABELS + OUT_LABELS + list(g["hid"].keys())


def dst_pool(g):
    return OUT_LABELS + list(g["hid"].keys())


def mutate(g, rng):
    g = clone(g)
    g["sigma"] = min(2.0, max(0.05, g["sigma"] * math.exp(rng.gauss(0, 0.2))))   # self-adaptive
    op = rng.random()
    syns = list(g["syn"].keys())
    if op < 0.30 and syns:
        g["syn"][rng.choice(syns)][2] += rng.gauss(0, g["sigma"])
    elif op < 0.45 and syns:                                  # delay: LOCAL random walk on [1,255]
        gene = g["syn"][rng.choice(syns)]
        step = 0
        while step == 0:                                      # small Gaussian step: mass on +-1/+-2,
            step = round(rng.gauss(0, 1.5))                   # occasional larger moves via the tail
        gene[3] = min(255, max(1, gene[3] + step))            # clamp to the valid delay range
    elif op < 0.58 and g["types"]:                            # ---- palette (neuron-TYPE) ops ----
        sub = rng.random()
        if sub < 0.5:                                         # mutate-type: jitter one param of a hidden type OR the shared output meta
            t = rng.choice(g["types"] + [g["out_type"]])
            p = rng.choice(["leak", "thr", "d"])
            if p == "leak":
                t["leak"] = max(0.0, t["leak"] + rng.gauss(0, 0.05))
            elif p == "thr":
                t["thr"] = max(0.2, t["thr"] + rng.gauss(0, 0.3))
            else:
                t["d"] = max(0.0, t["d"] + rng.gauss(0, 20))
        elif sub < 0.75 and g["hid"] and len(g["types"]) > 1:  # reassign: point a hidden at another type
            g["hid"][rng.choice(list(g["hid"].keys()))] = rng.randrange(len(g["types"]))
        elif sub < 0.9 and len(g["types"]) < MAX_TYPES:        # add-type: copy an existing type + jitter
            nt = dict(g["types"][rng.randrange(len(g["types"]))])
            nt["leak"] = max(0.0, nt["leak"] + rng.gauss(0, 0.05))
            nt["thr"] = max(0.2, nt["thr"] + rng.gauss(0, 0.3))
            nt["d"] = max(0.0, nt["d"] + rng.gauss(0, 20))
            g["types"].append(nt)
        elif len(g["types"]) > 1:                              # drop-type: remove an UNused type (keep >=1)
            used = set(g["hid"].values())
            unused = [i for i in range(len(g["types"])) if i not in used]
            if unused:
                di = rng.choice(unused)
                g["types"].pop(di)
                g["hid"] = {h: (ti - 1 if ti > di else ti) for h, ti in g["hid"].items()}
    elif op < 0.74 and len(g["syn"]) < MAX_SYN:
        s, t = rng.choice(src_pool(g)), rng.choice(dst_pool(g))
        if innov_of(s, t) not in g["syn"]:
            add_syn(g, s, t, rng.uniform(-1.5, 3.0), rng.randint(1, 5))
    elif op < 0.84 and len(g["syn"]) > 1:
        del g["syn"][rng.choice(syns)]
    elif op < 0.94 and len(g["hid"]) < MAX_HIDDEN and syns:
        k = rng.choice(syns); s, t, w, dl = g["syn"][k]
        del g["syn"][k]
        h = new_hidden(); g["hid"][h] = rng.randrange(len(g["types"]))   # new instance -> existing type
        add_syn(g, s, h, w, dl); add_syn(g, h, t, 1.0, rng.randint(1, 3))
    elif g["hid"]:
        h = rng.choice(list(g["hid"].keys())); del g["hid"][h]
        g["syn"] = {k: v for k, v in g["syn"].items() if v[0] != h and v[1] != h}
    return g


def crossover(gf, gw, rng):
    """gf = fitter parent, gw = weaker. NEAT: matching genes random; disjoint/excess from fitter.
    The TYPE PALETTE is inherited whole from the fitter parent (simplest correct choice); each
    hidden's type_index is clamped to that palette (a gw-only hidden may index a type gf lacks)."""
    child = {"types": [dict(t) for t in gf["types"]], "hid": {}, "syn": {}, "sigma": gf["sigma"],
             "out_type": dict(gf.get("out_type") or default_out_type())}
    ntypes = len(child["types"])
    for innov in set(gf["syn"]) | set(gw["syn"]):
        inf, inw = innov in gf["syn"], innov in gw["syn"]
        if inf and inw:
            child["syn"][innov] = list(rng.choice([gf["syn"][innov], gw["syn"][innov]]))
        elif inf:
            child["syn"][innov] = list(gf["syn"][innov])          # disjoint/excess from fitter only
    for h in set(gf["hid"]) | set(gw["hid"]):                      # hidden type_index: prefer fitter, clamp
        ti = gf["hid"][h] if h in gf["hid"] else gw["hid"][h]
        child["hid"][h] = min(ti, ntypes - 1)
    # keep only hidden actually referenced (avoid orphan metas bloating the build)
    used = set()
    for s, t, _, _ in child["syn"].values():
        used.add(s); used.add(t)
    child["hid"] = {h: mt for h, mt in child["hid"].items() if h in used}
    child["syn"] = {k: v for k, v in child["syn"].items()
                    if (v[0] in IN_LABELS or v[0] in OUT_LABELS or v[0] in child["hid"])
                    and (v[1] in OUT_LABELS or v[1] in child["hid"])}
    child["syn"] = one_edge_per_pair(child["syn"])   # simple digraph: one synapse per (src,tgt)
    return child


def tournament(scored, rng, k=3):
    """Return the (fitness, genome) tuple of the tournament winner (reuses scored fits)."""
    return min([scored[rng.randrange(len(scored))] for _ in range(k)], key=lambda z: -z[0])


def run_ea(pop_size, generations, seed, train_batch, val_n, elite=8, imm_frac=0.15, log=print):
    rng = random.Random(seed)
    vrng = random.Random(999983)
    val_xs = [[vrng.uniform(-1, 1) for _ in range(D)] for _ in range(val_n)]
    val_true = [oracle_order(x) for x in val_xs]

    pop = [random_genome(rng) for _ in range(pop_size)]
    history = []
    n_imm = int(imm_frac * pop_size)
    for gen in range(generations):
        txs = [[rng.uniform(-1, 1) for _ in range(D)] for _ in range(train_batch)]
        ttrue = [oracle_order(x) for x in txs]
        scored = [(eval_batch(g, txs, ttrue)[0], g) for g in pop]
        scored.sort(key=lambda z: -z[0])
        train_mean = sum(z[0] for z in scored) / len(scored)
        best_g = scored[0][1]
        vf, vs, vo = eval_batch(best_g, val_xs, val_true)          # TRUE performance
        # mean validation of the elite band (cheap health signal)
        elite_val = sum(eval_batch(z[1], val_xs, val_true)[0] for z in scored[:elite]) / elite
        history.append((gen, vf, vs, vo, elite_val, train_mean, len(best_g["hid"]), len(best_g["syn"])))
        log("gen %3d | VAL best %.4f (s %.3f o %.3f) | val-elite-mean %.4f | train-mean %.4f | net %dh/%ds"
            % (gen, vf, vs, vo, elite_val, train_mean, len(best_g["hid"]), len(best_g["syn"])))
        elites = [clone(z[1]) for z in scored[:elite]]
        children = []
        while len(children) < pop_size - elite - n_imm:
            f1, p1 = tournament(scored, rng)            # reuse this gen's fitnesses
            f2, p2 = tournament(scored, rng)
            gf, gw = (p1, p2) if f1 >= f2 else (p2, p1)
            children.append(mutate(crossover(gf, gw, rng), rng))
        immigrants = [random_genome(rng) for _ in range(n_imm)]
        pop = elites + children + immigrants
    # final: eval last population on val, pick best
    finals = sorted(pop, key=lambda g: -eval_batch(g, val_xs, val_true)[0])
    best = finals[0]
    bf, bs, bo = eval_batch(best, val_xs, val_true)
    return (best, bf, bs, bo), history


def run_ea_stream(pop_size, generations, seed, ss_batch, stream_n, val_n, val_stream_n,
                  elite=8, imm_frac=0.15, log=print):
    """Combined fitness = single_shot + streamed (windowed). T annealed WIDE->TIGHT."""
    rng = random.Random(seed)
    vrng = random.Random(999983)
    svrng = random.Random(424242)
    ss_val_xs = [[vrng.uniform(-1, 1) for _ in range(D)] for _ in range(val_n)]
    ss_val_true = [oracle_order(x) for x in ss_val_xs]
    stream_val = make_stream_trials(svrng, val_stream_n)
    pop = [random_genome(rng) for _ in range(pop_size)]
    n_imm = int(imm_frac * pop_size)
    history = []
    log("anneal schedule: T(gen0)=%d -> T(gen%d)=%d (WIN=%d, K=%d)"
        % (anneal_T(0, generations), generations - 1, anneal_T(generations - 1, generations), STREAM_WIN, STREAM_K))
    for gen in range(generations):
        T = anneal_T(gen, generations)
        ss_txs = [[rng.uniform(-1, 1) for _ in range(D)] for _ in range(ss_batch)]
        ss_ttrue = [oracle_order(x) for x in ss_txs]
        st_trials = make_stream_trials(rng, stream_n)
        if PACKED:                                            # population-parallel packed eval
            ss = eval_population(pop, ss_txs, ss_ttrue, device=DEVICE)
            st = eval_population_stream(pop, st_trials, T, device=DEVICE)
            scored = [(ss[i][0] + st[i][0], pop[i]) for i in range(len(pop))]
        else:                                                 # per-genome reference path
            scored = [(eval_batch(g, ss_txs, ss_ttrue)[0] + eval_stream(g, st_trials, T)[0], g) for g in pop]
        scored.sort(key=lambda z: -z[0])
        train_mean = sum(z[0] for z in scored) / len(scored)
        best = scored[0][1]
        ss_vf, ss_vs, ss_vo = eval_batch(best, ss_val_xs, ss_val_true)
        st_vf, st_vs, st_vo = eval_stream(best, stream_val, T)                    # held-out @ current T
        elite_val = sum(eval_batch(z[1], ss_val_xs, ss_val_true)[0]
                        + eval_stream(z[1], stream_val, T)[0] for z in scored[:elite]) / elite
        history.append((gen, T, ss_vf, ss_vs, ss_vo, st_vf, st_vs, st_vo, elite_val, train_mean,
                        len(best["hid"]), len(best["syn"])))
        log("gen %3d T=%2d | SS-val %.3f (s%.3f o%.3f) | STREAM-val@T%d %.3f (s%.3f o%.3f) | elite %.3f train %.3f | %dh/%ds"
            % (gen, T, ss_vf, ss_vs, ss_vo, T, st_vf, st_vs, st_vo, elite_val, train_mean,
               len(best["hid"]), len(best["syn"])), flush=True)
        elites = [clone(z[1]) for z in scored[:elite]]
        children = []
        while len(children) < pop_size - elite - n_imm:
            f1, p1 = tournament(scored, rng)
            f2, p2 = tournament(scored, rng)
            gf, gw = (p1, p2) if f1 >= f2 else (p2, p1)
            children.append(mutate(crossover(gf, gw, rng), rng))
        pop = elites + children + [random_genome(rng) for _ in range(n_imm)]
    # coping analysis: best of final pop, scored at WIDE vs TIGHT spacing
    best = sorted(pop, key=lambda g: -(eval_batch(g, ss_val_xs, ss_val_true)[0]
                                       + eval_stream(g, stream_val, T_TIGHT)[0]))[0]
    wide = eval_stream(best, stream_val, T_WIDE)
    tight = eval_stream(best, stream_val, T_TIGHT)
    ss = eval_batch(best, ss_val_xs, ss_val_true)
    return best, history, ss, wide, tight


if __name__ == "__main__" and os.environ.get("NE_STREAM"):
    P = int(os.environ.get("NE_POP", 150))
    G = int(os.environ.get("NE_GEN", 30))
    SB = int(os.environ.get("NE_SS_BATCH", 128))
    SN = int(os.environ.get("NE_STREAM_N", 64))
    VN = int(os.environ.get("NE_VAL", 800))
    VSN = int(os.environ.get("NE_VAL_STREAM", 250))
    print("neuroevo LUT STREAMED — pop=%d gens=%d ss_batch=%d stream_n=%d val=%d val_stream=%d K=%d WIN=%d"
          % (P, G, SB, SN, VN, VSN, STREAM_K, STREAM_WIN), flush=True)
    best, hist, ss, wide, tight = run_ea_stream(P, G, seed=1, ss_batch=SB, stream_n=SN,
                                                val_n=VN, val_stream_n=VSN)
    print("=" * 80)
    print("STREAMED RUN DONE — %d generations" % len(hist))
    print("  T annealed %d -> %d over the run" % (hist[0][1], hist[-1][1]))
    print("  gen0  SS-val %.3f  STREAM-val@T%d %.3f" % (hist[0][2], hist[0][1], hist[0][5]))
    print("  final SS-val %.3f (s%.3f o%.3f)" % (ss[0], ss[1], ss[2]))
    print("  best net: %d hidden / %d synapses" % (len(best["hid"]), len(best["syn"])))
    print("  COPING TEST (same best net):  STREAM@WIDE(T=%d) = %.3f (s%.3f o%.3f)  vs  STREAM@TIGHT(T=%d) = %.3f (s%.3f o%.3f)"
          % (T_WIDE, wide[0], wide[1], wide[2], T_TIGHT, tight[0], tight[1], tight[2]))
    print("  -> %s" % ("COPES with residual state (tight ~ wide)" if tight[0] > 0.85 * wide[0]
                       else "RELIES on wide spacing (tight << wide) — residual state hurts it"))
    print("=" * 80, flush=True)
    sys.exit(0)

if __name__ == "__main__":
    P = int(os.environ.get("NE_POP", 200))
    G = int(os.environ.get("NE_GEN", 150))
    TB = int(os.environ.get("NE_TRAIN_BATCH", 256))
    VN = int(os.environ.get("NE_VAL", 1500))
    print("neuroevo LUT v2 — pop=%d gens=%d train_batch=%d val=%d, real spnet CPU" % (P, G, TB, VN), flush=True)
    (best, bf, bs, bo), hist = run_ea(P, G, seed=1, train_batch=TB, val_n=VN)
    print("=" * 76)
    print("RUN DONE — held-out validation (fixed %d real x):" % VN)
    print("  gen0   VAL best fitness = %.4f (strict %.3f, ordering %.3f)" % (hist[0][1], hist[0][2], hist[0][3]))
    print("  final  VAL best fitness = %.4f (strict %.3f, ordering %.3f)" % (bf, bs, bo))
    kept = bf > hist[0][1] + 1e-4
    past2 = max(h[1] for h in hist[3:]) > hist[2][1] + 1e-4 if len(hist) > 3 else False
    print("  improved overall: %s (+%.4f) ; kept improving PAST gen2: %s" % (kept, bf - hist[0][1], past2))
    print("  best net: %d hidden neurons, %d synapses, sigma=%.3f" % (len(best["hid"]), len(best["syn"]), best["sigma"]))
    print("=" * 76, flush=True)
