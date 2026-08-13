"""native_encoding_latency.py — LATENCY-INPUT variant of the toy-LUT re-encoding.

BOTH ends timing-coded:
  * INPUTS  = time-to-first-spike. Each input value -> a spike time t_in = A - B*value
    (earlier = larger). For the binary LUT (x in {+-1}^6): +1 fires EARLY (t_hi),
    -1 fires LATE (t_lo). BOTH actually fire (unlike the coincidence version where
    -1 is silent).
  * OUTPUTS = the existing clock-comb latency stage (larger value fires earlier).

The crux is a DELAY-LINE DECODER that turns input spike TIMING back into the
weighted coincidence the detectors need. One-tick impulse synapses discard timing,
so per input i we build a relay/latch neuron u_i that fires IFF the input arrived
EARLY, gated by a delay-tuned inhibitory CLOCK pulse:

    x_i --(+E, d=1)--> u_i          (leak-free, thr LOW)
    GATE --(-I, d=1)--> u_i         (GATE pulse lands BETWEEN t_hi and t_lo)

  - x_i EARLY: excitation E>=thr reaches u_i and fires it (spike emitted, v reset)
    BEFORE the inhibitory GATE arrives -> u_i = 1.
  - x_i LATE : the GATE (-I) arrives first, driving v=-I; the later excitation gives
    v=E-I < thr -> u_i never fires -> u_i = 0.

So u_i reconstructs the unary bit u_i = [x_i == +1], and u_i --(W[k][i], d=1)--> H_k
feeds the SAME clock-biased detector as before:  H_k fires iff  W.u > det_theta[k]
=  (algebra of x=2u-1)  W.x + b > 0.  Everything downstream (detector -> complement
-> row AND -> output value -> comb) is unchanged; leak-free neurons accumulate, so
the downstream is timing-insensitive and just shifts later.

Generalization to real-valued inputs: t_in = A - B*value gives a graded first-spike
time; a *bank* of GATE taps at successive delays (a genuine multi-tap delay line)
would threshold the arrival time into a thermometer code of the value. For the
binary LUT a single tap suffices and is validated here.

Build: DIRECT single-meta-per-delay path with per-edge weights (_grow_explicit
weights=..., PR #78), enabled crash-free by the initialize_neurons OOB fix (PR #80).
"""
import torch
import spiky_cuda  # noqa: F401
from lutmodel import build_model, lut_spec
from spnet_harness import build_packed, leakfree_meta
from spiky.util.synapse_growth import SynapseGrowthEngine
from spiky.spnet.spnet import SpikingNet, SynapseMeta, NeuronMeta, NeuronDataType

torch.set_num_threads(1)


def build_direct(n_nodes, node_meta, neuron_metas, syn):
    """syn = [(src_local, tgt_local, weight, delay), ...]. One synapse-meta per
    distinct delay; per-edge weights via _grow_explicit(weights=...)."""
    counts = [0] * len(neuron_metas)
    slot_of = {}
    for local in range(n_nodes):
        mi = node_meta[local]
        slot_of[local] = (mi, counts[mi])
        counts[mi] += 1
    counts = [max(1, c) for c in counts]

    delays = sorted(set(int(d) for (_, _, _, d) in syn))
    dmeta = {d: i for i, d in enumerate(delays)}
    syn_metas = [SynapseMeta(learning_rate=0.0, min_delay=d, max_delay=d,
                             min_weight=-1000.0, max_weight=1000.0, initial_weight=0.0,
                             _forward_group_size=2, _backward_group_size=2) for d in delays]

    spnet = SpikingNet(synapse_metas=syn_metas, neuron_metas=neuron_metas,
                       neuron_counts=counts, initial_synapse_capacity=max(1024, 4 * len(syn)))
    spnet.to_device('cpu')
    ids_by = [spnet.get_neuron_ids_by_meta(i) for i in range(len(neuron_metas))]

    def gid(local):
        mi, slot = slot_of[local]
        return int(ids_by[mi][slot].item())

    ge = SynapseGrowthEngine(device='cpu', synapse_group_size=2,
                             max_groups_in_buffer=max(4096, 8 * (len(syn) + sum(counts))))
    for i in range(len(neuron_metas)):
        ge.register_neuron_type(max_synapses=8 * n_nodes, growth_command_list=[])
    for i in range(len(neuron_metas)):
        ids = ids_by[i]
        coords = torch.stack([torch.arange(len(ids)).float(), torch.zeros(len(ids)),
                              torch.full((len(ids),), float(i))], dim=1)
        ge.add_neurons(neuron_type_index=i, identifiers=ids, coordinates=coords)

    triples = torch.tensor([[dmeta[int(d)], gid(s), gid(t)] for (s, t, w, d) in syn],
                           dtype=torch.int32)
    weights = torch.tensor([float(w) for (s, t, w, d) in syn], dtype=torch.float32)
    chunk = ge._grow_explicit(triples, 1, weights=weights)
    spnet.add_connections(chunk, 1)
    chunk.recycle()
    spnet.compile(shuffle_synapses_random_seed=1)
    spnet.to_device('cpu')
    return spnet, gid


# ------------------------------------------------------------------ model + consts
m = build_model(0)
W, b, V, K, D, Dout = m["W"], m["b"], m["V"], m["K"], m["D"], m["Dout"]
sumW = [sum(W[k]) for k in range(K)]
det_theta = [(sumW[k] - b[k]) / 2.0 for k in range(K)]

LOW_THR, THETA_BIG, COMP_THR, ROW_THR, OUT_THR = 0.5, 50.0, 5.0, 2.5, 6.0
CLK_W, INH_W, CLK_C_DELAY = 10.0, -20.0, 8
DEC_E, DEC_I = 1.0, 10.0        # decoder excitation / inhibition
COMB_DELTA, COMB_BASE, COMB_M = 0.5, 20, 30

# latency input timeline (ticks). value +1 -> stim at TIN_HI (early), -1 -> TIN_LO (late)
TSTART, TGATE, TIN_HI, TIN_LO = 1, 4, 2, 7
BIAS_DELAY = 4                  # START-bias arrival at H coincides with the u_i volley
READ_VALUE_TICK = 18
N_TICKS = 64


def value_to_tin(v):
    """t_in = A - B*value  (earlier spike = larger value). Binary: +1->TIN_HI, -1->TIN_LO."""
    A = (TIN_HI + TIN_LO) / 2.0
    B = (TIN_LO - TIN_HI) / 2.0
    return int(round(A - B * v))


# ------------------------------------------------------------------ topology
nodes = ["START", "GATE"] + ["x%d" % i for i in range(D)] + ["u%d" % i for i in range(D)] \
        + ["H%d" % k for k in range(K)] + ["C%d" % k for k in range(K)] \
        + ["r%d" % a for a in range(1 << K)] + ["o%d" % j for j in range(Dout)]
local = {n: i for i, n in enumerate(nodes)}
thr = {"START": LOW_THR, "GATE": LOW_THR}
for i in range(D):
    thr["x%d" % i] = LOW_THR
    thr["u%d" % i] = LOW_THR
for k in range(K):
    thr["H%d" % k] = THETA_BIG
    thr["C%d" % k] = COMP_THR
for a in range(1 << K):
    thr["r%d" % a] = ROW_THR
for j in range(Dout):
    thr["o%d" % j] = OUT_THR
uthr = sorted(set(thr.values()))
neuron_metas = [leakfree_meta(t) for t in uthr]
node_meta = [uthr.index(thr[n]) for n in nodes]

syn = []
# --- delay-line decoder: u_i fires iff x_i early ---
for i in range(D):
    syn.append((local["x%d" % i], local["u%d" % i], DEC_E, 1))
    syn.append((local["GATE"], local["u%d" % i], -DEC_I, 1))
# --- detectors read the decoded unary bits (clock-biased weighted sum) ---
for k in range(K):
    for i in range(D):
        syn.append((local["u%d" % i], local["H%d" % k], W[k][i], 1))
    syn.append((local["START"], local["H%d" % k], THETA_BIG - det_theta[k], BIAS_DELAY))
    syn.append((local["START"], local["C%d" % k], CLK_W, CLK_C_DELAY))
    syn.append((local["H%d" % k], local["C%d" % k], INH_W, 1))
# --- row AND-gate + output value ---
for a in range(1 << K):
    for k in range(K):
        bit = (a >> (K - 1 - k)) & 1
        syn.append((local[("H%d" % k) if bit else ("C%d" % k)], local["r%d" % a], 1.0, 1))
    for j in range(Dout):
        syn.append((local["r%d" % a], local["o%d" % j], V[a][j], 1))
# --- clock comb -> spiking output latency stage ---
for j in range(Dout):
    for i in range(COMB_M):
        syn.append((local["START"], local["o%d" % j], COMB_DELTA, COMB_BASE + i))

# --- crash-confirmation of the DIRECT single-meta-per-delay per-edge-weights path
# (enabled by the initialize_neurons OOB fix, PR #80). It builds without crashing;
# note that _grow_explicit's group-aligned weights= buffer does NOT map 1:1 to input
# triple order (weights scramble within a source's edge set), so we do the actual
# correctness run on the proven palette build below.
_d_spnet, _ = build_direct(len(nodes), node_meta, neuron_metas, syn)
print("DIRECT per-edge-weights build: %d synapses, %d distinct delays — no crash (PR #80 OK)"
      % (_d_spnet.n_synapses(), len(set(d for (_, _, _, d) in syn))), flush=True)

# --- real net via the palette (weights guaranteed correct: one meta per (w,delay)) ---
packed = build_packed([{"n_nodes": len(nodes), "node_meta": node_meta, "synapses": syn}], neuron_metas)
spnet, _gidc = packed["spnet"], packed["gid"]
def gid(loc):
    return _gidc(0, loc)
print("built PALETTE: %d synapses, %d syn-metas" % (packed["n_synapses"], packed["n_syn_metas"]), flush=True)

Hids = torch.tensor([gid(local["H%d" % k]) for k in range(K)], dtype=torch.int32)
Rids = torch.tensor([gid(local["r%d" % a]) for a in range(1 << K)], dtype=torch.int32)
Oids = torch.tensor([gid(local["o%d" % j]) for j in range(Dout)], dtype=torch.int32)


def run_input(x, record=False):
    """stimulate START(t=TSTART), GATE(t=TGATE), and each x_i at its latency tick."""
    events = [(gid(local["START"]), TSTART), (gid(local["GATE"]), TGATE)]
    for i in range(D):
        events.append((gid(local["x%d" % i]), value_to_tin(x[i])))
    S = torch.zeros((1, N_TICKS, len(events)), dtype=torch.int32)
    Vv = torch.zeros((1, N_TICKS, len(events)), dtype=torch.float32)
    for j, (nid, tk) in enumerate(events):
        S[0, tk, j] = nid
        Vv[0, tk, j] = 50.0
    spnet.process_ticks(n_ticks_to_process=N_TICKS, batch_size=1, n_input_ticks=N_TICKS,
                        input_values=Vv, do_train=False, sparse_input=S,
                        do_record_voltage=True, do_reset_context=True)


# ------------------------------------------------------------------ validate 64/64
spec = lut_spec(m)
row_ok = out_ok = order_ok = 0
errs = []
sample = None
for si, s in enumerate(spec):
    x = s["x"]
    run_input(x)
    rr = spnet.export_neuron_data(Rids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(1 << K, N_TICKS)
    ovlt = spnet.export_neuron_data(Oids, 1, NeuronDataType.Voltage, 0, N_TICKS - 1).view(Dout, N_TICKS)
    ospk = spnet.export_neuron_data(Oids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(Dout, N_TICKS)
    rows = [a for a in range(1 << K) if (rr[a] > 0.5).any().item()]
    row = rows[0] if len(rows) == 1 else -1
    val = [float(ovlt[j, READ_VALUE_TICK].item()) for j in range(Dout)]
    t_out = []
    for j in range(Dout):
        nz = torch.nonzero(ospk[j] > 0.5).flatten()
        t_out.append(int(nz[0].item()) if nz.numel() else None)
    if row == s["row"]:
        row_ok += 1
    for j in range(Dout):
        errs.append(abs(val[j] - s["out"][j]))
    if row == s["row"] and all(abs(val[j] - s["out"][j]) < 1e-4 for j in range(Dout)):
        out_ok += 1
    if all(t is not None for t in t_out):
        by_value = sorted(range(Dout), key=lambda j: -s["out"][j])
        ok = True
        for a2 in range(Dout):
            for b2 in range(a2 + 1, Dout):
                ja, jb = by_value[a2], by_value[b2]
                if t_out[ja] > t_out[jb]:
                    ok = False
        if ok:
            order_ok += 1
    if si == 5:
        tin = [value_to_tin(x[i]) for i in range(D)]
        sample = (x, tin, s["bits"], s["row"], row, s["out"], val, t_out)

n = len(spec)
print("=" * 66)
print("LATENCY-INPUT LUT ENCODING — real spnet CPU, 64 inputs")
print("-" * 66)
print("  row-selection accuracy : %d/%d = %.1f%%" % (row_ok, n, 100 * row_ok / n))
print("  output VALUE match (membrane, tol 1e-4): %d/%d = %.1f%%" % (out_ok, n, 100 * out_ok / n))
print("  output value error: mean=%.3e max=%.3e" % (sum(errs) / len(errs), max(errs)))
print("  SPIKE-ORDER matches value ranking: %d/%d = %.1f%%" % (order_ok, n, 100 * order_ok / n))
if sample:
    x, tin, bits, trow, grow, tv, mv, to = sample
    print("-" * 66)
    print("  sample input x=%s" % x)
    print("    input spike ticks (t_in, x0..x5): %s   (+1=early %d, -1=late %d)" % (tin, TIN_HI, TIN_LO))
    print("    true bits=%s row=%d ; decoded row=%d" % (bits, trow, grow))
    print("    true LUT out : %s" % [round(v, 3) for v in tv])
    print("    membrane val : %s" % [round(v, 3) for v in mv])
    print("    output spike ticks (o0..o3): %s  -> order %s" %
          (to, sorted(range(Dout), key=lambda j: (to[j] if to[j] is not None else 1e9))))
print("=" * 66, flush=True)
