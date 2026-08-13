"""Run the ported 26/84 construction on REAL spnet (CPU) over all 64 inputs.
Two neuron configs: leak-free (v'=I via meta) and stock Izhikevich. Measure
row-match rate, output error, detector-bit flips vs the idealized lif reference,
plus solo-vs-packed no-cross-talk and smoothed-fitness sensitivity."""
import torch
from lutmodel import build_model, lut_spec, lut_bits, bits_to_row
from construction import build_construction, ALPHA, DBASE, KAPPA, T_READ
from spnet_harness import build_packed, run, leakfree_meta, stock_meta

m = build_model(0)
K, Dout = m["K"], m["Dout"]
neurons, syns = build_construction(m)         # lif neurons + (src,tgt,delay,w,kind)
names = [n.nid for n in neurons]
local = {nm: i for i, nm in enumerate(names)}
thr = {n.nid: n.theta for n in neurons}
N_NODES = len(names)

# node -> neuron-meta index, by threshold (leak-free) ; stock uses a single meta
uthr = sorted(set(thr.values()))
LF_METAS = [leakfree_meta(t) for t in uthr]
lf_node_meta = [uthr.index(thr[nm]) for nm in names]
ST_METAS = [stock_meta()]
st_node_meta = [0] * N_NODES

SYN = [(local[s], local[t], float(w), int(d)) for (s, t, d, w, kind) in syns]
GENOME = {"n_nodes": N_NODES, "node_meta": None, "synapses": SYN}

Hloc = [local["H%d" % k] for k in range(K)]
Rloc = [local["r%d" % a] for a in range(1 << K)]
Oloc = [local["o%d" % j] for j in range(Dout)]
READ = Hloc + Rloc + Oloc
IN_CUR = 100.0
N_TICKS = 30


def input_events(x):
    ev = [(local["START"], 0, IN_CUR)]
    for i, xi in enumerate(x):
        ev.append((local["x%d" % i], ALPHA + xi, IN_CUR))
    return ev


def decode_run(fired):
    hbits = [1 if fired[Hloc[k]] is not None else 0 for k in range(K)]
    rows = [a for a in range(1 << K) if fired[Rloc[a]] is not None]
    row = rows[0] if len(rows) == 1 else -1
    t_row = fired[Rloc[row]] if row >= 0 else None
    dec = []
    for j in range(Dout):
        to = fired[Oloc[j]]
        if to is not None and t_row is not None:
            dec.append((DBASE - (to - t_row)) / KAPPA)
        else:
            dec.append(None)
    return hbits, row, dec


def eval_config(neuron_metas, node_meta):
    g = dict(GENOME); g["node_meta"] = node_meta
    packed = build_packed([g], neuron_metas)
    spec = lut_spec(m)
    row_ok = 0
    bitflips = 0
    errs = []
    rows_seen = {}
    for s in spec:
        res = run(packed, [input_events(s["x"])], READ, N_TICKS)[0]
        hbits, row, dec = decode_run(res)
        rows_seen[row] = rows_seen.get(row, 0) + 1
        if row == s["row"]:
            row_ok += 1
        lb = s["bits"]
        bitflips += sum(1 for k in range(K) if hbits[k] != lb[k])
        tgt = s["out"]
        for j in range(Dout):
            if dec[j] is None:
                errs.append(abs(tgt[j]) + 5.0)  # no-spike penalty
            else:
                errs.append(abs(dec[j] - tgt[j]))
    n = len(spec)
    return {
        "n": n, "row_ok": row_ok, "match_rate": row_ok / n,
        "bitflips": bitflips, "bitflip_rate": bitflips / (n * K),
        "mean_err": sum(errs) / len(errs), "max_err": max(errs),
        "rows_seen": rows_seen, "packed": packed,
    }


print("=" * 70)
print("PORTED 26/84 CONSTRUCTION ON REAL SPNET (CPU) — 64 inputs, seed-0 LUT")
print("=" * 70)
for label, metas, nmeta in [("LEAK-FREE (v'=I via meta)", LF_METAS, lf_node_meta),
                            ("STOCK Izhikevich", ST_METAS, st_node_meta)]:
    r = eval_config(metas, nmeta)
    print("\n[%s]  synapses=%d" % (label, r["packed"]["n_synapses"]))
    print("  row match rate : %d/%d = %.1f%%" % (r["row_ok"], r["n"], 100 * r["match_rate"]))
    print("  detector-bit flips vs lif reference: %d / %d (%.1f%%)"
          % (r["bitflips"], r["n"] * K, 100 * r["bitflip_rate"]))
    print("  output error   : mean=%.3f  max=%.3f" % (r["mean_err"], r["max_err"]))
    print("  rows the net actually produced (row->count over 64): %s" % r["rows_seen"])

# ---- solo vs packed (no cross-talk) using the real construction (many metas) ----
print("\n" + "-" * 70)
print("SOLO vs PACKED (no cross-talk) — leak-free construction, 3 test inputs")
spec = lut_spec(m)
test_inputs = [spec[0]["x"], spec[21]["x"], spec[42]["x"]]
g = dict(GENOME); g["node_meta"] = lf_node_meta
solo_scores = []
for x in test_inputs:
    p = build_packed([g], LF_METAS)
    solo_scores.append(run(p, [input_events(x)], READ, N_TICKS)[0])
p3 = build_packed([g, g, g], LF_METAS)
packed_scores = run(p3, [input_events(x) for x in test_inputs], READ, N_TICKS)
identical = all(packed_scores[c] == solo_scores[c] for c in range(3))
print("  packed n_synapses=%d (expect 3x84=252)" % p3["n_synapses"])
for c in range(3):
    print("  input %d: solo==packed? %s" % (c, packed_scores[c] == solo_scores[c]))
print("  ALL identical (no cross-talk):", identical)

# ---- smoothed fitness varies under perturbation ----
print("\n" + "-" * 70)
print("SMOOTHED FITNESS sensitivity (mean output error) under weight perturbations")
import random
rng = random.Random(3)
def fitness(genome_syn):
    g2 = {"n_nodes": N_NODES, "node_meta": lf_node_meta, "synapses": genome_syn}
    p = build_packed([g2], LF_METAS)
    spec = lut_spec(m)
    errs = []
    for s in spec[:16]:
        res = run(p, [input_events(s["x"])], READ, N_TICKS)[0]
        _, row, dec = decode_run(res)
        for j in range(Dout):
            errs.append(abs(dec[j] - s["out"][j]) if dec[j] is not None else abs(s["out"][j]) + 5.0)
    return sum(errs) / len(errs)
base = fitness(SYN)
print("  base mean-error (16 inputs): %.4f" % base)
vals = [base]
for trial in range(4):
    pert = [(s, t, w * (1 + rng.uniform(-0.3, 0.3)), d) for (s, t, w, d) in SYN]
    f = fitness(pert)
    vals.append(f)
    print("  perturbed trial %d: %.4f  (delta %+.4f)" % (trial + 1, f, f - base))
spread = max(vals) - min(vals)
print("  fitness spread over trials: %.4f  -> %s" % (spread, "VARIES (usable gradient)" if spread > 1e-6 else "FLAT"))
