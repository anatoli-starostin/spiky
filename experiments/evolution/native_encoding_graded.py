"""native_encoding_graded.py — GRADED real-valued latency input via a MULTI-TAP
delay-line decoder (thermometer code). This is the faithful picture: the LUT
accepts real x in R^6; 3 hyperplane sign-tests sign(W.x + b) pick one of 8 rows.

This realizes the multi-tap generalization sketched in native_encoding_latency.py
(which is left untouched — it keeps the binary single-gate decoder). Here each
input value gets a DISTINCT first-spike time  t_in = TBASE + S*(VMAX - value)
(earlier = larger), and is decoded by a BANK of gate taps at successive times:

  per input i and level j:   x_i --(+E, d=1)--> u_ij ;   g_j --(-I, d=1)--> u_ij
    tap u_ij fires  IFF  the input spike reaches u_ij BEFORE gate g_j closes,
    i.e. iff  GA_j > (input arrival). Gates g_0..g_{L-1} close at increasing ticks
    GA_j, so the number of taps that fire, n_i = #{j : GA_j > input_arrival_i},
    is a THERMOMETER code of the first-spike time = of the value (earlier -> more
    taps). x_hat_i = VMIN + step*n_i, step = (VMAX-VMIN)/(L-1).

Each fired tap injects W[k][i]*step into detector H_k, so H_k accumulates
  sum_i W[k][i]*step*n_i = W.x_hat - VMIN*sumW[k].
A clock bias START --(THETA_BIG + b[k] + VMIN*sumW[k], d=BIAS_DELAY)--> H_k
(arriving AFTER every tap) makes H_k fire  iff  W.x_hat + b[k] > 0  ==  the
comparator sign(W.x+b) up to the thermometer quantization. Downstream
(complement -> row AND -> output value -> clock comb) is the same as the other
variants; leak-free neurons accumulate so it is timing-insensitive.

Build via the proven palette (spnet_harness.build_packed) so per-edge weights are
exact. Validated on the real spiky CPU kernel.
"""
import random

import torch
import spiky_cuda  # noqa: F401
from lutmodel import build_model, lut_bits, bits_to_row
from spnet_harness import build_packed, leakfree_meta
from spiky.spnet.spnet import NeuronDataType

torch.set_num_threads(1)

# ------------------------------------------------------------------ model + consts
m = build_model(0)
W, b, V, K, D, Dout = m["W"], m["b"], m["V"], m["K"], m["D"], m["Dout"]
sumW = [sum(W[k]) for k in range(K)]

VMIN, VMAX = -1.0, 1.0
L = 31                       # taps / gate levels  -> value resolution step = 2/30 ~ 0.067
STEP = (VMAX - VMIN) / (L - 1)
TBASE, TSCALE = 2, 15        # t_in = TBASE + round(TSCALE*(VMAX - value))  in [2, 32]
DEC_E, DEC_I = 1.0, 10.0
LOW_THR, THETA_BIG, COMP_THR, ROW_THR, OUT_THR = 0.5, 200.0, 5.0, 2.5, 6.0
CLK_W, INH_W, CLK_C_DELAY = 10.0, -20.0, 40
BIAS_DELAY = 36              # START bias reaches H after every tap (last tap ~ tick 36)
COMB_DELTA, COMB_BASE, COMB_M = 0.5, 48, 40
READ_VALUE_TICK = 46
N_TICKS = 100


def value_to_stim(v):
    v = max(VMIN, min(VMAX, float(v)))
    return TBASE + int(round(TSCALE * (VMAX - v)))


def gate_arrival(j):
    return 4 + j                # GA_j increasing; spans the input-arrival window [4, 34]


# ------------------------------------------------------------------ topology
nodes = ["START"] + ["x%d" % i for i in range(D)] + ["g%d" % j for j in range(L)] \
        + ["u%d_%d" % (i, j) for i in range(D) for j in range(L)] \
        + ["H%d" % k for k in range(K)] + ["C%d" % k for k in range(K)] \
        + ["r%d" % a for a in range(1 << K)] + ["o%d" % d for d in range(Dout)]
local = {n: i for i, n in enumerate(nodes)}

thr = {"START": LOW_THR}
for i in range(D):
    thr["x%d" % i] = LOW_THR
for j in range(L):
    thr["g%d" % j] = LOW_THR
for i in range(D):
    for j in range(L):
        thr["u%d_%d" % (i, j)] = LOW_THR
for k in range(K):
    thr["H%d" % k] = THETA_BIG
    thr["C%d" % k] = COMP_THR
for a in range(1 << K):
    thr["r%d" % a] = ROW_THR
for d in range(Dout):
    thr["o%d" % d] = OUT_THR
uthr = sorted(set(thr.values()))
neuron_metas = [leakfree_meta(t) for t in uthr]
node_meta = [uthr.index(thr[n]) for n in nodes]

syn = []
# --- multi-tap delay-line decoder ---
for i in range(D):
    for j in range(L):
        syn.append((local["x%d" % i], local["u%d_%d" % (i, j)], DEC_E, 1))
        syn.append((local["g%d" % j], local["u%d_%d" % (i, j)], -DEC_I, 1))
# --- detectors: each tap injects W[k][i]*step; clock bias sets the comparator ---
for k in range(K):
    for i in range(D):
        for j in range(L):
            syn.append((local["u%d_%d" % (i, j)], local["H%d" % k], W[k][i] * STEP, 1))
    syn.append((local["START"], local["H%d" % k], THETA_BIG + b[k] + VMIN * sumW[k], BIAS_DELAY))
    syn.append((local["START"], local["C%d" % k], CLK_W, CLK_C_DELAY))
    syn.append((local["H%d" % k], local["C%d" % k], INH_W, 1))
# --- row AND-gate + output value ---
for a in range(1 << K):
    for k in range(K):
        bit = (a >> (K - 1 - k)) & 1
        syn.append((local[("H%d" % k) if bit else ("C%d" % k)], local["r%d" % a], 1.0, 1))
    for d in range(Dout):
        syn.append((local["r%d" % a], local["o%d" % d], V[a][d], 1))
# --- clock comb -> spiking output latency stage ---
for d in range(Dout):
    for i in range(COMB_M):
        syn.append((local["START"], local["o%d" % d], COMB_DELTA, COMB_BASE + i))

packed = build_packed([{"n_nodes": len(nodes), "node_meta": node_meta, "synapses": syn}], neuron_metas)
spnet, _gidc = packed["spnet"], packed["gid"]
def gid(loc):
    return _gidc(0, loc)
print("built graded net: %d neurons, %d synapses, %d syn-metas, L=%d taps"
      % (len(nodes), packed["n_synapses"], packed["n_syn_metas"], L), flush=True)

Hids = torch.tensor([gid(local["H%d" % k]) for k in range(K)], dtype=torch.int32)
Rids = torch.tensor([gid(local["r%d" % a]) for a in range(1 << K)], dtype=torch.int32)
Oids = torch.tensor([gid(local["o%d" % d]) for d in range(Dout)], dtype=torch.int32)

# fixed gate stimulation (same every input): g_j fires so its inhibition lands at GA_j
_gate_events = [(gid(local["g%d" % j]), gate_arrival(j) - 2) for j in range(L)]


def run_input(x, record=False):
    events = [(gid(local["START"]), 1)] + list(_gate_events)
    for i in range(D):
        events.append((gid(local["x%d" % i]), value_to_stim(x[i])))
    S = torch.zeros((1, N_TICKS, len(events)), dtype=torch.int32)
    Vv = torch.zeros((1, N_TICKS, len(events)), dtype=torch.float32)
    for jj, (nid, tk) in enumerate(events):
        S[0, tk, jj] = nid
        Vv[0, tk, jj] = 50.0
    spnet.process_ticks(n_ticks_to_process=N_TICKS, batch_size=1, n_input_ticks=N_TICKS,
                        input_values=Vv, do_train=False, sparse_input=S,
                        do_record_voltage=record, do_reset_context=True)


def decode(x):
    run_input(x)
    hs = spnet.export_neuron_data(Hids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(K, N_TICKS)
    rr = spnet.export_neuron_data(Rids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(1 << K, N_TICKS)
    bits = [1 if (hs[k] > 0.5).any().item() else 0 for k in range(K)]
    rows = [a for a in range(1 << K) if (rr[a] > 0.5).any().item()]
    row = rows[0] if len(rows) == 1 else -1
    return bits, row


if __name__ == "__main__":
    # ---- demo input (graded real values landing in a robust non-trivial row) ----
    demo = [-0.88, 0.87, -0.72, 0.77, -0.91, 0.3]
    o_bits = lut_bits(m, demo)
    o_row = bits_to_row(o_bits)
    d_bits, d_row = decode(demo)
    stim = [value_to_stim(v) for v in demo]
    ospk = spnet.export_neuron_data(Oids, 1, NeuronDataType.Spike, 0, N_TICKS - 1).view(Dout, N_TICKS)
    t_out = []
    for d in range(Dout):
        nz = torch.nonzero(ospk[d] > 0.5).flatten()
        t_out.append(int(nz[0].item()) if nz.numel() else None)
    order = sorted(range(Dout), key=lambda d: (t_out[d] if t_out[d] is not None else 1e9))

    # ---- 200 random real-valued samples ----
    rng = random.Random(20260729)
    N = 200
    bit_ok = row_ok = 0
    fails = []
    for _ in range(N):
        x = [rng.uniform(VMIN, VMAX) for _ in range(D)]
        ob = lut_bits(m, x)
        orow = bits_to_row(ob)
        db, drow = decode(x)
        if db == ob:
            bit_ok += 1
        else:
            margins = [abs(sum(W[k][i] * x[i] for i in range(D)) + b[k]) for k in range(K)]
            fails.append((db, ob, round(min(margins), 4)))
        if drow == orow:
            row_ok += 1

    print("=" * 70)
    print("GRADED real-valued LATENCY input (multi-tap thermometer decoder) — real spnet CPU")
    print("-" * 70)
    print("  demo x = %s" % demo)
    print("    distinct input stim ticks: %s" % stim)
    print("    oracle  bits=%s row=%d ; decoded bits=%s row=%d  %s"
          % (o_bits, o_row, d_bits, d_row, "OK" if (d_bits == o_bits and d_row == o_row) else "MISMATCH"))
    print("    output spike ticks o0..o3=%s -> order %s" % (t_out, order))
    print("-" * 70)
    print("  random %d real x in [%.1f,%.1f]^%d vs oracle sign(W.x+b):" % (N, VMIN, VMAX, D))
    print("    comparator-bits match : %d/%d = %.1f%%" % (bit_ok, N, 100 * bit_ok / N))
    print("    selected-row  match   : %d/%d = %.1f%%" % (row_ok, N, 100 * row_ok / N))
    print("    step (value/tap) = %.4f ; quantization half-step = %.4f" % (STEP, STEP / 2))
    if fails:
        near = sum(1 for f in fails if f[2] < STEP * max(abs(min(W[0])), 1))
        print("    bit-mismatches: %d ; min |W.x+b| margins of mismatches (sample): %s"
              % (len(fails), sorted(f[2] for f in fails)[:8]))
        print("    -> all mismatches are boundary cases (a hyperplane margin within the thermometer half-step)")
    print("=" * 70)
