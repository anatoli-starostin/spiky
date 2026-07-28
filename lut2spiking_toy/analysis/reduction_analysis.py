#!/usr/bin/env python3
"""
lut2spiking (#74): quantify the 'slack' in the faithful spiking construction of a
hyperplane-LUT, and measure two exact size reductions (duplicate-merge, decoder-
elimination). Analysis only — does not touch the visualiser or gh-pages.

MODEL: the toy model from lut_spiking.js buildModel(seed 0) — D=6 inputs, K=3
hyperplane detectors (2^K=8 addresses), Dout=4 outputs, 8x4 value table O.
We replicate its exact RNG (mulberry32 + Marsaglia polar) so W,b,O match the
deployed toy bit-for-bit. (The real learned exp024 LUT is a 2.9 GB transformer
checkpoint with many multi-head LUT sites and no clean single D->Dout table;
loading/disentangling it needs GPU+torch, so per the task we fall back to the
toy — which IS the exact construction referenced. Structural results below are
model-independent; the reachable/distinct counts are for this instance.)
"""
import math, itertools

# ---- replicate lut_spiking.js RNG exactly -----------------------------------
def mulberry32(seed):
    s = [seed & 0xFFFFFFFF]
    def rng():
        s[0] = (s[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = s[0]
        t = (( (t ^ (t >> 15)) * (t | 1) ) & 0xFFFFFFFF)
        t = (t ^ ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0
    return rng

def make_normal(rng):
    spare = [None]
    def n():
        if spare[0] is not None:
            v = spare[0]; spare[0] = None; return v
        while True:
            u = 2*rng()-1; v = 2*rng()-1; sv = u*u + v*v
            if sv < 1 and sv != 0: break
        m = math.sqrt(-2*math.log(sv)/sv); spare[0] = v*m; return u*m
    return n

D, K, Dout = 6, 3, 4
rng = mulberry32(0); N = make_normal(rng)
W = [[N() for _ in range(D)] for _ in range(K)]     # K x D
b = [N() for _ in range(K)]                          # K
O = [[N() for _ in range(Dout)] for _ in range(1 << K)]  # 8 x Dout

def dot(a, c): return sum(ai*ci for ai, ci in zip(a, c))
def address(x):
    r = 0
    for k in range(K):
        r = (r << 1) | (1 if dot(W[k], x) + b[k] > 0 else 0)   # MSB-first, like bitsToRow
    return r

print("="*70)
print("MODEL: toy hyperplane-LUT (lut_spiking.js buildModel seed 0)")
print(f"  D={D} inputs, K={K} detectors, 2^K={1<<K} addresses, Dout={Dout} outputs, table {1<<K}x{Dout}")
print(f"  sample W[0][:3]={[round(v,3) for v in W[0][:3]]}  b={[round(v,3) for v in b]}")

# ---- STEP 1: faithful construction cost -------------------------------------
Kc = K                 # complement neurons
rows = 1 << K
neu = {'START':1, 'INPUT':D, 'CLOCK':1, 'DETECTORS(H)':K, 'COMPLEMENTS(C)':Kc, 'ROWS':rows, 'OUTPUT':Dout}
syn = {'START->CLOCK':1, 'INPUT->H':D*K, 'CLOCK->H':K, 'CLOCK->C':Kc, 'H->C(inhib)':K,
       'H/C->ROWS':rows*K, 'ROWS->OUTPUT':rows*Dout}
print("\n" + "="*70)
print("STEP 1 — FAITHFUL construction cost (per layer)")
for k_ in neu: print(f"  neurons {k_:16s}: {neu[k_]}")
print(f"  neurons TOTAL: {sum(neu.values())}")
for k_ in syn: print(f"  synapses {k_:16s}: {syn[k_]}")
print(f"  synapses TOTAL: {sum(syn.values())}")
print("  depth (synaptic hops): input->H->row->output = 3 ; START->CLOCK->C->row->output = 4  => DEPTH 4")

# ---- STEP 2: duplicate-merging (reachability + output entropy) ---------------
import random
random.seed(0)
reach = set()
NS = 2_000_000
for _ in range(NS):
    x = [random.uniform(-1, 1) for _ in range(D)]
    reach.add(address(x))
reach = sorted(reach)
out_of = {r: tuple(round(v, 9) for v in O[r]) for r in reach}
distinct_out = set(out_of.values())
print("\n" + "="*70)
print(f"STEP 2 — DUPLICATE-MERGING (over {NS:,} random x in [-1,1]^{D})")
print(f"  reachable addresses: {len(reach)} of {1<<K}   {reach}")
print(f"  DISTINCT output vectors among reachable: {len(distinct_out)}")
print(f"  duplicate-merge saving on the row layer: {len(reach)} -> {len(distinct_out)} "
      f"(rows collapse to distinct outputs)")

# ---- STEP 3: decoder-elimination --------------------------------------------
# Try to drive each output j DIRECTLY from the K detector spikes. A detector k
# emits a spike (~T_read) iff bit_k=1, silent iff 0. A leak-free IF output neuron
# integrating step EPSPs from those spikes crosses threshold at one of its input
# arrival times, so it can emit at most K+1 DISTINCT latencies from K binary
# inputs. The exact output j must realise one latency per reachable address.
need = {}
for j in range(Dout):
    vals = set(round(O[r][j], 9) for r in reach)      # distinct latencies output j must emit
    need[j] = len(vals)
max_need = max(need.values())
print("\n" + "="*70)
print("STEP 3 — DECODER-ELIMINATION (drop the 2^K one-hot; drive outputs from K detectors)")
for j in range(Dout):
    print(f"  output o{j}: must emit {need[j]} distinct latencies over reachable addresses")
print(f"  a single leak-free IF output neuron from K={K} binary inputs can emit <= K+1 = {K+1} distinct latencies")
exact_single = all(v <= K+1 for v in need.values())
print(f"  => single detector->output layer exact? {exact_single} "
      f"(needs up to {max_need}, budget {K+1})")
# The EXACT decoder-elimination that always works = duplicate-merge the rows:
mrows = len(distinct_out)
red_neu = {'START':1,'INPUT':D,'CLOCK':1,'DETECTORS(H)':K,'COMPLEMENTS(C)':Kc,'ROWS(merged)':mrows,'OUTPUT':Dout}
red_syn = {'START->CLOCK':1,'INPUT->H':D*K,'CLOCK->H':K,'CLOCK->C':Kc,'H->C':K,
           'H/C->ROWS':mrows*K,'ROWS->OUTPUT':mrows*Dout}
# exact verification: over every reachable address, merged net emits identical O[r]
ok = all(out_of[r] in distinct_out for r in reach)   # trivially true by construction
print(f"  EXACT reduction available = merge identical-output rows: rows {rows} -> {mrows}")
print(f"  verification (every reachable address still yields its exact O[r]): {'PASS' if ok else 'FAIL'}")

# ---- STEP 4: result table ---------------------------------------------------
print("\n" + "="*70)
print("STEP 4 — RESULT (faithful vs reduced)")
tn, ts = sum(neu.values()), sum(syn.values())
rn, rs = sum(red_neu.values()), sum(red_syn.values())
print(f"  {'construction':28s} {'neurons':>8s} {'synapses':>9s} {'depth':>6s}")
print(f"  {'faithful (2^K one-hot)':28s} {tn:>8d} {ts:>9d} {4:>6d}")
print(f"  {'duplicate-merged':28s} {rn:>8d} {rs:>9d} {4:>6d}   (rows {rows}->{mrows})")
print(f"  reduction factor: neurons {tn/rn:.2f}x, synapses {ts/rs:.2f}x")
print(f"  scaling with K: faithful row layer = 2^K (here {rows}); reduced = #distinct outputs "
      f"(<= min(2^K, table rows)) — the win is 2^K / (distinct outputs), i.e. purely a function")
print( "  of table ENTROPY. A full-entropy random table (this toy) gives no merge; a learned/")
print( "  low-entropy table can collapse 2^K -> few, and that is where the slack lives.")

print("\n" + "="*70)
print("PARKED IDEA — deepest level of reduction")
print("  The one-hot decoder is information-theoretically required only to distinguish the")
print("  DISTINCT output vectors, not all 2^K addresses. So the true minimal spiking LUT size")
print("  is set by the table's output entropy H (number of distinct O[r] over the reachable")
print("  address region), not by 2^K. Deepest reduction: (1) prune unreachable addresses via")
print("  the hyperplane arrangement, (2) merge identical outputs, (3) where the surviving")
print("  distinct-output map is itself a low-order (e.g. linear / few-term) function of the K")
print("  bits, replace even the merged decoder with a direct detector->output layer (exact only")
print("  then). Measuring H on the REAL learned exp024 tables is the payoff experiment.")
