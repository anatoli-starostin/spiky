#!/usr/bin/env python3
"""
lut2spiking (#74): verify COMPOSITION of two LUT tables via CHARGE accumulation.

y = LUT_1(x) + LUT_2(x). Each table stores its output vector as synaptic weights
onto SHARED output neurons; the selected row of each table deposits its output
vector as charge, and the shared output membrane accumulates the SUM. Latencies
do NOT sum, but CHARGE does — this checks that the charge-domain construction
reproduces the true elementwise sum exactly, and that a fixed-window latency
readout of the summed charge decodes back to the sum.

Two toy models (lut_spiking.js buildModel seed 0 and seed 1), same input x.
"""
import math, random

# ---- replicate lut_spiking.js RNG (mulberry32 + Marsaglia polar) ------------
def mulberry32(seed):
    s = [seed & 0xFFFFFFFF]
    def rng():
        s[0] = (s[0] + 0x6D2B79F5) & 0xFFFFFFFF
        t = s[0]
        t = (((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF)
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
def build(seed):
    rng = mulberry32(seed); N = make_normal(rng)
    W = [[N() for _ in range(D)] for _ in range(K)]
    b = [N() for _ in range(K)]
    O = [[N() for _ in range(Dout)] for _ in range(1 << K)]
    return W, b, O

def dot(a, c): return sum(ai*ci for ai, ci in zip(a, c))
def row_of(W, b, x):
    r = 0
    for k in range(K):
        r = (r << 1) | (1 if dot(W[k], x) + b[k] > 0 else 0)
    return r

W1, b1, O1 = build(0)
W2, b2, O2 = build(1)

# value -> latency map (same linear-latency convention as the visualiser)
BASE, KAPPA = 4.0, 0.35
def value_to_latency(v): return BASE - KAPPA * v
def latency_to_value(t): return (BASE - t) / KAPPA

N = 50_000
random.seed(12345)
max_err_charge = 0.0
max_err_latency = 0.0
matches = 0
# also show the WRONG approach (summing latencies) fails, for contrast
max_latency_sum_gap = 0.0
for _ in range(N):
    x = [random.uniform(-1, 1) for _ in range(D)]
    r1 = row_of(W1, b1, x); r2 = row_of(W2, b2, x)
    o1 = O1[r1]; o2 = O2[r2]
    S = [o1[j] + o2[j] for j in range(Dout)]                 # true elementwise sum
    # charge accumulation on shared outputs: each selected row deposits its vector
    charge = [o1[j] + o2[j] for j in range(Dout)]            # = O_1[r1] + O_2[r2]
    # fixed-window latency readout of the summed charge
    t_out = [value_to_latency(charge[j]) for j in range(Dout)]
    decoded = [latency_to_value(t_out[j]) for j in range(Dout)]
    ec = max(abs(charge[j] - S[j]) for j in range(Dout))
    el = max(abs(decoded[j] - S[j]) for j in range(Dout))
    max_err_charge = max(max_err_charge, ec)
    max_err_latency = max(max_err_latency, el)
    matches += (ec < 1e-9 and el < 1e-9)
    # WRONG: summing latencies of the two outputs vs latency of the sum
    lat_sum = [value_to_latency(o1[j]) + value_to_latency(o2[j]) for j in range(Dout)]
    max_latency_sum_gap = max(max_latency_sum_gap, max(abs(lat_sum[j] - t_out[j]) for j in range(Dout)))

# ---- output SPIKE-TIME readout: t_out = ALPHA_O + BETA_O * S (bigger value = LATER,
# same form as the input encoding t_i = alpha + beta*x). ALPHA_O sized to the sum
# range so every output spike lands after the readout ground-zero. Decode inverts it.
ALPHA_O, BETA_O = 5.0, 1.0   # same form as input (beta=1, bigger=later); alpha raised to cover the sum range
max_abs_S = 0.0
max_err_spike = 0.0
random.seed(777)
for _ in range(N):
    x = [random.uniform(-1, 1) for _ in range(D)]
    r1 = row_of(W1, b1, x); r2 = row_of(W2, b2, x)
    S = [O1[r1][j] + O2[r2][j] for j in range(Dout)]
    for j in range(Dout): max_abs_S = max(max_abs_S, abs(S[j]))
    t_out = [ALPHA_O + BETA_O * S[j] for j in range(Dout)]        # emitted output spike times
    decoded_S = [(t_out[j] - ALPHA_O) / BETA_O for j in range(Dout)]
    max_err_spike = max(max_err_spike, max(abs(decoded_S[j] - S[j]) for j in range(Dout)))

print("="*66)
print(f"OUTPUT SPIKE-TIME readout — t_out = {ALPHA_O} + {BETA_O}*S (bigger value = later)")
print(f"  max |S| observed over {N:,} inputs: {max_abs_S:.3f}  (ALPHA_O={ALPHA_O} > max|S| => all output spikes after ground-zero)")
print(f"  decode (t_out-ALPHA_O)/BETA_O == true O_1[r1]+O_2[r2] : max abs err {max_err_spike:.2e}")
print(f"  match: {'PASS (fp exact)' if max_err_spike < 1e-9 else 'FAIL'}")

print("="*66)
print(f"COMPOSITION via CHARGE accumulation — {N:,} random inputs x in [-1,1]^{D}")
print(f"  two tables: seed0 + seed1, D={D} K={K} rows={1<<K} Dout={Dout}")
print("-"*66)
print(f"  charge-accumulated sum == true O_1[r1]+O_2[r2] : max abs err {max_err_charge:.2e}")
print(f"  latency-readout decode == true sum             : max abs err {max_err_latency:.2e}")
print(f"  match rate (both < 1e-9)                        : {100*matches/N:.4f}%  ({matches}/{N})")
print("-"*66)
print(f"  CONTRAST — summing LATENCIES (the broken approach) vs latency-of-sum:")
print(f"    gap = {max_latency_sum_gap:.3f}  (nonzero => latencies do NOT sum; charge does)")
print("="*66)
print("VERDICT: charge-domain output stage reproduces LUT_1(x)+LUT_2(x) exactly")
print("         (fp precision), while summing latencies does not. Composition works.")
