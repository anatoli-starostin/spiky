#!/usr/bin/env python3
"""
Toy demo: a single hyperplane LUT table == a spiking (latency-coded, leak-free
integrate-and-fire) analogue, exactly.

No SPNet, no checkpoints, no GPU — pure numpy on CPU.

The math (see task spec):

  LUT:  bit_k = 1[<W_k, x> + b_k > 0];  row = K bits as int;  out = V[row]

  Spiking analogue (latency coding + fixed-window voltage readout):
    - encode x as first-spike latencies   t_i = alpha - beta * x_i   (alpha big
      enough that all t_i >= 0; beta > 0)
    - hyperplane k = a leak-free IF neuron: each input i adds a ramp of slope
      W[k,i] starting at t_i, so
          V_k(t) = sum_i W[k,i] * (t - t_i) * [t >= t_i]
    - read the sign bit at a FIXED time T_read > max latency. There every ramp
      has arrived, so
          V_k(T_read) = (T_read - alpha) * sum_i W[k,i]  +  beta * <W_k, x>
      which is affine in <W_k, x>. Choosing the threshold
          theta_k = (T_read - alpha) * sum_i W[k,i]  -  beta * b_k
      gives   V_k(T_read) - theta_k = beta * (<W_k, x> + b_k),
      and since beta > 0:  sign(V_k(T_read) - theta_k) == sign(<W_k, x> + b_k).
      So the spiking bit  [V_k(T_read) > theta_k]  equals the LUT bit exactly.
"""
import numpy as np

# ----------------------------------------------------------------------------
# 1. TOY HYPERPLANE LUT  (fixed seed for reproducibility)
# ----------------------------------------------------------------------------
rng = np.random.default_rng(0)
D, K, Dout = 8, 3, 4          # input dim, #hyperplanes (2^K rows), output dim
N = 10_000                    # number of random test inputs

W = rng.standard_normal((K, D))          # signed hyperplane weights  (K x D)
b = rng.standard_normal(K)               # biases                     (K,)
V = rng.standard_normal((2 ** K, Dout))  # value table          (2^K x Dout)

POWERS = (1 << np.arange(K - 1, -1, -1)) # MSB-first bit packing (lutorch conv.)


def lut_bits(X):
    """Exact LUT address bits: 1[<W_k,x> + b_k > 0].  X:(N,D) -> (N,K)."""
    return (X @ W.T + b > 0).astype(int)


def bits_to_row(bits):
    """Pack K bits (MSB-first) into a row index.  (N,K) -> (N,)."""
    return bits @ POWERS


# ----------------------------------------------------------------------------
# 2. SPIKING ANALOGUE  (latency encoding + fixed-window readout)
# ----------------------------------------------------------------------------
ALPHA, BETA = 3.0, 1.0        # t_i = alpha - beta*x_i ;  x in [-1,1] -> t in [2,4]
T_READ = 5.0                  # fixed readout time, > max latency (=alpha+beta=4)

SUMW = W.sum(axis=1)                              # sum_i W[k,i]           (K,)
THETA = (T_READ - ALPHA) * SUMW - BETA * b        # readout thresholds     (K,)


def spiking_membrane(X):
    """Leak-free IF membrane at the fixed readout time.
    V_k(T_read) = sum_i W[k,i]*(T_read - t_i),  t_i = alpha - beta*x_i.  (N,K)."""
    Tlat = ALPHA - BETA * X                        # latencies  (N,D)
    assert Tlat.min() >= 0, "latency went negative; raise ALPHA"
    assert T_READ > Tlat.max(), "T_READ must exceed the largest latency"
    return (T_READ - Tlat) @ W.T                   # (N,D)@(D,K) -> (N,K)


def spiking_bits(X):
    """Fixed-window spiking address bits: [V_k(T_read) > theta_k]."""
    return (spiking_membrane(X) > THETA).astype(int)


# ----------------------------------------------------------------------------
# 3. VERIFY: exact LUT vs spiking analogue over many random inputs
# ----------------------------------------------------------------------------
X = rng.uniform(-1.0, 1.0, size=(N, D))

lut_b = lut_bits(X)
spk_b = spiking_bits(X)

bit_agree = (lut_b == spk_b).mean()
row_lut, row_spk = bits_to_row(lut_b), bits_to_row(spk_b)
row_match = (row_lut == row_spk).mean()
out_match = np.all(V[row_lut] == V[row_spk], axis=1).mean()

print("=" * 68)
print(f"TOY HYPERPLANE LUT -> SPIKING ANALOGUE   (D={D}, K={K}, "
      f"rows={2**K}, Dout={Dout}, N={N})")
print("=" * 68)
print(f"per-bit sign agreement (spiking vs exact) : {bit_agree*100:.4f}%  "
      f"({int(bit_agree*N*K)}/{N*K} bits)")
print(f"full row-match rate                       : {row_match*100:.4f}%  "
      f"({int(row_match*N)}/{N})")
print(f"exact output-vector match rate            : {out_match*100:.4f}%  "
      f"({int(out_match*N)}/{N})")
# how close is the spiking membrane margin to beta*(<W,x>+b)?  (should be ~0)
resid = np.abs((spiking_membrane(X) - THETA) - BETA * (X @ W.T + b)).max()
print(f"max |(V-theta) - beta*(<W,x>+b)| residual : {resid:.2e}  (fp only)")


# ----------------------------------------------------------------------------
# 4a. EXTRA — weight quantization: the precision floor
#     Quantize W,b to `bq` bits and compare the resulting bits to the
#     FULL-PRECISION LUT.  (LUT and spiking share weights, so they still
#     agree with each other exactly; this measures fidelity-to-original.)
# ----------------------------------------------------------------------------
def quantize(A, bits):
    m = np.abs(A).max()
    if bits >= 30 or m == 0:
        return A.copy()
    levels = 2 ** (bits - 1) - 1
    scale = m / levels
    return np.round(A / scale) * scale


print("\n" + "-" * 68)
print("EXTRA (a): weight quantization to b bits  ->  vs full-precision LUT")
print("-" * 68)
print(f"{'bits':>4} | {'per-bit agree':>13} | {'row-match':>10}")
for bq in (2, 3, 4, 5, 6, 8, 10, 12, 16):
    Wq, bq_ = quantize(W, bq), quantize(b, bq)
    bits_q = (X @ Wq.T + bq_ > 0).astype(int)
    ba = (bits_q == lut_b).mean()
    rm = (bits_to_row(bits_q) == row_lut).mean()
    print(f"{bq:>4} | {ba*100:>12.3f}% | {rm*100:>9.3f}%")


# ----------------------------------------------------------------------------
# 4b. EXTRA — first-crossing (causal) readout vs fixed-window.
#     A real IF neuron fires the instant its membrane first exceeds threshold,
#     using only the spikes that have arrived so far. If the membrane overshoots
#     theta mid-way and later drifts back (cumulative slope can go negative),
#     the causal bit differs from the fixed-window bit -> causal-set truncation.
# ----------------------------------------------------------------------------
def firstcross_bits(X):
    Tlat = ALPHA - BETA * X
    out = np.zeros((N, K), dtype=int)
    for n in range(N):
        t = Tlat[n]
        order = np.argsort(t)          # process spikes in arrival order
        ts = t[order]
        for k in range(K):
            w = W[k, order]
            maxV = 0.0                 # V(t)=0 before the first spike
            Vcur = 0.0
            cum = w[0]                 # slope after 1st arrival
            tprev = ts[0]
            for j in range(1, D):
                Vcur += cum * (ts[j] - tprev)
                if Vcur > maxV:
                    maxV = Vcur
                cum += w[j]
                tprev = ts[j]
            Vcur += cum * (T_READ - tprev)   # final segment to T_read
            if Vcur > maxV:
                maxV = Vcur
            # neuron fires (bit=1) iff membrane ever rises above threshold
            out[n, k] = 1 if maxV > THETA[k] else 0
    return out


fc_b = firstcross_bits(X)
fc_vs_fixed_bit = (fc_b != spk_b).mean()
fc_row_match = (bits_to_row(fc_b) == row_spk).mean()
print("\n" + "-" * 68)
print("EXTRA (b): first-crossing (causal) readout vs fixed-window")
print("-" * 68)
print(f"per-bit disagreement (causal vs fixed) : {fc_vs_fixed_bit*100:.3f}%")
print(f"causal row-match vs fixed-window       : {fc_row_match*100:.3f}%")
print("(nonzero disagreement = the causal-set truncation effect: an early "
      "overshoot\n latches the neuron before all ramps have arrived.)")
