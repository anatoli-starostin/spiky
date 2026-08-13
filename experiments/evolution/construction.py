"""construction.py — our EXACT model-1 single-table circuit, expressed in the
general izhik_sim graph representation as leak-free-IF ('lif') neurons.

This is the correctness ORACLE / hard upper bound: 26 neurons, 84 synapses.
Integer-timed re-derivation of lut_spiking.js simulateCircuit (binary inputs):
  t_i = ALPHA + x_i (x in {-1,+1}); readout clock at T_READ; detector ramp
  weight = -W; theta_k = -(T_READ-ALPHA+1)*sumW_k - b_k keeps the discrete
  arrival+integrate-same-tick readout an EXACT sign test of <W_k,x>+b_k.
Layers: INPUT -> DETECT(H) -> COMPL(C=NOT H) -> ROWS(1-of-8) -> OUTPUT(o).
"""
from izhik_sim import Neuron

ALPHA = 2       # input latency base: t_i = ALPHA + x_i  (x in {-1,+1} -> {1,3})
T_READ = 5      # readout clock tick
THETA = 50.0    # detector firing threshold
INH = -20.0     # H -> C feedforward inhibition
OW = 5.0        # row -> output weight
DBASE, KAPPA = 3.0, 0.35   # output conduction-delay code (only affects o timing)
CLK_TICK = 4    # START -> CLK delay (CLK fires here); CLK->H arrives at T_READ


def build_construction(m):
    W, b, V, K, D, Dout = m["W"], m["b"], m["V"], m["K"], m["D"], m["Dout"]
    C = (T_READ - ALPHA + 1)                      # discrete ramp-count constant
    sumW = [sum(W[k]) for k in range(K)]
    th = [-C * sumW[k] - b[k] for k in range(K)]  # detector threshold offset

    neurons = [Neuron("START", "lif", theta=0.5), Neuron("CLK", "lif", theta=0.5)]
    for i in range(D):
        neurons.append(Neuron("x%d" % i, "lif", theta=0.5))
    for k in range(K):
        neurons.append(Neuron("H%d" % k, "lif", theta=THETA))
    for k in range(K):
        neurons.append(Neuron("C%d" % k, "lif", theta=0.5))
    for a in range(1 << K):
        neurons.append(Neuron("r%d" % a, "lif", theta=2.5))
    for j in range(Dout):
        neurons.append(Neuron("o%d" % j, "lif", theta=0.5))

    syn = []
    for k in range(K):
        for i in range(D):
            syn.append(("x%d" % i, "H%d" % k, 0, -W[k][i], "cur"))    # input ramps (Wd=-W)
        syn.append(("CLK", "H%d" % k, 1, THETA - th[k], "imp"))       # readout pulse @ T_READ
        syn.append(("CLK", "C%d" % k, 3, 1.0, "imp"))                 # complement excitation (tick 7)
        syn.append(("H%d" % k, "C%d" % k, 1, INH, "imp"))            # H -| C (C = NOT H)
    syn.append(("START", "CLK", CLK_TICK, 1.0, "imp"))
    for a in range(1 << K):
        for k in range(K):
            bit = (a >> (K - 1 - k)) & 1
            syn.append((("H%d" % k) if bit else ("C%d" % k), "r%d" % a, 1, 1.0, "imp"))
        for j in range(Dout):
            dly = max(1, round(DBASE - KAPPA * V[a][j]))
            syn.append(("r%d" % a, "o%d" % j, dly, OW, "imp"))
    return neurons, syn


def input_fire(x):
    """latency-coded forced source spikes for input x in {-1,+1}^D."""
    fire = {"START": 0}
    for i, xi in enumerate(x):
        fire["x%d" % i] = ALPHA + xi
    return fire


def read_row(fired_at, K):
    """which single row neuron fired (the emergent selected address), or -1."""
    rows = [a for a in range(1 << K) if fired_at.get("r%d" % a) is not None]
    return rows[0] if len(rows) == 1 else -1


def count_size(neurons, synapses):
    return len(neurons), len(synapses)
