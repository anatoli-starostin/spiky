"""t04 — turn the Izhikevich neuron into an IDEAL non-leaky integrate-and-fire neuron.

The Izhikevich update in the engine is
    v += dt * (cf_2*v^2 + cf_1*v + cf_0 - u + I)      (2 Euler steps, dt=0.5)
    u += a * (b*v - u)
    on spike: v = c, u += d

Setting cf_2 = cf_1 = cf_0 = 0, a = 0, b = 0, d = 0, c = 0 gives
    v += I  (per tick, exactly),  u == 0 forever
i.e. a perfect leak-free integrator with threshold `spike_threshold` and reset to 0.

That is the primitive we actually want for latency coding:
  * a single input of weight >= theta fires the neuron exactly 1 tick later,
  * k inputs of weight theta/k form an exact threshold-k gate with NO time limit
    between arrivals (no leak = unlimited integration window),
  * a negative weight is a PERMANENT cancellation -> a 1-neuron spike-order
    comparator with no latch and no relay chain.
"""
import torch
from spiky.spnet.spnet import NeuronMeta
from snn_harness import Net

DEV = "cuda:0"
THETA = 100.0
DRIVE = 200.0     # >= THETA, so an injected current always fires the neuron next tick

IF = NeuronMeta(neuron_type=0, cf_2=0.0, cf_1=0.0, cf_0=0.0,
                a=0.0, b=0.0, c=0.0, d=0.0, spike_threshold=THETA)


def net(n):
    return Net(n, neuron_meta=IF, device=DEV)


def A_exactness():
    print("=== (A) IF: single supra-threshold synapse, out = in + delay + 1 ? ===")
    delays = [1, 2, 3, 5, 8, 13, 21, 40, 100]
    g = net(1 + len(delays))
    for i, d in enumerate(delays):
        g.connect(0, 1 + i, THETA, d)
    g.build()
    st = torch.full((3, g.n_neurons), -1.0)
    for b, t0 in enumerate([0, 7, 20]):
        st[b, 0] = t0
    first, _, _ = g.run(st, n_ticks=200, amp=DRIVE)
    ok = True
    for b in range(3):
        ts = int(first[b, 0])
        offs = [int(first[b, 1 + i]) - ts for i in range(len(delays))]
        ok &= all(o == d + 1 for o, d in zip(offs, delays))
        print(f"  src fires at {ts:3}; (dst - src) = {offs}  expected {[d+1 for d in delays]}")
    print(f"  -> exact for all delays / all input ticks: {ok}")


def B_threshold_k(max_k=8):
    print("\n=== (B) IF: threshold-k gate with weight theta/k, arrivals spread over dt ===")
    for k in [2, 4, 8]:
        w = THETA / k
        row = []
        for dt in [d for d in [0, 1, 5, 20, 60] if 3 + (k - 1) * d <= 255]:
            g = net(k + 1)
            for i in range(k):
                g.connect(i, k, w, 3 + i * dt)
            g.build()
            st = torch.full((2, k + 1), -1.0)
            st[0, :k] = 0.0            # all k drivers fire  -> should spike
            st[1, :k - 1] = 0.0        # only k-1 drivers    -> must stay silent
            first, _, _ = g.run(st, n_ticks=256, amp=DRIVE)
            t_all, t_part = int(first[0, k]), int(first[1, k])
            last_arrival = 1 + 3 + (k - 1) * dt + 1
            row.append(f"dt={dt}:{'OK' if (t_all == last_arrival and t_part < 0) else f'BAD({t_all},{t_part})'}")
        print(f"  k={k} (w={w:g}): " + "  ".join(row))
    print("  OK = fires exactly 1 tick after the k-th arrival, and stays silent with k-1 inputs")


def C_comparator():
    print("\n=== (C) IF: 1-neuron spike-order comparator  bit = 1[t_a > t_b] ===")
    #   C  <- b  with +theta      (b earlier => fire)
    #   C  <- a  with -2*theta    (a earlier or equal => permanently vetoed)
    g = net(3)          # 0 = a, 1 = b, 2 = C
    g.connect(1, 2, THETA, 2)
    g.connect(0, 2, -2 * THETA, 2)
    g.build()
    W = 12
    ta, tb, exp, got = [], [], [], []
    st = []
    for i in range(W):
        for j in range(W):
            st.append([float(i), float(j), -1.0])
            ta.append(i); tb.append(j); exp.append(1 if i > j else 0)
    first, raster, _ = g.run(torch.tensor(st), n_ticks=64, amp=DRIVE)
    got = [(1 if int(first[n, 2]) >= 0 else 0) for n in range(len(st))]
    bad = [(ta[n], tb[n], exp[n], got[n]) for n in range(len(st)) if exp[n] != got[n]]
    print(f"  exhaustive over t_a, t_b in [0,{W}): {len(st)} cases, {len(bad)} mismatches")
    if bad:
        print("   first mismatches:", bad[:10])
    # latency of the '1' answer
    lat = [int(first[n, 2]) - tb[n] for n in range(len(st)) if got[n] == 1]
    print(f"  when it fires, latency after t_b is constant: {set(lat)}")
    print(f"  cost: 1 neuron, 2 synapses, <=1 spike per comparison")


if __name__ == "__main__":
    A_exactness()
    B_threshold_k()
    C_comparator()
