"""exp012: measure the substrate's timing primitives before designing anything with them.

Three numbers decide whether a spike-order detector is buildable at all:

  1 how many ticks after a suprathreshold input arrives does an excitatory cell spike?
  2 the same for an inhibitory cell driven by an excitatory one
  3 the VETO WINDOW -- at what arrival offsets does an inhibitory spike actually stop an
    excitatory cell from firing? If inhibition only works when it lands in the same tick as
    the excitation, the veto is a knife-edge and nothing else in the design matters.

Synthetic inputs are used so the two spike times are set exactly rather than sampled.
"""
import numpy as np
import torch

import tiny_grow as G
from harness import T_IN, LatencyEncoder

IA, IB = 0, 16
N_EXC, N_INH = 4, 2


class FixedEncoder:
    """Hands back the tick matrix it was given, so spike times are exact."""

    def __init__(self, ticks):
        self.ticks = ticks

    def __call__(self, x):
        return self.ticks


def spikes_of(H, ticks, kind, current=200.0):
    """-> [B, n_neurons] first spike tick over the whole 96-tick episode, 96 = never."""
    from spiky.spnet.spnet import NeuronDataType
    sp = H["spnet"]
    B = ticks.shape[0]
    enc = FixedEncoder(ticks)
    G.grow_run_episode(H, np.zeros((B, G.N_IN)), enc, current=current)
    ids = torch.as_tensor(H["ids"][kind], dtype=torch.int32, device=H["device"])
    R = sp.export_neuron_data(ids, B, NeuronDataType.Spike, 0, G.N_TICKS - 1)
    R = R.reshape(B, -1, G.N_TICKS)
    w = torch.arange(G.N_TICKS, 0, -1, device=R.device, dtype=R.dtype)
    first = G.N_TICKS - (R.ne(0) * w).amax(-1)
    return first.cpu().numpy()


def main():
    G.set_hidden_capacity(N_EXC, N_INH)
    G.set_out_per_target(1, "mean")
    G.set_weight_levels([-1.0] + [round(0.1 * i, 10) for i in range(11)])
    G.set_delay_levels(None)
    G.QUANTIZED = True
    G.FANOUT_CAP = None
    G.MAX_EPISODE_BATCH = 512
    GAIN = 200.0

    def g_drive(d_in=5, w=1.0):
        """in0 -> E0 only. Measures the excitatory response latency."""
        g = G.blank(n_exc=N_EXC, n_inh=N_INH)
        g["mask"][IA, 0] = True
        g["delay"][IA, 0] = d_in
        g["weight"][IA, 0] = w
        g["gain"] = GAIN
        g["inh_coeff"] = 1.0
        return G.enforce(g)

    B = 8
    ticks = np.zeros((B, G.N_IN), np.int64) + (T_IN - 1)
    ticks[:, IA] = np.arange(B) * 3                      # input 0 spikes at 0,3,6,...
    H = G.build([g_drive()], device="cuda")
    fe = spikes_of(H, ticks, 0)
    lat_e = fe[:, 0] - (ticks[:, IA] + 5)
    print("1 EXC latency: input arrives at t+5, E0 first-spikes at", fe[:, 0].tolist())
    print("   latency after arrival:", lat_e.tolist(), " -> lat_exc =",
          int(np.median(lat_e[np.isfinite(lat_e)])))
    del H
    torch.cuda.empty_cache()
    LE = int(np.median(lat_e))

    # ---- 2 the E -> I latency
    def g_ei(d_in=5, d_ei=3):
        g = g_drive(d_in)
        r_e0 = G.N_IN + 0
        g["mask"][r_e0, G.N_EXC_MAX + 0] = True
        g["delay"][r_e0, G.N_EXC_MAX + 0] = d_ei
        g["weight"][r_e0, G.N_EXC_MAX + 0] = 1.0
        return G.enforce(g)

    H = G.build([g_ei()], device="cuda")
    fi = spikes_of(H, ticks, 1)
    fe2 = spikes_of(H, ticks, 0)
    lat_i = fi[:, 0] - (fe2[:, 0] + 3)
    print(f"\n2 INH latency: I0 first-spikes at {fi[:, 0].tolist()}")
    print(f"   latency after arrival: {lat_i.tolist()} -> lat_inh ="
          f" {int(np.median(lat_i))}")
    del H
    torch.cuda.empty_cache()
    LI = int(np.median(lat_i))

    # ---- 3 THE VETO WINDOW. E0 is driven by in0; I0 is driven by in16 via E1, and the
    # inh->exc delay is pinned to 1. Sweep the arrival offset and see when E0 goes silent.
    def g_veto(dA, dB, dEI, coeff, w_inh=-1.0):
        g = G.blank(n_exc=N_EXC, n_inh=N_INH)
        for r, c, d, w in ((IA, 0, dA, 1.0), (IB, 1, dB, 1.0),
                           (G.N_IN + 1, G.N_EXC_MAX + 0, dEI, 1.0),
                           (G.N_IN + G.N_EXC_MAX + 0, 0, 1, w_inh)):
            g["mask"][r, c] = True
            g["delay"][r, c] = d
            g["weight"][r, c] = w
        g["gain"] = GAIN
        g["inh_coeff"] = coeff
        return G.enforce(g)

    print("\n3 VETO WINDOW  (E0 driven at a fixed tick; the veto's arrival is swept)")
    print("   offset = veto arrival tick − excitation arrival tick;  '.' = E0 fired, "
          "'X' = vetoed")
    dA, dB, dEI = 20, 2, 2
    exc_arrival = 0 + dA                                  # in0 spikes at tick 0
    best = {}
    for coeff in (1.0, 2.0, 4.0):
        line = []
        for off in range(-6, 7):
            # veto arrives at t16 + dB + LE + dEI + LI + 1
            t16 = exc_arrival + off - (dB + LE + dEI + LI + 1)
            if not (0 <= t16 <= T_IN - 1):
                line.append(" ")
                continue
            tk = np.zeros((1, G.N_IN), np.int64) + (T_IN - 1)
            tk[0, IA] = 0
            tk[0, IB] = t16
            H = G.build([g_veto(dA, dB, dEI, coeff)], device="cuda")
            f = spikes_of(H, tk, 0)
            fired = f[0, 0] < G.N_TICKS
            line.append("." if fired else "X")
            best.setdefault(coeff, []).append((off, bool(fired)))
            del H
            torch.cuda.empty_cache()
        print(f"   inh_coeff {coeff:4.1f}   offsets -6..+6:  {''.join(line)}")

    print(f"\n   lat_exc {LE}   lat_inh {LI}   total veto path = dB + {LE} + dEI + {LI} + 1")


if __name__ == "__main__":
    main()
