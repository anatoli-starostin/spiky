"""t01 — Phase-0 calibration.

(1) What input current makes a neuron fire, and after how many ticks?
(2) Does a synapse reproduce  out_latency == in_latency + delay  exactly?
(3) What synaptic weight is 'supra-threshold on its own' vs sub-threshold?
"""
import torch
from snn_harness import Net

DEV = "cuda:0"
T = 64


def sweep_input_amplitude():
    """One isolated neuron, kicked at tick 10 with various amplitudes."""
    net = Net(1, device=DEV)
    net.connect(0, 0, 0.0, 1)          # dummy self-synapse w=0 so the net has a synapse
    net.build()
    amps = [5, 10, 20, 40, 60, 80, 100, 150, 200, 300, 500]
    print("=== input amplitude -> first spike tick (kick at t=10) ===")
    for a in amps:
        st = torch.full((1, 1), 10.0)
        first, raster, n = net.run(st, n_ticks=T, amp=float(a))
        n_sp = int((raster > 0).sum().item())
        print(f"  amp={a:5} -> first_spike={int(first[0,0])}  total_spikes={n_sp}")


def sweep_synaptic_weight():
    """A -> B. A is kicked hard at t=10; sweep the A->B weight and the delay."""
    print("\n=== synaptic weight -> does B fire, and when? (delay=5) ===")
    for w in [5, 10, 20, 40, 60, 80, 100, 150, 200, 300, 400]:
        net = Net(2, device=DEV)
        net.connect(0, 1, float(w), 5)
        net.build()
        st = torch.tensor([[10.0, -1.0]])
        first, raster, _ = net.run(st, n_ticks=T, amp=400.0)
        print(f"  w={w:5} -> A first={int(first[0,0])}  B first={int(first[0,1])}"
              f"  B spikes={int((raster[0,1]>0).sum())}")


def delay_identity(w=400.0):
    """out_latency == in_latency + delay ?  Sweep delay and input tick."""
    print(f"\n=== out = in + delay, exactness check (w={w}) ===")
    delays = [1, 2, 3, 5, 8, 13, 21, 34]
    net = Net(1 + len(delays), device=DEV)
    for i, d in enumerate(delays):
        net.connect(0, 1 + i, w, d)
    net.build()
    in_ticks = [2, 5, 10, 17]
    st = torch.full((len(in_ticks), net.n_neurons), -1.0)
    for b, t0 in enumerate(in_ticks):
        st[b, 0] = t0
    first, raster, _ = net.run(st, n_ticks=T, amp=400.0)
    ok_all = True
    for b, t0 in enumerate(in_ticks):
        t_src = int(first[b, 0])
        row = []
        for i, d in enumerate(delays):
            t_dst = int(first[b, 1 + i])
            row.append(f"d={d}:{t_dst - t_src if t_dst >= 0 else 'X'}")
            ok_all &= (t_dst - t_src == d + 1) or (t_dst - t_src == d)
        print(f"  in_tick={t0:3} src_spike={t_src:3}  measured (dst-src): {' '.join(row)}")
    print(f"  -> constant offset across delays/input ticks: {ok_all}")


if __name__ == "__main__":
    torch.manual_seed(0)
    sweep_input_amplitude()
    sweep_synaptic_weight()
    delay_identity()
