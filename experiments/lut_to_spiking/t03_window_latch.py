"""t03 — (B') temporal integration window, and (D) the self-loop latch.

B' : does membrane voltage integrate ACROSS ticks?  Two inputs of weight w,
     dt ticks apart, where 2w is supra-threshold and w alone is not.
D  : a neuron with a supra-threshold self-synapse (delay 1) should fire on every
     tick once triggered -> a 1-neuron 'persistent inhibitor' (latch).
"""
import torch
from snn_harness import Net

DEV = "cuda:0"
T = 64
DRIVE = 400.0


def B_window():
    print("=== (B') temporal integration: two inputs of weight w, dt apart ===")
    for w in [10.0, 12.0, 15.0, 20.0]:
        row = []
        for dt in range(0, 10):
            net = Net(3, device=DEV)
            net.connect(0, 2, w, 3)
            net.connect(1, 2, w, 3 + dt)
            net.build()
            st = torch.tensor([[0.0, 0.0, -1.0]])
            first, _, _ = net.run(st, n_ticks=T, amp=DRIVE)
            t = int(first[0, 2])
            row.append(f"{dt}:{'.' if t < 0 else t}")
        print(f"  w={w:5} -> " + " ".join(row))
    print("  (first arrival at tick 5; '.' = silent)")


def D_latch():
    print("\n=== (D) self-loop latch: neuron 1 self-excites (w=100, delay=1) ===")
    net = Net(2, device=DEV)
    net.connect(0, 1, 100.0, 1)
    net.connect(1, 1, 100.0, 1)
    net.build()
    st = torch.tensor([[5.0, -1.0]])
    first, raster, n = net.run(st, n_ticks=40, amp=DRIVE)
    spikes = torch.nonzero(raster[0, 1] > 0).flatten().tolist()
    print(f"  trigger at t=5 -> latch spikes at ticks {spikes[:20]}{' ...' if len(spikes)>20 else ''}")
    print(f"  total latch spikes over 40 ticks: {len(spikes)}")


def D_ring(n_ring=4):
    print(f"\n=== (D') ring of {n_ring} neurons (delay 1 each) -> 1 spike every {n_ring} ticks ===")
    net = Net(1 + n_ring, device=DEV)
    net.connect(0, 1, 100.0, 1)
    for i in range(n_ring):
        net.connect(1 + i, 1 + (i + 1) % n_ring, 100.0, 1)
    net.build()
    st = torch.full((1, 1 + n_ring), -1.0)
    st[0, 0] = 5.0
    first, raster, _ = net.run(st, n_ticks=40, amp=DRIVE)
    tot = int((raster[0, 1:] > 0).sum())
    per = [torch.nonzero(raster[0, 1 + i] > 0).flatten().tolist()[:6] for i in range(n_ring)]
    print(f"  spikes per ring neuron (first 6): {per}")
    print(f"  total ring spikes over 40 ticks: {tot}  (vs {40-6} for a 1-neuron latch)")


if __name__ == "__main__":
    B_window()
    D_latch()
    D_ring()
