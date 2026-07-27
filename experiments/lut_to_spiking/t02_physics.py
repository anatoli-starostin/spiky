"""t02 — the physics table of one Izhikevich neuron under impulse synapses.

(A) k coincident inputs of weight w  -> fire? when?      (threshold-k gates)
(B) two inputs of weight w, dt apart -> fire? when?      (integration window)
(C) excitation + inhibition          -> veto? for how long? (comparators)
"""
import torch
from snn_harness import Net

DEV = "cuda:0"
T = 96
DRIVE = 400.0          # input current that always makes a driver neuron fire at t+1


def build_fan(k, w, delays):
    """k driver neurons -> 1 target, each with weight w and its own delay."""
    net = Net(k + 1, device=DEV)
    for i in range(k):
        net.connect(i, k, w, delays[i])
    return net.build()


def A_threshold_k(max_k=8):
    print("=== (A) k coincident inputs of weight w -> target fire tick (drivers fire at t=1) ===")
    ws = [4, 6, 8, 10, 12, 15, 20, 30, 50, 100]
    print("      w:  " + "".join(f"{w:>6}" for w in ws))
    for k in range(1, max_k + 1):
        row = []
        for w in ws:
            net = build_fan(k, float(w), [3] * k)
            st = torch.full((1, k + 1), -1.0)
            st[0, :k] = 0.0
            first, raster, _ = net.run(st, n_ticks=T, amp=DRIVE)
            t = int(first[0, k])
            row.append(f"{t:>6}" if t >= 0 else "     .")
        print(f"  k={k:2}:  " + "".join(row))
    print("  ('.' = silent; drivers spike at tick 1, synapse delay 3 -> earliest target tick 5)")


def B_window(w=8.0, k=2):
    print(f"\n=== (B) two inputs of weight {w}, second arrives dt ticks later -> fire tick ===")
    for dt in range(0, 12):
        net = Net(3, device=DEV)
        net.connect(0, 2, w, 3)
        net.connect(1, 2, w, 3 + dt)
        net.build()
        st = torch.tensor([[0.0, 0.0, -1.0]])
        first, raster, _ = net.run(st, n_ticks=T, amp=DRIVE)
        t = int(first[0, 2])
        print(f"  dt={dt:2} -> {'silent' if t < 0 else f'fires at {t} (={t-5} after 1st arrival)'}")


def C_veto(w_exc=100.0):
    print(f"\n=== (C) excitation w={w_exc} vs inhibition w_inh, inhibition arrives dt before ===")
    for w_inh in [-20.0, -50.0, -100.0, -150.0, -200.0, -400.0]:
        row = []
        for dt in range(0, 5):
            # exc arrives at tick 1+5=6 ; inh arrives at 1+5-dt
            net = Net(3, device=DEV)
            net.connect(0, 2, w_exc, 5)
            net.connect(1, 2, w_inh, 5 - dt)
            net.build()
            st = torch.tensor([[0.0, 0.0, -1.0]])
            first, raster, _ = net.run(st, n_ticks=T, amp=DRIVE)
            t = int(first[0, 2])
            row.append(f"dt={dt}:{'.' if t < 0 else t}")
        print(f"  w_inh={w_inh:7} -> " + "  ".join(row))
    print("  ('.' = vetoed; dt=0 means inhibition lands on the same tick as excitation)")


if __name__ == "__main__":
    A_threshold_k()
    B_window()
    C_veto()
