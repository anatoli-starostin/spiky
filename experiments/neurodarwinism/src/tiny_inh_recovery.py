"""exp012: how long does an excitatory cell stay hyperpolarized after one inhibitory pulse?

Standalone numpy replica of update_neuron_states_logic / detect_spikes_logic from
native/spiky/spnet/spnet_runtime_kernels_logic.proto. Nothing here touches the engine.

Per tick, in the engine's own order:
    detect_spikes      if V >= 30 -> emit (tests the V left by the previous tick)
    update_neuron_states
        if V >= 30:  V = c;  U += d                      (reset)
        2x:          V += 0.5 * ((0.04*V + 5)*V + 140 - U + I)
        U += a*(b*V - U)
        I := 0                                           (current does NOT persist)
"""
import numpy as np

CF2, CF1, CF0 = 0.04, 5.0, 140.0
A, B, C, D = 0.02, 0.2, -65.0, 8.0          # meta type 0, the excitatory cell
TH = 30.0
N_EULER, DT = 2, 0.5


def tick(V, U, I):
    fired = V >= TH
    if fired:
        V, U = C, U + D
    for _ in range(N_EULER):
        V = V + DT * ((CF2 * V + CF1) * V + CF0 - U + I)
    U = U + A * (B * V - U)
    return V, U, fired


def settle(n=50):
    V, U = C, B * C
    for _ in range(n):
        V, U, _ = tick(V, U, 0.0)
    return V, U


def main():
    V0, U0 = settle()
    print(f"resting state after 50 quiet ticks:   V = {V0:.4f}   U = {U0:.4f}")
    roots = np.roots([CF2, CF1, CF0 - U0])
    print(f"fixed points of 0.04V^2+5V+140-U at this U:  V* = {np.round(roots, 2).tolist()}"
          f"   (upper root = the regenerative threshold)\n")

    # ---------------------------------------------------------------- 2 recovery trace
    for amp in (-100.0, -160.0, -200.0):
        V, U = V0, U0
        print(f"single inhibitory pulse I = {amp:.0f} at tick 0, then I = 0")
        print(f"   tick   {'V':>10s} {'U':>9s}   |V - V_rest|")
        rec = None
        for t in range(-1, 9):
            I = amp if t == 0 else 0.0
            V, U, f = tick(V, U, I)
            dv = abs(V - V0)
            if t >= 0 and rec is None and dv < 1.0:
                rec = t
            print(f"   {t:+4d}   {V:10.3f} {U:9.3f}   {dv:9.3f}"
                  + ("   <- within 1.0 of rest" if rec == t else "")
                  + ("   SPIKED" if f else ""))
        print(f"   -> recovers to within 1.0 of rest after {rec} tick(s)\n")

    # ---------------------------------------------------------------- 3 coincidence test
    print("coincidence test: excitation I = +200 at tick E = 4; one inhibitory pulse "
          "I = -160 at E + delta")
    print(f"   {'delta':>6s}  {'spikes?':>8s}  {'spike tick':>11s}   V trace ticks 2..10")
    base = None
    for delta in (-3, -2, -1, 0, 1, 2, 3, None):
        V, U = V0, U0
        spike_tick, trace = None, []
        for t in range(0, 12):
            I = 0.0
            if t == 4:
                I += 200.0
            if delta is not None and t == 4 + delta:
                I += -160.0
            V, U, f = tick(V, U, I)
            if f and spike_tick is None:
                spike_tick = t
            if 2 <= t <= 10:
                trace.append(round(V, 1))
        lab = "none" if delta is None else f"{delta:+d}"
        if delta is None:
            base = spike_tick
        shift = "" if spike_tick is None or base is None else f"  ({spike_tick - base:+d})"
        print(f"   {lab:>6s}  {str(spike_tick is not None):>8s}  "
              f"{str(spike_tick):>11s}{shift}   {trace}")
    print(f"\n   (delta = 'none' is the control: excitation alone, spike at tick {base})")


if __name__ == "__main__":
    main()
