"""exp012: redo the inhibition probes at ORIGINAL Izhikevich current scale.

Hypothesis under test: the blow-up and the non-monotone hyperpolarization we measured are
artifacts of injecting I = +-200 into an explicit-Euler quadratic, far outside Izhikevich's
own regime (I ~ +10 excitatory, ~ -5 inhibitory, HELD rather than pulsed).

Same standalone replica of update_neuron_states_logic / detect_spikes_logic; engine untouched.

The distinction that matters for the conclusion:
  (a) MAGNITUDE problem -- blow-up / non-monotonicity -- would be fixed by small currents.
  (b) STRUCTURAL problem -- no after-hyperpolarization -- would NOT be, because the engine
      zeroes I every tick, so a synaptic input has no time constant at any scale.
"""
import numpy as np

CF2, CF1, CF0 = 0.04, 5.0, 140.0
A, B, C, D = 0.02, 0.2, -65.0, 8.0
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


def settle(n=200):
    V, U = C, B * C
    for _ in range(n):
        V, U, _ = tick(V, U, 0.0)
    return V, U


def main():
    V0, U0 = settle()
    print(f"rest: V = {V0:.4f}  U = {U0:.4f}   "
          f"regenerative threshold V* = {np.roots([CF2, CF1, CF0 - U0]).max():.2f}\n")

    # ---------------------------------------------------------------- 1 monotonicity
    print("1  SINGLE inhibitory pulse at tick 0, then I = 0   (V traces)")
    print(f"     {'I':>6s} | " + " ".join(f"{t:+4d}" for t in range(-1, 9)) + " |  min V   rec")
    rows = {}
    for amp in (-2.0, -5.0, -10.0, -20.0, -40.0, -100.0, -160.0, -200.0):
        V, U = V0, U0
        tr, rec, fired_at = [], None, None
        for t in range(-1, 9):
            V, U, f = tick(V, U, amp if t == 0 else 0.0)
            tr.append(V)
            if f and fired_at is None:
                fired_at = t
            if t >= 0 and rec is None and abs(V - V0) < 1.0:
                rec = t
        rows[amp] = (min(tr), rec, fired_at, U)
        flag = f"  FIRED t{fired_at}" if fired_at is not None else ""
        print(f"     {amp:6.0f} | " + " ".join(f"{v:6.1f}" for v in tr)
              + f" | {min(tr):7.1f}  {rec}{flag}")
    print("\n     monotonicity check (deeper trough for more negative I?)")
    prev = None
    for amp in (-2.0, -5.0, -10.0, -20.0, -40.0, -100.0, -160.0, -200.0):
        mn = rows[amp][0]
        tag = "" if prev is None else ("  ok" if mn < prev else "  <-- NON-MONOTONE")
        print(f"       I {amp:6.0f}  trough {mn:8.1f}{tag}")
        prev = mn
    print(f"     dU from the strongest small pulse (I=-40): "
          f"{rows[-40.0][3] - U0:+.4f}   (a spike adds d = {D})")

    # ---------------------------------------------------------------- 2 held-current regime
    print("\n2  HELD excitatory current (Izhikevich's own regime): natural spike latency")
    for iexc in (5.0, 10.0, 15.0, 20.0):
        V, U = V0, U0
        first = None
        for t in range(400):
            V, U, f = tick(V, U, iexc)
            if f:
                first = t
                break
        print(f"     I_exc = {iexc:5.1f} held -> first spike at tick {first}"
              + ("   (never within 400)" if first is None else ""))

    print("\n   coincidence test, HELD I_exc = +10 from tick 4; inhibition I = -5")
    print(f"     {'mode':>10s} {'delta':>6s} {'spike tick':>11s}   shift vs control")
    base = None
    for mode in ("control", "pulse", "held"):
        for delta in ([None] if mode == "control" else [-8, -4, -2, -1, 0, 1, 2, 4]):
            V, U = V0, U0
            first = None
            for t in range(400):
                I = 10.0 if t >= 4 else 0.0
                if mode == "pulse" and t == 4 + delta:
                    I += -5.0
                elif mode == "held" and t >= 4 + delta:
                    I += -5.0
                V, U, f = tick(V, U, I)
                if f:
                    first = t
                    break
            if mode == "control":
                base = first
                print(f"     {'control':>10s} {'-':>6s} {str(first):>11s}   (baseline)")
            else:
                sh = "never" if first is None else f"{first - base:+d}"
                print(f"     {mode:>10s} {delta:+6d} {str(first):>11s}   {sh}")

    # ---------------------------------------------------------------- 3 usability
    print("\n3  usability inside a 96-tick episode "
          "(readout window is ticks 64..95, input phase 0..31)")
    for iexc in (10.0, 15.0, 20.0, 50.0, 100.0, 200.0):
        V, U = V0, U0
        first = None
        for t in range(200):
            V, U, f = tick(V, U, iexc)
            if f:
                first = t
                break
        ok = first is not None and first <= 31
        print(f"     held I = {iexc:6.1f} -> latency {str(first):>5s} ticks   "
              f"{'usable as a one-hop relay' if ok else 'too slow / unusable'}")


if __name__ == "__main__":
    main()
