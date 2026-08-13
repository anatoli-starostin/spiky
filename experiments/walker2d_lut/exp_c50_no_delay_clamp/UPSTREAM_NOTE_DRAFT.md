# DRAFT — not sent

Note for nucstar on `LIFMultiHeadLUT`, branch `exp/lif-detectors-mhl` @ `24c0e60a`.
Drafted at Anatoli's instruction; **nothing has been sent, and nothing is committed.**

---

**Subject: `clamp(delay, 0, t_window)` silently kills the delay parameterisation when delays initialise at zero**

In `LIFMultiHeadLUT._membrane`:

```python
a = lat.view(B, 1, 1, -1) + torch.clamp(self.delay, 0.0, self.t_window).unsqueeze(0)
```

Below the floor, `torch.clamp` returns 0 in the forward **and** the gradient is exactly 0.
A delay pushed negative therefore has no path back — it is frozen at the floor for the rest
of training. That is fine if delays live safely inside the window, but the module's own
default is `delay_init_std=0.0`, which starts **every** delay exactly *on* the floor, where
roughly half the first updates push them across.

**Measured, on Walker2d SAC, 1 head × 128 tables × 1 detector × 16 buckets, 2,176 delays,
10,000 iterations, 3 seeds:**

| | learned delay range | delays ≤ 0 (dead: value and gradient both zero) |
|---|---|---:|
| `clamp(delay, 0, t_window)` | −0.006 … +6.7 / +11.3 / +10.1 | **94.6% / 94.9% / 94.9%** |
| `clamp(delay, -inf, t_window)` | −9.50 … +10.55 | 0% |
| predecessor module, no clamp | −10.09 … +12.67 | 0% |

The front-end's effective delay capacity collapses from 2,176 parameters to about 100. The
entries at −0.006 are not "nearly zero delays" — they are delays that crossed the floor
early and then received exactly zero gradient for the remaining ~9,900 iterations.

**Removing only the lower bound restores the predecessor's learned distribution exactly.**
Keeping the upper `t_window` cap (which is what holds arrivals inside `[·, 2·t_window]` so
`exp(a/tau)` stays float32-safe in the cumsum membrane) and dropping only the non-negativity
floor gives, seed for seed against the old unclamped module: means 0.533/0.464/0.505 vs
0.542/0.460/0.612, sds 1.910/1.915/1.982 vs 1.895/1.877/2.094, negative fractions
37.4/41.5/41.1% vs 37.7/40.8/38.9%. No delay ends on the retained upper cap.

**On the causality motivation.** The floor enforces "no synapse arrives before its latency
code". But the timeline origin is arbitrary: any learned delay tensor can be renormalised by
a single global shift so its minimum is 0, preserving every relative arrival and every
downstream comparison. So the floor forbids no reachable configuration — it only removes the
gradient of every delay that crosses it on the way there.

**Suggested fix, in order of preference:**

1. Drop the lower bound: `torch.clamp(self.delay, float("-inf"), self.t_window)`, or
   equivalently `torch.minimum(self.delay, t_window)`. Keeps the float32 safety, removes the
   trap.
2. If the floor must stay, at minimum change the default `delay_init_std` away from `0.0`,
   or add a positive `delay_init_const`, so training does not begin on the boundary. This is
   a strictly weaker fix — it makes the trap rarer, not absent.
3. A straight-through clamp (clamp the value, pass the gradient) would also work, but is a
   larger semantic change than either of the above.

**On the downstream effect.** Return went from 2233 ± 1259 (takeoff 1/3) with the floor to
3108 ± 1729 (takeoff 2/3) without it, against 4246 ± 298 for the predecessor module on the
same three seeds. Two of the three seeds recovered to within 5% of the predecessor. At n=3
that return-level comparison is **not** statistically decisive (|t| 0.71) and I would not
quote it as a headline; the parameter-level measurement above — 94.6–94.9% of 2,176 delays
dead, on every seed — is the part that stands on its own.

**Reproduction.** The gradient claim is directly testable without any training: set
`delay_init_std=0`, take one backward pass, and count `(delay < 0) & (delay.grad == 0)`. In
our parity harness, over a perturbed parameter set with 1,148 negative delays, the floored
module gives **1,148 of 1,148** with gradient exactly 0.0 and the floorless module gives
**0 of 1,148**.
