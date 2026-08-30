# exp_g_0050 — function-emitting cells, 12 gaussians per cell (STAGED, NEVER RUN)

Built, smoked and measured; never launched. Half of exp_g_0048's K, with `fe_k` as the only
change — 36 params per cell against 384, **10.67x fewer**, 57,026,316 params total.
Measured 1.288 s/step and 11.78 GiB, against K=24's 1.525 s/step under identical
methodology.

## Two findings from building it

**Halving K barely helps speed.** 1.288 against 1.525 s/step is only ~16%, not the ~2x one
might expect, because `torch.compile` already fused the synthesis away — the gather+sum now
dominates and it is K-independent. After the compile fix, K is a capacity and parameter
knob, **not** a speed knob.

**The requested "signed scale per bump" already existed.** The design change asked for was
fewer bumps plus a trainable signed scale so a bump could switch off or invert. `amp`
already is exactly that: it is an unconstrained signed real, no abs/softplus/exp is applied
to it anywhere, it enters the synthesis as a bare multiplicative factor, it initialises ~50%
negative, and in a direct test a fitted bump was driven from +0.5 **through zero to −1.0** to
match a downward target. A separate scale would make each bump `amp * scale`, a
reparameterisation of one signed coefficient and identical in function space, so it was
deliberately not added.

So this fork is strictly *less* expressive than exp_g_0048 with nothing gained in exchange —
worth knowing before spending the 1.43 h it would take, given exp_g_0048 was halted for
trailing.
