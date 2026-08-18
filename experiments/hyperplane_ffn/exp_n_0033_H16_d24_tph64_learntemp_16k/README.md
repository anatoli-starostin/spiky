# exp_n_0033 — H16/d24/tph128, learnable_temps=True, tied, 16k (A/B vs exp_n_0031)

IDENTICAL to **exp_n_0031** (H=16, inner d=24, tph=128, nap=6, tied, vanilla, 16k, std0.02 compress init)
EXCEPT **learnable_temps=True** on the FastMHL instances — so T_soft (soft_score_temp) and T_sel
(select_temp) become learnable nn.Parameters (one pair per head, init 0.5 via log-space), instead of the
current fixed non-trainable buffers.

**Motivation:** the strong PAST LUT results (the hyperplane/lutgpt line, e.g. exp010 = 1.19399 with
forward_mode=hard, soft_learnable_temps=true) all used learnable temps; the current CompressionMHL FFN-slot
line silently dropped it (FastMHL default False, never plumbed). This A/B tests whether re-enabling learnable
temps recovers loss. NOTE: forward_mode='hard', so the temps only shape the soft BACKWARD surrogate
(denom = T_soft+|d|, z = ts/T_sel) — making them learnable lets the gradient sharpness adapt per head.

**Plumbing:** added `learnable_temps: bool = False` to CompressionMultiHeadLUT.__init__ (default-preserving;
19/19 module tests pass) and forwarded it into the FastMHL `_lut_kw`; the trainer reads config
`lut_learnable_temps` (default False) and passes it. exp_n_0033 sets `lut_learnable_temps=True`.

**Params = 36,780,480 (SMOKE-confirmed)** = exp_n_0031's 36,780,288 + 192 learnable temp scalars
(16 heads × 2 temps × 6 layers). Negligible param delta, same ~4× cheaper FLOPs (H·d=384).

Runs 16k, serial after exp_n_0032. Compare: (a) vs exp_n_0031 (the fixed-temp twin) = the learnable-temps A/B;
(b) vs exp_n_0004 (1.21738); (c) vs tied dense 16k (1.19665).
