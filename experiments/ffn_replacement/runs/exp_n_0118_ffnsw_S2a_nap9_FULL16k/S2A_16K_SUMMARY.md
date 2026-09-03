# S2a nap9 — full 16k run (exp_n_0118) — results (task 4ce36b9a)

Full-length confirmation of the FFN-LUT sweep winner S2a: FFN CompressionMHL **H4 in48 out48 nap9
(512 cells) tph256**, untied head, standard LUT optimizer. Effective batch 24,576 (bs12×ga4) and LR
schedule family identical to exp_n_0084's 16k (lr 3e-4, warmup 1600, cosine over 16000); eval_every
200 == 0084 for step-aligned curves. 180.60M params / 2.286M FFN-FLOP / 2.375M vBW (= short-run S2a,
on the 0084 envelope, +2.2% FLOP).

## Headline
**final val_bpb 1.17460** — BEATS both targets:
- vs exp_n_0084 (dense-V FFN) **1.19866 → −0.02406**
- vs exp_n_0045 (tied) **1.1977 → −0.0231**
- best-ever S2a 1.17431 (~step 15600); both curves near-flat by 16k (converged).

**nap9 costs ~zero extra FLOP/vBW** (same envelope as 0084's FFN, +2.2% FLOP) — the −0.024 bpb is
bought purely with table cells (params: 180.6M vs 67.35M).

## Hypothesis test (is nap9's edge front-loaded or durable?) — REFUTED
Anatoli's hypothesis was "bigger nap mainly helps early convergence, advantage shrinks by the end."
The step-aligned gap (S2a − 0084) shows the **opposite** — the edge is smallest early and GROWS, then
HOLDS steady through the whole second half:

| window | mean gap (S2a − 0084) |
|---|---|
| early (≤4000) | −0.01865 |
| mid (4200–10000) | −0.02315 |
| late (10200–16000) | **−0.02439** |
| final @16000 | −0.02406 |

Trajectory: strong from step 200 (−0.013 → −0.025 by 1–2k), a brief **transient narrowing** around
step 2600–3400 (gap shrinks to ~−0.010…−0.016, 0084 momentarily catches up), then it **re-widens from
~4k onward and stabilizes at ≈−0.024 for the entire 10k–16k stretch**. So nap9's advantage is durable /
mildly back-loaded, not a front-loaded convergence-speed artifact.

(Note: the short-run −0.0275 was vs A0's own compressed-3500 schedule; the clean step-aligned gap vs
0084's true 16k curve is −0.0241 at the end — same order, and it holds.)

## Takeaway
A routed FFN (LUT) with a big table (nap9) genuinely **beats the dense-V-FFN reference by −0.024 bpb at
full length, at equal FLOP/vBW envelope**, spending only params. The nap (cells) lever is real and
durable — worth carrying into the next round (e.g. nap10, or nap9 at the cheaper Sweep-1 inner widths).

See `s2a_vs_0084_stepaligned.png`.
