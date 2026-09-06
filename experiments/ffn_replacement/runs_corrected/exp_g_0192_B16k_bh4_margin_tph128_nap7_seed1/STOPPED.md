# exp_g_0192 — STOPPED BY REQUEST at step 10,500 of 16,000. No final score.

**Read this before using any number from this folder.**

This run was **stopped on request**; it did not finish and was not killed by a fault. There
is therefore **no `summary.json`, no final score, and no `checkpoint.pt`** — the trainer
saves the checkpoint only after the last step, so nothing survives to re-score or resume
from. What remains is `config.json`, `metrics.csv` and `train.py`.

Last logged training step 10,800; last completed evaluation step 10,500.

## The design

BH4 addressing (`lut_impl="bh4"`): **both** `compress` and the anchor-pair addressing are
replaced. **There are no anchor pairs at all in this model.** The projection is BH4 — four
learnable block-diagonal factors interleaved with fixed Walsh–Hadamard mixes — and the
address is the sign of the projected **coordinates** rather than the sign of coordinate
**differences**. `decompress` is unchanged and still sits on top, and the tables keep our
narrow-rows-plus-decompress layout rather than LookupFFN's full-width rows. The score is
`margin`, which is algebraically LookupFFN's own score.

4 heads × 128 tables/head × 2^7 = 128 cells, NAP = 7. Each head has its own square BH4 at
working width 1024 (the first power of two at or above its 128 × 7 = 896 code
coordinates), with x zero-padded into it. BH4 costs 65,536 parameters per layer against a
parity target of 82,112 (compress 73,728 + bias 192 + anchor index buffers 8,192).

**48,427,008 parameters — −28.1% against the ladder's 67,352,256.**

## Two initialisation defects the sign-constancy diagnostic caught

Both were found by measurement before launch, and either alone would have wasted the run.

**1. Hadamard involutivity restored the padding.** The normalised Walsh–Hadamard transform
satisfies `H H = I`, so with the reference's near-identity initialisation of the `B_i` the
whole product collapses to `R ≈ H^4 = I`: BH4 at initialisation returns its input,
*including the zero padding*. About 512 of each head's 896 code coordinates were therefore
sitting on padded zeros, and the pooled code std of 0.65 ≈ √(384/896) is the fingerprint.
**Fix:** one fixed Hadamard applied to the padded input *outside* the learnable product, so
the `H^4 = I` cancellation can no longer restore the padding structure. Code std is now
0.606 ≈ √(384/1024) at every layer.

**2. The `decompress` BIAS produced a token-independent offset.** `model_build` zeroes the
decompress *weight* but not its *bias*, so every block emitted a constant vector (norm 0.82
at init) that accumulated down the residual stream. LayerNorm removes each token's mean
across dimensions, not a direction shared by every token. **Anchor-pair addressing cancels
exactly such an offset inside `d = z[a] − z[b]`, and silently did so for three previous
runs; coordinate-signing cannot, and dies on it.** Measured at initialisation on real
tokens, by depth:

| | L0 | L1 | L2 | L3 | L4 | L5 |
|---|---|---|---|---|---|---|
| constant-sign fraction, as built | 0.0000 | 0.1415 | 0.2846 | 0.3809 | 0.4325 | 0.4933 |
| distinct addresses / 128 | 127.8 | 38.5 | 18.5 | 17.2 | 10.2 | 9.8 |
| constant-sign fraction, bias zeroed | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| distinct addresses / 128 | 127.8 | 127.6 | 127.6 | 127.7 | 127.4 | 127.6 |

**Fix:** zero the decompress bias on this path — which is what the residual-branch zero-init
was supposed to achieve anyway, and what the dense baseline gets for free by using
`bias=False`. Scoped to `lut_impl="bh4"`, so every existing config still builds
bit-identically.

## Results

**The primary reference is exp_g_0191, not exp_g_0190.** exp_g_0191 is Light with the same
`margin` score, so comparing against it isolates the one thing this run changes — the
addressing. exp_g_0190 uses `bounded_norm` and therefore confounds addressing with the
score form.

| step | 0192 BH4 | 0191 Light+margin | gap vs 0191 | 0190 Light+bnorm | gap vs 0190 |
|---:|---:|---:|---:|---:|---:|
| 500 | 2.029904 | 1.974689 | +0.055215 | 2.009685 | +0.020219 |
| 1000 | 1.778964 | 1.728351 | +0.050613 | 1.759708 | +0.019256 |
| 1500 | 1.642208 | 1.600157 | +0.042051 | 1.621417 | +0.020791 |
| 2000 | 1.547904 | 1.507783 | +0.040121 | 1.532308 | +0.015596 |
| 2500 | 1.488624 | 1.432570 | +0.056054 | 1.476238 | +0.012386 |
| 3000 | 1.436983 | 1.384851 | +0.052132 | 1.431477 | +0.005506 |
| 3500 | 1.394982 | 1.352502 | +0.042480 | 1.388485 | +0.006497 |
| 4000 | 1.364515 | 1.327772 | +0.036743 | 1.359614 | +0.004901 |
| 4500 | 1.343402 | 1.309083 | +0.034319 | 1.336276 | +0.007126 |
| 5000 | 1.324696 | 1.293805 | +0.030891 | 1.320002 | +0.004694 |
| 5500 | 1.309525 | 1.279280 | +0.030245 | 1.303507 | +0.006018 |
| 6000 | 1.296275 | 1.267493 | +0.028782 | 1.292935 | +0.003340 |
| 6500 | 1.284939 | 1.257209 | +0.027730 | 1.281633 | +0.003306 |
| 7000 | 1.276162 | 1.249534 | +0.026628 | 1.272989 | +0.003173 |
| 7500 | 1.267051 | 1.240594 | +0.026457 | 1.264842 | +0.002209 |
| 8000 | 1.259685 | 1.233237 | +0.026448 | 1.257097 | +0.002588 |
| 8500 | 1.252293 | 1.226526 | +0.025767 | 1.251059 | +0.001234 |
| 9000 | 1.246797 | 1.220101 | +0.026696 | 1.244256 | +0.002541 |
| 9500 | 1.240637 | 1.214148 | +0.026489 | 1.238944 | +0.001693 |
| 10000 | 1.235493 | 1.209062 | +0.026431 | 1.233911 | +0.001582 |
| 10500 | 1.230552 | 1.204351 | +0.026201 | 1.229051 | +0.001501 |

At the last shared evaluation (step 10,500): **+0.026201 against exp_g_0191** and
+0.001501 against exp_g_0190.

The gap to exp_g_0191 is flat: it settles near +0.026 by step 4,000 and does not move for
the following 6,500 steps.

## Comparability — read before drawing a conclusion

This run is **not parameter-comparable** to the 0189/0190/0191 ladder, and the reason is
**not** BH4. At −28.1% the dominant term is `nap 8 → 7`, which halves every table
(6,291,456 → 3,145,728 per layer, −18,874,368 across the model). BH4 itself is small by
comparison. By our measured budget law of −0.007455 bpb per doubling of table parameters,
**halving the tables alone predicts about +0.0075 bpb of handicap** — 2.2× the 0.00335 seed
spread — before the addressing is judged at all.

Parameter-adjusted parity with exp_g_0191 therefore means landing within about +0.0075 of
it, i.e. roughly **1.186 at step 15,000** — *not* near exp_g_0190's 1.205.

Taking the handicap at face value, the addressing gap after adjustment is about
**+0.0187** at step 10,500, which is still ~5.6× the seed spread. On
this evidence coordinate-sign addressing is behind anchor-pair addressing at a matched
score form — but the run was stopped, the handicap is an extrapolation from a fitted law
rather than a measurement of this configuration, and one seed cannot settle it.

**The right control does not exist yet: a Light run at nap=7, tph=128, `margin`** — the
identical table budget and score, with only the projection and addressing differing. That
is the run that would isolate BH4 cleanly, and it costs about an hour.

## Throughput note

The reference `fwht` is a Python loop over log2(d) butterfly stages, each allocating an
`[N, H, d]` temporary; that made the layer memory-bound at 2.2 s/step against Light's
0.203. We replaced it with a GEMM against the Hadamard matrix — built *by* `fwht`, and
verified equal in forward and both gradients to 1e-13 — giving 0.571 s/step, 3.85× faster.
`torch.compile` compiles the module cleanly (1 graph, 0 graph breaks) but adds nothing.

**Consequently this run computes an O(d log d) structured transform as a dense GEMM,
because that is faster at our shapes. Its wall-clock therefore says nothing whatsoever
about BH4's deployment cost** — only about its quality. The paper's FLOP argument concerns
the butterfly, which we did not use.
