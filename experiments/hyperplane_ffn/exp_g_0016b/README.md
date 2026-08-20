# exp_g_0014 / 0015 / 0016 — does autoencoder warmup of the LUT slot beat random init?

Tracking issue: **#108**. Exploratory 3-arm set. This README covers all three arms; each arm's
folder carries a copy.

| arm | folder | init |
|---|---|---|
| **A — control** | `exp_g_0014` | plain random init (reference) |
| **B — warmup-random** | `exp_g_0015` | slot warmed as an autoencoder on N(0,1) input |
| **C — warmup-structured** | `exp_g_0016` | slot warmed on **real captured activations** |

Base config is the lean tph64 hard arm (H8/d48/tph64/nap6, tied, learnable temps) — the same
geometry as `exp_g_0006` (1.228335 @ 27,343,200).

**All three share a byte-identical `train.py`** (`cmp`-verified), derived from `exp_n_0040`'s.
The arms differ **only in `config.json`**. With `lut_warmup_init: false` (arm A) the warmup code
is inert and the file behaves exactly like `exp_n_0040`'s. `src/spiky/lutorch/` is untouched.

## The warmup harness

A fresh slot `S` and a structurally identical mirror `M` (different seed, so not a twin) are
trained together so that `M(S(u)) ≈ u` under MSE — Adam, lr 1e-3, 500 steps, batch 4,096. `M` is
then discarded and `S`'s warmed weights initialise the real model's FFN slot.

**Per-layer, not clone.** Each of the 6 layers gets its own warmup, and in arm C each is warmed on
*that layer's own* captured input distribution. (The brief allowed warm-one-and-clone as a
simplification; per-layer was preferred and is what was done.)

**What this objective actually is.** `S` maps 384 → 384 and its internal LUT width is `H·d = 384`
too, so there is **no bottleneck** — this is not compression. The objective is "carry information
rather than destroy it": it pushes the LUT tables off near-zero random noise toward values that
make the slot invertible on the distribution it saw.

### Arm C: why a throwaway backbone

At init, real activations are near-Gaussian, so warming on them would be **indistinguishable from
arm B**. Arm C therefore first trains a **throwaway backbone for 1,000 steps** on real data, then
captures the `ln2(x)` tensor fed to each of the 6 slots (65,536 token rows per layer). Those
activations carry real manifold structure. **The backbone is then discarded** — the comparison run
starts fresh at step 0 with only the warmed LUT slots carried over.

That this matters is confirmed by the numbers, and in an instructive way:

```
captured activations: mean ≈ 0.0000, std ≈ 1.005–1.014   (per layer)
```

The captured inputs are **statistically almost identical to N(0,1) in their first two moments** —
yet they warm to a far lower floor. The entire difference between arms B and C is higher-order
structure, not scale.

## Warmup results (measured)

```
arm B (N(0,1) input)                    arm C (captured activations)
L0  0.999381 -> 0.360364   63.9%        L0  1.028551 -> 0.161760   84.3%
L1  1.000477 -> 0.358639   64.2%        L1  1.009908 -> 0.067543   93.3%
L2  0.999226 -> 0.361311   63.8%        L2  1.009923 -> 0.073809   92.7%
L3  0.999493 -> 0.361339   63.8%        L3  1.010134 -> 0.078799   92.2%
L4  1.000230 -> 0.361056   63.9%        L4  1.012150 -> 0.079984   92.1%
L5  1.001898 -> 0.362272   63.8%        L5  1.013941 -> 0.071686   92.9%
```

Both converged to a floor. **Real activations are far more reconstructible than isotropic
Gaussian** (~0.07 vs ~0.36), which is what you would expect if they lie on a lower-dimensional
manifold — the hard-routed LUT can capture that but cannot capture isotropic noise. Layer 0 is the
outlier in arm C (0.162), consistent with first-layer activations being less structured.

Per-step curves are in each arm's `warmup_mse.csv`.

## ⚠ Two things that are not free

**1. The zero-init collision — this is a second changed variable.**
`MinimalGPT` deliberately zero-inits every slot's `decompress.weight`, so the FFN residual branch
outputs exactly **0** at step 0. Every other arm in this sweep trains under that convention.
Loading a warmed `decompress` **destroys it** — the branch is live from step 0. So B and C differ
from A in *two* ways: warmed tables **and** a non-zero residual branch at init. If a warmup arm
wins, that alone will not say which change did it.

`warmup_load_decompress` (default `true`, per the brief) controls this. Setting it `false` keeps
compress+tables warmed with the zero-init intact, which is the strictly cleaner single-variable
comparison and is worth running if either arm shows an effect.

**2. The warmup is not free compute, though it is cheap.**
Arm B: 6 × 500 autoencoder steps. Arm C: additionally 1,000 backbone steps plus the capture pass.
Both are small against 16,000 training steps, but arm C's 1,000 backbone steps do mean it has
"seen" real data before its step 0 — the *tables* carry that, even though the backbone is thrown
away.

## What would count as a result

Control is arm A. The metric that matters is the **early-curve delta at steps 2,000–4,000**: a
better init should show up as faster early descent, and may wash out by 16k. `metrics.csv` logs
every 200 steps in all arms, so early curves are directly comparable.

Reference points: `exp_g_0006` (same geometry, random init) finished at **1.228335**; dense
baseline `exp073` is **1.196646**.

## Reproducing

```
PREPARE=1 python train.py   # arms B/C only: runs the warmup, writes warmed_init.pt
python train.py             # the 16k run (loads warmed_init.pt when warmup is on)
```
`warmed_init.pt` is gitignored (`experiments/**/*.pt`) and regenerable from the committed config.
`train.py` asserts that the cached file's `source` matches the config, so a stale arm-B cache
cannot silently be used for arm C.

## RESULT — warmup does not help. It buys an early lead and then pays it back with interest.

| arm | init | final val_bpb | vs control |
|---|---|--:|--:|
| **exp_g_0014** | A — random (control) | **1.228063** | — |
| **exp_g_0016b** | C — real-activation warmed, zero-init decompress | **1.236579** | **+0.008516** |
| **exp_g_0015** | B — N(0,1) warmed, warmed decompress | **1.239134** | **+0.011071** |
| exp_g_0015b | B — N(0,1) warmed, zero-init decompress | *skipped @ 8,400* | −0.001091 (ahead) |
| exp_g_0016 | C — warmed decompress | *never launched* | — |

All at 27,343,116 params, ~1.01 h each.

### The shape of the failure

`exp_g_0016b` is the cleanest arm — real-activation-warmed tables, zero-init decompress preserved,
single variable against the control. Its trajectory:

```
   400:  -0.046886   <- best lead
  2000:  -0.040217
  4000:  -0.005649
  6000:  +0.000624   <- crosses over
  8000:  +0.002411
 10000:  +0.006017
 12000:  +0.007477
 14000:  +0.008001
 16000:  +0.008516   <- final
```

**Ahead at 29/80 evals, behind at 51/80 — and behind at 51 of the 51 evals after the crossing.**
The deficit widens monotonically after step 6,000; it is not noise.

So warmup works exactly as advertised *early* — a −0.047 head start at step 400 is a large effect —
and that head start is not merely lost but **reversed**. Whatever the warmed tables encode helps
before the model has learned anything and actively constrains it afterwards.

### This falsifies the intermediate reading

Partway through this set, `exp_g_0015` (warmed decompress) was worse at 80/80 evals while
`exp_g_0015b` (zero-init decompress) was ahead at 42/42, and the natural conclusion — which was
recorded at the time — was that **the harm came from the zero-init collision, not from the warmed
tables**.

`exp_g_0016b` refutes that. It preserves the zero-init and *still* ends **+0.008516 worse**. The
flag accounts for only part of the gap:

```
warmed decompress   (exp_g_0015)   +0.011071
zero-init preserved (exp_g_0016b)  +0.008516
difference attributable to the flag   0.002555
```

The flag is real but secondary. **The warmed tables themselves are what cost the run**, and the
zero-init collision was a ~0.0026 aggravation on top.

It also means `exp_g_0015b` was stopped mid-reversal: it was ahead at 42/42 evals at step 8,400,
but `exp_g_0016b` had already crossed by step 6,000 and ended worse. On the family evidence,
`exp_g_0015b` was very likely heading the same way.

### Reconstruction quality bought nothing

Arm C's warmup reached a floor **~5× lower** than arm B's (MSE ~0.07 vs ~0.36 — real activations
are far more reconstructible than isotropic Gaussian). That bought essentially nothing downstream:

```
step    0016b (real acts)   0015b (N(0,1))
 200      -0.044969           -0.045674
 400      -0.046886           -0.044005
 600      -0.028017           -0.026443
 800      -0.019001           -0.020956
1000      -0.015945           -0.021072
```

Indistinguishable. **The early advantage comes from having non-random tables at all, not from how
well they invert their input.** A 5× better autoencoder produced no better initialisation.

### How strongly to hold this

- **Solid:** warmup is not a win at this budget. Two independent warmup sources, both worse than
  random init at 16k, with the better-controlled arm's deficit widening monotonically over its
  last 51 evals.
- **Single seed per arm**, as everywhere in this sweep. The +0.0085 is ~14× the ±0.0006 noise band
  used elsewhere here, so seed noise is an unlikely explanation for the sign.
- **Not tested:** whether a *smaller* warmup (fewer steps, or warming only `compress`) keeps the
  early gain without the late cost, and whether the crossing moves with the step budget. The early
  lead is genuinely large, so a shorter-budget regime might still favour it.

## Status

Complete. `exp_g_0014`, `exp_g_0015`, `exp_g_0016b` ran; `exp_g_0015b` and `exp_g_0016` were
stopped/skipped by decision and carry `SKIPPED.txt` rather than fabricated summaries.
