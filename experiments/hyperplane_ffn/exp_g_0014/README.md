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

## Status

All three built, smoke-tested at **27,343,200 params** each (6 CompressionMHL, 48 FastMultiHeadLut,
`forward_mode` hard live on all 48). Warmups for B and C complete and cached. Code committed before
any launch.
